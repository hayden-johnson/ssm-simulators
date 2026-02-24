# Global settings for cython
# cython: cdivision=True
# cython: wraparound=False
# cython: boundscheck=False
# cython: initializedcheck=False

"""
Bayesian Poisson model simulator.

This simulator implements an N-choice model where each choice corresponds to a
Poisson process (neuron) with baseline rate ``noise``. For each sample, one
latent "true" choice is drawn from ``prior`` and receives an additional
choice-specific ``signal`` boost. A Bayesian decoder tracks posterior
probabilities over hypotheses and terminates when posterior entropy drops below
``entropy_threshold``.
"""

import numpy as np
cimport numpy as cnp
from libc.math cimport log as c_log
from libc.math cimport exp as c_exp
from libc.math cimport lgamma as c_lgamma

# Import utility functions from the _utils module
from cssm._utils import (
    set_seed,
    compute_smooth_unif,
    compute_deadline_tmp,
    build_param_dict_from_2d_array,
    build_full_metadata,
    build_minimal_metadata,
    build_return_dict,
)

DTYPE = np.float32
cdef double C_NEG_INF = -1.0e300


def _as_2d_float_array(arr, name):
    """Convert input to a 2D float32 array."""
    out = np.asarray(arr, dtype=DTYPE)
    if out.ndim == 0:
        out = np.expand_dims(out, axis=0)
    if out.ndim == 1:
        out = np.expand_dims(out, axis=1)
    if out.ndim != 2:
        raise ValueError(f"{name} must be a scalar, 1D, or 2D array.")
    return out


def _broadcast_to_choices(arr, name, n_trials, n_choices):
    """Broadcast a parameter to shape (n_trials, n_choices)."""
    out = _as_2d_float_array(arr, name)
    if out.shape[0] != n_trials:
        raise ValueError(
            f"{name} first dimension ({out.shape[0]}) does not match n_trials ({n_trials})."
        )
    if out.shape[1] == n_choices:
        return out
    if out.shape[1] == 1:
        return np.tile(out, (1, n_choices)).astype(DTYPE)
    raise ValueError(
        f"{name} second dimension must be 1 or n_choices ({n_choices}), got {out.shape[1]}."
    )


def _normalize_prior(prior, n_trials, n_choices):
    """Normalize prior to shape (n_trials, n_choices)."""
    if prior is None:
        return np.full((n_trials, n_choices), 1.0 / float(n_choices), dtype=np.float64)

    prior_arr = np.asarray(prior, dtype=np.float64)
    if prior_arr.ndim == 0:
        raise ValueError("prior must be 1D or 2D when provided.")
    if prior_arr.ndim == 1:
        prior_arr = np.expand_dims(prior_arr, axis=0)
    if prior_arr.ndim != 2:
        raise ValueError("prior must be 1D or 2D.")
    if prior_arr.shape[1] != n_choices:
        raise ValueError(
            f"prior must have length n_choices ({n_choices}), got {prior_arr.shape[1]}."
        )
    if prior_arr.shape[0] == 1 and n_trials > 1:
        prior_arr = np.tile(prior_arr, (n_trials, 1))
    elif prior_arr.shape[0] != n_trials:
        raise ValueError(
            f"prior first dimension ({prior_arr.shape[0]}) does not match n_trials ({n_trials})."
        )
    if (prior_arr < 0).any():
        raise ValueError("prior must be non-negative.")
    if not np.isfinite(prior_arr).all():
        raise ValueError("prior must contain only finite values.")

    row_sums = prior_arr.sum(axis=1, keepdims=True)
    if (row_sums <= 0).any():
        raise ValueError("Each prior row must sum to a positive value.")

    return prior_arr / row_sums


cdef inline double _log_poisson_pmf_fast(
    long long k,
    double lam,
    double log_lam,
    double lgamma_kp1,
):
    """Numerically stable scalar log Poisson PMF using precomputed values."""
    if lam <= 0.0:
        if k == 0:
            return 0.0
        return C_NEG_INF
    return (<double>k) * log_lam - lam - lgamma_kp1


def bayes_poisson(
    signal,  # signal boost per choice, shape: (n_trials, n_choices)
    noise,  # baseline firing rate, shape: (n_trials, 1) or (n_trials, n_choices)
    t,  # non-decision time, shape: (n_trials, 1)
    entropy_threshold=None,  # optional stopping entropy, shape: (n_trials, 1)
    time_step=None,  # optional decoding step size, shape: (n_trials, 1)
    prior=None,  # optional prior over choices, shape: (n_choices,) or (n_trials, n_choices)
    s=None,  # unused, kept for interface compatibility
    deadline=None,  # deadline per trial
    float delta_t=0.001,  # fallback step size if time_step is not provided
    float max_t=20.0,  # maximal time horizon
    int n_samples=2000,  # number of samples
    int n_trials=1,  # number of trials
    boundary_fun=None,  # unused, kept for interface compatibility
    boundary_multiplicative=True,
    boundary_params={},
    random_state=None,
    return_option="full",
    smooth_unif=False,
    **kwargs,
):
    """
    Simulate RTs and choices for the Bayesian Poisson decoder model.
    """

    if signal is None or noise is None or t is None:
        raise ValueError("bayes_poisson requires signal, noise, and t.")
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")

    signal = _as_2d_float_array(signal, "signal")
    n_trials = signal.shape[0]
    n_choices = signal.shape[1]

    if n_choices < 2:
        raise ValueError("bayes_poisson requires at least two choices.")

    noise = _broadcast_to_choices(noise, "noise", n_trials, n_choices)

    t = _as_2d_float_array(t, "t")
    if t.shape[0] != n_trials:
        raise ValueError("t must have the same number of trials as signal.")
    if t.shape[1] != 1:
        if t.shape[1] == n_choices:
            t = t[:, :1]
        else:
            raise ValueError("t must have shape (n_trials, 1) or (n_trials, n_choices).")

    if entropy_threshold is None:
        entropy_threshold = np.full((n_trials, 1), 0.1, dtype=DTYPE)
    entropy_threshold = _as_2d_float_array(entropy_threshold, "entropy_threshold")
    if entropy_threshold.shape[0] != n_trials:
        raise ValueError("entropy_threshold must match n_trials.")
    if entropy_threshold.shape[1] != 1:
        raise ValueError("entropy_threshold must have shape (n_trials, 1).")

    if time_step is None:
        default_step = delta_t if delta_t > 0.0 else 0.01
        time_step = np.full((n_trials, 1), default_step, dtype=DTYPE)
    time_step = _as_2d_float_array(time_step, "time_step")
    if time_step.shape[0] != n_trials:
        raise ValueError("time_step must match n_trials.")
    if time_step.shape[1] != 1:
        raise ValueError("time_step must have shape (n_trials, 1).")

    if deadline is None:
        deadline = np.full(n_trials, max_t, dtype=DTYPE)
    else:
        deadline = np.asarray(deadline, dtype=DTYPE)
        deadline = np.squeeze(deadline)
        if deadline.ndim == 0:
            deadline = np.full(n_trials, float(deadline), dtype=DTYPE)
        if deadline.ndim != 1:
            raise ValueError("deadline must be a scalar or 1D array.")
        if deadline.shape[0] != n_trials:
            raise ValueError("deadline length must match n_trials.")

    if s is None:
        s = np.zeros((n_trials, n_choices), dtype=DTYPE)
    else:
        s = _broadcast_to_choices(s, "s", n_trials, n_choices)

    if (not np.isfinite(signal).all()) or (signal < 0).any():
        raise ValueError("All signal values must be finite and >= 0.")
    if (not np.isfinite(noise).all()) or (noise < 0).any():
        raise ValueError("All noise values must be finite and >= 0.")
    if (not np.isfinite(t).all()) or (t < 0).any():
        raise ValueError("All t values must be finite and >= 0.")
    if (not np.isfinite(entropy_threshold).all()) or (entropy_threshold < 0).any():
        raise ValueError("entropy_threshold values must be finite and >= 0.")
    if (not np.isfinite(time_step).all()) or (time_step <= 0).any():
        raise ValueError("time_step values must be finite and > 0.")
    if not np.isfinite(deadline).all():
        raise ValueError("deadline values must be finite.")

    prior_arr = _normalize_prior(prior, n_trials, n_choices)
    cdef double inv_log2 = 1.0 / c_log(2.0)
    cdef double uniform_entropy = c_log(float(n_choices)) * inv_log2

    set_seed(random_state)
    rng = np.random.default_rng(random_state)

    # Use contiguous float64 buffers for fast typed indexing in core loops.
    signal_arr64 = np.ascontiguousarray(signal, dtype=np.float64)
    noise_arr64 = np.ascontiguousarray(noise, dtype=np.float64)
    prior_arr64 = np.ascontiguousarray(prior_arr, dtype=np.float64)
    t_arr64 = np.ascontiguousarray(t, dtype=np.float64)
    entropy_arr64 = np.ascontiguousarray(entropy_threshold, dtype=np.float64)
    time_step_arr64 = np.ascontiguousarray(time_step, dtype=np.float64)
    deadline_arr64 = np.ascontiguousarray(deadline, dtype=np.float64)

    cdef double[:, :] signal_view = signal_arr64
    cdef double[:, :] noise_view = noise_arr64
    cdef double[:, :] prior_view = prior_arr64
    cdef double[:, :] t_view = t_arr64
    cdef double[:, :] entropy_view = entropy_arr64
    cdef double[:, :] time_step_view = time_step_arr64
    cdef double[:] deadline_view = deadline_arr64

    rts = np.full((n_samples, n_trials, 1), -999.0, dtype=DTYPE)
    cdef float[:, :, :] rts_view = rts
    choices = np.zeros((n_samples, n_trials, 1), dtype=np.intc)
    cdef int[:, :, :] choices_view = choices

    cdef Py_ssize_t trial_ix, sample_ix, step_ix, choice_ix
    cdef int n_steps, true_choice, chosen_idx, fallback_choice, best_idx
    cdef bint stopped
    cdef long long k_count
    cdef double dt_trial, deadline_tmp, entropy_thr, t_nd
    cdef double t_elapsed, smooth_u, rt_value
    cdef double lam_noise, lam_signal, lgamma_kp1
    cdef double lpn, lps, total_log_noise, max_lp, lp
    cdef double sum_exp, exp_shifted, entropy, posterior_i, best_post
    cdef double prior_val, best_prior

    cdef cnp.ndarray signal_plus_noise_arr
    cdef double[:] signal_plus_noise_view
    cdef cnp.ndarray log_prior_arr
    cdef double[:] log_prior_view
    cdef cnp.ndarray rates_dt_by_true_arr
    cdef double[:, :] rates_dt_by_true_view
    cdef cnp.ndarray lam_noise_arr
    cdef double[:, :] lam_noise_view
    cdef cnp.ndarray lam_signal_arr
    cdef double[:, :] lam_signal_view
    cdef cnp.ndarray log_lam_noise_arr
    cdef double[:, :] log_lam_noise_view
    cdef cnp.ndarray log_lam_signal_arr
    cdef double[:, :] log_lam_signal_view
    cdef cnp.ndarray counts_arr
    cdef long long[:] counts_view
    cdef cnp.ndarray delta_arr
    cdef double[:] delta_view
    cdef cnp.ndarray lp_arr
    cdef double[:] lp_view
    cdef cnp.ndarray unnorm_arr
    cdef double[:] unnorm_view
    cdef cnp.ndarray true_choices_arr
    cdef long long[:] true_choices_view
    cdef cnp.ndarray spike_increments_arr
    cdef long long[:, :] spike_increments_view

    for trial_ix in range(n_trials):
        dt_trial = time_step_view[trial_ix, 0]
        t_nd = t_view[trial_ix, 0]
        entropy_thr = entropy_view[trial_ix, 0]
        deadline_tmp = <double>compute_deadline_tmp(
            max_t,
            <float>deadline_view[trial_ix],
            <float>t_nd,
        )

        # Compute fallback choice (MAP under prior) and log-prior once per trial.
        log_prior_arr = np.empty(n_choices, dtype=np.float64)
        log_prior_view = log_prior_arr
        best_prior = -1.0
        fallback_choice = 0
        for choice_ix in range(n_choices):
            prior_val = prior_view[trial_ix, choice_ix]
            if prior_val > best_prior:
                best_prior = prior_val
                fallback_choice = <int>choice_ix
            if prior_val > 0.0:
                log_prior_view[choice_ix] = c_log(prior_val)
            else:
                log_prior_view[choice_ix] = C_NEG_INF

        if deadline_tmp <= 0.0:
            for sample_ix in range(n_samples):
                choices_view[sample_ix, trial_ix, 0] = fallback_choice
            continue

        n_steps = <int>(deadline_tmp / dt_trial)
        if n_steps <= 0:
            for sample_ix in range(n_samples):
                choices_view[sample_ix, trial_ix, 0] = fallback_choice
            continue

        # Precompute per-trial constants used in inner loops.
        signal_plus_noise_arr = np.empty(n_choices, dtype=np.float64)
        signal_plus_noise_view = signal_plus_noise_arr
        for choice_ix in range(n_choices):
            signal_plus_noise_view[choice_ix] = (
                noise_view[trial_ix, choice_ix] + signal_view[trial_ix, choice_ix]
            )

        rates_dt_by_true_arr = np.empty((n_choices, n_choices), dtype=np.float64)
        rates_dt_by_true_view = rates_dt_by_true_arr
        for true_choice in range(n_choices):
            for choice_ix in range(n_choices):
                rates_dt_by_true_view[true_choice, choice_ix] = (
                    noise_view[trial_ix, choice_ix] * dt_trial
                )
            rates_dt_by_true_view[true_choice, true_choice] += (
                signal_view[trial_ix, true_choice] * dt_trial
            )

        lam_noise_arr = np.empty((n_steps, n_choices), dtype=np.float64)
        lam_signal_arr = np.empty((n_steps, n_choices), dtype=np.float64)
        log_lam_noise_arr = np.empty((n_steps, n_choices), dtype=np.float64)
        log_lam_signal_arr = np.empty((n_steps, n_choices), dtype=np.float64)
        lam_noise_view = lam_noise_arr
        lam_signal_view = lam_signal_arr
        log_lam_noise_view = log_lam_noise_arr
        log_lam_signal_view = log_lam_signal_arr

        for step_ix in range(n_steps):
            t_elapsed = (<double>(step_ix + 1)) * dt_trial
            for choice_ix in range(n_choices):
                lam_noise = noise_view[trial_ix, choice_ix] * t_elapsed
                lam_signal = signal_plus_noise_view[choice_ix] * t_elapsed
                lam_noise_view[step_ix, choice_ix] = lam_noise
                lam_signal_view[step_ix, choice_ix] = lam_signal
                if lam_noise > 0.0:
                    log_lam_noise_view[step_ix, choice_ix] = c_log(lam_noise)
                else:
                    log_lam_noise_view[step_ix, choice_ix] = C_NEG_INF
                if lam_signal > 0.0:
                    log_lam_signal_view[step_ix, choice_ix] = c_log(lam_signal)
                else:
                    log_lam_signal_view[step_ix, choice_ix] = C_NEG_INF

        # Draw latent true choice once per sample.
        true_choices_arr = rng.choice(
            n_choices,
            size=n_samples,
            p=prior_arr64[trial_ix, :],
        )
        true_choices_view = true_choices_arr

        # Reusable work buffers.
        counts_arr = np.zeros(n_choices, dtype=np.int64)
        delta_arr = np.empty(n_choices, dtype=np.float64)
        lp_arr = np.empty(n_choices, dtype=np.float64)
        unnorm_arr = np.empty(n_choices, dtype=np.float64)
        counts_view = counts_arr
        delta_view = delta_arr
        lp_view = lp_arr
        unnorm_view = unnorm_arr

        for sample_ix in range(n_samples):
            true_choice = <int>true_choices_view[sample_ix]
            for choice_ix in range(n_choices):
                counts_view[choice_ix] = 0

            chosen_idx = fallback_choice
            stopped = False

            spike_increments_arr = rng.poisson(
                lam=rates_dt_by_true_arr[true_choice, :],
                size=(n_steps, n_choices),
            )
            spike_increments_view = spike_increments_arr

            for step_ix in range(n_steps):
                for choice_ix in range(n_choices):
                    counts_view[choice_ix] += spike_increments_view[step_ix, choice_ix]

                total_log_noise = 0.0
                for choice_ix in range(n_choices):
                    k_count = counts_view[choice_ix]
                    lam_noise = lam_noise_view[step_ix, choice_ix]
                    lam_signal = lam_signal_view[step_ix, choice_ix]

                    if (lam_noise > 0.0) or (lam_signal > 0.0):
                        lgamma_kp1 = c_lgamma((<double>k_count) + 1.0)
                    else:
                        lgamma_kp1 = 0.0

                    lpn = _log_poisson_pmf_fast(
                        k_count,
                        lam_noise,
                        log_lam_noise_view[step_ix, choice_ix],
                        lgamma_kp1,
                    )
                    lps = _log_poisson_pmf_fast(
                        k_count,
                        lam_signal,
                        log_lam_signal_view[step_ix, choice_ix],
                        lgamma_kp1,
                    )

                    total_log_noise += lpn
                    delta_view[choice_ix] = lps - lpn

                max_lp = C_NEG_INF
                for choice_ix in range(n_choices):
                    lp = total_log_noise + delta_view[choice_ix] + log_prior_view[choice_ix]
                    lp_view[choice_ix] = lp
                    if lp > max_lp:
                        max_lp = lp

                if max_lp <= (C_NEG_INF / 2.0):
                    chosen_idx = 0
                    entropy = uniform_entropy
                else:
                    sum_exp = 0.0
                    for choice_ix in range(n_choices):
                        exp_shifted = c_exp(lp_view[choice_ix] - max_lp)
                        unnorm_view[choice_ix] = exp_shifted
                        sum_exp += exp_shifted

                    if sum_exp <= 0.0:
                        chosen_idx = 0
                        entropy = uniform_entropy
                    else:
                        entropy = 0.0
                        best_idx = 0
                        best_post = -1.0
                        for choice_ix in range(n_choices):
                            posterior_i = unnorm_view[choice_ix] / sum_exp
                            if posterior_i > best_post:
                                best_post = posterior_i
                                best_idx = <int>choice_ix
                            if posterior_i > 0.0:
                                entropy -= posterior_i * (c_log(posterior_i) * inv_log2)
                        chosen_idx = best_idx

                if entropy < entropy_thr:
                    t_elapsed = (<double>(step_ix + 1)) * dt_trial
                    smooth_u = <double>compute_smooth_unif(
                        smooth_unif,
                        <float>t_elapsed,
                        <float>deadline_tmp,
                        <float>dt_trial,
                    )
                    rt_value = t_elapsed + t_nd + smooth_u

                    if (deadline_view[trial_ix] > 0.0) and (rt_value < deadline_view[trial_ix]):
                        rts_view[sample_ix, trial_ix, 0] = <float>rt_value
                    else:
                        rts_view[sample_ix, trial_ix, 0] = -999.0
                    choices_view[sample_ix, trial_ix, 0] = chosen_idx
                    stopped = True
                    break

            if not stopped:
                rts_view[sample_ix, trial_ix, 0] = -999.0
                choices_view[sample_ix, trial_ix, 0] = chosen_idx

    possible_choices = list(np.arange(0, n_choices, dtype=np.intc))
    minimal_meta = build_minimal_metadata(
        simulator_name="bayes_poisson",
        possible_choices=possible_choices,
        n_samples=n_samples,
        n_trials=n_trials,
        boundary_fun_name=None,
    )
    # Required by simulator post-processing (binning) even when return_option='minimal'.
    minimal_meta["max_t"] = max_t
    minimal_meta["delta_t"] = delta_t

    if return_option == "full":
        signal_dict = build_param_dict_from_2d_array(signal, "signal", n_choices)
        noise_dict = build_param_dict_from_2d_array(noise, "noise", n_choices)

        sim_config = {"delta_t": delta_t, "max_t": max_t}
        params = {
            "signal": signal,
            "noise": noise,
            "t": t,
            "deadline": deadline,
            "s": s,
            "entropy_threshold": entropy_threshold,
            "time_step": time_step,
            "prior": prior_arr.astype(DTYPE),
        }
        full_meta = build_full_metadata(
            minimal_metadata=minimal_meta,
            params=params,
            sim_config=sim_config,
            extra_params={**signal_dict, **noise_dict},
        )
        return build_return_dict(rts, choices, full_meta)

    if return_option == "minimal":
        return build_return_dict(rts, choices, minimal_meta)

    raise ValueError('return_option must be either "full" or "minimal"')
