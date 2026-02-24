"""Bayesian Poisson model configuration."""

import cssm
from ssms.basic_simulators import boundary_functions as bf
from ssms.transforms import (
    ColumnStackParameters,
    ExpandDimension,
    RenameParameter,
)


def get_bayes_poisson_config():
    """Get configuration for 2-choice Bayesian Poisson decoder model."""
    return {
        "name": "bayes_poisson",
        "params": [
            "signal1",
            "signal2",
            "noise",
            "t",
            "entropy",
            "step",
        ],
        "param_bounds": [
            [0.0, 0.0, 1e-6, 0.0, 0.0, 1e-4],
            [30.0, 30.0, 30.0, 2.0, 1.0, 0.2],
        ],
        "boundary_name": "constant",
        "boundary": bf.constant,
        "n_params": 6,
        "default_params": [4.0, 4.0, 2.0, 1e-3, 0.1, 0.01],
        "nchoices": 2,
        "choices": [0, 1],
        "n_particles": 2,
        "simulator": cssm.bayes_poisson,
        "parameter_transforms": {
            "sampling": [],
            "simulation": [
                ColumnStackParameters(
                    ["signal1", "signal2"], "signal", delete_sources=False
                ),
                RenameParameter("entropy", "entropy_threshold"),
                RenameParameter("step", "time_step"),
                ExpandDimension(["noise", "t", "entropy_threshold", "time_step"]),
            ],
        },
    }
