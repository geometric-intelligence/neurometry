import skdim
import numpy as np
import os

os.environ["GEOMSTATS_BACKEND"] = "pytorch"
import geomstats.backend as gs

import neurometry.datasets.synthetic as synthetic


def skdim_dimension_estimation(
    methods, dimensions, num_trials, num_points, num_neurons, poisson_multiplier=1
):
    if methods == "all":
        methods = [method for method in dir(skdim.id) if not method.startswith("_")]

    id_estimates = {}
    for method_name in methods:
        method = getattr(skdim.id, method_name)()
        estimates = np.zeros((len(dimensions), num_trials))
        for dim_idx, dim in enumerate(dimensions):
            torus_points = synthetic.hypertorus(dim, num_points)
            neural_torus, _ = synthetic.synthetic_neural_manifold(
                torus_points,
                num_neurons,
                "sigmoid",
                poisson_multiplier,
                scales=gs.ones(num_neurons),
            )
            for trial_idx in range(num_trials):
                method.fit(neural_torus)
                estimates[dim_idx, trial_idx] = np.mean(method.dimension_)
        id_estimates[method_name] = estimates

    return id_estimates
