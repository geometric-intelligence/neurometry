import skdim
import numpy as np
import os

os.environ["GEOMSTATS_BACKEND"] = "pytorch"
import geomstats.backend as gs

import neurometry.datasets.synthetic as synthetic
import matplotlib.pyplot as plt


def skdim_dimension_estimation(
    methods, dimensions, manifold_type, num_trials, num_points, num_neurons, poisson_multiplier=1,ref_frequency=200
):
    if methods == "all":
        methods = [method for method in dir(skdim.id) if not method.startswith("_")]

    point_generator = getattr(synthetic, manifold_type)

    noise_level = np.sqrt(1 / (ref_frequency*poisson_multiplier))

    id_estimates = {}
    for method_name in methods:
        method = getattr(skdim.id, method_name)()
        estimates = np.zeros((len(dimensions), num_trials))
        for dim_idx, dim in enumerate(dimensions):
            points = point_generator(dim, num_points)
            neural_manifold, _ = synthetic.synthetic_neural_manifold(
                points,
                num_neurons,
                "sigmoid",
                poisson_multiplier,
                ref_frequency,
                scales=gs.ones(num_neurons),
            )
            for trial_idx in range(num_trials):
                method.fit(neural_manifold)
                estimates[dim_idx, trial_idx] = np.mean(method.dimension_)
        id_estimates[method_name] = estimates

    return id_estimates, noise_level


def plot_dimension_experiments(dim_estimates, dimensions, max_id_dim, manifold_type, noise_level):
    num_methods = len(dim_estimates)

    # Creating a subplot grid - adjust the number of rows and columns as needed
    rows = int(np.ceil(np.sqrt(num_methods)))
    cols = int(np.ceil(num_methods / rows))

    # Creating the figure
    fig, axs = plt.subplots(rows, cols, figsize=(20, 20))

    if manifold_type == "hypersphere":
        extrinsic_dims = [dim + 1 for dim in dimensions]
        gt_label = "Ground Truth Extrinsic Dimension $(d + 1)$"
        y_lim = [0, max_id_dim + 1]
    elif manifold_type == "hypertorus":
        extrinsic_dims = [2 * dim for dim in dimensions]
        y_lim = [0, 2 * max_id_dim]
        gt_label = "Ground Truth Extrinsic Dimension $(2d)$"

    fig.suptitle(
        f"Dimension Estimation for {manifold_type}, noise level={100*noise_level:.1f}%",
        fontsize=40,
    )

    for i, (method, estimates) in enumerate(dim_estimates.items()):
        ax = axs[i // cols, i % cols] 
        mean_dim = np.mean(estimates, axis=1)
        std_dim = np.std(estimates, axis=1)

        ax.errorbar(
            dimensions,
            mean_dim,
            yerr=std_dim,
            fmt="o",
            label=method,
            capsize=5,
            marker="o",
            markersize=10,
        )
        ax.plot(dimensions, dimensions, "k--", label="Ground Truth Intrinsic Dimension")
        ax.plot(
            dimensions,
            extrinsic_dims,
            "r--",
            label=gt_label,
        )
        ax.set_xlabel("Intrinsic Dimension $d$", fontsize=30)
        ax.set_ylabel("Estimated Dimension", fontsize=30)

        ax.set_title(method, fontsize=30)
        ax.set_aspect("auto", adjustable="box")
        ax.legend(fontsize=20)
        ax.set_xlim([0, max_id_dim])
        ax.set_ylim(y_lim)

    plt.tight_layout()
    
    plt.show()
