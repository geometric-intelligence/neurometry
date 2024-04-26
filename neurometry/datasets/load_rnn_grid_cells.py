import os

import matplotlib.pyplot as plt
import numpy as np

# sys.path.append(str(Path(__file__).parent.parent))
from .rnn_grid_cells import (config, dual_agent_activity,
                             single_agent_activity, utils)


def load_activations(epochs, file_path, version="single", verbose=True, save=True):
    activations = []
    rate_maps = []
    state_points = []
    positions = []
    g_s = []

    activations_dir = os.path.join(file_path, "activations")

    for epoch in epochs:
        activations_epoch_path = os.path.join(
            activations_dir, f"activations_{version}_agent_epoch_{epoch}.npy"
        )
        rate_map_epoch_path = os.path.join(
            activations_dir, f"rate_map_{version}_agent_epoch_{epoch}.npy"
        )
        positions_epoch_path = os.path.join(
            activations_dir, f"positions_{version}_agent_epoch_{epoch}.npy"
        )
        gs_epoch_path = os.path.join(
            activations_dir, f"g_{version}_agent_epoch_{epoch}.npy"
        )

        if (
            os.path.exists(activations_epoch_path)
            and os.path.exists(rate_map_epoch_path)
            and os.path.exists(positions_epoch_path)
            and os.path.exists(gs_epoch_path)
        ):
            activations.append(np.load(activations_epoch_path))
            rate_maps.append(np.load(rate_map_epoch_path))
            positions.append(np.load(positions_epoch_path))
            g_s.append(np.load(gs_epoch_path))
            if verbose:
                print(f"Epoch {epoch} found.")
        else:
            print(f"Epoch {epoch} not found. Loading ...")
            parser = config.parser
            options, _ = parser.parse_known_args()
            options.run_ID = utils.generate_run_ID(options)
            if version == "single":
                (
                    activations_single_agent,
                    rate_map_single_agent,
                    g_single_agent,
                    positions_single_agent,
                ) = single_agent_activity.main(options, file_path, epoch=epoch)
                activations.append(activations_single_agent)
                rate_maps.append(rate_map_single_agent)
                positions.append(positions_single_agent)
                g_s.append(g_single_agent)
            elif version == "dual":
                (
                    activations_dual_agent,
                    rate_map_dual_agent,
                    g_dual_agent,
                    positions_dual_agent,
                ) = dual_agent_activity.main(options, file_path, epoch=epoch)
                activations.append(activations_dual_agent)
                rate_maps.append(rate_map_dual_agent)
                positions.append(positions_dual_agent)
                g_s.append(g_dual_agent)

            if save:
                np.save(activations_epoch_path, activations[-1])
                np.save(rate_map_epoch_path, rate_maps[-1])
                np.save(positions_epoch_path, positions[-1])
                np.save(gs_epoch_path, g_s[-1])

        state_points_epoch = activations[-1].reshape(activations[-1].shape[0], -1)
        state_points.append(state_points_epoch)

    if verbose:
        print(f"Loaded epochs {epochs} of {version} agent model.")
        print(
            f"activations has shape {activations[0].shape}. There are {activations[0].shape[0]} grid cells with {activations[0].shape[1]} x {activations[0].shape[2]} environment resolution, averaged over {activations[0].shape[3]} trajectories."
        )
        print(
            f"state_points has shape {state_points[0].shape}. There are {state_points[0].shape[1]} data points in the {state_points[0].shape[0]}-dimensional state space."
        )
        print(
            f"rate_maps has shape {rate_maps[0].shape}. There are {rate_maps[0].shape[1]} data points averaged over {activations[0].shape[3]} trajectories in the {rate_maps[0].shape[0]}-dimensional state space."
        )
        print(f"positions has shape {positions[0].shape}.")

    return activations, rate_maps, state_points, positions, g_s


# def plot_rate_map(indices, num_plots, activations, title):
#     rng = np.random.default_rng(seed=0)
#     if indices is None:
#         idxs = rng.integers(0, 4095, num_plots)
#     else:
#         idxs = indices
#         num_plots = len(indices)

#     rows = 4
#     cols = num_plots // rows + (num_plots % rows > 0)

#     plt.rcParams["text.usetex"] = False

#     fig, axes = plt.subplots(rows, cols, figsize=(20, 8))

#     for i in range(rows):
#         for j in range(cols):
#             if i * cols + j < num_plots:
#                 gc = np.mean(activations[idxs[i * cols + j]], axis=2)
#                 axes[i, j].imshow(gc)
#                 axes[i, j].set_title(f"grid cell id: {idxs[i * cols + j]}")
#                 axes[i, j].axis("off")
#             else:
#                 axes[i, j].axis("off")

#     plt.suptitle(title)
#     plt.tight_layout()
#     plt.show()


def plot_rate_map(indices, num_plots, activations, title, seed=None):
    rng = np.random.default_rng(seed=seed)
    if indices is None:
        idxs = rng.integers(0, activations.shape[0] - 1, num_plots)
    else:
        idxs = indices
        num_plots = len(indices)

    rows = 4
    cols = num_plots // rows + (num_plots % rows > 0)

    plt.rcParams["text.usetex"] = False

    fig, axes = plt.subplots(rows, cols, figsize=(20, 8))

    for i in range(rows):
        for j in range(cols):
            if i * cols + j < num_plots:
                if len(activations.shape) == 4:
                    gc = np.mean(activations[idxs[i * cols + j]], axis=2)
                else:
                    gc = activations[idxs[i * cols + j]]
                if axes.ndim > 1:  # Check if axes is a 2D array
                    ax = axes[i, j]
                else:  # If axes is flattened (e.g., only one row of subplots)
                    ax = axes[i * cols + j]
                ax.imshow(gc)
                ax.set_title(f"grid cell id: {idxs[i * cols + j]}", fontsize=10)
                ax.axis("off")
            else:
                if axes.ndim > 1:
                    axes[i, j].axis("off")
                else:
                    axes[i * cols + j].axis("off")

    fig.suptitle(title, fontsize=30)
    plt.tight_layout()
    plt.show()
