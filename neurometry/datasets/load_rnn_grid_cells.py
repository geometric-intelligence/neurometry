import os
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from .rnn_grid_cells import config, dual_agent_activity, single_agent_activity, utils

# Loading single agent model

parent_dir = os.getcwd() + "/datasets/rnn_grid_cells/"


single_model_folder = "Single agent path integration/Seed 1 weight decay 1e-06/"
single_model_parameters = "steps_20_batch_200_RNN_4096_relu_rf_012_DoG_True_periodic_False_lr_00001_weight_decay_1e-06/"


dual_model_folder = (
    "Dual agent path integration disjoint PCs/Seed 1 weight decay 1e-06/"
)
dual_model_parameters = "steps_20_batch_200_RNN_4096_relu_rf_012_DoG_True_periodic_False_lr_00001_weight_decay_1e-06/"


def load_activations(epochs, version="single", verbose=True):
    activations = []
    rate_maps = []
    state_points = []

    if version == "single":
        activations_dir = (
            parent_dir + single_model_folder + single_model_parameters + "activations/"
        )
    elif version == "dual":
        activations_dir = (
            parent_dir + dual_model_folder + dual_model_parameters + "activations/"
        )

    random.seed(0)
    for epoch in epochs:
        activations_epoch_path = (
            activations_dir + f"activations_{version}_agent_epoch_{epoch}.npy"
        )
        rate_map_epoch_path = (
            activations_dir + f"rate_map_{version}_agent_epoch_{epoch}.npy"
        )
        if os.path.exists(activations_epoch_path) and os.path.exists(
            rate_map_epoch_path
        ):
            activations.append(np.load(activations_epoch_path))
            rate_maps.append(np.load(rate_map_epoch_path))
            if verbose:
                print(f"Epoch {epoch} found!!! :D")
        else:
            print(f"Epoch {epoch} not found. Loading ...")
            parser = config.parser
            options, _ = parser.parse_known_args()
            options.run_ID = utils.generate_run_ID(options)
            if type == "single":
                activations_single_agent, rate_map_single_agent = (
                    single_agent_activity.main(options, epoch=epoch)
                )
                activations.append(activations_single_agent)
                rate_maps.append(rate_map_single_agent)
            elif type == "dual":
                activations_dual_agent, rate_map_dual_agent = dual_agent_activity.main(
                    options, epoch=epoch
                )
                activations.append(activations_dual_agent)
                rate_maps.append(rate_map_dual_agent)
        state_points_epoch = activations[-1].reshape(activations[-1].shape[0], -1)
        state_points.append(state_points_epoch)

    if verbose:
        print(f"Loaded epochs {epochs} of {version} agent model.")
        print(
            f"There are {activations[0].shape[0]} grid cells with {activations[0].shape[1]} x {activations[0].shape[2]} environment resolution, averaged over {activations[0].shape[3]} trajectories."
        )
        print(
            f"There are {state_points[0].shape[1]} data points in the {state_points[0].shape[0]}-dimensional state space."
        )
        print(
            f"There are {rate_maps[0].shape[1]} data points averaged over {activations[0].shape[3]} trajectories in the {rate_maps[0].shape[0]}-dimensional state space."
        )

    return activations, rate_maps, state_points


def plot_rate_map(num_plots, activations):
    idxs = np.random.randint(0, 4095, num_plots)

    rows = 4
    cols = num_plots // rows + (num_plots % rows > 0)

    fig, axes = plt.subplots(rows, cols, figsize=(20, 8))

    for i in range(rows):
        for j in range(cols):
            if i * cols + j < num_plots:
                gc = np.mean(activations[idxs[i * cols + j]], axis=2)
                axes[i, j].imshow(gc)
                axes[i, j].set_title(f"grid cell id: {idxs[i * cols + j]}")
                axes[i, j].axis("off")
            else:
                axes[i, j].axis("off")

    plt.tight_layout()
    plt.show()
