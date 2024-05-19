import os
import pickle

import default_config
import matplotlib.pyplot as plt
import numpy as np
import wandb
import yaml
from sklearn.decomposition import PCA

from neurometry.datasets.load_rnn_grid_cells import get_scores, umap_dbscan
from neurometry.dimension.dim_reduction import (
    plot_2d_manifold_projections,
    plot_pca_projections,
)
from neurometry.topology.persistent_homology import compute_diagrams_shuffle
from neurometry.topology.plotting import plot_all_barcodes_with_null

pretrained_run_id = "20240418-180712"
pretrained_run_dir = os.path.join(
    os.getcwd(),
    f"logs/rnn_isometry/{pretrained_run_id}",
)

pretrained_config_file = os.path.join(pretrained_run_dir, "config.txt")
with open(pretrained_config_file) as f:
    pretrained_config = yaml.safe_load(f)

pretrained_activations_file = os.path.join(pretrained_run_dir, "ckpt/activations/activations-step25000.pkl")
with open(pretrained_activations_file, "rb") as f:
    pretrained_activations = pickle.load(f)

scores = get_scores(pretrained_run_dir, pretrained_activations, pretrained_config)

pretrained_clusters, umap_cluster_labels = umap_dbscan(
    pretrained_activations["v"], pretrained_run_dir, pretrained_config, sac_array=None, plot=False
)

neural_points_pretrained = {}
rate_maps_pretrained = {}
for id in np.unique(umap_cluster_labels):
    rate_maps_pretrained[id] = pretrained_activations["v"][umap_cluster_labels == id]
    neural_points_pretrained[id] = rate_maps_pretrained[id].reshape(len(rate_maps_pretrained[id]), -1).T


def plot_experiment(run_name,figs_dir):
    expt_config = _load_expt_config(run_name)
    neural_points_expt, rate_maps_expt = _get_expt_activations_per_cluster(run_name)

    fig_saliency_kernel = _plot_gaussian_kernel(expt_config["s_0"], location=expt_config["x_saliency"], scale=expt_config["sigma_saliency"])
    wandb.log({"saliency_kernel": wandb.Image(fig_saliency_kernel)})
    #save fig_saliency_kernel
    fig_saliency_kernel.savefig(os.path.join(figs_dir, f"{run_name}_saliency_kernel.pdf"))

    for module in neural_points_expt:
        fig_rate_maps_pretrained = _draw_heatmap(rate_maps_pretrained[module], f"Module {module} pretrained")
        wandb.log({f"rate_maps_pretrained_module_{module}": wandb.Image(fig_rate_maps_pretrained)})
        fig_rate_maps_pretrained.savefig(os.path.join(figs_dir, f"{run_name}_rate_maps_pretrained_module_{module}.pdf"))
        fig_rate_maps_expt = _draw_heatmap(rate_maps_expt[module], f"Module {module} experiment")
        wandb.log({f"rate_maps_expt_module_{module}": wandb.Image(fig_rate_maps_expt)})
        fig_rate_maps_expt.savefig(os.path.join(figs_dir, f"{run_name}_rate_maps_expt_module_{module}.pdf"))
        fig_pca = plot_pca_projections(
            neural_points_pretrained[module],
            neural_points_expt[module],
            f"Module {module} pretrained",
            f"Module {module} experiment",
            K=min(6, neural_points_pretrained[module].shape[1]),
        )
        wandb.log({f"pca_module_{module}": wandb.Image(fig_pca)})
        fig_pca.savefig(os.path.join(figs_dir, f"{run_name}_pca_module_{module}.pdf"))

        fig_nonlinear = plot_2d_manifold_projections(
            neural_points_pretrained[module],
            neural_points_expt[module],
            f"Module {module} pretrained",
            f"Module {module} experiment",
        )
        wandb.log({f"nonlinear_projection_module_{module}": wandb.Image(fig_nonlinear)})
        fig_nonlinear.savefig(os.path.join(figs_dir, f"{run_name}_nonlinear_projection_module_{module}.pdf"))
        if neural_points_pretrained[module].shape[1] >= 6 and neural_points_expt[module].shape[1] >= 6:
            pca_pretrained = PCA(n_components=min(6, neural_points_pretrained[module].shape[1]))
            neural_points_pretrained_pca = pca_pretrained.fit_transform(neural_points_pretrained[module])
            module_pretrained_diagrams = compute_diagrams_shuffle(neural_points_pretrained_pca, num_shuffles=10, homology_dimensions=(0,1,2))
            pca_expt = PCA(n_components=min(6, neural_points_expt[module].shape[1]))
            neural_points_expt_pca = pca_expt.fit_transform(neural_points_expt[module])
            module_expt_diagrams = compute_diagrams_shuffle(neural_points_expt_pca, num_shuffles=10, homology_dimensions=(0,1,2))
            tda_fig = plot_all_barcodes_with_null(
                module_pretrained_diagrams,
                f"Module {module} pretrained",
                module_expt_diagrams,
                f"Module {module} experiment",
            )
            wandb.log({f"tda_module_{module}": wandb.Image(tda_fig)})
            tda_fig.savefig(os.path.join(figs_dir, f"{run_name}_tda_module_{module}.pdf"))



def _get_expt_activations_per_cluster(run_name):
    activations = _load_expt_rate_maps(run_name)["v"]
    neural_points_expt = {}
    rate_maps_expt = {}
    for id in np.unique(umap_cluster_labels):
        rate_maps_expt[id] = activations[umap_cluster_labels == id]
        neural_points_expt[id] = rate_maps_expt[id].reshape(len(rate_maps_expt[id]), -1).T
    return neural_points_expt, rate_maps_expt


def _load_expt_rate_maps(run_name):
    activations_dir = default_config.activations_dir
    activations_file = os.path.join(activations_dir, f"{run_name}_activations.pkl")
    with open(activations_file, "rb") as f:
        return pickle.load(f)



def _load_expt_config(run_name):
    configs_dir = default_config.configs_dir
    config_file = os.path.join(configs_dir, f"{run_name}.json")

    with open(config_file) as file:
        return yaml.safe_load(file)



def _draw_heatmap(activations, title):
    # activations should be a 3-D tensor: [num_rate_maps, H, W]
    num_rate_maps = min(activations.shape[0], 100)
    #H, W = activations.shape[1], activations.shape[2]

    # Determine the number of rows and columns for the plot grid
    ncol = int(np.ceil(np.sqrt(num_rate_maps)))
    nrow = int(np.ceil(num_rate_maps / ncol))

    fig, axs = plt.subplots(nrow, ncol, figsize=(ncol * 2, nrow * 2))
    fig.suptitle(title, fontsize=20, fontweight="bold", verticalalignment="top")

    for i in range(num_rate_maps):
        row, col = divmod(i, ncol)
        if nrow == 1:
            ax = axs[col]
        elif ncol == 1:
            ax = axs[row]
        else:
            ax = axs[row, col]

        weight = activations[i]
        vmin, vmax = weight.min() - 0.01, weight.max()

        cmap = plt.get_cmap("jet", 1000)
        cmap.set_under("w")

        ax.imshow(
            weight,
            interpolation="nearest",
            cmap=cmap,
            aspect="auto",
            vmin=vmin,
            vmax=vmax,
        )
        ax.axis("off")

    # Hide any remaining empty subplots
    if num_rate_maps < nrow * ncol:
        for j in range(num_rate_maps, nrow * ncol):
            row, col = divmod(j, ncol)
            if nrow == 1:
                ax = axs[col]
            elif ncol == 1:
                ax = axs[row]
            else:
                fig.delaxes(axs[row, col])

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig

def _plot_gaussian_kernel(intensity, location=0.8, scale=0.1):
    x = np.linspace(0, 1, 40)
    y = np.linspace(0, 1, 40)
    X, Y = np.meshgrid(x, y)
    Z = np.exp(-((X - location) ** 2 + (Y - location) ** 2) / (2 * scale**2))
    kernel = 1 + intensity * Z

    fig, ax = plt.subplots()
    cax = ax.imshow(kernel, cmap="hot", extent=[0, 1, 0, 1])

    ax.set_xticks(np.linspace(0, 1, num=11))  # Set x-ticks from 0 to 1
    ax.set_yticks(np.linspace(0, 1, num=11))  # Set y-ticks from 0 to 1
    ax.set_xticklabels(np.round(np.linspace(0, 1, num=11), 2))
    ax.set_yticklabels(np.round(np.linspace(0, 1, num=11), 2))

    plt.colorbar(cax, ax=ax, orientation="vertical")
    return fig
