import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import torch

FIGURES = os.path.join(os.getcwd(), "results/figures/")
if not os.path.exists(FIGURES):
    os.makedirs(FIGURES)

HTML_FIGURES = os.path.join(os.getcwd(), "results/html_figures/")
if not os.path.exists(HTML_FIGURES):
    os.makedirs(HTML_FIGURES)


def plot_loss(train_losses, test_losses, config):
    fig, ax = plt.subplots(figsize=(20, 20))
    epochs = [epoch for epoch in range(1, config.n_epochs + 1)]
    ax.plot(
        epochs,
        np.log(1 + np.array(np.log(1 + np.array(train_losses)))),
        linewidth=10,
        label="train",
    )
    ax.plot(
        epochs,
        np.log(1 + np.array(np.log(1 + np.array(test_losses)))),
        linewidth=10,
        label="test",
    )
    ax.set_title("Losses", fontsize=40)
    ax.set_xlabel("epoch", fontsize=40)
    ax.set_ylabel("Log(1+Loss)", fontsize=40)
    ax.set_ylabel("Log(1+Loss)", fontsize=40)
    ax.legend(prop={"size": 40})
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.savefig(os.path.join(FIGURES, f"{config.results_prefix}_losses.png"))
    # plt.savefig(os.path.join(FIGURES, f"{config.results_prefix}_losses.svg"))
    return fig


def plot_recon_per_time(model, dataset_torch, labels, config):

    if config.dataset_name in ["s1_synthetic", "experimental"]:
        fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(24, 12), sharex=True)
    else:
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(24, 12), sharex=True)

    z_latent, recon, _ = model(dataset_torch)

    dataset = dataset_torch.cpu().detach().numpy()
    recon = recon.cpu().detach().numpy()

    axes[0].imshow(dataset[:1000, :].T, aspect="auto")
    axes[0].set_xlabel("Times", fontsize=16)
    axes[0].set_ylabel("True Neurons", fontsize=16)
    axes[1].imshow(recon[:1000, :].T, aspect="auto")
    axes[1].set_xlabel("Times", fontsize=16)
    axes[1].set_ylabel("Recon Neurons", fontsize=16)

    if config.dataset_name in ["s1_synthetic", "experimental"]:
        angles = np.array(labels["angles"])
        axes[2].plot(angles[:1000], linewidth=10)
        axes[2].set_xlabel("Times", fontsize=16)
        axes[2].set_ylabel("True Lab Angles", fontsize=16)
        axes[2].set_xlim(xmin=0)

        z_latent = z_latent.cpu().detach().numpy()
        z_norms = np.linalg.norm(z_latent, axis=1)
        if not np.all(np.allclose(z_norms, 1.0, atol=0.1)):
            print("WARNING: Latent variables are not on a circle.")
        angles_latent = (np.arctan2(z_latent[:, 1], z_latent[:, 0]) + 2 * np.pi) % (
            2 * np.pi
        )
        axes[3].plot(angles_latent[:1000], linewidth=10)
        axes[3].set_xlabel("Times", fontsize=16)
        axes[3].set_ylabel("Latent Angles", fontsize=16)
        axes[3].set_xlim(xmin=0)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES, f"{config.results_prefix}_recon_per_time.png"))
    # plt.savefig(os.path.join(FIGURES, f"{config.results_prefix}_recon_per_time.svg"))
    return fig


def plot_recon_per_positional_angle(model, dataset_torch, labels, config):
    fig = plt.figure(figsize=(24, 12))

    if config.dataset_name == "s1_synthetic":
        ax_data = fig.add_subplot(1, 2, 1)
        colormap = plt.get_cmap("hsv")
        x_data = dataset_torch[:, 0].cpu().detach().cpu().numpy()
        y_data = dataset_torch[:, 1].cpu().detach().cpu().numpy()
        _, rec, _ = model(dataset_torch)
        x_rec = rec[:, 0]
        x_rec = [x.item() for x in x_rec]
        y_rec = rec[:, 1]
        y_rec = [y.item() for y in y_rec]
        ax_data.set_title("Synthetic data", fontsize=40)
        sc_data = ax_data.scatter(
            x_data, y_data, s=400, c=labels["angles"], cmap=colormap
        )
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        ax_rec = fig.add_subplot(1, 2, 2)
        ax_rec.set_title("Reconstruction", fontsize=40)
        sc_rec = ax_rec.scatter(x_rec, y_rec, s=400, c=labels["angles"], cmap=colormap)
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
    elif config.dataset_name in (
        "s2_synthetic",
        "t2_synthetic",
        "grid_cells",
        "three_place_cells_synthetic",
    ):
        ax_data = fig.add_subplot(1, 2, 1, projection="3d")
        x_data = dataset_torch[:, 0].cpu().detach().cpu().numpy()
        y_data = dataset_torch[:, 1].cpu().detach().cpu().numpy()
        z_data = dataset_torch[:, 2].cpu().detach().cpu().numpy()
        norms_data = torch.linalg.norm(dataset_torch, axis=1).cpu().detach().numpy()
        _, rec, _ = model(dataset_torch)
        norms_rec = torch.linalg.norm(rec, axis=1).cpu().detach().cpu().numpy()
        x_rec = rec[:, 0]
        x_rec = [x.item() for x in x_rec]
        y_rec = rec[:, 1]
        y_rec = [y.item() for y in y_rec]
        z_rec = rec[:, 2]
        z_rec = [z.item() for z in z_rec]

        ax_data.set_title("Synthetic data", fontsize=40)
        ax_data.scatter3D(x_data, y_data, z_data, s=400, c=norms_data)
        plt.axis("off")
        # ax_data.view_init(elev=60, azim=45, roll=0)
        ax_rec = fig.add_subplot(1, 2, 2, projection="3d")
        ax_rec.set_title("Reconstruction", fontsize=40)
        ax_rec.scatter3D(x_rec, y_rec, z_rec, s=400, c=norms_rec)
        plt.axis("off")
        if config.dataset_name == "t2_synthetic":
            ax_data.set_xlim(
                -(config.major_radius + config.minor_radius),
                (config.major_radius + config.minor_radius),
            )
            ax_data.set_ylim(
                -(config.major_radius + config.minor_radius),
                (config.major_radius + config.minor_radius),
            )
            ax_data.set_zlim(
                -(config.major_radius + config.minor_radius),
                (config.major_radius + config.minor_radius),
            )
            ax_rec.set_xlim(
                -(config.major_radius + config.minor_radius),
                (config.major_radius + config.minor_radius),
            )
            ax_rec.set_ylim(
                -(config.major_radius + config.minor_radius),
                (config.major_radius + config.minor_radius),
            )
            ax_rec.set_zlim(
                -(config.major_radius + config.minor_radius),
                (config.major_radius + config.minor_radius),
            )
        plotly_fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=x_rec,
                    y=y_rec,
                    z=z_rec,
                    mode="markers",
                    marker=dict(
                        size=5,
                        color=norms_rec,  # set color to an array/list of desired values
                        colorscale="Viridis",  # choose a colorscale
                        opacity=0.8,
                    ),
                )
            ]
        )
        plotly_fig.update_layout(
            scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
            title=dict(
                text="Neural Manifold Reconstruction", font=dict(size=24), x=0.5
            ),
        )
        pio.write_html(
            plotly_fig,
            os.path.join(HTML_FIGURES, f"{config.results_prefix}_recon.html"),
        )
        pio.write_html(
            plotly_fig,
            os.path.join(HTML_FIGURES, f"{config.results_prefix}_recon.html"),
        )
    elif config.dataset_name == "experimental":
        thetas = np.array(labels["angles"])
        sort = np.argsort(thetas)
        dataset = dataset_torch.cpu().detach().numpy()
        sorted_dataset = dataset[sort, :]

        _, rec, _ = model(dataset_torch)

        rec = rec.cpu().detach().numpy()
        sorted_rec = rec[sort, :]

        color_norm = mpl.colors.Normalize(0.0, np.max(sorted_dataset))

        ax_data = fig.add_subplot(121)

        ax_data.set_xlabel("angle")
        ax_data.set_ylabel("neuron")

        ax_data.set_title("recorded place cell activity", fontsize=30)

        ax_data.imshow(
            sorted_dataset.T,
            extent=[0, 360, dataset_torch.shape[1], 0],
            aspect=5,
            norm=color_norm,
            cmap="viridis",
        )

        ax_rec = fig.add_subplot(122)

        ax_rec.set_xlabel("angle")
        ax_rec.set_ylabel("neuron")

        ax_rec.set_title("reconstructed place cell activity", fontsize=30)
        # breakpoint()

        ax_rec.imshow(
            sorted_rec.T,
            extent=[0, 360, dataset_torch.shape[1], 0],
            aspect=5,
            norm=color_norm,
            cmap="viridis",
        )

        # plt.colorbar(im_rec)
    else:
        raise NotImplementedError

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig(os.path.join(FIGURES, f"{config.results_prefix}_recon.png"))
    # plt.savefig(os.path.join(FIGURES, f"{config.results_prefix}_recon.svg"))
    return fig


def plot_latent_space(model, dataset_torch, labels, config):
    fig = plt.figure(figsize=(20, 20))
    if config.dataset_name in (
        "s1_synthetic",
        "experimental",
        "three_place_cells_synthetic",
    ):
        ax = fig.add_subplot(111)
        z, _, _ = model(dataset_torch.to(config.device))
        colormap = plt.get_cmap("twilight")
        z0 = z[:, 0]
        z0 = [_.item() for _ in z0]
        z1 = z[:, 1]
        z1 = [_.item() for _ in z1]
        sc = ax.scatter(z0, z1, c=labels["angles"], s=400, cmap=colormap)
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
    elif config.dataset_name in ("s2_synthetic", "t2_synthetic", "grid_cells"):
        ax = fig.add_subplot(111, projection="3d")
        z, _, _ = model(dataset_torch.to(config.device))
        z0 = z[:, 0]
        z0 = [_.item() for _ in z0]
        z1 = z[:, 1]
        z1 = [_.item() for _ in z1]
        z2 = z[:, 2]
        z2 = [_.item() for _ in z2]
        sc = ax.scatter3D(z0, z1, z2, s=400)
        ax.view_init(elev=60, azim=45, roll=0)
        if config.dataset_name == "t2_synthetic":
            ax.set_xlim(
                -(config.major_radius + config.minor_radius),
                (config.major_radius + config.minor_radius),
            )
            ax.set_ylim(
                -(config.major_radius + config.minor_radius),
                (config.major_radius + config.minor_radius),
            )
            ax.set_zlim(
                -(config.major_radius + config.minor_radius),
                (config.major_radius + config.minor_radius),
            )
        else:
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_zlim(-1, 1)

        plotly_fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=z0,
                    y=z1,
                    z=z2,
                    mode="markers",
                    marker=dict(
                        size=5,
                        color="blue",  # set color to an array/list of desired values
                        colorscale="Viridis",  # choose a colorscale
                        opacity=0.8,
                    ),
                )
            ]
        )
        # Set the layout properties
        plotly_fig.update_layout(
            scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
            title=dict(text="Latent Space", font=dict(size=24), x=0.5),
        )
        # Save the plot as an interactive HTML file
        pio.write_html(
            plotly_fig,
            os.path.join(HTML_FIGURES, f"{config.results_prefix}_latent_plot.html"),
        )

    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    ax.set_title("Latent space", fontsize=40)
    plt.savefig(os.path.join(FIGURES, f"{config.results_prefix}_latent_plot.png"))
    # plt.savefig(os.path.join(FIGURES, f"{config.results_prefix}_latent_plot.svg"))

    return fig


def plot_curvature_norms(angles, curvature_norms, config, norm_val, profile_type):
    fig = plt.figure(figsize=(24, 12))
    colormap = plt.get_cmap("hsv")
    if norm_val is not None:
        color_norm = mpl.colors.Normalize(0.0, norm_val)
    else:
        color_norm = mpl.colors.Normalize(0.0, max(curvature_norms))
    if config.dataset_name in ("s1_synthetic", "experimental"):
        ax1 = fig.add_subplot(121)
        ax1.plot(angles, curvature_norms, linewidth=10)
        ax1.set_xlabel("angle", fontsize=30)
        ax1.set_ylabel("mean curvature norm", fontsize=30)

        ax2 = fig.add_subplot(122, projection="polar")
        sc = ax2.scatter(
            angles,
            np.ones_like(angles),
            c=curvature_norms,
            s=400,
            cmap=colormap,
            norm=color_norm,
            linewidths=0,
        )
        ax2.set_yticks([])
        ax2.set_xlabel("angle", fontsize=30)

        ax1.set_title(f"{profile_type} mean curvature norm profile", fontsize=30)
        ax2.set_title(f"{profile_type} mean curvature norm profile", fontsize=30)

    elif config.dataset_name == "s2_synthetic":
        ax = fig.add_subplot(111, projection="3d")
        x = config.radius * [np.sin(angle[0]) * np.cos(angle[1]) for angle in angles]
        y = config.radius * [np.sin(angle[0]) * np.sin(angle[1]) for angle in angles]
        z = config.radius * [np.cos(angle[0]) for angle in angles]
        sc = ax.scatter3D(
            x, y, z, s=400, c=curvature_norms, cmap="Spectral", norm=color_norm
        )
        plt.colorbar(sc)
        ax.set_title(f"{profile_type} mean curvature norm profile", fontsize=30)
    elif config.dataset_name == "t2_synthetic":
        ax = fig.add_subplot(111, projection="3d")
        x = [
            (config.major_radius - config.minor_radius * np.cos(angle[0]))
            * np.cos(angle[1])
            for angle in angles
        ]
        y = [
            (config.major_radius - config.minor_radius * np.cos(angle[0]))
            * np.sin(angle[1])
            for angle in angles
        ]
        z = [config.minor_radius * np.sin(angle[0]) for angle in angles]
        sc = ax.scatter3D(
            x, y, z, s=400, c=curvature_norms, cmap="Spectral", norm=color_norm
        )
        plt.colorbar(sc)
        ax.set_title(f"{profile_type} mean curvature norm profile", fontsize=30)
        ax.set_xlim(
            -(config.major_radius + config.minor_radius),
            (config.major_radius + config.minor_radius),
        )
        ax.set_ylim(
            -(config.major_radius + config.minor_radius),
            (config.major_radius + config.minor_radius),
        )
        ax.set_zlim(
            -(config.major_radius + config.minor_radius),
            (config.major_radius + config.minor_radius),
        )
        plt.axis("off")

    if config.dataset_name in ["s2_synthetic", "t2_synthetic", "grid_cells"]:
        if norm_val != None:
            plotly_fig = go.Figure(
                data=[
                    go.Scatter3d(
                        x=x,
                        y=y,
                        z=z,
                        mode="markers",
                        marker=dict(
                            size=10,
                            color=curvature_norms,  # set color to an array/list of desired values
                            colorscale="plasma",  # choose a colorscale
                            opacity=0.8,
                            cmin=0,
                            cmax=float(norm_val),
                            colorbar=dict(title="Norm of curvature", tickmode="auto"),
                        ),
                    )
                ]
            )
        else:
            plotly_fig = go.Figure(
                data=[
                    go.Scatter3d(
                        x=x,
                        y=y,
                        z=z,
                        mode="markers",
                        marker=dict(
                            size=10,
                            color=curvature_norms,  # set color to an array/list of desired values
                            colorscale="plasma",  # choose a colorscale
                            opacity=0.8,
                            colorbar=dict(title="Norm of curvature", tickmode="auto"),
                        ),
                    )
                ]
            )

        plotly_fig.update_layout(
            scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
            title=dict(text="Profile of Curvature Norm", font=dict(size=24), x=0.5),
        )

        plotly_fig.update_layout(
            scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
            title=dict(text="Profile of Curvature Norm", font=dict(size=24), x=0.5),
        )

        pio.write_html(
            plotly_fig,
            os.path.join(
                HTML_FIGURES,
                f"{config.results_prefix}_curv_profile_{profile_type}.html",
            ),
        )

    plt.savefig(
        os.path.join(
            FIGURES, f"{config.results_prefix}_curv_profile_{profile_type}.png"
        )
    )
    # plt.savefig(
    #     os.path.join(
    #         FIGURES, f"{config.results_prefix}_curv_profile_{profile_type}.svg"
    #     )
    # )

    return fig


def plot_neural_manifold_learned(curv_norm_learned_profile, config, labels):

    if config.dataset_name == "experimental":
        stats = [
            "mean_velocities",
            "median_velocities",
            "std_velocities",
            "min_velocities",
            "max_velocities",
        ]
        cmaps = ["viridis", "viridis", "magma", "Blues", "Reds"]

        fig, axes = plt.subplots(
            nrows=1,
            ncols=len(stats),
            figsize=(20, 4),
            subplot_kw={"projection": "polar"},
        )
        for i_stat, stat_velocities in enumerate(stats):
            ax = axes[i_stat]
            ax.scatter(
                # Note: using the geodesic distance makes the plot
                # reparameterization invariant.
                # However, the computation is extremely slow, thus
                # we recommend using z_grid for the main pipeline
                # and computing geodesic_dist in the notebook 07
                # after having selected a run.
                curv_norm_learned_profile["z_grid"],
                1 / curv_norm_learned_profile["curv_norm_learned"],
                c=curv_norm_learned_profile[stat_velocities],
                cmap=cmaps[i_stat],
            )
            ax.plot(
                curv_norm_learned_profile["z_grid"],
                1 / curv_norm_learned_profile["curv_norm_learned"],
                c="black",
            )
            ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
            ax.grid(True)
            ax.set_title("Color: " + stat_velocities, va="bottom")
            fig.tight_layout()
    else:
        fig, ax = plt.subplots(
            nrows=1, ncols=1, figsize=(20, 4), subplot_kw={"projection": "polar"}
        )

        ax.scatter(
            # Note: using the geodesic distance makes the plot
            # reparameterization invariant.
            # However, the computation is extremely slow, thus
            # we recommend using z_grid for the main pipeline
            # and computing geodesic_dist in the notebook 07
            # after having selected a run.
            curv_norm_learned_profile["z_grid"],
            1 / curv_norm_learned_profile["curv_norm_learned"],
        )
        ax.plot(
            curv_norm_learned_profile["z_grid"],
            1 / curv_norm_learned_profile["curv_norm_learned"],
            c="black",
        )
        ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
        ax.grid(True)
        fig.tight_layout()

    plt.savefig(
        os.path.join(FIGURES, f"{config.results_prefix}_neural_manifold_learned.png")
    )
    # plt.savefig(
    #     os.path.join(FIGURES, f"{config.results_prefix}_neural_manifold_learned.svg")
    # )

    return fig


def plot_comparison_curvature_norms(
    angles, curvature_norms_true, curvature_norms_learned, error, config
):

    if config.dataset_name == "s1_synthetic":
        fig, ax = plt.subplots(figsize=(20, 20))
        ax.plot(angles, curvature_norms_true, "--", label="true")
        ax.plot(angles, curvature_norms_learned, label="learned")
        ax.set_xlabel("angle", fontsize=40)
        ax.legend(prop={"size": 40}, loc="upper right")
        ax.set_title("Error = " + "%.3f" % error, fontsize=30)
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
    elif config.dataset_name == "s2_synthetic":
        fig = plt.figure(figsize=(20, 20))
        ax_analytic = plt.add_subplot(121, projection="3d")
        ax_learned = plt.add_subplot(122, projection="3d")
        x = [np.sin(angle[0]) * np.cos(angle[1]) for angle in angles]
        y = [np.sin(angle[0]) * np.sin(angle[1]) for angle in angles]
        z = [np.cos(angle[0]) for angle in angles]
        ax_analytic.scatter3D(x, y, z, s=5, c=curvature_norms_true)
        ax_learned.scatter3D(x, y, z, s=5, c=curvature_norms_learned)

    plt.savefig(os.path.join(FIGURES, f"{config.results_prefix}_comparison.png"))
    # plt.savefig(os.path.join(FIGURES, f"{config.results_prefix}_comparison.svg"))
    return fig


def plot_persistence_diagrams(diagrams):
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_xlabel("Birth")
    ax.set_ylabel("Death")
    ax.set_title("Persistence Diagram")
    for n, H_n in enumerate(diagrams):
        birth_n = H_n[:, 0]
        death_n = H_n[:, 1]
        ax.scatter(birth_n, death_n, s=10, label="H_" + str(n))
        ax.plot([0, 1], [0, 1], transform=ax.transAxes, c="gray", ls="--")

    ax.legend(loc="lower right")

    # plt.savefig(os.path.join(FIGURES, f"{config.results_prefix}_persistence_diagrams.png"))


def plot_grids(grids, arena_dims):
    """Visualize the the firing lattices for all grid cells."""
    colormap = plt.get_cmap("hsv")
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.set_aspect("equal")
    ax.set_facecolor("darkblue")
    ax.set_xlim(-arena_dims[0] / 2 - 0.5, arena_dims[0] / 2 + 0.5)
    ax.set_ylim(-arena_dims[1] / 2 - 0.5, arena_dims[1] / 2 + 0.5)
    ax.set_title("Hexagonal grids")
    ax.set_xlabel("x-position")
    ax.set_ylabel("y-position")
    for lattice in grids:
        plt.scatter(lattice[:, 0], lattice[:, 1])
    # plt.savefig(os.path.join(FIGURES, f"{config.results_prefix}_grid_lattices.png"))


def plot_grid_rate_maps(rate_maps):
    fig = plt.figure(figsize=(15, 12))
    n_cells = rate_maps.shape[0]
    for cell_index in range(n_cells):
        ax = fig.add_subplot(int(np.ceil(n_cells / 2)), 2, cell_index + 1)
        img = ax.imshow(rate_maps[cell_index])
        plt.colorbar(img, label="Relative Firing Rate")
        ax.set_title("Grid Cell #" + str(cell_index + 1) + " Firing Rate Map")
    # plt.savefig(os.path.join(FIGURES, f"{config.results_prefix}_grid_rate_maps.png"))



def plot_activity_with_mi(expt_id,name,neural_activity,task_variable,mutual_info):
    """ Visualize neural activity perisitimulus histogram and show mutual information between neural activity and task variable.
    
    """
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, gridspec_kw={'width_ratios': [1, 30]}, figsize=(20,10))

    vector = mutual_info

    img1 = ax1.imshow(vector[:, np.newaxis], aspect='auto', cmap='viridis', vmin=vector.min(), vmax=vector.max())
    ax1.set_ylabel(f"Neuron - {name} Mutual Information",fontsize=25)
    ax1.set_yticks(np.arange(0,len(mutual_info)))
    ax1.set_xticks([])


    img2 = ax2.imshow(neural_activity.T, aspect="auto",cmap="viridis",norm="symlog",interpolation='none')
    ax2.set_yticks([])
    ax2.set_title(f"Neural Activity vs {name}, Experiment {expt_id} -- Symlog scale",fontsize=25)
    ax2.set_xticks(np.arange(len(task_variable))[::500])
    ax2.set_xticklabels(task_variable[::500].astype(int))
    ax2.set_xlabel(f"{name}",fontsize=25)

    for i in range(len(neural_activity.T)):
        ax2.axhline(i-0.5, color='black', lw=1)
        ax1.axhline(i-0.5, color='black', lw=1)

    ax2.tick_params(axis='both', which='major', labelsize=15)
    plt.subplots_adjust(wspace=0.0)

    # Create a new axis for the colorbar at the bottom
    cbar_ax = fig.add_axes([0.95, 0.1, 0.02, 0.75])

    # Add a colorbar with label
    cbar = fig.colorbar(img2, cax=cbar_ax, orientation='vertical')
    cbar.set_label('Firing rate (spikes/second)', fontsize=25, labelpad=15)
    cbar_ax.tick_params(axis='both', which='major', labelsize=15)

    plt.show()

