import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch

FIGURES = "results/figures/"
if not os.path.exists(FIGURES):
    os.makedirs(FIGURES)


def plot_loss(train_losses, test_losses, config):
    fig, ax = plt.subplots(figsize=(20, 20))
    epochs = [epoch for epoch in range(1, config.n_epochs + 1)]
    ax.plot(epochs, train_losses, label="train")
    ax.plot(epochs, test_losses, label="test")
    ax.set_title("Losses", fontsize=40)
    ax.set_xlabel("epoch", fontsize=40)
    ax.legend(prop={"size": 40})
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.savefig(os.path.join(FIGURES, f"{config.results_prefix}_losses.png"))
    # plt.savefig(os.path.join(FIGURES, f"{config.results_prefix}_losses.svg"))
    return fig


def plot_recon(model, dataset_torch, labels, config):
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
        sc_data = ax_data.scatter(x_data, y_data, c=labels["angles"], cmap=colormap)
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        ax_rec = fig.add_subplot(1, 2, 2)
        ax_rec.set_title("Reconstruction", fontsize=40)
        sc_rec = ax_rec.scatter(x_rec, y_rec, c=labels["angles"], cmap=colormap)
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        # plt.colorbar(sc_rec)
    elif config.dataset_name in ("s2_synthetic", "t2_synthetic"):
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
        ax_data.scatter3D(x_data, y_data, z_data, s=5, c=norms_data)
        plt.axis("off")
        # ax_data.view_init(elev=60, azim=45, roll=0)
        ax_rec = fig.add_subplot(1, 2, 2, projection="3d")
        ax_rec.set_title("Reconstruction", fontsize=40)
        ax_rec.scatter3D(x_rec, y_rec, z_rec, s=5, c=norms_rec)
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
    plt.savefig(os.path.join(FIGURES, f"{config.results_prefix}_recon.svg"))
    return fig


def plot_latent_space(model, dataset_torch, labels, config):
    fig = plt.figure(figsize=(20, 20))
    if config.dataset_name in ("s1_synthetic", "experimental"):
        ax = fig.add_subplot(111)
        z, _, _ = model(dataset_torch.to(config.device))
        colormap = plt.get_cmap("twilight")
        z0 = z[:, 0]
        z0 = [_.item() for _ in z0]
        z1 = z[:, 1]
        z1 = [_.item() for _ in z1]
        sc = ax.scatter(z0, z1, c=labels["angles"], s=10, cmap=colormap)
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
    elif config.dataset_name in ("s2_synthetic", "t2_synthetic"):
        ax = fig.add_subplot(111, projection="3d")
        z, _, _ = model(dataset_torch.to(config.device))
        z0 = z[:, 0]
        z0 = [_.item() for _ in z0]
        z1 = z[:, 1]
        z1 = [_.item() for _ in z1]
        z2 = z[:, 2]
        z2 = [_.item() for _ in z2]
        sc = ax.scatter3D(z0, z1, z2)
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
        ax1.plot(angles, curvature_norms)
        ax1.set_xlabel("angle", fontsize=30)
        ax1.set_ylabel("mean curvature norm", fontsize=30)

        ax2 = fig.add_subplot(122, projection="polar")
        sc = ax2.scatter(
            angles,
            np.ones_like(angles),
            c=curvature_norms,
            s=20,
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
            x, y, z, s=30, c=curvature_norms, cmap="Spectral", norm=color_norm
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
            x, y, z, s=30, c=curvature_norms, cmap="Spectral", norm=color_norm
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

    plt.savefig(
        os.path.join(
            FIGURES, f"{config.results_prefix}_curv_profile_{profile_type}.png"
        )
    )
    plt.savefig(
        os.path.join(
            FIGURES, f"{config.results_prefix}_curv_profile_{profile_type}.svg"
        )
    )

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
    plt.savefig(os.path.join(FIGURES, f"{config.results_prefix}_comparison.svg"))
    return fig
