import os

os.environ["GEOMSTATS_BACKEND"] = "pytorch"


import geomstats.backend as gs
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch

import neural_metric
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
import numpy as np
import torch.nn.functional as F
from datasets.synthetic import get_synth_immersion
import scipy.signal
import main


def get_model_immersion(model):
    def model_immersion(angle):
        z = gs.array([gs.cos(angle), gs.sin(angle)])
        z = z.to(main.config.device)
        x_mu = model.decode(z)
        return x_mu

    return model_immersion


def compute_extrinsic_curvature(angles, immersion, embedding_dim):

    mean_curvature = gs.zeros(len(angles), embedding_dim)
    for _, angle in enumerate(angles):
        for i in range(embedding_dim):
            mean_curvature[_, i] = torch.autograd.functional.hessian(
                func=lambda x: immersion(x)[i], inputs=angle, strict=True
            )

    mean_curvature_norm = torch.linalg.norm(mean_curvature, dim=1, keepdim=True)
    mean_curvature_norm = [_.item() for _ in mean_curvature_norm]

    return mean_curvature, mean_curvature_norm


def compute_intrinsic_curvature():
    return NotImplementedError


def get_mean_curvature(model, angles, embedding_dim):
    model.eval()
    immersion = get_model_immersion(model)
    mean_curvature, mean_curvature_norm = compute_extrinsic_curvature(
        angles, immersion, embedding_dim
    )
    return mean_curvature, mean_curvature_norm


def get_mean_curvature_synth(angles, config, synth_rotation):
    immersion = get_synth_immersion(
        amp_func=config.amp_func,
        radius=config.radius,
        n_wiggles=config.n_wiggles,
        amp_wiggles=config.amp_wiggles,
        embedding_dim=config.embedding_dim,
        rot=synth_rotation,
    )
    mean_curvature_synth, mean_curvature_norm_synth = compute_extrinsic_curvature(
        angles, immersion, config.embedding_dim
    )

    return mean_curvature_synth, mean_curvature_norm_synth


def get_cross_corr(signal1, signal2):
    s1 = np.squeeze(signal1)
    s1 = s1 - np.mean(s1)
    s1 = s1 / np.linalg.norm(s1)
    s2 = np.squeeze(signal2)
    s2 = s2 - np.mean(s2)
    s2 = s2 / np.linalg.norm(s2)
    correlation = np.correlate(s1, s2, mode="same")
    lags = scipy.signal.correlation_lags(s1.size, s2.size, mode="same")
    lag = lags[np.argmax(correlation)]
    s1 = np.roll(s1, -lag)
    return s1, s2, correlation


def plot_curvature_profile(angles, mean_curvature_norms):

    colormap = plt.get_cmap("twilight")
    color_norm = mpl.colors.Normalize(0.0, 1.2 * max(mean_curvature_norms))
    plt.figure(figsize=(12, 5))

    ax2 = plt.subplot(1, 2, 1, polar=True)
    sc = ax2.scatter(
        angles,
        np.ones_like(angles),
        c=mean_curvature_norms,
        s=10,
        cmap=colormap,
        norm=color_norm,
        linewidths=0,
    )
    # ax1.set_yticks([])
    ax2.set_yticks([])

    plt.colorbar(sc)

    ax1 = plt.subplot(1, 2, 2)

    pt = ax1.plot(angles, mean_curvature_norms)

    ax1.set_xlabel("angle")


def plot_curv_synth(figure, angles, mean_curvature_norms):
    colormap = plt.get_cmap("twilight")
    color_norm = mpl.colors.Normalize(0.0, 1.2 * max(mean_curvature_norms))

    ax_circle_synth = figure.add_subplot(3, 3, 4, polar=True)
    sc_circle = ax_circle_synth.scatter(
        angles,
        np.ones_like(angles),
        c=mean_curvature_norms,
        s=20,
        cmap=colormap,
        norm=color_norm,
        linewidths=0,
    )
    plt.xticks(fontsize=30)
    ax_circle_synth.set_yticks([])

    plt.colorbar(sc_circle)

    ax_profile_synth = figure.add_subplot(3, 3, 5)

    ax_profile_synth.plot(angles, mean_curvature_norms)
    ax_profile_synth.set_title("Analytic mean curvature profile", fontsize=40)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)

    ax_profile_synth.set_xlabel("angle", fontsize=40)


def plot_curv_learned(figure, angles, mean_curvature_norms):
    colormap = plt.get_cmap("twilight")
    color_norm = mpl.colors.Normalize(0.0, 1.2 * max(mean_curvature_norms))

    ax_circle = figure.add_subplot(3, 3, 7, polar=True)
    sc_circle = ax_circle.scatter(
        angles,
        np.ones_like(angles),
        c=mean_curvature_norms,
        s=20,
        cmap=colormap,
        norm=color_norm,
        linewidths=0,
    )
    ax_circle.set_yticks([])
    plt.xticks(fontsize=30)

    plt.colorbar(sc_circle)

    ax_profile = figure.add_subplot(3, 3, 8)

    ax_profile.plot(angles, mean_curvature_norms)
    ax_profile.set_title("Learned mean curvature profile", fontsize=40)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)

    ax_profile.set_xlabel("angle", fontsize=40)


def plot_recon(figure, model, dataset_torch, labels):
    colormap = plt.get_cmap("twilight")
    x_data = dataset_torch[:, 0]
    y_data = dataset_torch[:, 1]
    angles = torch.linspace(0, 2 * gs.pi, 2000)
    z = torch.stack([torch.cos(angles), torch.sin(angles)], axis=-1)
    z = z.to(main.config.device)
    rec = model.decode(z)
    x_rec = rec[:, 0]#.cpu().detach().numpy()
    x_rec = [x.item() for x in x_rec]
    y_rec = rec[:, 1]#.cpu().detach().numpy()
    y_rec = [y.item() for y in y_rec]
    ax_data = figure.add_subplot(3, 3, 2)
    ax_data.set_title("Synthetic data", fontsize=40)
    sc_data = ax_data.scatter(x_data, y_data, c=labels["angles"], cmap=colormap)
    ax_rec = figure.add_subplot(3, 3, 3)
    ax_rec.set_title("Reconstruction", fontsize=40)
    sc_rec = ax_rec.scatter(x_rec, y_rec, c=labels["angles"], cmap=colormap)
    plt.colorbar(sc_rec)


def plot_latent_space(figure, model, dataset_torch, labels):
    _, posterior_params = model(dataset_torch.to(main.config.device))

    z, _, _ = model.reparameterize(posterior_params)

    ax_latent = figure.add_subplot(3, 3, 6)
    colormap = plt.get_cmap("twilight")

    z0 = z[:, 0]
    z0 = [_.item() for _ in z0]
    z1 = z[:, 1]
    z1 = [_.item() for _ in z1]

    ax_latent.set_title("Latent space", fontsize=40)

    sc_latent = ax_latent.scatter(z0, z1, c=labels["angles"], s=5, cmap=colormap)


def plot_loss(figure, train_losses, test_losses):
    ax_loss = figure.add_subplot(3, 3, 1)
    ax_loss.plot(train_losses, label="train")
    ax_loss.plot(test_losses, label="test")
    ax_loss.set_title("Losses", fontsize=40)
    ax_loss.set_xlabel("epoch", fontsize=40)
    ax_loss.legend(prop={"size": 40})
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)


def master_plot(
    model,
    dataset_torch,
    labels,
    angles,
    mean_curvature_norms,
    mean_curvature_norms_synth,
    s1,
    s2,
    pearson,
    train_losses,
    test_losses,
    config,
):
    model.eval()
    figure = plt.figure(figsize=(40, 40))
    plot_loss(figure, train_losses, test_losses)
    plot_recon(figure, model, dataset_torch, labels)
    plot_latent_space(figure, model, dataset_torch, labels)
    plot_curv_synth(figure, angles, mean_curvature_norms_synth)
    plot_curv_learned(figure, angles, mean_curvature_norms)
    ax1 = figure.add_subplot(3, 3, 9)
    ax1.plot(s1)
    ax1.plot(s2)
    plt.xticks(fontsize=30)
    ax1.set_title("Pearson correlation = " + str(pearson), fontsize=30)
    plt.savefig(f"results/figures/{config.results_prefix}_master_plot.png")
    return figure

