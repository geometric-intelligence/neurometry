import os

os.environ["GEOMSTATS_BACKEND"] = "pytorch"

import geomstats.backend as gs
import numpy as np
import torch
from datasets.synthetic import get_s1_synthetic_immersion, get_s2_synthetic_immersion
from geomstats.geometry.pullback_metric import PullbackMetric


def get_neural_immersion(model, config):
    """Define immersion from latent angles to neural manifold."""

    def neural_immersion(angle):
        if config.dataset_name in ("s1_synthetic", "experimental"):
            z = gs.array([gs.cos(angle[0]), gs.sin(angle[0])])

        elif config.dataset_name == "s2_synthetic":
            theta = angle[0]
            phi = angle[1]
            z = gs.array(
                [
                    gs.sin(theta) * gs.cos(phi),
                    gs.sin(theta) * gs.sin(phi),
                    gs.cos(theta),
                ]
            )

        z = z.to(config.device)
        x_mu = model.decode(z)

        return x_mu

    return neural_immersion


# Compute mean curvature vector at each point, along with their corresponding magnitudes using Geomstats
def compute_mean_curvature(latent_angle, neural_immersion, dim, embedding_dim):
    neural_metric = PullbackMetric(
        dim=dim, embedding_dim=embedding_dim, immersion=neural_immersion
    )
    mean_curvature_vec = gs.zeros(len(latent_angle), embedding_dim)
    for _, angle in enumerate(latent_angle):
        angle = torch.unsqueeze(angle, dim=0)
        mean_curvature_vec[_, :] = neural_metric.mean_curvature_vector(angle)

    mean_curvature_norms = torch.linalg.norm(mean_curvature_vec, dim=1, keepdim=True)
    mean_curvature_norms = gs.array([norm.item() for norm in mean_curvature_norms])

    return mean_curvature_vec, mean_curvature_norms


# Uses compute_mean_curvature to find mean curvature profile from analytic expression of the immersion
def get_mean_curvature_analytic(angles, config):
    if config.dataset_name == "s1_synthetic":
        immersion = get_s1_synthetic_immersion(
            distortion_func=config.distortion_func,
            radius=config.radius,
            n_wiggles=config.n_wiggles,
            distortion_amp=config.distortion_amp,
            embedding_dim=config.embedding_dim,
            rot=config.synthetic_rotation,
        )
        dim = 1
    elif config.dataset_name == "s2_synthetic":
        immersion = get_s2_synthetic_immersion(
            radius=config.radius,
            distortion_amp=config.distortion_amp,
            embedding_dim=config.embedding_dim,
            rot=config.synthetic_rotation,
        )
        dim = 2

    mean_curvature_analytic, mean_curvature_norm_analytic = compute_mean_curvature(
        latent_angle=angles,
        neural_immersion=immersion,
        dim=dim,
        embedding_dim=config.embedding_dim,
    )

    return mean_curvature_analytic, mean_curvature_norm_analytic


# Computes "error" of learned curvature profile given analytic profile, for S^1
def get_difference(thetas, h1, h2):
    h1 = np.array(h1)
    h2 = np.array(h2)
    diff = np.trapz((h1 - h2) ** 2, thetas)
    normalization = np.trapz(h1**2, thetas) + np.trapz(h2**2, thetas)

    return diff / normalization


# Helper function for get_difference_s2, integrates over S^2
def integrate_sphere(thetas, phis, h):
    sum_phis = torch.zeros_like(thetas)
    for t, theta in enumerate(thetas):
        sum_phis[t] = torch.trapz(
            y=h[len(phis) * t : len(phis) * (t + 1)], x=phis
        ) * np.sin(theta)
    integral = torch.trapz(y=sum_phis, x=thetas)
    return integral


# Computes "error" of learned curvature profile given analytic profile, for S^2
def get_difference_s2(
    thetas, phis, mean_curvature_norms, mean_curvature_norms_analytic
):
    diff = integrate_sphere(
        thetas, phis, (mean_curvature_norms - mean_curvature_norms_analytic) ** 2
    )
    normalization = integrate_sphere(
        thetas, phis, (mean_curvature_norms) ** 2 + (mean_curvature_norms_analytic) ** 2
    )
    return diff / normalization
