import os

os.environ["GEOMSTATS_BACKEND"] = "pytorch"

import geomstats.backend as gs
import numpy as np
import torch
from datasets.synthetic import (
    get_s1_synthetic_immersion,
    get_s2_synthetic_immersion,
    get_t2_synthetic_immersion,
)
from geomstats.geometry.pullback_metric import PullbackMetric


def get_learned_immersion(model, config):
    """Define immersion from latent angles to neural manifold."""

    def immersion(angle):
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
        elif config.dataset_name == "t2_synthetic":
            theta = angle[0]
            phi = angle[1]
            z = gs.array(
                [
                    (config.major_radius + config.minor_radius*gs.cos(theta)) * gs.cos(phi),
                    (config.major_radius + config.minor_radius*gs.cos(theta)) * gs.sin(phi),
                    config.minor_radius*gs.sin(theta),
                ]
            )

        z = z.to(config.device)
        x_mu = model.decode(z)

        return x_mu

    return immersion


def get_true_immersion(config):
    if config.dataset_name == "s1_synthetic":
        immersion = get_s1_synthetic_immersion(
            distortion_func=config.distortion_func,
            radius=config.radius,
            n_wiggles=config.n_wiggles,
            distortion_amp=config.distortion_amp,
            embedding_dim=config.embedding_dim,
            rot=config.synthetic_rotation,
        )
    elif config.dataset_name == "s2_synthetic":
        immersion = get_s2_synthetic_immersion(
            radius=config.radius,
            distortion_amp=config.distortion_amp,
            embedding_dim=config.embedding_dim,
            rot=config.synthetic_rotation,
        )
    elif config.dataset_name == "t2_synthetic":
        immersion = get_t2_synthetic_immersion(
            major_radius=config.major_radius,
            minor_radius=config.minor_radius,
            distortion_amp=config.distortion_amp,
            embedding_dim=config.embedding_dim,
            rot=config.synthetic_rotation,
        )
    return immersion


def get_z_grid(config): 
    if config.dataset_name in ("s1_synthetic", "experimental"):
        z_grid = torch.linspace(0, 2 * gs.pi, config.n_times)
    elif config.dataset_name == "s2_synthetic":
        thetas = gs.linspace(0.01, gs.pi, config.n_times)
        phis = gs.linspace(0, 2 * gs.pi, config.n_times)
        z_grid = torch.cartesian_prod(thetas, phis)
    elif config.dataset_name == "t2_synthetic":
        thetas = gs.linspace(0, 2 * gs.pi, config.n_times)
        phis = gs.linspace(0, 2 * gs.pi, config.n_times)
        z_grid = torch.cartesian_prod(thetas, phis)
    return z_grid


# Compute mean curvature vector at each point, along with their corresponding magnitudes using Geomstats
def _compute_mean_curvature(z_grid, immersion, dim, embedding_dim):
    neural_metric = PullbackMetric(
        dim=dim, embedding_dim=embedding_dim, immersion=immersion
    )
    # curv = gs.zeros(len(z_grid), embedding_dim)
    # for _, z in enumerate(z_grid):
    #     # TODO(nina): Vectorize in geomstats to avoid this for loop
    #     z = torch.unsqueeze(z, dim=0)
    #     curv[_, :] = neural_metric.mean_curvature_vector(z)

    curv = neural_metric.mean_curvature_vector(z_grid)

    curv_norm = torch.linalg.norm(curv, dim=1, keepdim=True)
    curv_norm = gs.array([norm.item() for norm in curv_norm])

    return curv, curv_norm


# Uses compute_mean_curvature to find mean curvature profile from true expression of the immersion
def compute_mean_curvature_learned(model, config):
    z_grid = get_z_grid(config)
    immersion = get_learned_immersion(model, config)
    curv, curv_norm = _compute_mean_curvature(
        z_grid=z_grid,
        immersion=immersion,
        dim=config.manifold_dim,
        embedding_dim=config.embedding_dim,
    )
    return z_grid, curv, curv_norm


# Uses compute_mean_curvature to find mean curvature profile from true expression of the immersion
def compute_mean_curvature_true(config):
    z_grid = get_z_grid(config)
    immersion = get_true_immersion(config)
    curv, curv_norm = _compute_mean_curvature(
        z_grid=z_grid,
        immersion=immersion,
        dim=config.manifold_dim,
        embedding_dim=config.embedding_dim,
    )
    return z_grid, curv, curv_norm


# Computes "error" of learned curvature profile given true profile, for S^1
def _compute_error_s1(thetas, curv_norms_learned, curv_norms_true):
    curv_norms_learned = np.array(curv_norms_learned)
    curv_norms_true = np.array(curv_norms_true)
    diff = np.trapz((curv_norms_learned - curv_norms_true) ** 2, thetas)
    normalization = np.trapz(curv_norms_learned**2, thetas) + np.trapz(
        curv_norms_true**2, thetas
    )

    return diff / normalization


# Helper function for compute_error_s2, integrates over S^2
def _integrate_s2(thetas, phis, h):
    sum_phis = torch.zeros_like(thetas)
    for t, theta in enumerate(thetas):
        sum_phis[t] = torch.trapz(
            y=h[len(phis) * t : len(phis) * (t + 1)], x=phis
        ) * np.sin(theta)
    integral = torch.trapz(y=sum_phis, x=thetas)
    return integral


# Computes "error" of learned curvature profile given true profile, for S^2
def _compute_error_s2(thetas, phis, curv_norms_learned, curv_norms_true):
    diff = _integrate_s2(thetas, phis, (curv_norms_learned - curv_norms_true) ** 2)
    normalization = _integrate_s2(
        thetas, phis, (curv_norms_learned) ** 2 + (curv_norms_true) ** 2
    )
    return diff / normalization


def _integrate_t2(thetas, phis, h):
    # TODO
    return 0


def _compute_error_t2(thetas, phis, curv_norms_learned, curv_norms_true):
    # TODO
    return 0


def compute_error(
    z_grid, curv_norms_learned, curv_norms_true, config
):  # Calculate method error
    if config.dataset_name == "s1_synthetic":
        thetas = z_grid
        error = _compute_error_s1(thetas, curv_norms_learned, curv_norms_true)
    elif config.dataset_name == "s2_synthetic":
        thetas = z_grid[:, 0]
        phis = z_grid[:, 1]
        error = _compute_error_s2(thetas, phis, curv_norms_learned, curv_norms_true)
    elif config.dataset_name == "t2_synthetic":
        thetas = z_grid[:, 0]
        phis = z_grid[:, 1]
        error = _compute_error_s2(thetas, phis, curv_norms_learned, curv_norms_true)
    return error
