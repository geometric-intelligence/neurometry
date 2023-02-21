import os

os.environ["GEOMSTATS_BACKEND"] = "pytorch"

import time

import geomstats.backend as gs
import gph
import numpy as np
import torch
from datasets.synthetic import (
    get_s1_synthetic_immersion,
    get_s2_synthetic_immersion,
    get_t2_synthetic_immersion,
)
from geomstats.geometry.pullback_metric import PullbackMetric
from geomstats.geometry.special_orthogonal import SpecialOrthogonal  # NOQA


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
                    (config.major_radius - config.minor_radius * gs.cos(theta))
                    * gs.cos(phi),
                    (config.major_radius - config.minor_radius * gs.cos(theta))
                    * gs.sin(phi),
                    config.minor_radius * gs.sin(theta),
                ]
            )

        z = z.to(config.device)
        x_mu = model.decode(z)

        return x_mu

    return immersion


def get_true_immersion(config):
    rot = torch.eye(n=config.embedding_dim)
    if config.synthetic_rotation == "random":
        rot = SpecialOrthogonal(n=config.embedding_dim).random_point()
    if config.dataset_name == "s1_synthetic":
        immersion = get_s1_synthetic_immersion(
            geodesic_distortion_func=config.geodesic_distortion_func,
            radius=config.radius,
            n_wiggles=config.n_wiggles,
            geodesic_distortion_amp=config.geodesic_distortion_amp,
            embedding_dim=config.embedding_dim,
            rot=rot,
        )
    elif config.dataset_name == "s2_synthetic":
        immersion = get_s2_synthetic_immersion(
            radius=config.radius,
            geodesic_distortion_amp=config.geodesic_distortion_amp,
            embedding_dim=config.embedding_dim,
            rot=rot,
        )
    elif config.dataset_name == "t2_synthetic":
        immersion = get_t2_synthetic_immersion(
            major_radius=config.major_radius,
            minor_radius=config.minor_radius,
            geodesic_distortion_amp=config.geodesic_distortion_amp,
            embedding_dim=config.embedding_dim,
            rot=rot,
        )
    return immersion


def get_z_grid(config, n_grid_points=100):
    if config.dataset_name in ("s1_synthetic", "experimental"):
        z_grid = torch.linspace(0, 2 * gs.pi, n_grid_points)
    elif config.dataset_name == "s2_synthetic":
        thetas = gs.linspace(0.01, gs.pi, n_grid_points)
        phis = gs.linspace(0, 2 * gs.pi, n_grid_points)
        z_grid = torch.cartesian_prod(thetas, phis)
    elif config.dataset_name == "t2_synthetic":
        thetas = gs.linspace(0, 2 * gs.pi, n_grid_points)
        phis = gs.linspace(0, 2 * gs.pi, n_grid_points)
        z_grid = torch.cartesian_prod(thetas, phis)
    return z_grid


def _compute_curvature(z_grid, immersion, dim, embedding_dim):
    """Compute mean curvature vector and its norm at each point."""
    neural_metric = PullbackMetric(
        dim=dim, embedding_dim=embedding_dim, immersion=immersion
    )
    z0 = torch.unsqueeze(z_grid[0], dim=0)
    if dim == 1:
        curv = gs.zeros(len(z_grid), embedding_dim)
        geodesic_dist = gs.zeros(len(z_grid))

        for i_z, z in enumerate(z_grid):
            # TODO(nina): Vectorize in geomstats to
            # - avoid this for loop
            # - be able to use batch normalization (needs batch's len > 1)
            z = torch.unsqueeze(z, dim=0)
            curv[i_z, :] = neural_metric.mean_curvature_vector(z)
            # Note: these lines are commented out (see PR description)
            # as it makes the computations extremely long.
            # Recommendation: compute these offline in a notebook
            # if i_z > 1:
            #     geodesic_dist[i_z] = neural_metric.dist(z0, z)
    else:
        curv = neural_metric.mean_curvature_vector(z_grid)

    curv_norm = torch.linalg.norm(curv, dim=1, keepdim=True)
    curv_norm = gs.array([norm.item() for norm in curv_norm])

    return geodesic_dist, curv, curv_norm


def compute_curvature_learned(model, config, embedding_dim, n_grid_points=100):
    """Use _compute_curvature to find mean curvature profile from learned immersion"""
    z_grid = get_z_grid(config=config, n_grid_points=n_grid_points)
    immersion = get_learned_immersion(model, config)
    start_time = time.time()
    geodesic_dist, curv, curv_norm = _compute_curvature(
        z_grid=z_grid,
        immersion=immersion,
        dim=config.manifold_dim,
        embedding_dim=embedding_dim,
    )
    end_time = time.time()
    print("Computation time: " + "%.3f" % (end_time - start_time) + " seconds.")
    return z_grid, geodesic_dist, curv, curv_norm


def compute_curvature_true(config, n_grid_points=100):
    """Use compute_mean_curvature to find mean curvature profile from true immersion"""
    z_grid = get_z_grid(config=config, n_grid_points=n_grid_points)
    immersion = get_true_immersion(config)
    start_time = time.time()
    geodesic_dist, curv, curv_norm = _compute_curvature(
        z_grid=z_grid,
        immersion=immersion,
        dim=config.manifold_dim,
        embedding_dim=config.embedding_dim,
    )
    end_time = time.time()
    print("Computation time: " + "%.3f" % (end_time - start_time) + " seconds.")
    return z_grid, geodesic_dist, curv, curv_norm


def _compute_curvature_error_s1(thetas, curv_norms_learned, curv_norms_true):
    """Compute "error" of learned curvature profile given true profile for S1."""
    curv_norms_learned = np.array(curv_norms_learned)
    curv_norms_true = np.array(curv_norms_true)
    diff = np.trapz((curv_norms_learned - curv_norms_true) ** 2, thetas)
    normalization = np.trapz(curv_norms_learned**2, thetas) + np.trapz(
        curv_norms_true**2, thetas
    )

    return diff / normalization


def _integrate_s2(thetas, phis, h):
    """Helper function for compute_curvature_error_s2.

    This function integrates over S^2.
    """
    sum_phis = torch.zeros_like(thetas)
    for t, theta in enumerate(thetas):
        sum_phis[t] = torch.trapz(
            y=h[len(phis) * t : len(phis) * (t + 1)], x=phis
        ) * np.sin(theta)
    integral = torch.trapz(y=sum_phis, x=thetas)
    return integral


def _compute_curvature_error_s2(thetas, phis, curv_norms_learned, curv_norms_true):
    """Compute "error" of learned curvature profile given true profile for S2."""
    diff = _integrate_s2(thetas, phis, (curv_norms_learned - curv_norms_true) ** 2)
    normalization = _integrate_s2(
        thetas, phis, (curv_norms_learned) ** 2 + (curv_norms_true) ** 2
    )
    return diff / normalization


def _integrate_t2(thetas, phis, h):
    # TODO
    return 0


def _compute_curvature_error_t2(thetas, phis, curv_norms_learned, curv_norms_true):
    # TODO
    return 0


def compute_curvature_error(
    z_grid, curv_norms_learned, curv_norms_true, config
):  # Calculate method error
    start_time = time.time()

    if config.dataset_name == "s1_synthetic":
        thetas = z_grid
        error = _compute_curvature_error_s1(thetas, curv_norms_learned, curv_norms_true)
    elif config.dataset_name == "s2_synthetic":
        thetas = z_grid[:, 0]
        phis = z_grid[:, 1]
        error = _compute_curvature_error_s2(
            thetas, phis, curv_norms_learned, curv_norms_true
        )
    elif config.dataset_name == "t2_synthetic":
        thetas = z_grid[:, 0]
        phis = z_grid[:, 1]
        error = _compute_curvature_error_s2(
            thetas, phis, curv_norms_learned, curv_norms_true
        )
    end_time = time.time()
    print("Computation time: " + "%.3f" % (end_time - start_time) + " seconds.")
    return error


def compute_persistence_diagrams(point_cloud, maxdim=2, n_threads=-1):
    pers = gph.ripser_parallel(X=point_cloud, maxdim=maxdim, n_threads=n_threads)
    diagrams = pers["dgms"]

    return diagrams
