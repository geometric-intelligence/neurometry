"""Generate and load synthetic datasets."""

import logging
import os

import numpy as np
import pandas as pd
import skimage
import torch
from torch.distributions.multivariate_normal import MultivariateNormal

from neurometry.estimators.topology.persistent_homology import (
    cohomological_circular_coordinates,
    cohomological_toroidal_coordinates,
)

os.environ["GEOMSTATS_BACKEND"] = "pytorch"
import geomstats.backend as gs
from geomstats.geometry.special_orthogonal import SpecialOrthogonal


def load_projected_images(n_scalars=5, n_angles=1000, img_size=128):
    """Load a dataset of 2D images projected into 1D projections.

    The actions are:
    - action of SO(2): rotation
    - action of R^_+: blur

    Parameters
    ----------
    n_scalars : int
        Number of scalar used for action of scalings.
    n_angles : int
        Number of angles used for action of SO(2).

    Returns
    -------
    projections : array-like, shape=[n_scalars * n_angles, img_size]]
        Projections with different orientations and blurs.
    labels : pd.DataFrame, shape=[n_scalars * n_angles, 2]
        Labels organized in 2 columns: angles, and scalars.
    """
    images, labels = load_images(
        n_scalars=n_scalars, n_angles=n_angles, img_size=img_size
    )

    projections = np.sum(images, axis=-1)
    return projections, labels


def load_images(n_scalars=10, n_angles=1000, img_size=256):
    """Load a dataset of images.

    The actions are:
    - action of SO(2): rotation
    - action of R^_+: blur

    Parameters
    ----------
    n_scalars : int
        Number of scalar used for action of scalings.
    n_angles : int
        Number of angles used for action of SO(2).

    Returns
    -------
    images : array-like, shape=[n_scalars * n_angles, img_size, img_size]]
        Images with different orientations and blurs.
    labels : pd.DataFrame, shape=[n_scalars * n_angles, 2]
        Labels organized in 2 columns: angles, and scalars.
    """
    logging.info("Generating dataset of synthetic images.")
    image = skimage.data.camera()
    image = skimage.transform.resize(image, (img_size, img_size), anti_aliasing=True)

    images = []
    angles = []
    scalars = []
    rng = np.random.default_rng(seed=0)
    for i_angle in range(n_angles):
        angle = 360 * i_angle / n_angles
        rot_image = skimage.transform.rotate(image, angle)
        for i_scalar in range(n_scalars):
            scalar = 1 + 0.2 * i_scalar
            blur_image = skimage.filters.gaussian(rot_image, sigma=scalar)
            noise = rng.normal(loc=0.0, scale=0.05, size=blur_image.shape)
            images.append((blur_image + noise).astype(np.float32))
            angles.append(angle)
            scalars.append(scalar)

    labels = pd.DataFrame(
        {
            "angles": angles,
            "scalars": scalars,
        }
    )
    return np.array(images), labels


def load_points(n_scalars=1, n_angles=1000):
    """Load a dataset of points in R^3.

    The actions are:
    - action of SO(2): along z-axis
    - action of R^_+

    Parameters
    ----------
    n_scalars : int
        Number of scalar used for action of scalings.
    n_angles : int
        Number of angles used for action of SO(2).

    Returns
    -------
    points : array-like, shape=[n_scalars * n_angles, 3]
        Points sampled on a cone.
    labels : pd.DataFrame, shape=[n_scalars * n_angles, 2]
        Labels organized in 2 columns: angles, and scalars.
    """
    points = []
    angles = []
    scalars = []
    point = np.array([1, 0, 1])
    for i_angle in range(n_angles):
        angle = 2 * np.pi * i_angle / n_angles
        rotmat = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1.0],
            ]
        )
        rot_point = rotmat @ point
        for i_scalar in range(n_scalars):
            scalar = 1 + i_scalar
            points.append(scalar * rot_point)

            angles.append(angle)
            scalars.append(scalar)

    labels = pd.DataFrame(
        {
            "angles": angles,
            "scalars": scalars,
        }
    )

    return np.array(points), labels


def _create_bump_array(position, width):
    """Create array with Gaussian bump of specified position and width.

    Array is of length 360 filled with zeros, except for the bump.

    Parameters
    ----------
    - position (int): The index of the center of the bump.
    - width (int): The width of the bump.

    Returns
    -------
    - bump_array (numpy.ndarray): The array with the Gaussian bump.
    """
    # Create array of zeros with length 360
    bump_array = np.zeros(360)

    # Define the range of indices for the bump
    left = position - width // 2
    right = position + width // 2

    # Create the Gaussian bump
    x = np.linspace(-1, 1, width)
    bump = np.exp(-(x**2) * 5)  # Gaussian bump
    bump /= np.max(bump)  # Normalize to maximum amplitude of 1

    # Add the bump to the array
    bump_array[left:right] += bump

    return bump_array


def load_three_place_cells(n_times=360, centers=None, widths=None):
    """Load dataset of three synthetic place cells.

    This is a dataset of synthetic place cell firings, that
    simulates a rat walking in a circle.

    Three place cells are chosen, with their position and
    with the width of their place field that can vary.

    Parameters
    ----------
    n_times : int
        Number of times.

    Returns
    -------
    place_cells : array-like, shape=[n_times, 3]
        Number of firings per time step and per cell.
    labels : pd.DataFrame, shape=[n_times, 1]
        Labels organized in 1 column: angles.
    """
    if n_times != 360:
        raise NotImplementedError
    if centers is None:
        centers = [40, 150, 270]
    if widths is None:
        widths = [80, 300, 180]

    place_cell_0 = _create_bump_array(40, 80)
    place_cell_1 = _create_bump_array(150, 300)
    place_cell_2 = _create_bump_array(270, 180)
    place_cells = np.vstack([place_cell_0, place_cell_1, place_cell_2]).T

    assert place_cells.shape == (360, 3)

    labels = np.arange(0, 360, step=1)
    return place_cells, pd.DataFrame({"angles": labels})


def load_place_cells(n_times=10000, n_cells=40):
    """Load synthetic place cells.

    This is a dataset of synthetic place cell firings, that
    simulates a rat walking in a circle.

    Each place cell activated (2 firings) also activates
    its neighbors (1 firing each) to simulate the circular
    relationship.

    Parameters
    ----------
    n_times : int
        Number of times.
    n_cells : int
        Number of place cells.

    Returns
    -------
    place_cells : array-like, shape=[n_times, n_cells]
        Number of firings per time step and per cell.
    labels : pd.DataFrame, shape=[n_times, 1]
        Labels organized in 1 column: angles.
    """
    n_firing_per_cell = int(n_times / n_cells)
    place_cells = []
    labels = []
    rng = np.random.default_rng(seed=0)
    for _ in range(n_firing_per_cell):
        for i_cell in range(n_cells):
            cell_firings = np.zeros(n_cells)

            if i_cell == 0:
                cell_firings[-2] = rng.poisson(1.0)
                cell_firings[-1] = rng.poisson(2.0)
                cell_firings[0] = rng.poisson(4.0)
                cell_firings[1] = rng.poisson(2.0)
                cell_firings[2] = rng.poisson(1.0)
            elif i_cell == 1:
                cell_firings[-1] = rng.poisson(1.0)
                cell_firings[0] = rng.poisson(2.0)
                cell_firings[1] = rng.poisson(4.0)
                cell_firings[2] = rng.poisson(2.0)
                cell_firings[3] = rng.poisson(1.0)
            elif i_cell == n_cells - 2:
                cell_firings[-4] = rng.poisson(1.0)
                cell_firings[-3] = rng.poisson(2.0)
                cell_firings[-2] = rng.poisson(4.0)
                cell_firings[-1] = rng.poisson(2.0)
                cell_firings[0] = rng.poisson(1.0)
            elif i_cell == n_cells - 1:
                cell_firings[-3] = rng.poisson(1.0)
                cell_firings[-2] = rng.poisson(2.0)
                cell_firings[-1] = rng.poisson(4.0)
                cell_firings[0] = rng.poisson(2.0)
                cell_firings[1] = rng.poisson(1.0)
            else:
                cell_firings[i_cell - 2] = rng.poisson(1.0)
                cell_firings[i_cell - 1] = rng.poisson(2.0)
                cell_firings[i_cell] = rng.poisson(4.0)
                cell_firings[i_cell + 1] = rng.poisson(2.0)
                cell_firings[i_cell - 3] = rng.poisson(1.0)
            place_cells.append(cell_firings)
            labels.append(i_cell / n_cells * 360)

    return np.array(place_cells), pd.DataFrame({"angles": labels})


def load_s1_synthetic(
    synthetic_rotation,
    n_times=1500,
    radius=1,
    n_wiggles=6,
    geodesic_distortion_amp=0.4,
    embedding_dim=10,
    noise_var=0.01,
    geodesic_distortion_func="wiggles",
):
    """Create "wiggly" circles with noise.

    Parameters
    ----------
    n_times : int

    circle_radius : float
        Primary circle radius.
    n_wiggles : int
        Number of "wiggles".
    amp_wiggles : float, < 1
        Amplitude of "wiggles".
    embedding_dim : int
        Dimension of embedding dimension.
    noise_var : float
        Variance (sigma2) of the Gaussian noise.

    Returns
    -------
    noisy_data : array-like, shape=[n_times, embedding_dim]
        Number of firings per time step and per cell.
    labels : pd.DataFrame, shape=[n_times, 1]
        Labels organized in 1 column: angles.
    """
    rot = torch.eye(n=embedding_dim)
    if synthetic_rotation == "random":
        rot = SpecialOrthogonal(n=embedding_dim).random_point()

    immersion = get_s1_synthetic_immersion(
        geodesic_distortion_func=geodesic_distortion_func,
        radius=radius,
        n_wiggles=n_wiggles,
        geodesic_distortion_amp=geodesic_distortion_amp,
        embedding_dim=embedding_dim,
        rot=rot,
    )

    angles = gs.linspace(0, 2 * gs.pi, n_times)

    labels = pd.DataFrame(
        {
            "angles": angles,
        }
    )

    data = torch.zeros(n_times, embedding_dim)

    for _, angle in enumerate(angles):
        data[_, :] = immersion(angle)

    noise_dist = MultivariateNormal(
        loc=torch.zeros(embedding_dim),
        covariance_matrix=noise_var * torch.eye(embedding_dim),
    )

    noisy_data = data + radius * noise_dist.sample((n_times,))

    circular_coords = cohomological_circular_coordinates(noisy_data)

    labels = pd.DataFrame({"angles": circular_coords})

    return noisy_data, labels


def load_s2_synthetic(
    synthetic_rotation,
    n_times,
    radius,
    geodesic_distortion_amp,
    embedding_dim,
    noise_var,
):
    rot = torch.eye(n=embedding_dim)
    if synthetic_rotation == "random":
        rot = SpecialOrthogonal(n=embedding_dim).random_point()

    immersion = get_s2_synthetic_immersion(
        radius, geodesic_distortion_amp, embedding_dim, rot
    )

    sqrt_ntimes = int(gs.sqrt(n_times))
    thetas = gs.linspace(0.01, gs.pi, sqrt_ntimes)

    phis = gs.linspace(0, 2 * gs.pi, sqrt_ntimes)

    points = torch.cartesian_prod(thetas, phis)

    labels = pd.DataFrame({"thetas": points[:, 0], "phis": points[:, 1]})

    data = torch.zeros(sqrt_ntimes**2, embedding_dim)

    for _, point in enumerate(points):
        point = gs.array(point)
        data[_, :] = immersion(point)

    noise_dist = MultivariateNormal(
        loc=torch.zeros(embedding_dim),
        covariance_matrix=radius * noise_var * torch.eye(embedding_dim),
    )

    noisy_data = data + noise_dist.sample((sqrt_ntimes**2,))

    return noisy_data, labels


def load_t2_synthetic(
    synthetic_rotation,
    n_times,
    major_radius,
    minor_radius,
    geodesic_distortion_amp,
    embedding_dim,
    noise_var,
):
    rot = torch.eye(n=embedding_dim)
    if synthetic_rotation == "random":
        rot = SpecialOrthogonal(n=embedding_dim).random_point()

    immersion = get_t2_synthetic_immersion(
        major_radius, minor_radius, geodesic_distortion_amp, embedding_dim, rot
    )
    sqrt_ntimes = int(gs.sqrt(n_times))

    thetas = gs.linspace(0, 2 * gs.pi, sqrt_ntimes)

    phis = gs.linspace(0, 2 * gs.pi, sqrt_ntimes)

    angles = torch.cartesian_prod(thetas, phis)

    labels = pd.DataFrame({"thetas": angles[:, 0], "phis": angles[:, 1]})

    data = torch.zeros(sqrt_ntimes**2, embedding_dim)

    for _, angle_pair in enumerate(angles):
        data[_, :] = immersion(angle_pair)

    noise_dist = MultivariateNormal(
        loc=torch.zeros(embedding_dim),
        covariance_matrix=major_radius * noise_var * torch.eye(embedding_dim),
    )

    noisy_data = data + noise_dist.sample((sqrt_ntimes**2,))

    toroidal_coords = cohomological_toroidal_coordinates(noisy_data)

    labels = pd.DataFrame(
        {"thetas": toroidal_coords[:, 0], "phis": toroidal_coords[:, 1]}
    )

    return noisy_data, labels


def get_s1_synthetic_immersion(
    geodesic_distortion_func,
    radius,
    n_wiggles,
    geodesic_distortion_amp,
    embedding_dim,
    rot,
):
    """Creates function whose image is "wiggly" circles in high-dim space.

    Parameters
    ----------
    circle_radius : float
        Primary circle radius.
    n_wiggles : int
        Number of "wiggles".
    amp_wiggles : float, < 1
        Amplitude of "wiggles".
    embedding_dim : int
        Dimension of immersion codomain.

    Returns
    -------
    synth_immersion : function
        Synthetic immersion from S1 to R^N.
    """

    def polar(angle):
        """Extrinsic coordinates of embedded 1-sphere parameterized by angle.

        Parameters
        ----------
        angle : float

        """
        return gs.array([gs.cos(angle), gs.sin(angle)])

    def synth_immersion(angle):
        """Synthetic immersion function.

        Parameters
        ----------
        angle : float
            Angle coordinate on circle.

        Returns
        -------
        padded_point : array-like, shape=[embedding_dim, ]
            Yiels an embedding_dim-dimensional point making up wiggly circle
        """
        if geodesic_distortion_func == "wiggles":
            amplitude = radius * (
                1 + geodesic_distortion_amp * gs.cos(n_wiggles * angle)
            )
        elif geodesic_distortion_func == "bump":
            amplitude = radius * (
                1
                + geodesic_distortion_amp * gs.exp(-5 * (angle - gs.pi / 2) ** 2)
                + geodesic_distortion_amp * gs.exp(-5 * (angle - 3 * gs.pi / 2) ** 2)
            )
        else:
            raise NotImplementedError

        point = amplitude * polar(angle)
        point = gs.squeeze(point, axis=-1)
        if embedding_dim > 2:
            point = gs.concatenate([point, gs.zeros(embedding_dim - 2)])
        return gs.einsum("ij,j->i", rot, point)

    return synth_immersion


def get_s2_synthetic_immersion(radius, geodesic_distortion_amp, embedding_dim, rot):
    def spherical(theta, phi):
        x = gs.sin(theta) * gs.cos(phi)
        y = gs.sin(theta) * gs.sin(phi)
        z = gs.cos(theta)
        return gs.array([x, y, z])

    def s2_synthetic_immersion(angle_pair):
        theta = angle_pair[0]
        phi = angle_pair[1]

        amplitude = radius * (
            1
            + geodesic_distortion_amp * gs.exp(-5 * theta**2)
            + geodesic_distortion_amp * gs.exp(-5 * (theta - gs.pi) ** 2)
        )

        point = amplitude * spherical(theta, phi)
        point = gs.squeeze(point, axis=-1)
        if embedding_dim > 3:
            point = gs.concatenate([point, gs.zeros(embedding_dim - 3)])

        return gs.einsum("ij,j->i", rot, point)

    return s2_synthetic_immersion


def get_t2_synthetic_immersion(
    major_radius, minor_radius, geodesic_distortion_amp, embedding_dim, rot
):
    def torus_proj(theta, phi):
        x = (major_radius - minor_radius * gs.cos(theta)) * gs.cos(phi)
        y = (major_radius - minor_radius * gs.cos(theta)) * gs.sin(phi)
        z = minor_radius * gs.sin(theta)
        return gs.array([x, y, z])

    def t2_synthetic_immersion(angle_pair):
        theta = angle_pair[0]
        phi = angle_pair[1]
        amplitude = (
            1
            + geodesic_distortion_amp
            * gs.exp(-2 * (phi - gs.pi / 2) ** 2)
            * gs.exp(-2 * (theta - gs.pi) ** 2)
            + geodesic_distortion_amp
            * gs.exp(-2 * (phi - 3 * gs.pi / 2) ** 2)
            * gs.exp(-2 * (theta - gs.pi) ** 2)
        )

        point = amplitude * torus_proj(theta, phi)
        point = gs.squeeze(point, axis=-1)
        if embedding_dim > 3:
            point = gs.concatenate([point, gs.zeros(embedding_dim - 3)])

        return gs.einsum("ij,j->i", rot, point)

    return t2_synthetic_immersion
