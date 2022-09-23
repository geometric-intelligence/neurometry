"""Generate and load synthetic datasets."""
import logging

import os

os.environ["GEOMSTATS_BACKEND"] = "pytorch"
import geomstats.backend as gs
import numpy as np
import pandas as pd
import skimage
import torch
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal


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
    for i_angle in range(n_angles):
        angle = 360 * i_angle / n_angles
        rot_image = skimage.transform.rotate(image, angle)
        for i_scalar in range(n_scalars):
            scalar = 1 + 0.2 * i_scalar
            blur_image = skimage.filters.gaussian(rot_image, sigma=scalar)
            noise = np.random.normal(loc=0.0, scale=0.05, size=blur_image.shape)
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
    for _ in range(n_firing_per_cell):
        for i_cell in range(n_cells):
            cell_firings = np.zeros(n_cells)

            if i_cell == 0:
                cell_firings[-2] = np.random.poisson(1.0)
                cell_firings[-1] = np.random.poisson(2.0)
                cell_firings[0] = np.random.poisson(4.0)
                cell_firings[1] = np.random.poisson(2.0)
                cell_firings[2] = np.random.poisson(1.0)
            elif i_cell == 1:
                cell_firings[-1] = np.random.poisson(1.0)
                cell_firings[0] = np.random.poisson(2.0)
                cell_firings[1] = np.random.poisson(4.0)
                cell_firings[2] = np.random.poisson(2.0)
                cell_firings[3] = np.random.poisson(1.0)
            elif i_cell == n_cells - 2:
                cell_firings[-4] = np.random.poisson(1.0)
                cell_firings[-3] = np.random.poisson(2.0)
                cell_firings[-2] = np.random.poisson(4.0)
                cell_firings[-1] = np.random.poisson(2.0)
                cell_firings[0] = np.random.poisson(1.0)
            elif i_cell == n_cells - 1:
                cell_firings[-3] = np.random.poisson(1.0)
                cell_firings[-2] = np.random.poisson(2.0)
                cell_firings[-1] = np.random.poisson(4.0)
                cell_firings[0] = np.random.poisson(2.0)
                cell_firings[1] = np.random.poisson(1.0)
            else:
                cell_firings[i_cell - 2] = np.random.poisson(1.0)
                cell_firings[i_cell - 1] = np.random.poisson(2.0)
                cell_firings[i_cell] = np.random.poisson(4.0)
                cell_firings[i_cell + 1] = np.random.poisson(2.0)
                cell_firings[i_cell - 3] = np.random.poisson(1.0)
            place_cells.append(cell_firings)
            labels.append(i_cell / n_cells * 360)

    return np.array(place_cells), pd.DataFrame({"angles": labels})


def load_s1_synthetic(
    rot,
    n_times=1500,
    radius=1,
    n_wiggles=6,
    distortion_amp=0.4,
    embedding_dim=10,
    noise_var=0.01,
    distortion_func="wiggles",
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
    immersion = get_s1_synthetic_immersion(
        distortion_func=distortion_func,
        radius=radius,
        n_wiggles=n_wiggles,
        distortion_amp=distortion_amp,
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
        covariance_matrix= noise_var * torch.eye(embedding_dim),
    )

    noisy_data = data + radius*noise_dist.sample((n_times,))

    return noisy_data, labels


def load_s2_synthetic(rot, n_times, radius, distortion_amp, embedding_dim, noise_var):

    immersion = get_s2_synthetic_immersion(radius, distortion_amp, embedding_dim, rot)

    thetas = gs.linspace(0, gs.pi, n_times)

    phis = gs.linspace(0, 2 * gs.pi, n_times)

    angles = torch.cartesian_prod(thetas, phis)


    labels = pd.DataFrame(
        {
        "thetas": angles[:,0],
        "phis": angles[:,1]
        }
    )

    data = torch.zeros(n_times*n_times, embedding_dim)

    for _, angle_pair in enumerate(angles):
        data[_, :] = immersion(angle_pair)

    noise_dist = MultivariateNormal(
        loc=torch.zeros(embedding_dim),
        covariance_matrix=radius * noise_var * torch.eye(embedding_dim),
    )

    noisy_data = data + noise_dist.sample((n_times*n_times,))

    return noisy_data, labels



def load_t2_synthetic(rot, n_times, major_radius, minor_radius, distortion_amp, embedding_dim, noise_var):

    immersion = get_t2_synthetic_immersion(major_radius, minor_radius, distortion_amp, embedding_dim, rot)

    thetas = gs.linspace(0, 2*gs.pi, n_times)

    phis = gs.linspace(0, 2 * gs.pi, n_times)

    angles = torch.cartesian_prod(thetas, phis)


    labels = pd.DataFrame(
        {
        "thetas": angles[:,0],
        "phis": angles[:,1]
        }
    )

    data = torch.zeros(n_times*n_times, embedding_dim)

    for _, angle_pair in enumerate(angles):
        data[_, :] = immersion(angle_pair)

    noise_dist = MultivariateNormal(
        loc=torch.zeros(embedding_dim),
        covariance_matrix=major_radius * noise_var * torch.eye(embedding_dim),
    )

    noisy_data = data + noise_dist.sample((n_times*n_times,))

    return noisy_data, labels



def get_s1_synthetic_immersion(
    distortion_func, radius, n_wiggles, distortion_amp, embedding_dim, rot
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
        if distortion_func == "wiggles":
            amplitude = radius * (1 + distortion_amp * gs.cos(n_wiggles * angle))
        elif distortion_func == "bump":
            amplitude = radius * (
                1
                + distortion_amp * gs.exp(-5 * (angle - gs.pi / 2) ** 2)
                + distortion_amp * gs.exp(-5 * (angle - 3 * gs.pi / 2) ** 2)
            )
        else:
            raise NotImplementedError

        point = amplitude * polar(angle)

        padded_point = F.pad(
            input=point, pad=(0, embedding_dim - 2), mode="constant", value=0.0
        )
        return gs.einsum("ij,j->i", rot, padded_point)

    return synth_immersion


def get_s2_synthetic_immersion(radius, distortion_amp, embedding_dim, rot):
    def spherical(theta, phi):
        x = gs.sin(theta) * gs.cos(phi)
        y = gs.sin(theta) * gs.sin(phi)
        z = gs.cos(theta)
        return gs.array([x, y, z])

    def s2_synthetic_immersion(angle_pair):
        theta = angle_pair[0]
        phi = angle_pair[1]
        
        amplitude = radius * (1 + distortion_amp * gs.exp(-5 * theta**2)
        + distortion_amp * gs.exp(-5 * (theta-gs.pi)**2))

        point = amplitude * spherical(theta, phi)

        padded_point = F.pad(
            input=point, pad=(0, embedding_dim - 3), mode="constant", value=0.0
        )

        return gs.einsum("ij,j->i", rot, padded_point)

    return s2_synthetic_immersion


def get_t2_synthetic_immersion(
    major_radius, minor_radius, distortion_amp, embedding_dim, rot
):
    def torus_proj(theta, phi):
        x = (major_radius + minor_radius * gs.cos(theta)) * gs.cos(phi)
        y = (major_radius + minor_radius * gs.cos(theta)) * gs.sin(phi)
        z = minor_radius * gs.sin(theta)
        return gs.array([x, y, z])

    def t2_synthetic_immersion(angle_pair):

        theta = angle_pair[0]
        phi = angle_pair[1]
        amplitude = 1 + distortion_amp * gs.exp(-5 * (phi-gs.pi/2)**2) + distortion_amp * gs.exp(-5 * (phi-3*gs.pi/2)**2)

        point = amplitude * torus_proj(theta, phi)

        padded_point = F.pad(
            input=point, pad=(0, embedding_dim - 3), mode="constant", value=0.0
        )

        return gs.einsum("ij,j->i", rot, padded_point)

    return t2_synthetic_immersion
