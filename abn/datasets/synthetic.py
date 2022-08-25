"""Generate and load synthetic datasets."""
import logging

import geomstats.backend as gs
import numpy as np
import pandas as pd
import skimage
import torch
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
import torch.nn.functional as F


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


def load_wiggles(
    n_times=1000,
    synth_radius=1,
    n_wiggles=6,
    amp_wiggles=0.4,
    embedding_dim=10,
    noise_var=0.01,
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
    noisy_data : array-like, shape=[embedding_dim, n_times]
        Number of firings per time step and per cell.
    labels : pd.DataFrame, shape=[n_times, 1]
        Labels organized in 1 column: angles.
    """

    def polar(angle):
        return gs.stack([gs.cos(angle), gs.sin(angle)], axis=0)

    def synth_immersion(angles):
        amplitudes = synth_radius * (1 + amp_wiggles * gs.cos(n_wiggles * angles))
        wiggly_circle = gs.einsum("ik,jk->ij", polar(angles), np.diag(amplitudes))

        # padded_wiggly_circle = gs.vstack(
        #     [wiggly_circle, gs.zeros((embedding_dim - 2, len(angle)))]
        # )
        padded_wiggly_circle = F.pad(input = wiggly_circle, pad = (0,0,0,embedding_dim-2),mode="constant", value=0.0 )

        so = SpecialOrthogonal(n=embedding_dim)

        rot = so.random_point()

        return gs.einsum("ik,kj->ij", rot, padded_wiggly_circle)

    angles = gs.linspace(0, 2 * gs.pi, n_times)

    labels = pd.DataFrame(
        {
            "angles": angles,
        }
    )

    noise_cov = np.diag(noise_var * gs.ones(embedding_dim))

    noisy_data = (
        synth_immersion(angles)
        + gs.random.multivariate_normal(
            mean=gs.zeros(embedding_dim), cov=noise_cov, size=len(angles)
        ).T
    )

    return noisy_data, labels
