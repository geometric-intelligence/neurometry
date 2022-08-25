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


def get_synth_immersion(radius, n_wiggles, amp_wiggles, embedding_dim):
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
        amplitude = radius * (1 + amp_wiggles * gs.cos(n_wiggles * angle))

        point = amplitude * polar(angle)

        padded_point = F.pad(
            input=point, pad=(0, embedding_dim - 2), mode="constant", value=0.0
        )

        so = SpecialOrthogonal(n=embedding_dim)

        rot = so.random_point()

        return gs.einsum("ij,j->i", rot, padded_point)

    return synth_immersion


def mean_curv_vector(base_points, params):

    if params["immersion_type"] == "analytic":
        immersion = get_synth_immersion(
            radius=params["radius"],
            n_wiggles=params["n_wiggles"],
            amp_wiggles=params["amp_wiggles"],
            embedding_dim=params["embedding_dim"],
        )
        metric = neural_metric.NeuralMetric(
            dim=1, embedding_dim=params["embedding_dim"], immersion=immersion
        )
    elif params["immersion_type"] == "VAE":
        model = torch.load(params["model_filename"])
        model.eval()
        immersion = neural_metric.get_neural_immersion(model)
        metric = neural_metric.NeuralMetric(
            dim=1, embedding_dim=params["embedding_dim"], immersion=immersion
        )

    mean_curv = [metric.mean_curvature(base_point) for base_point in base_points]

    return mean_curv


def plot(angles, mean_curvature_norms):
    
    colormap = plt.get_cmap("hsv")
    color_norm = mpl.colors.Normalize(0.0, 1.2 * max(mean_curvature_norms))
    plt.figure(figsize=(12,5 ))

    ax2 = plt.subplot(1,2,1,polar=True)
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

    ax1 = plt.subplot(1,2,2)
    
    pt = ax1.plot(angles,mean_curvature_norms)

    ax1.set_xlabel("angle")

    


