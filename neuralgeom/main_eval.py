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


def get_model_immersion(model):
    def model_immersion(angle):
        z = gs.array([gs.cos(angle), gs.sin(angle)])
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

    return mean_curvature, mean_curvature_norm


def compute_intrinsic_curvature():
    return NotImplementedError


def plot_curvature_profile(angles, mean_curvature_norms):

    colormap = plt.get_cmap("hsv")
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
