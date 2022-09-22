import os

os.environ["GEOMSTATS_BACKEND"] = "pytorch"

import geomstats.backend as gs

from geomstats.geometry.pullback_metric import PullbackMetric


import geomstats.backend as gs
import torch
import numpy as np
from datasets.synthetic import get_s1_synthetic_immersion
import scipy.signal


def get_model_immersion(model,device):
    def model_immersion(angle):
        z = gs.array([gs.cos(angle), gs.sin(angle)])
        z = z.to(device)
        x_mu = model.decode(z)
        return x_mu

    return model_immersion


def get_metric(model, config):
    immersion = get_model_immersion(model,config.device)
    embedding_dim = config.embedding_dim
    metric = PullbackMetric(dim=2,embedding_dim=embedding_dim,immersion=immersion)
    return metric



# def compute_extrinsic_curvature(angles, immersion, embedding_dim, radius):
#     mean_curvature = gs.zeros(len(angles), embedding_dim)
#     for _, angle in enumerate(angles):
#         for i in range(embedding_dim):
            
#             hessian = torch.autograd.functional.hessian(
#                 func=lambda x: immersion(x)[i], inputs=angle, strict=True
#             )
#             mean_curvature[_, i] = einsum

#     for _, angle in enumerate(angles):
#         hessian = gs.zeros()
#         for component in range(embedding_dim):
#         hessian_row = 


#     mean_curvature_norm = torch.linalg.norm(mean_curvature, dim=1, keepdim=True)
#     mean_curvature_norm = [_.item() for _ in mean_curvature_norm]

#     return mean_curvature, mean_curvature_norm

def compute_extrinsic_curvature(angles, immersion, embedding_dim, radius):
    mean_curvature = gs.zeros(len(angles), embedding_dim)
    for _, angle in enumerate(angles):
        for i in range(embedding_dim):
            hessian = torch.autograd.functional.hessian(
                func=lambda x: immersion(x)[i], inputs=angle, strict=True
            )
            mean_curvature[_, i] = (1/radius**2)*hessian

    mean_curvature_norm = torch.linalg.norm(mean_curvature, dim=1, keepdim=True)
    mean_curvature_norm = [_.item() for _ in mean_curvature_norm]

    return mean_curvature, mean_curvature_norm


def compute_intrinsic_curvature():
    return NotImplementedError


def get_mean_curvature(model, angles, config, embedding_dim):
    model.eval()
    immersion = get_model_immersion(model, config.device)
    mean_curvature, mean_curvature_norm = compute_extrinsic_curvature(
        angles, immersion, embedding_dim, config.radius
    )
    return mean_curvature, mean_curvature_norm


def get_mean_curvature_analytic(angles, config):
    immersion = get_s1_synthetic_immersion(
        distortion_func=config.distortion_func,
        radius=config.radius,
        n_wiggles=config.n_wiggles,
        distortion_amp=config.distortion_amp,
        embedding_dim=config.embedding_dim,
        rot=config.synthetic_rotation,
    )
    mean_curvature_synth, mean_curvature_norm_synth = compute_extrinsic_curvature(
        angles, immersion, config.embedding_dim, config.radius
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

def get_difference(thetas, h1, h2):
    h1 = np.array(h1)
    h2 = np.array(h2)
    diff = np.trapz((h1-h2)**2,thetas)/(np.trapz(h1**2,thetas)*np.trapz(h2**2,thetas))
    return diff



