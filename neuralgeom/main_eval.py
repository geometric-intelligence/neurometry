import os

os.environ["GEOMSTATS_BACKEND"] = "pytorch"

import geomstats.backend as gs

from geomstats.geometry.pullback_metric import PullbackMetric


import geomstats.backend as gs
import torch
import numpy as np
from datasets.synthetic import get_s1_synthetic_immersion, get_s2_synthetic_immersion
import scipy.signal


def get_model_immersion(model,config):
    def model_immersion(angle):
        if config.dataset_name == "s1_synthetic":
            z = gs.array([gs.cos(angle), gs.sin(angle)])
        elif config.dataset_name == "s2_synthetic":
            theta = angle[0][0] 
            phi = angle[0][1]
            z = gs.array([gs.sin(theta)*gs.cos(phi),gs.sin(theta)*gs.sin(phi),gs.cos(theta)])
        z = z.to(config.device)
        x_mu = model.decode(z)

        return x_mu

    return model_immersion


#TODO: check this is right
def get_second_fundamental_form(immersion, point, dim, embedding_dim):
    metric = PullbackMetric(dim,embedding_dim,immersion)
    metric.inner_product_derivative_matrix = get_patch_inner_product_derivative_matrix(
        embedding_dim, dim, immersion
    )
    christoffels = metric.christoffels(point)
    christoffels = gs.squeeze(christoffels)
    assert christoffels.shape == (dim, dim, dim), christoffels.shape

    second_fundamental_form = gs.zeros(embedding_dim,dim,dim)
    for a in range(embedding_dim):
        hessian_a = torch.autograd.functional.hessian(
            func=lambda x: immersion(x)[a], inputs=point, strict=True
        )
        hessian_a = gs.squeeze(hessian_a)
        assert hessian_a.shape == (dim, dim), hessian_a.shape
        jacobian_a = torch.autograd.functional.jacobian(
            func=lambda x: immersion(x)[a], inputs=point, strict=True
        )
        
        jacobian_a = torch.squeeze(jacobian_a, dim=0)
        if len(jacobian_a.shape) == 0:
            jacobian_a = gs.to_ndarray(jacobian_a, to_ndim=1)
        assert jacobian_a.shape == (dim,), jacobian_a.shape
        
        second_fundamental_form[a] = hessian_a - torch.einsum("kij,k->ij", christoffels, jacobian_a)

    return second_fundamental_form

def get_patch_inner_product_derivative_matrix(embedding_dim, dim, immersion):
    def patch_inner_product_derivative_matrix(point):
        hessian_aij = gs.zeros((embedding_dim, dim, dim))
        jacobian_ai = gs.zeros((embedding_dim, dim))

        for a in range(embedding_dim):
            hessian_a = torch.autograd.functional.hessian(
                func=lambda x: immersion(x)[a], inputs=point, strict=True
            )
            hessian_a = gs.squeeze(hessian_a)
            assert hessian_a.shape == (dim, dim), hessian_a.shape
            hessian_aij[a, :, :] = hessian_a
            jacobian_a = torch.autograd.functional.jacobian(
                func=lambda x: immersion(x)[a], inputs=point, strict=True
            )
            jacobian_a = torch.squeeze(jacobian_a, dim=0)
            if len(jacobian_a.shape) == 0:
                jacobian_a = gs.to_ndarray(jacobian_a, to_ndim=1)
            assert jacobian_a.shape == (dim,), jacobian_a.shape
            jacobian_ai[a, :] = jacobian_a

        derivative_matrix = gs.einsum(
            "aki,aj->kij", hessian_aij, jacobian_ai
        ) + gs.einsum("akj,ai->kij", hessian_aij, jacobian_ai)

        # for compatibility with geomstats
        derivative_matrix = gs.transpose(derivative_matrix, axes=(2, 1, 0))
        return derivative_matrix

    return patch_inner_product_derivative_matrix

def compute_mean_curvature(points, immersion, dim, embedding_dim):
    metric = PullbackMetric(dim,embedding_dim,immersion)
    mean_curvature = torch.zeros((len(points), embedding_dim))
    for i_point, point in enumerate(points):
        point = gs.array([point])
        second_fundamental_form = get_second_fundamental_form(immersion, point, dim, embedding_dim)
        cometric_matrix = gs.squeeze(metric.cometric_matrix(point))
        mean_curvature[i_point,:] = torch.einsum("ij,aij->a",cometric_matrix,second_fundamental_form)
    
    mean_curvature_norms = torch.linalg.norm(mean_curvature, dim=1, keepdim=True)
    mean_curvature_norms = gs.array([norm.item() for norm in mean_curvature_norms])

    return mean_curvature, mean_curvature_norms



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
            radius = config.radius,
            distortion_amp=config.distortion_amp,
            embedding_dim=config.embedding_dim,
            rot=config.synthetic_rotation,
        )

    mean_curvature_analytic, mean_curvature_norm_analytic = compute_mean_curvature(
        points=angles,immersion=immersion,dim=2,embedding_dim=config.embedding_dim
    )

    # mean_curvature_synth, mean_curvature_norm_synth = compute_extrinsic_curvature(
    #     angles, immersion, config.embedding_dim, config.radius
    # )

    return mean_curvature_analytic, mean_curvature_norm_analytic


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



def integrate_sphere(thetas,phis,h):
    sum_phis = torch.zeros_like(thetas)
    for t, theta in enumerate(thetas):
        sum_phis[t] = torch.trapz(y=h[len(phis)*t:len(phis)*(t+1)],x=phis)*np.sin(theta)
    integral = torch.trapz(y=sum_phis,x=thetas)
    return integral

def get_difference_s2(thetas,phis,mean_curvature_norms,mean_curvature_norms_analytic):
    diff = integrate_sphere(thetas,phis,(mean_curvature_norms-mean_curvature_norms_analytic)**2)
    normalization = integrate_sphere(thetas,phis,(mean_curvature_norms)**2+(mean_curvature_norms_analytic)**2)
    return diff/normalization
    



