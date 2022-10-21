import os

os.environ["GEOMSTATS_BACKEND"] = "pytorch"
import geomstats.backend as gs
import torch
import itertools
import sys

from main_eval import get_patch_inner_product_derivative_matrix
from main_eval import compute_mean_curvature
from main_eval import get_second_fundamental_form as sff
from datasets.synthetic import get_s1_synthetic_immersion
from geomstats.geometry.pullback_metric import PullbackMetric
from geomstats.geometry.hypersphere import Hypersphere

sphere = Hypersphere(dim=2)


def get_immersion(radius):
    def immersion(theta):
        return gs.array([radius * gs.cos(theta), radius * gs.sin(theta)])

    return immersion


# Note: same immersion as geomstats' test_pullback_metric.py
def get_sphere_immersion(radius):
    def immersion(point):
        theta = point[0]
        phi = point[1]
        x = gs.sin(theta) * gs.cos(phi)
        y = gs.sin(theta) * gs.sin(phi)
        z = gs.cos(theta)
        return gs.array([radius * x, radius * y, radius * z])

    return immersion



def test_second_fundamental_form_s1():
    dim = 1
    embedding_dim = 2
    radius = 3

    point = gs.array([gs.pi / 3])
    print("point s1:", point.shape)

    immersion = get_immersion(radius=radius)
    sec_fun = sff(immersion, point, dim, embedding_dim)
    print("s1 second fundamental form", sec_fun)
    expected = gs.array(
        [
            -radius * gs.cos(point),
            -radius * gs.sin(point),
        ]
    ).reshape((embedding_dim, dim, dim))

    assert gs.allclose(sec_fun.shape, expected.shape), sec_fun.shape
    assert gs.allclose(sec_fun, expected), sec_fun

def test_second_fundamental_form_bump():
    dim = 1
    embedding_dim = 2
    radius = 3

    point = gs.array([gs.pi / 3])
    print("point s1:", point.shape)

    immersion = get_s1_synthetic_immersion(
        distortion_func="bump",radius=1,n_wiggles=3,distortion_amp=0.3,embedding_dim=2,rot=torch.eye(2))

    sec_fun = sff(immersion, point, dim, embedding_dim)

    assert gs.allclose(sec_fun.shape, (embedding_dim, dim, dim)), sec_fun.shape


def test_jacobian_s2():
    dim = 2
    embedding_dim = 3
    radius = 3

    points = gs.array([gs.pi / 3, gs.pi])
    theta = points[0]
    phi = points[1]

    immersion = get_sphere_immersion(radius=radius)
    # k : rows index the derivatives
    # a (alpha) : columns index the output coordinates
    jacobian_ka = torch.autograd.functional.jacobian(
        func=lambda x: immersion(x)[:], inputs=points, strict=True
    )
    jacobian_ka = torch.squeeze(jacobian_ka, dim=0)
    expected_k1 = gs.array(
        [
            radius * gs.cos(theta) * gs.cos(phi),
            -radius * gs.sin(theta) * gs.sin(phi),
        ]
    )
    expected_k2 = gs.array(
        [
            radius * gs.cos(theta) * gs.sin(phi),
            radius * gs.sin(theta) * gs.cos(phi),
        ]
    )
    expected_k3 = gs.array(
        [
            -radius * gs.sin(theta),
            0,
        ]
    )
    expected_ka = gs.stack([expected_k1, expected_k2, expected_k3], axis=0)
    assert jacobian_ka.shape == (embedding_dim, dim), jacobian_ka.shape
    assert jacobian_ka.shape == expected_ka.shape

    assert gs.allclose(jacobian_ka, expected_ka), jacobian_ka


def test_second_fundamental_form_s2():
    dim = 2
    embedding_dim = 3
    radius = 3

    points = gs.array([gs.pi / 3, gs.pi])
    theta = points[0]
    phi = points[1]

    immersion = get_sphere_immersion(radius=radius)
    sec_fun = sff(immersion, points, dim, embedding_dim)

    expected_11 = gs.array(
        [
            -radius * gs.sin(theta) * gs.cos(phi),
            -radius * gs.sin(theta) * gs.sin(phi),
            -radius * gs.cos(theta),
        ]
    )
    expected_22 = gs.array(
        [
            -radius * gs.sin(theta) ** 2 * gs.sin(theta) * gs.cos(phi),
            -radius * gs.sin(theta) ** 2 * gs.sin(theta) * gs.sin(phi),
            -radius * gs.sin(theta) ** 2 * gs.cos(theta),
        ]
    )

    sec_fun_11 = sec_fun[:, 0, 0]
    sec_fun_22 = sec_fun[:, 1, 1]

    assert gs.allclose(sec_fun_11.shape, expected_11.shape), sec_fun_11.shape
    assert gs.allclose(sec_fun_22.shape, expected_22.shape), sec_fun_22.shape

    assert gs.allclose(sec_fun_11, expected_11), sec_fun_11
    assert gs.allclose(sec_fun_22, expected_22, atol=1e-5), sec_fun_22


def test_metric_matrix_s1():
    dim, embedding_dim, radius = 1, 2, 3

    immersion = get_immersion(radius=radius)
    metric = PullbackMetric(dim=dim, embedding_dim=embedding_dim, immersion=immersion)
    point = gs.array([gs.pi / 4])
    matrix = metric.metric_matrix(point)

    expected_matrix = gs.array([radius ** 2]).reshape((dim, dim))

    assert gs.allclose(matrix.shape, expected_matrix.shape), matrix.shape
    assert gs.allclose(matrix, expected_matrix), matrix


def test_cometric_matrix_s1():
    dim, embedding_dim, radius = 1, 2, 3
    immersion = get_immersion(radius=radius)
    metric = PullbackMetric(dim=dim, embedding_dim=embedding_dim, immersion=immersion)
    point = gs.array([gs.pi / 3])

    comatrix = metric.cometric_matrix(point)

    expected_comatrix = gs.array([1 / radius ** 2]).reshape((dim, dim))

    assert gs.allclose(comatrix.shape, expected_comatrix.shape), comatrix.shape
    assert gs.allclose(comatrix, expected_comatrix), comatrix


def test_metric_matrix_s2():
    dim, embedding_dim, radius = 2, 3, 1

    immersion = get_sphere_immersion(radius=radius)
    metric = PullbackMetric(dim=dim, embedding_dim=embedding_dim, immersion=immersion)
    point = gs.array([gs.pi / 3, gs.pi])
    theta, phi = point[0], point[1]
    matrix = metric.metric_matrix(point)

    expected_matrix = gs.array(
        [[radius ** 2, 0], [0, radius ** 2 * gs.sin(theta) ** 2]]
    )

    assert gs.allclose(matrix.shape, expected_matrix.shape), matrix.shape
    print(matrix)
    print(expected_matrix)
    assert gs.allclose(matrix, expected_matrix), matrix


def test_cometric_matrix_s2():
    dim, embedding_dim, radius = 2, 3, 4
    immersion = get_sphere_immersion(radius=radius)
    metric = PullbackMetric(dim=dim, embedding_dim=embedding_dim, immersion=immersion)
    point = gs.array([gs.pi / 3, gs.pi])
    theta, phi = point[0], point[1]

    comatrix = metric.cometric_matrix(point)

    expected_comatrix = gs.array(
        [[1 / (radius ** 2), 0], [0, 1 / (radius ** 2 * gs.sin(theta) ** 2)]]
    )

    assert gs.allclose(comatrix.shape, expected_comatrix.shape), comatrix.shape
    assert gs.allclose(comatrix, expected_comatrix), comatrix


def test_inner_product_derivative_matrix_s2():
    dim, embedding_dim, radius = 2, 3, 1
    immersion = get_sphere_immersion(radius=radius)
    metric = PullbackMetric(dim=dim, embedding_dim=embedding_dim, immersion=immersion)

    point = gs.array([gs.pi / 3, gs.pi])
    theta, phi = point[0], point[1]

    # PATCH: assign new method in pullback metric
    metric.inner_product_derivative_matrix = get_patch_inner_product_derivative_matrix(
        embedding_dim, dim, immersion
    )

    derivative_matrix = metric.inner_product_derivative_matrix(point)

    # derivative with respect to theta
    expected_1 = gs.array(
        [[0, 0], [0, 2 * radius ** 2 * gs.cos(theta) * gs.sin(theta)]]
    )
    # derivative with respect to phi
    expected_2 = gs.zeros(1)

    assert gs.allclose(derivative_matrix.shape, (2, 2, 2)), derivative_matrix.shape
    assert gs.allclose(derivative_matrix[:, :, 0], expected_1), derivative_matrix[0]
    assert gs.allclose(derivative_matrix[:, :, 1], expected_2), derivative_matrix[1]


def test_christoffels_bump():
    dim, embedding_dim, radius = 1, 2, 1
    immersion = get_s1_synthetic_immersion(distortion_func="bump",radius=1,n_wiggles=3,distortion_amp=0.3,embedding_dim=2,rot=torch.eye(2))
    metric = PullbackMetric(dim=dim, embedding_dim=embedding_dim, immersion=immersion)
    # PATCH: assign new method in pullback metric
    metric.inner_product_derivative_matrix = get_patch_inner_product_derivative_matrix(
        embedding_dim, dim, immersion
    )
    point = gs.array([gs.pi / 3])
    christoffels = metric.christoffels(point)
    print("WORKING WITH ONE POINT")

    # points =  gs.linspace(0,2*gs.pi,100)
    # christoffels = metric.christoffels(points)

def test_christoffels_s1():
    dim, embedding_dim, radius = 1, 2, 1
    immersion = get_immersion(radius=radius)
    metric = PullbackMetric(dim=dim, embedding_dim=embedding_dim, immersion=immersion)
    point = gs.array([gs.pi / 3])

    christoffels = metric.christoffels(point)

    assert gs.allclose(christoffels.shape, (1, 1, 1)), christoffels.shape
    assert gs.allclose(christoffels, gs.zeros((1, 1, 1))), christoffels.shape

def test_christoffels_s2():
    dim, embedding_dim, radius = 2, 3, 1
    immersion = get_sphere_immersion(radius=radius)
    metric = PullbackMetric(dim=dim, embedding_dim=embedding_dim, immersion=immersion)
    # PATCH: assign new method in pullback metric
    metric.inner_product_derivative_matrix = get_patch_inner_product_derivative_matrix(
        embedding_dim, dim, immersion
    )
    point = gs.array([gs.pi / 3, gs.pi])
    theta, phi = point[0], point[1]

    christoffels = metric.christoffels(point)

    assert gs.allclose(christoffels.shape, (2, 2, 2)), christoffels.shape
    assert ~gs.allclose(christoffels, gs.zeros((2, 2, 2))), "christoffels are zero"

    expected_1_11 = expected_2_11 = expected_2_22 = expected_1_12 = 0

    assert gs.allclose(christoffels[0, 0, 0], expected_1_11), christoffels[0, 0, 0]
    assert gs.allclose(christoffels[1, 0, 0], expected_2_11), christoffels[1, 0, 0]
    assert gs.allclose(christoffels[1, 1, 1], expected_2_22), christoffels[1, 1, 1]
    assert gs.allclose(christoffels[0, 0, 1], expected_1_12), christoffels[0, 0, 1]

    expected_1_22 = -gs.sin(theta) * gs.cos(theta)
    expected_2_12 = expected_2_21 = gs.cos(theta) / gs.sin(theta)

    assert gs.allclose(christoffels[0, 1, 1], expected_1_22), christoffels[0, 1, 1]
    assert gs.allclose(christoffels[1, 0, 1], expected_2_12), christoffels[1, 0, 1]
    assert gs.allclose(christoffels[1, 1, 0], expected_2_21), christoffels[1, 1, 0]


def test_compute_mean_curvature_s1():
    dim = 1
    embedding_dim = 2
    radius = 3

    points = gs.array([[gs.pi / 3]])

    immersion = get_immersion(radius=radius)
    mean_curvature, mean_curvature_norms = compute_mean_curvature(
        points=points, immersion=immersion, dim=dim, embedding_dim=embedding_dim
    )
    print("s1 mean curvature norms", mean_curvature_norms)
    assert gs.allclose(
        mean_curvature_norms,
        [
            1 / radius,
        ]
        * len(points),
    )


def test_compute_mean_curvature_s2():
    dim = 2
    embedding_dim = 3
    radius = 3

    points = gs.array(
        [[gs.pi / 3, gs.pi], [gs.pi / 6, gs.pi / 2], [gs.pi / 4, gs.pi / 2]]
    )
    print("points s2:", points.shape)
    # sec_fun = sff(immersion,point,dim,embedding_dim)

    immersion = get_sphere_immersion(radius=radius)
    mean_curvature, mean_curvature_norms = compute_mean_curvature(
        points=points, immersion=immersion, dim=dim, embedding_dim=embedding_dim
    )
    print("s2 mean curvature norms", mean_curvature_norms)
    assert gs.allclose(
        mean_curvature_norms,
        [
            2 / radius,
        ]
        * len(points),
    ), mean_curvature_norms
