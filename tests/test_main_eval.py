import os

os.environ["GEOMSTATS_BACKEND"] = "pytorch"
import geomstats.backend as gs
import torch

from neuralgeom.main_eval import compute_mean_curvature
from neuralgeom.main_eval import get_second_fundamental_form as sff


def get_immersion(radius):
    def immersion(theta):
        return radius * gs.array([gs.cos(theta), gs.sin(theta)])

    return immersion


# Note: same immersion as geomstats' test_pullback_metric.py
def get_sphere_immersion(radius):
    def immersion(point):
        theta = point[0]
        phi = point[1]
        x = gs.sin(theta) * gs.cos(phi)
        y = gs.sin(theta) * gs.sin(phi)
        z = gs.cos(theta)
        return radius * gs.array([x, y, z])

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
    expected_k1 = gs.array([
        radius * gs.cos(theta) * gs.cos(phi),
        - radius * gs.sin(theta) * gs.sin(phi),
    ])
    expected_k2 = gs.array([
        radius * gs.cos(theta) * gs.sin(phi),
        radius * gs.sin(theta) * gs.cos(phi),])
    expected_k3 = gs.array([
        - radius * gs.sin(theta),
        0,
    ])
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
    print("s2 second fundamental form", sec_fun)

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
    print("sec_fun_11", sec_fun_11.shape)
    sec_fun_22 = sec_fun[:, 1, 1]
    assert gs.allclose(sec_fun_11.shape, expected_11.shape), sec_fun_11.shape
    assert gs.allclose(sec_fun_22.shape, expected_22.shape), sec_fun_22.shape

    assert gs.allclose(sec_fun_11, expected_11), sec_fun_11
    assert gs.allclose(sec_fun_22, expected_22, atol=1e-5), sec_fun_22


def test_compute_mean_curvature_s1():
    dim = 1
    embedding_dim = 2
    radius = 3

    # Note: error for point = angle = 0
    points = gs.linspace(1, 2 * gs.pi, 3)
    points = points.reshape((len(points), 1))
    print("points s1:", points.shape)
    # sec_fun = sff(immersion,point,dim,embedding_dim)

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
