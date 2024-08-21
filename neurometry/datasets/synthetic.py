import os

os.environ["GEOMSTATS_BACKEND"] = "pytorch"
import geomstats.backend as gs
import torch
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.klein_bottle import KleinBottle
from geomstats.geometry.product_manifold import ProductManifold


def synthetic_neural_manifold(
    points,
    encoding_dim,
    nonlinearity,
    poisson_multiplier=1,
    ref_frequency=200,
    **kwargs,
):
    """Generate points on a synthetic neural manifold.

    Parameters
    ----------
    points : array-like, shape=[num_points, intrinsic_dim]
        Points on the manifold.
    encoding_dim : int
        Dimension of the encoded points. This is the dimension of the neural state space.
    nonlinearity : str
        Nonlinearity to apply. Must be one of 'relu', 'sigmoid', 'tanh', or 'linear'.
    **kwargs : dict
        Keyword arguments for the manifold generation.

    Returns
    -------
    manifold_points : array-like, shape=[num_points, intrinsic_dim]
        Points on the manifold.
    """
    manifold_extrinsic_dim = points.shape[1]
    encoding_matrix = random_encoding_matrix(manifold_extrinsic_dim, encoding_dim)
    encoded_points = encode_points(points, encoding_matrix)
    manifold_points = ref_frequency * apply_nonlinearity(
        encoded_points, nonlinearity, **kwargs
    )
    try:
        noisy_points = poisson_spikes(manifold_points, poisson_multiplier)
    except Exception:
        print("WARNING! Poisson spikes not generated: mean must be non-negative")
        noisy_points = None

    noise_level = gs.sqrt(1 / (ref_frequency * poisson_multiplier))
    print(f"noise level: {100*noise_level:.2f}%")

    return noisy_points, manifold_points


def hypersphere(intrinsic_dim, num_points, radius=1):
    """Generate points on a hypersphere of given intrinsic dimension and radius.

    Parameters
    ----------
    intrinsic_dim : int
        Intrinsic dimension of the hypersphere.
    num_points : int
        Number of points to generate.
    radius : float, optional
        Radius of the hypersphere. Default is 1.

    Returns
    -------
    hypersphere_points : array-like, shape=[num_points, minimal_embedding_dim]
        Points on the hypersphere.

    """
    unit_hypersphere = Hypersphere(dim=intrinsic_dim)
    unit_hypersphere_points = unit_hypersphere.random_point(n_samples=num_points)
    return radius * unit_hypersphere_points


def hypertorus(intrinsic_dim, num_points, radii=None, parameterization="flat"):
    """Generate points on a flat hypertorus of given intrinsic dimension and radii.

    The n-hypertorus is the product manifold of n circles.

    Parameters
    ----------
    intrinsic_dim : int
        Intrinsic dimension of the hypertorus.
    num_points : int
        Number of points to generate.
    radii : list of floats, optional
        Radii of the circles. If None, the hypertorus is the unit hypertorus.

    Returns
    -------
    hypertorus_points : array-like, shape=[num_points, 2*intrinsic_dim]
        Points on the hypertorus.
    """
    factors = [Hypersphere(dim=1) for _ in range(intrinsic_dim)]
    unit_hypertorus = ProductManifold(factors=factors)
    unit_hypertorus_points = unit_hypertorus.random_point(n_samples=num_points)
    hypertorus_points = unit_hypertorus_points
    if radii is not None:
        assert (
            len(radii) == intrinsic_dim
        ), f"radii must be a list of length {intrinsic_dim}"
        for _ in range(intrinsic_dim):
            hypertorus_points[:, _, :] = radii[_] * unit_hypertorus_points[:, _, :]
    return gs.reshape(hypertorus_points, (num_points, intrinsic_dim * 2))


def cylinder(num_points, radius=1):
    """Generate points on a cylinder of given radius.

    The cylinder is the product manifold of a circle and the interval [-1,1].

    Parameters
    ----------
    num_points : int
        Number of points to generate.
    radius : float, optional
        Radius of the cylinder. Default is 1.

    """
    factors = [Hypersphere(dim=1), Euclidean(dim=1)]
    cylinder = ProductManifold(factors=factors)
    cylinder_points = cylinder.random_point(n_samples=num_points, bound=1)
    cylinder_points[:, :2] = radius * cylinder_points[:, :2]
    return cylinder_points


def klein_bottle(num_points, size_factor=1, coords_type="bottle"):
    """Generate points on a Klein bottle manifold.

    Parameters
    ----------
    num_points : int
        Number of points to generate.
    size_factor : int
        Multiplies all coordinates by this size factor.
    coords_type: str
        Choose the type of parametrization desired. Options: extrinsic, bottle or bagel

    Returns
    -------
    kleinbottle_points : array-like, shape=[num_points, n] (n depends on the parametrization)
        Points on the Klein bottle.
    """
    possible_coord_types = ["bottle", "bagel", "extrinsic"]
    if coords_type not in possible_coord_types:
        raise Exception(
            "Please pick a valid parametrization for the random points on the Klein Bottle"
        )
    unit_klein_bottle = KleinBottle()
    unit_klein_bottle_points = unit_klein_bottle.random_point(n_samples=num_points)
    unit_klein_bottle_points = unit_klein_bottle.to_coords(
        unit_klein_bottle_points, coords_type
    )
    return size_factor * unit_klein_bottle_points


def random_encoding_matrix(manifold_extrinsic_dim, encoding_dim):
    """Generate a random encoding matrix.

    Parameters
    ----------
    manifold_extrinsic_dim : int
        Extrinsic dimension of the manifold. This is the minimal embedding dimension of the manifold.
    encoding_dim : int
        Dimension of the encoded points. This is the dimension of the neural state space.

    Returns
    -------
    encoding_matrix : array-like, shape=[manifold_extrinsic_dim, encoding_dim]
        Random encoding matrix.
    """
    return gs.random.uniform(-1, 1, (manifold_extrinsic_dim, encoding_dim))


def encode_points(manifold_points, encoding_matrix):
    """Encode points on a manifold using a given encoding matrix.

    Parameters
    ----------
    manifold_points : array-like, shape=[num_points, manifold_extrinsic_dim]
        Points on the manifold.
    encoding_matrix : array-like, shape=[manifold_extrinsic_dim, encoding_dim]
        Encoding matrix.

    Returns
    -------
    encoded_points : array-like, shape=[num_points, encoding_dim]
        Encoded points.
    """
    return gs.einsum("ij,jk->ik", manifold_points, encoding_matrix)


def apply_nonlinearity(encoded_points, nonlinearity, **kwargs):
    """Apply a nonlinearity to the encoded points.

    Parameters
    ----------
    encoded_points : array-like, shape=[num_points, encoding_dim]
        Encoded points.
    nonlinearity : str
        Nonlinearity to apply. Must be one of 'relu', 'sigmoid', 'tanh', or 'linear'.
    **kwargs : dict
        Keyword arguments for the nonlinearity.

    Returns
    -------
    nonlinearity_points : array-like, shape=[num_points, encoding_dim]
        Encoded points after applying the nonlinearity.
    """
    if nonlinearity == "relu":
        return relu(encoded_points, **kwargs)
    if nonlinearity == "sigmoid":
        return scaled_sigmoid(encoded_points, **kwargs)
    if nonlinearity == "tanh":
        return scaled_tanh(encoded_points, **kwargs)
    if nonlinearity == "linear":
        return encoded_points
    raise ValueError("Nonlinearity not recognized")


def relu(tensor, threshold=0):
    return gs.maximum(threshold, tensor)


def scaled_sigmoid(tensor, scales):
    assert tensor.shape[1] == scales.shape[0], "scales must have same shape as tensor"
    return 1 / (1 + gs.exp(-scales * tensor))


def scaled_tanh(tensor, scales):
    assert tensor.shape[1] == scales.shape[0], "scales must have same shape as tensor"
    return 1 + gs.tanh(scales * tensor)


def poisson_spikes(data, multiplier=1):
    """Generate Poisson spike trains from data.

    Parameters
    ----------
    data : array-like, shape=[num_points, num_neurons]
        Points on the manifold.
    multiplier : int
        Multiplier for the number of spikes to generate.

    Returns
    -------
    spikes : array-like, shape=[num_points, num_neurons]
        Poisson spike trains.
    """
    return torch.poisson(data * multiplier) / multiplier
