import os

os.environ["GEOMSTATS_BACKEND"] = "pytorch"
import geomstats.backend as gs
from geomstats.geometry.hypersphere import Hypersphere



def hypersphere(intrinsic_dim,num_points,radius=1):
    unit_hypersphere = Hypersphere(dim=intrinsic_dim)
    unit_hypersphere_points = unit_hypersphere.random_point(n_samples=num_points)
    hypersphere_points = radius*unit_hypersphere_points
    return hypersphere_points


def random_encoding_matrix(manifold_extrinsic_dim,encoding_dim):
    return gs.random.uniform(-1,1,(manifold_extrinsic_dim,encoding_dim))


def encode_manifold(manifold_points,encoding_matrix):
    encoded_points = gs.einsum("ij,jk->ik",manifold_points,encoding_matrix)
    return encoded_points












