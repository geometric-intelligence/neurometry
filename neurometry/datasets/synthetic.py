import os

os.environ["GEOMSTATS_BACKEND"] = "pytorch"
import geomstats.backend as gs
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.product_manifold import ProductManifold


### Synthetic Latent Manifolds ###

def hypersphere(intrinsic_dim,num_points,radius=1):
    unit_hypersphere = Hypersphere(dim=intrinsic_dim)
    unit_hypersphere_points = unit_hypersphere.random_point(n_samples=num_points)
    hypersphere_points = radius*unit_hypersphere_points
    return hypersphere_points

def hypertorus(intrinsic_dim,num_points,radii=None):
    factors = [Hypersphere(dim=1) for _ in range(intrinsic_dim)]
    unit_hypertorus = ProductManifold(factors=factors)
    unit_hypertorus_points = unit_hypertorus.random_point(n_samples=num_points)
    hypertorus_points = gs.zeros_like(unit_hypertorus_points)
    if radii is not None:
        assert len(radii)==intrinsic_dim, f"radii must be a list of length {intrinsic_dim}"
        for _ in range(intrinsic_dim):
            hypertorus_points[:,_,:] = radii[_]*unit_hypertorus_points[:,_,:]
    hypertorus_points = gs.reshape(hypertorus_points,(num_points,intrinsic_dim*2))
    return hypertorus_points

def cylinder(num_points,radius=1):
    factors = [Hypersphere(dim=1),Euclidean(dim=1)]
    cylinder = ProductManifold(factors=factors)
    cylinder_points = cylinder.random_point(n_samples=num_points,bound=1)
    cylinder_points[:,:2] = radius*cylinder_points[:,:2]
    return cylinder_points

def klein_bottle():
    # waiting for geomstats implementation 
    return NotImplementedError


### Synthetic Encoding Scheme ###

def random_encoding_matrix(manifold_extrinsic_dim,encoding_dim):
    return gs.random.uniform(-1,1,(manifold_extrinsic_dim,encoding_dim))


def encode_points(manifold_points,encoding_matrix):
    encoded_points = gs.einsum("ij,jk->ik",manifold_points,encoding_matrix)
    return encoded_points

    
def apply_nonlinearity(encoded_points, nonlinearity, **kwargs):
    if nonlinearity == 'relu':
        return relu(encoded_points, **kwargs)
    elif nonlinearity == 'sigmoid':
        return scaled_sigmoid(encoded_points, **kwargs)
    elif nonlinearity == 'tanh':
        return scaled_tanh(encoded_points, **kwargs)
    else:
        raise ValueError("Nonlinearity not recognized")


def relu(tensor, threshold=0):
    return gs.maximum(threshold, tensor)

def scaled_sigmoid(tensor, scales):
    assert(tensor.shape[1] == scales.shape[0]), "scales must have same shape as tensor"
    return 1 / (1 + gs.exp(-scales*tensor))

def scaled_tanh(tensor, scales):
    assert(tensor.shape[1] == scales.shape[0]), "scales must have same shape as tensor"
    return gs.tanh(scales*tensor)




















