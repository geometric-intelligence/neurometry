from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.product_manifold import ProductManifold
from neurometry.neural_simulators.population_manifold.base import PopulationManifold


class PopulationHypertorus(PopulationManifold):
    def __init__(self, n_neurons, intrinsic_dim, nonlinearity="relu", ref_frequency=200, fano_factor=1.0):
        factors = [Hypersphere(dim=1) for _ in range(intrinsic_dim)]
        unit_hypertorus = ProductManifold(factors=factors)

        super().__init__(n_neurons=n_neurons, manifold=unit_hypertorus, nonlinearity=nonlinearity, ref_frequency=ref_frequency, fano_factor=fano_factor)