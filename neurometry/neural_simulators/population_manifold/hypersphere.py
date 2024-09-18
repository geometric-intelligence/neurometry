from geomstats.geometry.hypersphere import Hypersphere
from neurometry.neural_simulators.population_manifold.base import PopulationManifold


class PopulationHypersphere(PopulationManifold):
    def __init__(self, n_neurons, intrinsic_dim, nonlinearity="relu", ref_frequency=200, fano_factor=1.0):
        super().__init__(n_neurons=n_neurons, manifold=Hypersphere(dim=intrinsic_dim), nonlinearity=nonlinearity, ref_frequency=ref_frequency, fano_factor=fano_factor)