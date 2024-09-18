from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.product_manifold import ProductManifold
from neurometry.stimuli.base import Stimuli


class StimuliHypertorus(Stimuli):
    def __init__(self, intrinsic_dim):
        factors = [Hypersphere(dim=1) for _ in range(intrinsic_dim)]
        unit_hypertorus = ProductManifold(factors=factors)
        super().__init__(manifold=unit_hypertorus)