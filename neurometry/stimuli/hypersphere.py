from geomstats.geometry.hypersphere import Hypersphere
from neurometry.stimuli.base import Stimuli


class StimuliHypersphere(Stimuli):
    def __init__(self, intrinsic_dim):
        super().__init__(manifold=Hypersphere(dim=intrinsic_dim))