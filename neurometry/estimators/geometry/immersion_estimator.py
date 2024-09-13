
from sklearn.base import BaseEstimator


class ImmersionEstimator(BaseEstimator):
    def __init__(self, num_neurons, intrinsic_dimension, topology):

        self.estimate_ = None
        self.num_neurons = num_neurons
        self.intrinsic_dimension = intrinsic_dimension
        self.topology = topology






