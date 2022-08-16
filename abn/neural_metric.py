
import os
os.environ["GEOMSTATS_BACKEND"] = "pytorch"
import geomstats.backend as gs

from geomstats.geometry.pullback_metric import PullbackMetric
import torch
import numpy
import matplotlib.pyplot as plt
from geomstats.geometry.riemannian_metric import RiemannianMetric


class NeuralMetric(PullbackMetric):
    
    def __init__(self, dim, embedding_dim, immersion):
        super(NeuralMetric, self).__init__(dim=dim, embedding_dim = embedding_dim, immersion=immersion)
        self.dim = dim
        self.embedding_dim = embedding_dim
        self.immersion = immersion

    def injectivity_radius(self, base_point):
        return gs.pi

    # def immersion_i(self, i, x):
    #     return self.immersion(x)[i]


    def mean_curvature(self, base_point):

        H = gs.zeros((self.embedding_dim,))
        for i in range(self.embedding_dim):
            H[i] = torch.autograd.functional.hessian(func = lambda x: self.immersion(x)[i],inputs = base_point)  #self.f_i[self.immersion,x,i])
        return H
        



