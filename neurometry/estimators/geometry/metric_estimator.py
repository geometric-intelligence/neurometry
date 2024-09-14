# import os

# os.environ["GEOMSTATS_BACKEND"] = "pytorch"
# from geomstats.geometry.base import ImmersedSet
# from geomstats.geometry.euclidean import Euclidean
# from geomstats.geometry.pullback_metric import PullbackMetric


# class NeuralManifoldIntrinsic(ImmersedSet):
#     def __init__(self, dim, neural_embedding_dim, neural_immersion, equip=True):
#         self.neural_embedding_dim = neural_embedding_dim
#         super().__init__(dim=dim, equip=equip)
#         self.neural_immersion = neural_immersion

#     def immersion(self, point):
#         return self.neural_immersion(point)

#     def _define_embedding_space(self):
#         return Euclidean(dim=self.neural_embedding_dim)

#     neural_manifold = NeuralManifoldIntrinsic(
#         dim, embedding_dim, immersion, equip=False
#     )
#     neural_manifold.equip_with_metric(PullbackMetric)
