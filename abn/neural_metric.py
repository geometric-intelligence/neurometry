import os

os.environ["GEOMSTATS_BACKEND"] = "pytorch"
import geomstats.backend as gs

from geomstats.geometry.pullback_metric import PullbackMetric
import torch


class NeuralMetric(PullbackMetric):
    def __init__(self, dim, embedding_dim, immersion):
        super(NeuralMetric, self).__init__(
            dim=dim, embedding_dim=embedding_dim, immersion=immersion
        )
        self.dim = dim
        self.embedding_dim = embedding_dim
        self.immersion = immersion

    def injectivity_radius(self, base_point):
        return gs.pi

    def mean_curvature(self, base_point):

        H = gs.zeros((self.embedding_dim,))
        for i in range(self.embedding_dim):
            H[i] = torch.autograd.functional.hessian(
                func=lambda x: self.immersion(x)[i], inputs=base_point
            )
        return H


def get_neural_immersion(model):
    """NEED TO FIX DOCUMENTATION HERE


    Compute the immersion from S^1 [position of rat along circle] to R^N.

    R^N represents the place cell neural state space.
    This function f is the composition of:
        a function p : S^1 -> R^2   {takes an angle, returns (x,y) coords}
        and a function Dec : R^2 -> R^N     {takes (x,y) coords in latent space, returns reconstruction}

        f(theta) = Dec(p(theta))

    Parameters
    ----------

    model : models.fc_vae.VAE
        VAE model. The decoder that reconstructs neural activity is used to define the immersion.
    theta : array_like, shape=[batch_size,]
        Position of rat along circular track, expressed as an angle.

    Returns
    -------
    x_mu : array_like, shape=[batch_size, data_dim]
        Mean of generative model likelihood distribution. [CURRENTLY ASSUMED TO BE GAUSSIAN]
    """

    def neural_immersion(theta):
        z = gs.array([gs.cos(theta),gs.sin(theta)])
        recon_x = model.decode(z)
        x_mu, _ = recon_x
        return x_mu

    return neural_immersion
