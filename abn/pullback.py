import os
os.environ["GEOMSTATS_BACKEND"] = "pytorch"
import geomstats.backend as gs

import sys
import warnings

sys.path.append(os.path.dirname(os.getcwd()))
warnings.filterwarnings("ignore")

from geomstats.geometry.pullback_metric import PullbackMetric
import torch


def get_metric(intrinsic_dim, model_filename):
    """ Calculate metric induced on latent space manifold using pull-back method.
        See: https://geomstats.github.io/notebooks/09_practical_methods__implement_your_own_riemannian_geometry.html

    Parameters
    ----------

    intrinsic_dim : int
        Dimension of underlying manifold. 
    immersion : function
        Immersion.
    model_filename : string
        Filename of saved generative model.

    Returns
    -------
    metric : geomstats.geometry.pullback_metric.PullbackMetric object
        Riemannian metric induced on underlying manifold by given immersion.

    """
    model = torch.load(model_filename)
    model.eval()

    def neural_immersion(theta):
        """Compute the immersion from S^1 [position of rat along circle] to R^N. 
        
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
    
        z = gs.stack([gs.cos(theta),gs.sin(theta)],axis=-1)
        recon_x = model.decode(z)
        x_mu, _ = recon_x
        return x_mu

    def polar_immersion(theta):
        return gs.stack([gs.cos(theta),gs.sin(theta)],axis=-1)

    data_dim = model.data_dim
    
    #metric = PullbackMetric(dim=intrinsic_dim, embedding_dim=data_dim, immersion = neural_immersion)
    metric = PullbackMetric(dim=intrinsic_dim, embedding_dim=2, immersion = polar_immersion)

    return metric


def compute_curvature(metric):
    return TODO


comment = "hello"