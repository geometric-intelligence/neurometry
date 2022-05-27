"""Losses."""

import torch
from torch.nn import functional as F


def elbo(recon_x, x, mu, logvar):
    """Compute VAE loss function.

    The VAE loss is defined as:
    = reconstruction loss + Kl divergence
    over all elements and batch

    Notes
    -----
    see Appendix B from VAE paper:
    Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    https://arxiv.org/abs/1312.6114
    0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)

    Parameters
    ----------
    recon_x : array-like, shape=[batch_size, data_dim]
        Reconstructed data corresponding to input data x.
    x : array-like, shape=[batch_size, data_dim]
        Input data.
    mu : array-like, shape=[batch_size, latent_dim]
        Mean of multivariate Gaussian in latent space.
    logvar : array-like, shape=[batch_size, latent_dim]
        Vector representing the diagonal covariance of the
        multivariate Gaussian in latent space.

    Returns
    -------
    _ : array-like, shape=[batch_size,]
        Loss function on each batch element.
    """
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction="sum")

    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + kld
