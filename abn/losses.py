"""Losses."""

import torch
from torch.nn import functional as F


def elbo(x, gen_likelihood_type, posterior_type, gen_likelihood_params, posterior_params, beta=1.0):
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

    x : array-like, shape=[batch_size, data_dim]
        Input data.
    gen_likelihood_type : string
        Specifies type of distribution used for generative model
    posterior_type : string
        Specifies type of distribution used as approximate posterior
    gen_likelihood_params : tuple
        Distributional parameters of generative model. ((e.g.), (x_mu,x_logvar) for Gaussian.
    posterior_params : tuple
        Distributional parameters of approximate posterior. ((e.g.), (z_mu,z_logvar) for Gaussian.
    beta : float
        multiplicative factor in front of Kld term in loss, should help with 
        disentangling latent space. Classic VAE has beta = 1. See beta-VAE (Higgins, et al.)

    Returns
    -------
    _ : array-like, shape=[batch_size,]
        Loss function on each batch element.
    """
    if posterior_type == "Gaussian":
        z_mu, z_logvar = posterior_params
        z_var = torch.exp(z_logvar)
        kld = -0.5 * torch.sum(1 + z_logvar - z_mu.pow(2) - z_var)
    
    if gen_likelihood_type == "Gaussian":
        x_mu, x_logvar = gen_likelihood_params
        x_var = torch.exp(x_logvar)
        #manually setting x_var = 0.001
        x_var = torch.zeros(x_var.shape) + 1e-3
        recon_loss = torch.sum(0.5*torch.log(x_var) + 0.5*torch.div((x-x_mu).pow(2),x_var)) # + constant
    elif gen_likelihood_type == "Poisson":
        x_lambda = gen_likelihood_params
        from scipy import special
        #TODO: check why there are "nan"'s coming up 
        recon_loss = torch.sum(-x*torch.log(x_lambda) + x_lambda + torch.log(special.factorial(x)))

    return recon_loss + beta * kld
