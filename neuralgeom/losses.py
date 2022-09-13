"""Losses."""

import torch

from hyperspherical_vae.distributions import VonMisesFisher
from hyperspherical_vae.distributions import HypersphericalUniform


def elbo(x, x_mu, posterior_params, z, labels, config):
    """Compute VAE elbo loss.

    The VAE elbo loss is defined as:
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
    gen_likelihood_params : tuple
        Learned distributional parameters of generative model. (e.g., (x_mu,x_logvar) for Gaussian).
    posterior_params : tuple
        Learned distributional parameters of approximate posterior. (e.g., (z_mu,z_logvar) for Gaussian).
    config : module
        Module specifying various model hyperparameters

    Returns
    -------
    _ : array-like, shape=[batch_size,]
        VAE elbo loss function on each batch element.
    """

    if config.posterior_type == "gaussian":
        z_mu, z_logvar = posterior_params
        z_var = torch.exp(z_logvar)
        kld = -0.5 * torch.sum(1 + z_logvar - z_mu.pow(2) - z_var)

    if config.posterior_type == "hyperspherical":
        z_mu, z_kappa = posterior_params
        q_z = VonMisesFisher(z_mu, z_kappa)
        p_z = HypersphericalUniform(config.latent_dim - 1)
        kld = torch.distributions.kl.kl_divergence(q_z, p_z).mean()

    if config.gen_likelihood_type == "gaussian":
        recon_loss = torch.sum((x - x_mu).pow(2))
    # + constant
    # elif config.gen_likelihood_type == "laplacian":
    #     x_mu, x_logvar = gen_likelihood_params
    #     recon_loss = (
    #         torch.nn.BCEWithLogitsLoss(reduction="none")(x_mu, x).sum(-1).mean()
    #     )

    # elif config.gen_likelihood_type == "poisson":
    #     x_lambda = gen_likelihood_params
    #     from scipy import special
    #      # TODO: check why there are "nan"'s coming up
    #     recon_loss = torch.sum(
    #         -x * torch.log(x_lambda) + x_lambda + torch.log(special.factorial(x))
    #     )

    return (
        recon_loss
        + config.beta * kld
        + config.gamma * latent_regularization_loss(labels, z, config)
    )


def latent_regularization_loss(labels, z, config):
    """Compute the latent space regularization loss.
    This loss is intended to enforce a certain structure on the latent space.
    Implemented here is a 'circle' regularization loss, where the latent variables
    are penalized for deviating from a radius = 1 from the origin, and
    for representing the wrong 'angle' compared to the ground truth labels.

    Parameters
    ----------
    labels : array-like, shape=[batch_size]
        Input labels.
    posterior_params : tuple
        Learned distributional parameters of approximate posterior. (e.g., (z_mu,z_logvar) for Gaussian).
    config : module
        Module specifying various model hyperparameters

    Returns
    -------
    _ : array_like, shape=[batch_size]
        Latent regularization loss on each batch element.

    """

    # latent_angles = (torch.atan2(z[:, 1], z[:, 0]) + 2 * torch.pi) % (
    #     2 * torch.pi
    # )

    latent_angles = (torch.atan2(z[:, 1], z[:, 0]) + 2 * torch.pi) % (2 * torch.pi)

    if config.dataset_name == "experimental":
        labels = labels * (torch.pi / 180)

    angle_loss = torch.sum(((latent_angles - labels) % (2 * torch.pi)) ** 2)

    return angle_loss