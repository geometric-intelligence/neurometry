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
        recon_loss = torch.mean((x - x_mu).pow(2))

    if config.dataset_name == "s1_synthetic":
        recon_loss = recon_loss / (config.radius**2)


    return (
        recon_loss
        + config.beta * kld
        + config.gamma * latent_regularization_loss(labels, z, config)
    )


def latent_regularization_loss(labels, z, config):

    if config.dataset_name == "s1_synthetic":
        latent_angles = (torch.atan2(z[:, 1], z[:, 0]) + 2 * torch.pi) % (2 * torch.pi)
        angle_loss = torch.mean(1 - torch.cos(latent_angles - labels))
        latent_loss = angle_loss
    elif config.dataset_name == "experimental":
        labels = labels * (torch.pi / 180)
        latent_angles = (torch.atan2(z[:, 1], z[:, 0]) + 2 * torch.pi) % (2 * torch.pi)
        angle_loss = torch.mean(1 - torch.cos(latent_angles - labels))
        latent_loss = angle_loss
    elif config.dataset_name == "s2_synthetic":
        latent_thetas = torch.arccos(z[:,2])
        latent_phis = (torch.atan2(z[:, 1], z[:, 0]) + 2 * torch.pi) % (2 * torch.pi)
        thetas_loss = torch.mean(1-torch.cos(latent_thetas-labels[:,0]))
        phis_loss = torch.mean(torch.sin(latent_thetas)*torch.sin(labels[:,0])*(1-torch.cos(latent_phis-labels[:,1])))
        latent_loss = thetas_loss + phis_loss
    elif config.dataset_name == "t2_synthetic":
        # TODO
        latent_loss = 0

    return latent_loss





