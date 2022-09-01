"""Losses."""

import torch


def elbo(x, model, config):
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

    posterior_params, (q_z, p_z), z, x_rec = model(x)

    recon_loss = torch.nn.BCEWithLogitsLoss(reduction="mean")(x_rec, x).sum(-1)

    if model.latent_geometry == "normal":
        kl_loss = torch.distributions.kl.kl_divergence(q_z, p_z).sum(-1).mean()
    elif model.latent_geometry == "hyperspherical":
        kl_loss = torch.distributions.kl.kl_divergence(q_z, p_z).mean()
    else:
        raise NotImplementedError

    elbo_loss = recon_loss + config.beta * kl_loss

    return elbo_loss
