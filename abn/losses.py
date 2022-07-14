"""Losses."""

import torch

def compute_loss(x, labels, gen_likelihood_params, posterior_params, config):
    """ Compute total loss function.

    The total loss consists of the VAE elbo loss plus 
    a latent space regularization loss to enforce a certain
    structure in the latent space. 

    Parameters
    ----------

    x : array-like, shape=[batch_size, input_dim]
        Input data.
    labels : array-like, shape=[batch_size]
        Input labels.
    gen_likelihood_params : tuple
        Learned distributional parameters of generative model. (e.g., (x_mu,x_logvar) for Gaussian).
    posterior_params : tuple
        Learned distributional parameters of approximate posterior. (e.g., (z_mu,z_logvar) for Gaussian).   
    config : module
        Module specifying various hyperparameters
    
    Hyperparameters in config:
    -------------------
    gen_likelihood_type : string
        Specifies type of distribution used for generative model
    posterior_type : string
        Specifies type of distribution used as approximate posterior
    beta : float
        Multiplicative factor in front of Kld term in loss, should help with 
        disentangling latent space. Classic VAE has beta = 1. See beta-VAE (Higgins, et al.)
    alpha : float
        Hyperparameter weighing contribution of 'circle loss' to latent regularization loss
    gamma: float
        Hyperparameter weighing contribution of 'angle loss' to latent regularization loss

    Returns
    -------
    _ : array-like, shape=[batch_size]
        Total loss function on each batch element. 

    """

    elbo_loss = elbo(x, gen_likelihood_params, posterior_params, config)
    lsr_loss = latent_regularization_loss(labels, posterior_params, config)
    return elbo_loss + lsr_loss


def elbo(x, gen_likelihood_params, posterior_params, config):
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

    
    if config.gen_likelihood_type == "gaussian":
        x_mu, x_logvar = gen_likelihood_params
        x_var = torch.exp(x_logvar)
        #manually setting x_var = 0.001
        x_var = torch.zeros(x_var.shape) + 1e-3
        recon_loss = torch.sum(0.5*torch.log(x_var) + 0.5*torch.div((x-x_mu).pow(2),x_var)) # + constant
    elif config.gen_likelihood_type == "poisson":
        x_lambda = gen_likelihood_params
        from scipy import special
        #TODO: check why there are "nan"'s coming up 
        recon_loss = torch.sum(-x*torch.log(x_lambda) + x_lambda + torch.log(special.factorial(x)))
    
    return recon_loss + config.beta * kld


def latent_regularization_loss(labels,posterior_params, config):
    """ Compute the latent space regularization loss.
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
    if config.posterior_type == "gaussian":
        z_mu, _ = posterior_params
    
    circle_loss = torch.sum((1-torch.linalg.norm(z_mu,dim=1))**2)

    latent_angles = ((torch.atan2(z_mu[:,1],z_mu[:,0]) + 2*torch.pi)%(2*torch.pi))*(180/torch.pi)

    angle_loss = torch.sum((latent_angles-labels)**2)

    return config.alpha*circle_loss + config.gamma*angle_loss

