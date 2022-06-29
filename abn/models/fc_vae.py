"""G-manifold deep learning.

This file gathers deep learning models related to G-manifold learning.
"""

import torch
from torch.nn import functional as F
import torch.distributions as distributions

# TODO: Investigate this:
# proceedings.neurips.cc//paper/2020/file/510f2318f324cf07fce24c3a4b89c771-Paper.pdf


class VAE(torch.nn.Module):
    """VAE with Linear (fully connected) layers.

    Parameters
    ----------
    data_dim : int
        Dimension of input data.
        Example: 40 for neural recordings of 40 units/clusters.
    latent_dim : int
        Dimension of the latent space.
        Example: 2.
    """

    def __init__(self, data_dim, latent_dim, posterior_type = "Gaussian", gen_likelihood_type = "Gaussian"):
        super(VAE, self).__init__()
        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.posterior_type = posterior_type
        self.gen_likelihood_type = gen_likelihood_type
        self.fc1 = torch.nn.Linear(self.data_dim, 400)
        self.fc2 = torch.nn.Linear(400, 400)
        
        if posterior_type == "Gaussian":
            self.fc_z_mu = torch.nn.Linear(400, self.latent_dim)
            self.fc_z_logvar = torch.nn.Linear(400, self.latent_dim)
        #hello
        
        self.fc3 = torch.nn.Linear(self.latent_dim, 400)
        self.fc4 = torch.nn.Linear(400,400)
        
        
        if gen_likelihood_type == "Gaussian":
            self.fc_x_mu = torch.nn.Linear(400, self.data_dim)
            #adding hidden layer to logvar
            self.fc_x_logvar1 = torch.nn.Linear(400,400)
            self.fc_x_logvar2 = torch.nn.Linear(400,self.data_dim)
        elif gen_likelihood_type == "Poisson":
            self.fc_x_lambda = torch.nn.Linear(400,self.data_dim)

    def encode(self, x):
        """Encode input into mean and log-variance.

        The parameters mean (mu) and variance (computed
        from logvar) defines a multivariate Gaussian
        that represents the approximate posterior of the
        latent variable z given the input x.

        Parameters
        ----------
        x : array-like, shape=[batch_size, data_dim]
            Input data.

        Returns
        -------
        mu : array-like, shape=[batch_size, latent_dim]
            Mean of multivariate Gaussian in latent space.
        logvar : array-like, shape=[batch_size, latent_dim]
            Vector representing the diagonal covariance of the
            multivariate Gaussian in latent space.
        """
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        if self.posterior_type == "Gaussian":
            z_mu = self.fc_z_mu(h2)
            z_logvar = self.fc_z_logvar(h2)
            posterior_params = z_mu, z_logvar

        return posterior_params

    # def reparameterize(self, mu, logvar):
    #     """Sample from Gaussian of parameters mu, logvar.

    #     Parameters
    #     ----------
    #     mu : array-like, shape=[batch_size, latent_dim]
    #         Mean of multivariate Gaussian in latent space.
    #     logvar : array-like, shape=[batch_size, latent_dim]
    #         Vector representing the diagonal covariance of the
    #         multivariate Gaussian in latent space.

    #     Returns
    #     -------
    #     _ : array-like, shape=[batch_size, latent_dim]
    #         Sample of the multivariate Gaussian of parameters
    #         mu and logvar.
    #     """
    #     std = torch.exp(0.5 * logvar)
    #     eps = torch.randn_like(std)
    #     return mu + eps * std

    def reparameterize(self, posterior_params):
        if self.posterior_type == "Gaussian":
            z_mu, z_logvar = posterior_params
            z_std = torch.exp(0.5 * z_logvar)
            eps = torch.randn_like(z_std)
            z = z_mu + eps*z_std     
        
        return z

    def decode(self, z):
        """Decode latent variable z into data.

        Parameters
        ----------
        z : array-like, shape=[batch_size, latent_dim]
            Input to the decoder.

        Returns
        -------
        _ : array-like, shape=[batch_size, data_dim]
            Reconstructed data corresponding to z.
        """
        h3 = F.relu(self.fc3(z))
        h4 = F.relu(self.fc4(h3))
        if self.gen_likelihood_type == "Gaussian":
            x_mu = self.fc_x_mu(h4)
            #adding hidden layer to x_logvar
            h_x_logvar = self.fc_x_logvar1(h4)
            x_logvar = self.fc_x_logvar2(h_x_logvar)
            gen_likelihood_params = x_mu, x_logvar
        elif self.gen_likelihood_type == "Poisson":
            x_lambda = self.fc_x_lambda(h4)
            gen_likelihood_params = x_lambda
        
        return gen_likelihood_params


    def forward(self, x):
        """Run VAE: Encode, sample and decode.

        Parameters
        ----------
        x : array-like, shape=[batch_size, data_dim]
            Input data.

        Returns
        -------
        _ : array-like, shape=[batch_size, data_dim]
            Reconstructed data corresponding to z.
        mu : array-like, shape=[batch_size, latent_dim]
            Mean of multivariate Gaussian in latent space.
        logvar : array-like, shape=[batch_size, latent_dim]
            Vector representing the diagonal covariance of the
            multivariate Gaussian in latent space.
        """
        
        posterior_params = self.encode(x)
        z = self.reparameterize(posterior_params)
        gen_likelihood_params = self.decode(z)
        return gen_likelihood_params, posterior_params 



