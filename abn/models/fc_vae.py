"""G-manifold deep learning.

This file gathers deep learning models related to G-manifold learning.
"""

import torch
from torch.nn import functional as F


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

    def __init__(
        self,
        data_dim,
        latent_dim,
        encoder_width=400,
        encoder_depth=2,
        decoder_width=400,
        decoder_depth=2,
        posterior_type="gaussian",
        gen_likelihood_type="gaussian",
    ):
        super(VAE, self).__init__()
        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.posterior_type = posterior_type
        self.gen_likelihood_type = gen_likelihood_type

        

        self.encoder_fc = torch.nn.Linear(self.data_dim, encoder_width)
        self.encoder_linears = torch.nn.ModuleList(
            [torch.nn.Linear(encoder_width, encoder_width) for _ in range(encoder_depth)])

        if posterior_type == "gaussian":
            self.fc_z_mu = torch.nn.Linear(encoder_width, self.latent_dim)
            self.fc_z_logvar = torch.nn.Linear(encoder_width, self.latent_dim)

        self.decoder_fc = torch.nn.Linear(self.latent_dim, decoder_width)
        self.decoder_linears = torch.nn.ModuleList(
            [torch.nn.Linear(decoder_width, decoder_width) for _ in range(decoder_depth)])

        if gen_likelihood_type == "gaussian":
            self.fc_x_mu = torch.nn.Linear(decoder_width, self.data_dim)
            # adding hidden layer to logvar
            self.fc_x_logvar1 = torch.nn.Linear(decoder_width, decoder_width)
            self.fc_x_logvar2 = torch.nn.Linear(decoder_width, self.data_dim)
        elif gen_likelihood_type == "poisson":
            self.fc_x_lambda = torch.nn.Linear(decoder_width, self.data_dim)

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
        h = F.relu(self.encoder_fc(x))

        for layer in self.encoder_linears:
            h = F.relu(layer(h))

        if self.posterior_type == "gaussian":
            z_mu = self.fc_z_mu(h)
            z_logvar = self.fc_z_logvar(h)
            posterior_params = z_mu, z_logvar

        return posterior_params

    def reparameterize(self, posterior_params):
        """
        Apply reparameterization trick. We 'eternalize' the
        randomness in z by re-parameterizing the variable as
        a deterministic and differentiable function of x,
        the encoder weights, and a new random variable eps.

        Parameters
        ----------
        posterior_params : tuple
            Distributional parameters of approximate posterior. ((e.g.), (z_mu,z_logvar) for Gaussian.

        Returns
        -------

        z: array-like, shape = [batch_size, latent_dim]
            Re-parameterized latent variable.
        """

        if self.posterior_type == "gaussian":
            z_mu, z_logvar = posterior_params
            z_std = torch.exp(0.5 * z_logvar)
            eps = torch.randn_like(z_std)
            z = z_mu + eps * z_std

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
        h = F.relu(self.decoder_fc(z))

        for layer in self.decoder_linears:
            h = F.relu(layer(h))

        if self.gen_likelihood_type == "gaussian":
            x_mu = self.fc_x_mu(h)
            # adding hidden layer to x_logvar
            h_x_logvar = self.fc_x_logvar1(h)
            x_logvar = self.fc_x_logvar2(h_x_logvar)
            gen_likelihood_params = x_mu, x_logvar
        elif self.gen_likelihood_type == "poisson":
            x_lambda = self.fc_x_lambda(h)
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
