"""G-manifold deep learning.

This file gathers deep learning models related to G-manifold learning.
"""

import torch
from hyperspherical.distributions import HypersphericalUniform, VonMisesFisher
from torch.distributions.normal import Normal as Normal
from torch.nn import functional as F


class ToroidalVAE(torch.nn.Module):
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
        sftbeta,
        encoder_width=400,
        encoder_depth=2,
        decoder_width=400,
        decoder_depth=2,
        posterior_type="gaussian",
    ):
        super(VAE, self).__init__()
        self.data_dim = data_dim
        self.sftbeta = sftbeta
        self.latent_dim = latent_dim
        self.posterior_type = posterior_type

        decoder_width = encoder_width
        decoder_depth = encoder_depth

        self.encoder_fc = torch.nn.Linear(self.data_dim, encoder_width)
        self.encoder_linears = torch.nn.ModuleList(
            [
                torch.nn.Linear(encoder_width, encoder_width)
                for _ in range(encoder_depth)
            ]
        )

        if posterior_type == "gaussian":
            self.fc_z_mu = torch.nn.Linear(encoder_width, self.latent_dim)
            self.fc_z_logvar = torch.nn.Linear(encoder_width, self.latent_dim)
        elif posterior_type == "hyperspherical":
            self.fc_z_theta_mu = torch.nn.Linear(encoder_width, self.latent_dim)
            self.fc_z_theta_logvar = torch.nn.Linear(encoder_width, 1)  # kappa

            self.fc_z_phi_mu = torch.nn.Linear(encoder_width, self.latent_dim)
            self.fc_z_phi_logvar = torch.nn.Linear(encoder_width, 1)

        self.decoder_fc = torch.nn.Linear(self.latent_dim, decoder_width)
        self.decoder_linears = torch.nn.ModuleList(
            [
                torch.nn.Linear(decoder_width, decoder_width)
                for _ in range(decoder_depth)
            ]
        )

        self.fc_x_mu = torch.nn.Linear(decoder_width, self.data_dim)

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
        h = F.softplus(self.encoder_fc(x.double()), beta=self.sftbeta)

        for layer in self.encoder_linears:
            h = F.softplus(layer(h), beta=self.sftbeta)

        if self.posterior_type == "hyperspherical":
            z_theta_mu = self.fc_z_theta_mu(h)
            z_theta_kappa = F.softplus(self.fc_z_theta_logvar(h)) + 1
            posterior_params_theta = z_theta_mu, z_theta_kappa

            z_phi_mu = self.fc_z_phi_mu(h)
            z_phi_kappa = F.softplus(self.fc_z_phi_logvar(h)) + 1
            posterior_params_phi = z_phi_mu, z_phi_kappa

        return posterior_params_theta, posterior_params_phi

    def reparameterize(self, posterior_params_theta, posterior_params_phi):
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

        if self.posterior_type == "hyperspherical":
            z_theta_mu, z_theta_kappa = posterior_params_theta
            q_z_theta = VonMisesFisher(z_theta_mu, z_theta_kappa)
            p_z_theta = HypersphericalUniform(self.latent_dim - 1)

            z_phi_mu, z_phi_kappa = posterior_params_phi
            q_z_phi = VonMisesFisher(z_phi_mu, z_phi_kappa)
            p_z_phi = HypersphericalUniform(self.latent_dim - 1)

        z_theta = q_z_theta.rsample()

        z_phi = q_z_phi.sample()

        return z_theta, q_z_theta, p_z_theta, z_phi, q_z_phi, p_z_phi

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

        h = F.softplus(self.decoder_fc(z), beta=self.sftbeta)

        for layer in self.decoder_linears:
            h = F.softplus(layer(h), beta=self.sftbeta)

        x_mu = self.fc_x_mu(h)

        return x_mu

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
        z, _, _ = self.reparameterize(posterior_params)
        x_mu = self.decode(z)
        return x_mu, posterior_params
