"""G-manifold deep learning.

This file gathers deep learning models related to G-manifold learning.
"""

import geomstats.backend as gs
import torch
from hyperspherical.distributions import VonMisesFisher
from torch.nn import functional as F


class KleinBottleVAE(torch.nn.Module):
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
        encoder_width,
        encoder_depth,
        decoder_width,
        decoder_depth,
        posterior_type,
    ):
        super().__init__()
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

        self.fc_z_theta_mu = torch.nn.Linear(encoder_width, self.latent_dim)
        self.fc_z_theta_kappa = torch.nn.Linear(encoder_width, 1)

        self.fc_z_u_mu = torch.nn.Linear(encoder_width, self.latent_dim)
        self.fc_z_u_kappa = torch.nn.Linear(encoder_width, 1)

        self.decoder_fc = torch.nn.Linear(3, decoder_width)
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
        h = F.softplus(self.encoder_fc(x), beta=self.sftbeta)

        for layer in self.encoder_linears:
            h = F.softplus(layer(h), beta=self.sftbeta)

        z_theta_mu = self.fc_z_theta_mu(h)
        z_theta_kappa = F.softplus(self.fc_z_theta_kappa(h)) + 1

        z_u_mu = self.fc_z_u_mu(h)
        z_u_kappa = F.softplus(self.fc_z_u_kappa(h)) + 1

        return z_theta_mu, z_theta_kappa, z_u_mu, z_u_kappa


    def _build_klein_bottle(self, z_theta, z_u):
        # theta = torch.atan2(z_theta[:, 1] / z_theta[:, 0])
        # phi = torch.atan2(z_u[:, 1] / z_u[:, 0])

        r = 5

        theta = z_theta
        v = z_u

        x = (
            r + gs.cos(theta / 2) * gs.sin(v) - gs.sin(theta / 2) * gs.sin(2 * v)
        ) * gs.cos(theta)
        y = (
            r + gs.cos(theta / 2) * gs.sin(v) - gs.sin(theta / 2) * gs.sin(2 * v)
        ) * gs.sin(theta)
        z = gs.sin(theta / 2) * gs.sin(v) + gs.cos(theta / 2) * gs.sin(2 * v)

        return gs.stack([x, y, z], axis=-1)

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

        z_theta_mu, z_theta_kappa, z_u_mu, z_u_kappa = posterior_params

        q_z_theta = VonMisesFisher(z_theta_mu, z_theta_kappa)

        q_z_u = VonMisesFisher(z_u_mu, z_u_kappa)

        z_theta = q_z_theta.rsample()

        z_u = q_z_u.rsample()

        return self._build_torus(z_theta, z_u)


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

        return self.fc_x_mu(h)


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

        x_mu = self.decode(z)

        return z, x_mu, posterior_params
