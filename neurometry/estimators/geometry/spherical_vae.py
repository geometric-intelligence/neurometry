"""G-manifold deep learning.

This file gathers deep learning models related to G-manifold learning.
"""

import torch
from torch.nn import functional as F

from neurometry.estimators.curvature.hyperspherical.distributions.von_mises_fisher import (
    VonMisesFisher,
)


class SphericalVAE(torch.nn.Module):
    """VAE with Linear (fully connected) layers.

    Parameters
    ----------
    num_neurons : int
        Number of neurons in the input data. This is the ambient dimension of the neural state space.
        Example: 40 for neural recordings of 40 neurons.
    latent_dim : int
        Dimension of the latent space.
        Example: 2.
    """

    def __init__(
        self,
        num_neurons,
        latent_dim,
        sftbeta,
        encoder_width=400,
        encoder_depth=2,
        decoder_width=400,
        decoder_depth=2,
        drop_out_p=0.0,
    ):
        super().__init__()
        self.num_neurons = num_neurons
        self.sftbeta = sftbeta
        self.latent_dim = latent_dim
        self.drop_out_p = drop_out_p

        decoder_width = encoder_width
        decoder_depth = encoder_depth

        self.encoder_fc = torch.nn.Linear(self.num_neurons, encoder_width)
        self.encoder_linears = torch.nn.ModuleList(
            [
                torch.nn.Linear(encoder_width, encoder_width)
                for _ in range(encoder_depth)
            ]
        )

        self.fc_z_mu = torch.nn.Linear(encoder_width, self.latent_dim)
        self.fc_z_logvar = torch.nn.Linear(encoder_width, 1)  # kappa

        self.decoder_fc = torch.nn.Linear(self.latent_dim, decoder_width)
        self.decoder_linears = torch.nn.ModuleList(
            [
                torch.nn.Linear(decoder_width, decoder_width)
                for _ in range(decoder_depth)
            ]
        )

        self.fc_x_mu = torch.nn.Linear(decoder_width, self.num_neurons)

        self.drop_out = torch.nn.Dropout(p=self.drop_out_p)

    def encode(self, x):
        """Encode input into mean and log-variance.

        The parameters mean (mu) and variance (computed
        from logvar) defines a multivariate Gaussian
        that represents the approximate posterior of the
        latent variable z given the input x.

        Parameters
        ----------
        x : array-like, shape=[batch_size, num_neurons]
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

        for _i_layer, layer in enumerate(self.encoder_linears):
            # h = self.drop_out(h)
            h = layer(h)
            h = F.softplus(h, beta=self.sftbeta)

        z_mu = self.fc_z_mu(h)
        z_kappa = F.softplus(self.fc_z_logvar(h)) + 1

        return z_mu, z_kappa

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

        z_mu, z_kappa = posterior_params
        q_z = VonMisesFisher(z_mu, z_kappa)

        return q_z.rsample()

    def decode(self, z):
        """Decode latent variable z into data.

        Parameters
        ----------
        z : array-like, shape=[batch_size, latent_dim]
            Input to the decoder.

        Returns
        -------
        _ : array-like, shape=[batch_size, num_neurons]
            Reconstructed data corresponding to z.
        """

        h = F.softplus(self.decoder_fc(z), beta=self.sftbeta)

        for _i_layer, layer in enumerate(self.decoder_linears):
            # h = self.drop_out(h)
            h = layer(h)
            h = F.softplus(h, beta=self.sftbeta)

        return self.fc_x_mu(h)

    def forward(self, x):
        """Run VAE: Encode, sample and decode.

        Parameters
        ----------
        x : array-like, shape=[batch_size, num_neurons]
            Input data.

        Returns
        -------
        _ : array-like, shape=[batch_size, num_neurons]
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
