"""G-manifold deep learning.

This file gathers deep learning models related to G-manifold learning.
"""

import torch
from torch.nn import functional as F

# TODO: Investigate this:
# proceedings.neurips.cc//paper/2020/file/510f2318f324cf07fce24c3a4b89c771-Paper.pdf


class LieVAE(torch.nn.Module):
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

    def __init__(self, data_dim, group_dim=2, latent_dim=2):
        super(LieVAE, self).__init__()
        self.latent_dim = latent_dim

        self.fc1 = torch.nn.Linear(data_dim, 400)
        self.fc21 = torch.nn.Linear(400, group_dim)
        self.fc22 = torch.nn.Linear(400, group_dim)
        self.fc3 = torch.nn.Linear(latent_dim, 400)
        self.fc4 = torch.nn.Linear(400, data_dim)

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
        mu = self.fc21(h1)

        mu = mu / torch.linalg.norm(mu, axis=1).reshape((-1, 1))
        logvar = self.fc22(h1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """Sample from Gaussian of parameters mu, logvar.

        Parameters
        ----------
        mu : array-like, shape=[batch_size, latent_dim]
            Mean of multivariate Gaussian in latent space.
        logvar : array-like, shape=[batch_size, latent_dim]
            Vector representing the diagonal covariance of the
            multivariate Gaussian in latent space.

        Returns
        -------
        _ : array-like, shape=[batch_size, latent_dim]
            Sample of the multivariate Gaussian of parameters
            mu and logvar.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z_sample = mu + eps * std
        return z_sample / torch.linalg.norm(z_sample, axis=1).reshape((-1, 1))

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
        return torch.sigmoid(self.fc4(h3))

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
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
