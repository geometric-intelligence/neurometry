"""Deep learning models."""

import torch
import torch.nn as nn
from torch.nn import functional as F


class VAE(nn.Module):
    """VAE with Linear (fully connected) layers."""

    def __init__(self, data_dim, latent_dim):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(data_dim, 400)
        self.fc21 = nn.Linear(400, latent_dim)
        self.fc22 = nn.Linear(400, latent_dim)
        self.fc3 = nn.Linear(latent_dim, 400)
        self.fc4 = nn.Linear(400, data_dim)

    def encode(self, x):
        """Encode input into mean and log-variance."""
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        """Sample from Gaussian of parameters mu, logvar."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """Decode latent variable z into data."""
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        """Encode and decode."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
