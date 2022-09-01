import torch
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import os

os.environ["GEOMSTATS_BACKEND"] = "pytorch"
import geomstats.backend as gs
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal


class Wiggles(Dataset):
    def __init__(
        self,
        n_times=1500,
        n_wiggles=6,
        synth_radius=1,
        amp_wiggles=0.4,
        embedding_dim=10,
        noise_var=0.01,
    ):

        """
        Create "wiggly" circles with noise.

        Parameters
        ----------
        n_times : int

        circle_radius : float
            Primary circle radius.
        n_wiggles : int
            Number of "wiggles".
        amp_wiggles : float, < 1
            Amplitude of "wiggles".
        embedding_dim : int
            Dimension of embedding dimension.
        noise_var : float
            Variance (sigma2) of the Gaussian noise.

        Returns
        -------
        noisy_data : array-like, shape=[embedding_dim, n_times]
            Number of firings per time step and per cell.
        labels : pd.DataFrame, shape=[n_times, 1]
            Labels organized in 1 column: angles.
        """

        super().__init__()
        self.n_times = n_times
        self.synth_radius = synth_radius
        self.n_wiggles = n_wiggles
        self.amp_wiggles = amp_wiggles
        self.embedding_dim = embedding_dim
        self.noise_var = noise_var

        self.data, self.labels = self.generate_wiggles()

    def generate_wiggles(self):
        def polar(angle):
            return torch.tensor(
                torch.stack([torch.cos(angle), torch.sin(angle)], axis=0)
            )

        def synth_immersion(angles):
            amplitudes = self.synth_radius * (
                1 + self.amp_wiggles * torch.cos(self.n_wiggles * angles)
            )
            wiggly_circle = torch.einsum(
                "ik,jk->ij", polar(angles), torch.diag(torch.tensor(amplitudes))
            )
            wiggly_circle = torch.tensor(wiggly_circle)

            padded_wiggly_circle = F.pad(
                input=wiggly_circle,
                pad=(0, 0, 0, self.embedding_dim - 2),
                mode="constant",
                value=0.0,
            )

            so = SpecialOrthogonal(n=self.embedding_dim)

            rot = so.random_point()

            return torch.einsum("ik,kj->ij", rot, padded_wiggly_circle)

        labels = torch.linspace(0, 2 * np.pi, self.n_times)

        data = synth_immersion(labels).T

        noise_dist = MultivariateNormal(
            loc=torch.zeros(self.embedding_dim),
            covariance_matrix=self.noise_var * torch.eye(self.embedding_dim),
        )

        noisy_data = data + noise_dist.sample((self.n_times,))

        return noisy_data, labels

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def __len__(self):
        return len(self.data)
