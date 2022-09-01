import torch
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import os

os.environ["GEOMSTATS_BACKEND"] = "pytorch"
import geomstats.backend as gs
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal
from geomstats.geometry.special_orthogonal import SpecialOrthogonal


class Wiggles(Dataset):
    def __init__(
        self,
        n_times=1500,
        n_wiggles=6,
        synth_radius=1,
        amp_wiggles=0.4,
        embedding_dim=10,
        noise_var=0.01,
        rotation=True,
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
        self.rotation = rotation
        self.noise_var = noise_var

        self.data, self.labels = self.generate_wiggles()


    def generate_wiggles(self):
        
        if self.rotation == True:
            self.rot = SpecialOrthogonal(n=self.embedding_dim).random_point()
        else:
            self.rot = torch.eye(self.embedding_dim)
        
        def polar(angle):
            return torch.stack([torch.cos(angle), torch.sin(angle)], axis=0)
            

        def synth_immersion(angles):
            amplitudes = self.synth_radius * (
                1 + self.amp_wiggles * torch.cos(self.n_wiggles * angles)
            )
            wiggly_circle = torch.einsum(
                "ik,jk->ij", polar(angles), torch.diag(amplitudes)
            )
            wiggly_circle = wiggly_circle

            padded_wiggly_circle = F.pad(
                input=wiggly_circle,
                pad=(0, 0, 0, self.embedding_dim - 2),
                mode="constant",
                value=0.0,
            )
            
            points = torch.einsum("ik,kj->ij", self.rot, padded_wiggly_circle)

            return points

        angles = torch.linspace(0, 2 * np.pi, self.n_times)

        data = synth_immersion(angles).T

        noise_dist = MultivariateNormal(
            loc=torch.zeros(self.embedding_dim),
            covariance_matrix=self.noise_var * torch.eye(self.embedding_dim),
        )

        noisy_data = data + noise_dist.sample((self.n_times,))

        return noisy_data, angles

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def __len__(self):
        return len(self.data)
