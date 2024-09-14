import os

os.environ["GEOMSTATS_BACKEND"] = "pytorch"
import geomstats.backend as gs
import torch
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

import neurometry.estimators.geometry.models.train_config as train_config
from neurometry.estimators.geometry.models.spherical_vae import SphericalVAE
from neurometry.estimators.geometry.models.toroidal_vae import ToroidalVAE
from neurometry.estimators.geometry.models.train import train_test


class ImmersionEstimator(BaseEstimator):
    def __init__(self, extrinsic_dim, topology, device):
        self.estimate_ = None
        self.device = device
        self.extrinsic_dim = extrinsic_dim
        self.topology = topology
        self.latent_dims = {"circle": 2, "sphere": 3, "torus": 3}
        self.model = self._get_model()

    def _get_model(self):
        if self.topology == "circle" or self.topology == "sphere":
            return SphericalVAE(
                extrinsic_dim=self.extrinsic_dim,
                latent_dim=self.latent_dims[self.topology],
            ).to(self.device)
        if self.topology == "torus":
            return ToroidalVAE(
                extrinsic_dim=self.extrinsic_dim,
                latent_dim=self.latent_dims[self.topology],
            ).to(self.device)
        raise ValueError("Topology not supported")

    def _intrinsic_to_extrinsic(self):
        if self.topology == "circle":
            return lambda x: gs.array([gs.cos(x), gs.sin(x)])
        if self.topology == "sphere":
            return lambda x: gs.array(
                [
                    gs.sin(x[0]) * gs.cos(x[1]),
                    gs.sin(x[0]) * gs.sin(x[1]),
                    gs.cos(x[0]),
                ]
            )
        if self.topology == "torus":
            return lambda x: gs.array(
                [
                    (1 - gs.cos(x[0])) * gs.cos(x[1]),
                    (1 - gs.cos(x[0])) * gs.sin(x[1]),
                    gs.sin(x[0]),
                ]
            )
        raise ValueError("Topology not supported")

    def fit(self, X, y=None):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        X_train = torch.tensor(X_train).to(self.device)
        X_test = torch.tensor(X_test).to(self.device)
        y_train = torch.tensor(y_train).to(self.device)
        y_test = torch.tensor(y_test).to(self.device)
        train_loader = DataLoader(
            TensorDataset(X_train, y_train),
            batch_size=train_config.batch_size,
            shuffle=True,
        )
        test_loader = DataLoader(
            TensorDataset(X_test, y_test),
            batch_size=train_config.batch_size,
            shuffle=True,
        )

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=train_config.lr, amsgrad=True
        )
        train_losses, test_losses, best_model = train_test(
            model=self.model,
            train_loader=train_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            scheduler=None,
            config=train_config,
        )

        self.estimate_ = lambda task_variable: best_model.decode(
            self._intrinsic_to_extrinsic()(task_variable).T
        )

        return self
