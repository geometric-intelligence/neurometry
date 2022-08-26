import torch
import torch.nn.functional as F
from collections import OrderedDict

import sys

sys.path.append("../")

from hyperspherical_vae.distributions.von_mises_fisher import VonMisesFisher
from hyperspherical_vae.distributions.hyperspherical_uniform import (
    HypersphericalUniform,
)


class SphericalVAE(torch.nn.Module):
    def __init__(
        self,
        input_dim=256,
        encoder_dims=[64, 64, 32],
        latent_dim=2,
        distribution="vmf",
        weight_init=torch.nn.init.xavier_uniform_,
        device="cpu",
    ):

        """
        Spherical Variational Autoencoder (VAE), based off of:
        https://github.com/nicola-decao/s-vae-pytorch/blob/master/examples/mnist.py

        Parameters
        ----------
        input_dim (int):
            dimensionality of the input
        encoder_dims (lst):
            length specifying a dimensionality for each desired layer
        latent_dim (int):
            dimensionality of the latent space
        distribution (str):
            either "normal" for a gaussian latent space (ordinary VAE)
            or "vmf" for von Mises Fisher (Spherical VAE)
        weight_init (function):
            weight initializer function from PyTorch
        device (str):
            device to run the model on. may be a string, i.e. "cpu" or "cuda:1"
            or an integer specifying the gpu number.
        """

        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder_dims = encoder_dims
        self.decoder_dims = encoder_dims[::-1]
        self.distribution = distribution
        self.weight_init = weight_init
        self.device = device
        self.build()

    def build(self):
        self.generate_encoder()
        self.generate_decoder()

    def generate_module(self, layer_dims, out_fn=None):
        layers = OrderedDict()

        for i in range(len(layer_dims) - 1):
            layers["fullyconnected_{}".format(i)] = torch.nn.Linear(
                layer_dims[i], layer_dims[i + 1], bias=True
            )
            self.weight_init(layers["fullyconnected_{}".format(i)].weight)
            torch.nn.init.zeros_(layers["fullyconnected_{}".format(i)].bias)
            if i < len(layer_dims) - 2:
                layers["activation_{}".format(i)] = torch.nn.ReLU()
            if i == len(layer_dims) - 2 and out_fn is not None:
                layers["out_{}".format(i)] = out_fn
            else:
                continue

        module = torch.nn.Sequential(layers)
        return module

    def generate_encoder(self):
        encoder_dims = [self.input_dim] + self.encoder_dims
        self.encoder = self.generate_module(encoder_dims, out_fn=torch.nn.ReLU())
<<<<<<< HEAD
        
        # TODO: Initialize weights and biases
        if self.distribution == 'normal':
            # compute mean and std of the normal distribution
            self.mu = torch.nn.Linear(self.encoder_dims[-1], self.latent_dim, bias=True)
            self.var = torch.nn.Linear(self.encoder_dims[-1], self.latent_dim, bias=True)
        elif self.distribution == 'vmf':
=======
        if self.distribution == "normal":
            # compute mean and std of the normal distribution
            self.mu = torch.nn.Linear(self.encoder_dims[-1], self.latent_dim)
            self.var = torch.nn.Linear(self.encoder_dims[-1], self.latent_dim)
        elif self.distribution == "vmf":
>>>>>>> 4c6a3db3124bc689cd05dde83e8f3f51d549b3ea
            # compute mean and concentration of the von Mises-Fisher
            self.mu = torch.nn.Linear(self.encoder_dims[-1], self.latent_dim, bias=True)
            self.var = torch.nn.Linear(self.encoder_dims[-1], 1, bias=True)
        else:
            raise NotImplemented
<<<<<<< HEAD
        self.weight_init(self.mu.weight)
        self.weight_init(self.var.weight)
        torch.nn.init.zeros_(self.mu.bias)
        torch.nn.init.zeros_(self.var.bias)
        
=======

>>>>>>> 4c6a3db3124bc689cd05dde83e8f3f51d549b3ea
    def generate_decoder(self):
        decoder_dims = [self.latent_dim] + self.decoder_dims + [self.input_dim]
        self.decoder = self.generate_module(decoder_dims, out_fn=None)

    def encode(self, x):
        latent_rep = self.encoder(x)
        if self.distribution == "normal":
            # compute mean and std of the normal distribution
            z_mean = self.mu(latent_rep)
            z_var = F.softplus(self.var(latent_rep))
        elif self.distribution == "vmf":
            # compute mean and concentration of the von Mises-Fisher
            z_mean = self.mu(latent_rep)
            z_mean = z_mean / z_mean.norm(dim=-1, keepdim=True)
            # the `+ 1` prevent collapsing behaviors
            z_var = F.softplus(self.var(latent_rep)) + 1
        else:
            raise NotImplemented
        return z_mean, z_var

    def decode(self, z):
        return self.decoder(z)

    def reparameterize(self, z_mean, z_var):
        if self.distribution == "normal":
            q_z = torch.distributions.normal.Normal(z_mean, z_var)
            p_z = torch.distributions.normal.Normal(
                torch.zeros_like(z_mean), torch.ones_like(z_var)
            )
        elif self.distribution == "vmf":
            q_z = VonMisesFisher(z_mean, z_var)
            p_z = HypersphericalUniform(self.latent_dim - 1)
        else:
            raise NotImplemented

        return q_z, p_z

    def sample(self, n_samples):
        raise NotImplemented

    def forward(self, x):
        (z_mean, z_var), (q_z, p_z), z = self.to_latent(x)
        x_ = self.decode(z)
        return (z_mean, z_var), (q_z, p_z), z, x_

    def to_latent(self, x):
        z_mean, z_var = self.encode(x)
        q_z, p_z = self.reparameterize(z_mean, z_var)
        z = q_z.rsample()
        return (z_mean, z_var), (q_z, p_z), z

    def log_likelihood(self, x, n=10):
        """
        :param model: model object
        :param optimizer: optimizer object
        :param n: number of MC samples
        :return: MC estimate of log-likelihood
        """

        z_mean, z_var = self.encode(x.reshape(-1, self.input_dim))
        q_z, p_z = self.reparameterize(z_mean, z_var)
        z = q_z.rsample(torch.Size([n]))
        x_mb_ = self.decode(z)

        log_p_z = p_z.log_prob(z)

        if self.distribution == "normal":
            log_p_z = log_p_z.sum(-1)

        log_p_x_z = -torch.nn.BCEWithLogitsLoss(reduction="none")(
            x_mb_, x.reshape(-1, self.input_dim).repeat((n, 1, 1))
        ).sum(-1)

        log_q_z_x = q_z.log_prob(z)

        if self.distribution == "normal":
            log_q_z_x = log_q_z_x.sum(-1)

        return (
            (log_p_x_z + log_p_z - log_q_z_x).t().logsumexp(-1)
            - torch.log(torch.tensor(n))
        ).mean()
