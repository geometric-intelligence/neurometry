import torch
import torch.nn.functional as F
from torch.distributions.normal import Normal as Normal
from hyperspherical_vae.distributions import VonMisesFisher
from hyperspherical_vae.distributions import HypersphericalUniform


class VAE(torch.nn.Module):
    """New VAE implementation.

    Parameters
    ----------


    """

    def __init__(
        self,
        input_dim,
        encoder_dims,
        latent_dim,
        latent_geometry,
        decoder_dims,
        weight_init=torch.nn.init.xavier_uniform_,
        bias_init=torch.nn.init.zeros_,
    ):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.encoder_dims = encoder_dims
        self.latent_dim = latent_dim
        self.latent_geometry = latent_geometry
        self.decoder_dims = decoder_dims
        self.weight_init = weight_init
        self.bias_init = bias_init

        self.encoder_layers, self.distribution_param_layers = self.generate_encoder(
            input_dim=self.input_dim,
            encoder_dims=self.encoder_dims,
            latent_dim=self.latent_dim,
        )
        self.decoder_layers = self.generate_decoder(
            latent_dim=self.latent_dim,
            decoder_dims=self.decoder_dims,
            input_dim=self.input_dim,
        )

    def generate_encoder(self, input_dim, encoder_dims, latent_dim):

        encoder_layers = torch.nn.ModuleList(
            [
                torch.nn.Linear(encoder_dims[i], encoder_dims[i + 1], bias=True)
                for i in range(len(encoder_dims) - 1)
            ]
        )
        encoder_layers.insert(0, torch.nn.Linear(input_dim, encoder_dims[0], bias=True))

        distribution_param_layers = torch.nn.ModuleDict()

        if self.latent_geometry == "normal":
            distribution_param_layers["mu"] = torch.nn.Linear(
                self.encoder_dims[-1], latent_dim, bias=True
            )
            distribution_param_layers["var"] = torch.nn.Linear(
                self.encoder_dims[-1], latent_dim, bias=True
            )
        elif self.latent_geometry == "hyperspherical":
            distribution_param_layers["mu"] = torch.nn.Linear(
                self.encoder_dims[-1], latent_dim, bias=True
            )
            distribution_param_layers["kappa"] = torch.nn.Linear(
                self.encoder_dims[-1], 1, bias=True
            )
        else:
            raise NotImplementedError

        for layer in encoder_layers:
            self.weight_init(layer.weight)
            self.bias_init(layer.bias)

        for param_layer in distribution_param_layers.values():
            self.weight_init(param_layer.weight)
            self.bias_init(param_layer.bias)

        return encoder_layers, distribution_param_layers

    def generate_decoder(self, latent_dim, decoder_dims, input_dim):

        decoder_layers = torch.nn.ModuleList(
            [
                torch.nn.Linear(decoder_dims[i], decoder_dims[i + 1], bias=True)
                for i in range(len(decoder_dims) - 1)
            ]
        )
        decoder_layers.insert(
            0, torch.nn.Linear(latent_dim, decoder_dims[0], bias=True)
        )
        decoder_layers.insert(
            len(decoder_layers), torch.nn.Linear(decoder_dims[-1], input_dim, bias=True)
        )

        for layer in decoder_layers:
            self.weight_init(layer.weight)
            self.bias_init(layer.bias)

        return decoder_layers

    def encode(self, x):

        posterior_params = {}
        h = x
        for layer in self.encoder_layers:
            h = F.relu(layer(h))

        if self.latent_geometry == "normal":
            z_mu = self.distribution_param_layers["mu"](h)
            z_var = F.softplus(self.distribution_param_layers["var"](h))
            posterior_params["mu"] = z_mu
            posterior_params["var"] = z_var
        elif self.latent_geometry == "hyperspherical":
            z_mu = self.distribution_param_layers["mu"](h)
            z_mu = z_mu / z_mu.norm(dim=-1, keepdim=True)
            # the `+ 1` prevent collapsing behaviors (check?)
            z_kappa = F.softplus(self.distribution_param_layers["kappa"](h)) + 1
            posterior_params["mu"] = z_mu
            posterior_params["kappa"] = z_kappa
        else:
            raise NotImplementedError

        return posterior_params

    def decode(self, z):

        h = z
        for layer in self.decoder_layers:
            h = F.relu(layer(h))

        x_rec = h

        return x_rec

    def reparameterize(self, posterior_params):
        if self.latent_geometry == "normal":
            q_z = Normal(posterior_params["mu"], posterior_params["var"])
            p_z = Normal(
                torch.zeros_like(
                    posterior_params["mu"], torch.ones_like(posterior_params["mu"])
                )
            )
        elif self.latent_geometry == "hyperspherical":
            q_z = VonMisesFisher(posterior_params["mu"], posterior_params["kappa"])
            p_z = HypersphericalUniform(self.latent_dim - 1)
        else:
            raise NotImplementedError

        return q_z, p_z

    def forward(self, x):
        posterior_params = self.encode(x)
        q_z, p_z = self.reparameterize(posterior_params)
        z = q_z.rsample()

        x_rec = self.decode(z)

        return posterior_params, (q_z, p_z), z, x_rec
