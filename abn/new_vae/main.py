"""Main script."""
import os

os.environ["GEOMSTATS_BACKEND"] = "pytorch"

import datasets.utils
import default_config
import matplotlib.pyplot as plt
import newvae
import torch
import train
import wandb
from datasets.wiggles import Wiggles
from datasets.data_loader import TrainValLoader
import pandas as pd


wandb.init(
    project="neural_geometry",
    config={
        "run_name": default_config.run_name,
        "device": default_config.device,
        "batch_size": default_config.batch_size,
        "log_interval": default_config.log_interval,
        "checkpt_interval": default_config.checkpt_interval,
        "n_epochs": default_config.n_epochs,
        "learning_rate": default_config.learning_rate,
        "beta": default_config.beta,
        "dataset_name": default_config.dataset_name,
        "expt_id": default_config.expt_id,
        "timestep_microsec": default_config.timestep_microsec,
        "img_size": default_config.img_size,
        "n_times": default_config.n_times,
        "amp_wiggles": default_config.amp_wiggles,
        "synth_radius": default_config.synth_radius,
        "n_wiggles": default_config.n_wiggles,
        "embedding_dim ": default_config.embedding_dim,
        "noise_var": default_config.noise_var,
        "encoder_dims": default_config.encoder_dims,
        "latent_dim": default_config.latent_dim,
        "latent_geometry": default_config.latent_geometry,
        "decoder_dims": default_config.decoder_dims,
        "results_prefix": default_config.results_prefix,
    },
)

config = wandb.config

wandb.run.name = config.run_name
results_prefix = config.results_prefix

dataset = Wiggles(
    n_times=1500,
    n_wiggles=6,
    synth_radius=1,
    amp_wiggles=0.2,
    embedding_dim=2,
    noise_var=0.001,
)

"""
Min 0 Max 1
"""
dataset.data = dataset.data - dataset.data.min()
dataset.data = dataset.data / dataset.data.max()


data = dataset.data
labels = dataset.labels

labels = pd.DataFrame(
    {
        "angles": labels,
    }
)

data_loader = TrainValLoader(batch_size=config.batch_size)

data_loader.load(dataset)

train_loader = data_loader.train

test_loader = data_loader.val


# dataset_torch, labels, train_loader, test_loader = datasets.utils.load(default_config)

# _, data_dim = dataset_torch.shape

_, data_dim = data.shape


model = newvae.VAE(
    input_dim=data_dim,
    encoder_dims=config.encoder_dims,
    latent_dim=config.latent_dim,
    latent_geometry=config.latent_geometry,
    decoder_dims=config.decoder_dims,
).to(config.device)


optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)


losses = train.train_model(
    model, data, labels, train_loader, test_loader, optimizer, config
)


train_losses, test_losses = losses


for data, labels in test_loader:
    data, labels = data.to(config.device), labels.to(config.device)

# torch.onnx.export(
#     model, data, f"results/trained_models/{config.results_prefix}_model.onnx"
# )


plt.figure()
plt.plot(train_losses, label="train")
plt.plot(test_losses, label="test")
plt.legend()
plt.savefig(f"results/figures/{config.results_prefix}_losses.png")
plt.close()

torch.save(model, f"results/trained_models/{config.results_prefix}_model.pt")


wandb.finish()
