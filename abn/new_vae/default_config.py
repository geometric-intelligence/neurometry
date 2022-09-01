import logging

from datetime import datetime


import torch


run_name = "hello"

# Can be replaced by logging.DEBUG or logging.WARNING
logging.basicConfig(level=logging.INFO)

# Hardware
device = "cuda" if torch.cuda.is_available() else "cpu"

# Training
batch_size = 128  # 128
log_interval = 20
checkpt_interval = 2
n_epochs = 5
learning_rate = 1e-3
beta = 0.2


# Dataset
dataset_name = "wiggles"

if dataset_name == "experimental":
    expt_id = "34"  # hd: with head direction
    timestep_microsec = 1000000
else:
    timestep_microsec = -1
    expt_id = -1

if dataset_name in ["images", "projected_images"]:
    img_size = 64
else:
    img_size = -1


if dataset_name == "wiggles":
    n_times = 1000
    amp_wiggles = 0.3
    synth_radius = 1
    n_wiggles = 5
    embedding_dim = 5
    noise_var = 0.01

else:
    n_times = -1
    amp_wiggles = -1
    synth_radius = -1
    n_wiggles = -1
    embedding_dim = -1
    noise_var = -1


# Models

encoder_dims = [40, 30, 20, 10]
latent_dim = 2
latent_geometry = "hyperspherical"
decoder_dims = encoder_dims[::-1]


# Results
now = str(datetime.now().replace(second=0, microsecond=0))
results_prefix = f"{dataset_name}_{now}"
trained_model_path = None
