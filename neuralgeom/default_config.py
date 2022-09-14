"""Default configuration for a run."""

import logging
from datetime import datetime
from platform import architecture

import torch
from geomstats.geometry.special_orthogonal import SpecialOrthogonal

run_name = "trial run"

# Can be replaced by logging.DEBUG or logging.WARNING
logging.basicConfig(level=logging.INFO)

# Hardware
device = "cuda" if torch.cuda.is_available() else "cpu"

# # Training
# batch_size = 128  # 128
# log_interval = 20
# checkpt_interval = 20
# n_epochs = 120
# learning_rate = 1e-3
# beta = 0.1*radius
# alpha = 1
# gamma = 0.0


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


if dataset_name == "wiggles":
    amp_func = "wiggles"
    n_times = 1000
    amp_wiggles = 0.2
    radius = 10
    n_wiggles = 3
    embedding_dim = 2
    noise_var = 0.001 * radius
    rot = SpecialOrthogonal(n=embedding_dim).random_point()
else:
    n_times = -1
    amp_wiggles = -1
    radius = -1
    n_wiggles = -1
    embedding_dim = -1
    noise_var = -1
    rot = -1
    amp_func = ""


# Training
batch_size = 128  # 128
log_interval = 20
checkpt_interval = 20
n_epochs = 120
learning_rate = 1e-3
beta = 0.1 * radius
alpha = 1
gamma = 0.0


# Models
model_type = "fc_vae"
encoder_width = 400
#decoder_width = 40
decoder_width = encoder_width
encoder_depth = 4
#decoder_depth = 4
decoder_depth = encoder_depth
latent_dim = 2
posterior_type = "hyperspherical"
gen_likelihood_type = "gaussian"
with_regressor = False
if with_regressor:
    weight_regressor = 1.0
    h_dim_regressor = 20

# Results
now = str(datetime.now().replace(second=0, microsecond=0))
results_prefix = f"{dataset_name}_{now}"
trained_model_path = None
