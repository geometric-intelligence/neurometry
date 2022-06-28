"""Default configuration for a run."""

import logging
from datetime import datetime
from platform import architecture

import torch

# Can be replaced by logging.DEBUG or logging.WARNING
logging.basicConfig(level=logging.INFO)

# Hardware
device = "cuda" if torch.cuda.is_available() else "cpu"

# Training
batch_size = 128 #128
log_interval = 10
checkpt_interval = 10
n_epochs = 500
learning_rate = 1e-3
beta = 1

# Dataset
dataset_name = "points"
if dataset_name == "experimental":
    expt_id = "15_hd"  # hd: with head direction
    timestep_ns = 1000000

if dataset_name in ["images", "projected_images"]:
    img_size = 64

# Models
model_type = "fc_vae"
latent_dim = 2
posterior_type = "Gaussian"
gen_likelihood_type = "Gaussian"
with_regressor = False
if with_regressor:
    weight_regressor = 1.0
    h_dim_regressor = 20

# Results
now = str(datetime.now().replace(second=0, microsecond=0))
results_prefix = f"results/{dataset_name}_{now}"

