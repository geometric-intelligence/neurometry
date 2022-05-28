"""Default configuration for a run."""

import logging
from datetime import datetime

import torch

# Can be replaced by logging.DEBUG or logging.WARNING
logging.basicConfig(level=logging.INFO)

# Hardware
device = "cuda" if torch.cuda.is_available() else "cpu"

# Training
batch_size = 128
log_interval = 10
checkpt_interval = 10
n_epochs = 100
learning_rate = 1e-3

# Dataset
dataset_name = "projected_images"
if dataset_name == "experimental":
    expt_id = "16_hd"  # hd: with head direction

if dataset_name in ["images", "projected_images"]:
    img_size = 64

# Models
latent_dim = 2
with_regressor = False
if with_regressor:
    weight_regressor = 1.0
    h_dim_regressorr = 20

# Results
now = str(datetime.now().replace(second=0, microsecond=0))
results_prefix = f"results/{dataset_name}_{now}"
