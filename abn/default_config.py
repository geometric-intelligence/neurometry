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

# Dataset
dataset = "projections"

# Models
with_regressor = False
weight_regressor = 1.0
latent_dim = 2

# Results
now = str(datetime.now().replace(second=0, microsecond=0))
results_prefix = f"results/{dataset}_{now}"
