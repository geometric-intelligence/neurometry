"""Default configuration for launching experiments."""

import logging
import os
from datetime import datetime

import torch
from ray.tune.search.hyperopt import HyperOptSearch

os.environ["GEOMSTATS_BACKEND"] = "pytorch"

# WANDB API KEY
# Find it here: https://wandb.ai/authorize
# Story it in file: api_key.txt (without extra line break)
with open("api_key.txt") as f:
    api_key = f.read()

# Directories
work_dir = os.getcwd()
configs_dir = os.path.join(work_dir, "results/configs")
if not os.path.exists(configs_dir):
    os.makedirs(configs_dir)
trained_models_dir = os.path.join(work_dir, "results/trained_models")
if not os.path.exists(trained_models_dir):
    os.makedirs(trained_models_dir)
ray_sweep_dir = os.path.join(work_dir, "results/ray_sweep")

# Hardware
device = "cuda" if torch.cuda.is_available() else "cpu"

# Can be replaced by logging.DEBUG or logging.WARNING
logging.basicConfig(level=logging.INFO)

# Results
project = "neuralgeom"
trained_model_path = None

### Fixed experiment parameters ###
### ---> Dicts giving values that are not changed in experiments

manifold_dim = {
    "experimental": 1,
    "s1_synthetic": 1,
    "s2_synthetic": 2,
    "t2_synthetic": 2,
}

latent_dim = {
    "experimental": 2,
    "s1_synthetic": 2,
    "s2_synthetic": 3,
    "t2_synthetic": 2,
}

posterior_type = {
    "experimental": "hyperspherical",
    "s1_synthetic": "hyperspherical",
    "s2_synthetic": "hyperspherical",
    "t2_synthetic": "toroidal",
}

distortion_func = {
    "experimental": None,
    "s1_synthetic": "bump",
    "s2_synthetic": None,
    "t2_synthetic": None,
}

n_wiggles = {
    "experimental": None,
    "s1_synthetic": 3,
    "s2_synthetic": None,
    "t2_synthetic": None,
}

radius = {
    "experimental": None,
    "s1_synthetic": 1,
    "s2_synthetic": 1,
    "t2_synthetic": None,
}

major_radius = {
    "experimental": None,
    "s1_synthetic": None,
    "s2_synthetic": None,
    "t2_synthetic": 2,
}

minor_radius = {
    "experimental": None,
    "s1_synthetic": None,
    "s2_synthetic": None,
    "t2_synthetic": 1,
}

synthetic_rotation = {
    "experimental": None,
    "s1_synthetic": "identity",  # or "random"
    "s2_synthetic": "identity",  # or "random"
    "t2_synthetic": "identity",  # or "random"
}

### Variable experiment parameters ###
### ---> Lists of values to try for each parameter

# Datasets
dataset_name = ["experimental"]
for one_dataset_name in dataset_name:
    if one_dataset_name not in [
        "s1_synthetic",
        "s2_synthetic",
        "t2_synthetic",
        "experimental",
    ]:
        raise ValueError(f"Dataset name {one_dataset_name} not recognized.")

# Ignored if dataset_name != "experimental"
expt_id = ["41"]  # hd: with head direction
timestep_microsec = [int(1e6), int(1e5)]
smooth = [True]
# Note: if there is only one gain (gain 1), it will be selected
# even if select gain 1 is false
select_gain_1 = [True, False]

# Ignored if dataset_name == "experimental"
n_times = [1000]  # actual number of times is sqrt_ntimes ** 2
embedding_dim = [2, 3]
distortion_amp = [0.4]
noise_var = [1e-3]

# Models
model_type = "neural_vae"
gen_likelihood_type = "gaussian"

# Training
scheduler = False
log_interval = 20
checkpt_interval = 20
n_epochs = 150  # 240
sftbeta = 4.5
beta = 0.03  # 0.03  # weight for KL term
gamma = 30  # 20  # weight for latent loss term

### Ray sweep hyperparameters ###
# --> Lists of values to sweep for each hyperparameter
# Except for lr_min and lr_max which are floats
lr_min = 0.00001
lr_max = 0.1
batch_size = [8, 16, 32]
encoder_width = [100]
encoder_depth = [4]
decoder_width = [50, 100, 200, 300]
decoder_depth = [5, 10, 20, 50, 100]

# Number of times to sample from the
# hyperparameter space. Defaults to 1. If `grid_search` is
# provided as an argument, the grid will be repeated
# `num_samples` of times. If this is -1, (virtually) infinite
# samples are generated until a stopping condition is met.
# Given that 8/10 gpus can run at the same time,
# We choose a multiple of 8.
num_samples = 256
sweep_metric = "test_loss"
# Doc on tune.run:
# https://docs.ray.io/en/latest/_modules/ray/tune/tune.html
