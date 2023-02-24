"""Default configuration for launching experiments."""

import logging
import os
from datetime import datetime

import torch
import numpy as np
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
curvature_profiles_dir = os.path.join(os.getcwd(), "results/curvature_profiles/")
if not os.path.exists(curvature_profiles_dir):
    os.makedirs(curvature_profiles_dir)


# Hardware
device = "cuda" if torch.cuda.is_available() else "cpu"

# Can be replaced by logging.DEBUG or logging.WARNING
logging.basicConfig(level=logging.INFO)

# Results
project = "gridcells"
trained_model_path = None

### Fixed experiment parameters ###
### ---> Dicts giving values that are not changed in experiments

manifold_dim = {
    "experimental": 1,
    "s1_synthetic": 1,
    "s2_synthetic": 2,
    "t2_synthetic": 2,
    "grid_cells": 2,
}

latent_dim = {
    "experimental": 2,
    "s1_synthetic": 2,
    "s2_synthetic": 3,
    "t2_synthetic": 3,
    "grid_cells": 3,
}

posterior_type = {
    "experimental": "hyperspherical",
    "s1_synthetic": "hyperspherical",
    "s2_synthetic": "hyperspherical",
    "t2_synthetic": "toroidal",
    "grid_cells": "toroidal",
}

geodesic_distortion_func = {
    "experimental": None,
    "s1_synthetic": "bump",
    "s2_synthetic": None,
    "t2_synthetic": None,
    "grid_cells": None,
}

n_wiggles = {
    "experimental": None,
    "s1_synthetic": 3,
    "s2_synthetic": None,
    "t2_synthetic": None,
    "grid_cells": None,
}

radius = {
    "experimental": None,
    "s1_synthetic": 1,
    "s2_synthetic": 1,
    "t2_synthetic": None,
    "grid_cells": None,
}

major_radius = {
    "experimental": None,
    "s1_synthetic": None,
    "s2_synthetic": None,
    "t2_synthetic": 2,
    "grid_cells": 1,
}

minor_radius = {
    "experimental": None,
    "s1_synthetic": None,
    "s2_synthetic": None,
    "t2_synthetic": 1,
    "grid_cells": 1,
}

synthetic_rotation = {
    "experimental": None,
    "s1_synthetic": "identity",  # or "random"
    "s2_synthetic": "identity",  # or "random"
    "t2_synthetic": "identity",  # or "random"
    "grid_cells": None,
}

### Variable experiment parameters ###
### ---> Lists of values to try for each parameter

# Datasets
dataset_name = ["grid_cells"]
for one_dataset_name in dataset_name:
    if one_dataset_name not in [
        "s1_synthetic",
        "s2_synthetic",
        "t2_synthetic",
        "experimental",
        "grid_cells",
    ]:
        raise ValueError(f"Dataset name {one_dataset_name} not recognized.")

# Only used if dataset_name == "experimental"
expt_id = ["41"]  # , "34"]  # hd: with head direction
timestep_microsec = [int(1e6)]  # , int(1e6)]  # , int(1e5)]
smooth = [True]  # , False]
# Note: if there is only one gain (gain 1), it will be selected
# even if select gain 1 is false
select_gain_1 = [True]  # , False]  # , False]


# Only used of dataset_name in ["s1_synthetic", "s2_synthetic", "t2_synthetic"]
n_times = [2000]  # , 2000]  # actual number of times is sqrt_ntimes ** 2
embedding_dim = [3]  # , 5, 8, 10, 20, 50]
geodesic_distortion_amp = [0.4]
noise_var = [1e-3]  # , 1e-2, 1e-1]

# Only used if dataset_name == "grid_cells"
grid_scale = [1.0]
arena_dims = [np.array([8, 8])]
n_cells = [3]
grid_orientation_mean = [0.0]
grid_orientation_std = [0.0]
field_width = [0.05]
resolution = [10]

# Models
gen_likelihood_type = "gaussian"

# Training
batch_shuffle = (
    True  # do not shuffle train/test set when moving forward or dynamic loss are used
)
scheduler = False
log_interval = 4
checkpt_interval = 20
n_epochs = 10 # 50  # 200  # 150  # 240
sftbeta = 4.5  # beta parameter for softplus
alpha = 1.0  # weight for the reconstruction loss
beta = 0.001  # 0.03  # weight for KL loss
gamma = 10 # 20  # weight for latent regularization loss
gamma_moving = 0  # weight for moving forward loss, put 0. if unused
gamma_dynamic = 0  # weight for dynamic loss - TODO

### Ray sweep hyperparameters ###
# --> Lists of values to sweep for each hyperparameter
# Except for lr_min and lr_max which are floats
lr_min = 0.000001
lr_max = 0.1
batch_size = [16]#[16,32,64]
encoder_width = [40]#[100,400]  # , 100, 200, 300]
encoder_depth = [2]#[4,6,8]  # , 10, 20, 50, 100]
decoder_width = [40]#[100,400]  # , 100, 200, 300]
decoder_depth = [2]#[4,6,8]  # , 10, 20, 50, 100]
drop_out_p = [0]#[0,0.1,0.2]  # put probability p at 0. for no drop out
for p in drop_out_p:
    assert p >= 0.0 and p <= 1, "Probability needs to be in [0, 1]"

# Number of times to sample from the
# hyperparameter space. Defaults to 1. If `grid_search` is
# provided as an argument, the grid will be repeated
# `num_samples` of times. If this is -1, (virtually) infinite
# samples are generated until a stopping condition is met.
# Given that 8/10 gpus can run at the same time,
# We choose a multiple of 8.
num_samples = 1 #128
sweep_metric = "test_loss"
n_grid_points = 2000  # number of points on the z grid for curvature
# Doc on tune.run:
# https://docs.ray.io/en/latest/_modules/ray/tune/tune.html
