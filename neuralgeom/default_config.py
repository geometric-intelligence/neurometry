"""Default configuration for a run."""

import logging
import os
from datetime import datetime

import torch

os.environ["GEOMSTATS_BACKEND"] = "pytorch"

# Can be replaced by logging.DEBUG or logging.WARNING
logging.basicConfig(level=logging.INFO)

# Hardware
device = "cuda" if torch.cuda.is_available() else "cpu"

# Results
project = "neuralgeom"
now = str(datetime.now().replace(second=0, microsecond=0))
trained_model_path = None

### Fixed experiment parameters (dicts) ###

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

### Variable experiment parameters (lists) ###

# Datasets
dataset_name = ["experimental", "s1_synthetic", "s2_synthetic", "t2_synthetic"]
for one_dataset_name in dataset_name:
    if one_dataset_name not in [
        "s1_synthetic",
        "s2_synthetic",
        "t2_synthetic",
        "experimental",
    ]:
        raise ValueError(f"Dataset name {one_dataset_name} not recognized.")

# Ignored if dataset_name != "experimental"
expt_id = ["41", "34"]  # hd: with head direction
timestep_microsec = [int(1e6)]
smooth = [True, False]
select_gain_1 = [True, False]

# Ignored if dataset_name == "experimental"
n_times = [10, 100]  # actual number of times is sqrt_ntimes ** 2
embedding_dim = [2, 3, 4]
distortion_amp = [0.4]
noise_var = [1e-3]


# Models
model_type = "neural_vae"
encoder_width = 600
decoder_width = encoder_width
encoder_depth = 4
decoder_depth = encoder_depth
gen_likelihood_type = "gaussian"

# Training
batch_size = 20
scheduler = False
log_interval = 20
checkpt_interval = 20
n_epochs = 2  # 240  #
learning_rate = 1e-3
sftbeta = 4.5
beta = 0.03  # 0.03  # hyperparameter weighting KL term
gamma = 20  # 20  # hyperparameter weighting latent loss term
