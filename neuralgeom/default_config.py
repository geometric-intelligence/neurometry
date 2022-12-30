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
project = "neural_geom"
now = str(datetime.now().replace(second=0, microsecond=0))
trained_model_path = None

dataset_to_manifold_dim = {
    "experimental": 1,
    "s1_synthetic": 1,
    "s2_synthetic": 2,
    "t2_synthetic": 2,
}

dataset_to_latent_dim = {
    "experimental": 1,
    "s1_synthetic": 1,
    "s2_synthetic": 2,
    "t2_synthetic": 2,
}

dataset_to_posterior_type = {
    "experimental": "hyperspherical",
    "s1_synthetic": "hyperspherical",
    "s2_synthetic": "hyperspherical",
    "t2_synthetic": "toroidal",
}

### Choose main experiments parameters: ###

# Datasets
dataset_name = ["s1_synthetic", "s2_synthetic"]
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
select_first_gain = [True, False]

# Ignored if dataset_name == "experimental"
n_times = [10000]  # actual number of times is sqrt_ntimes ** 2
distortion_amp = [0.4, 0.8]
embedding_dim = [2, 3]
noise_var = [1e-3]

distortion_func = "bump"  # s1 only
n_wiggles = 3  # s1 only
radius = 1  # s1 and s2 only
major_radius = 2  # t2
minor_radius = 1  # t2
synthetic_rotation = "identity"  # or "random"
if synthetic_rotation not in ["identity", "random"]:
    raise ValueError(f"Rotation {synthetic_rotation} not recognized.")


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
n_epochs = 10  # 240  #
learning_rate = 1e-3
sftbeta = 4.5
beta = 0.03  # 0.03
gamma = 20  # 20
