"""Default configuration for a run."""

import logging
from datetime import datetime
import torch
import os


os.environ["GEOMSTATS_BACKEND"] = "pytorch"
from geomstats.geometry.special_orthogonal import SpecialOrthogonal

run_name = "testing"

# Can be replaced by logging.DEBUG or logging.WARNING
logging.basicConfig(level=logging.INFO)

# Hardware
device = "cuda" if torch.cuda.is_available() else "cpu"


# Training
batch_size = 128
scheduler = False
log_interval = 20
checkpt_interval = 20
n_epochs = 150
learning_rate = 1e-3
sftbeta = 4.5
beta = 0.03
gamma = 10

# Dataset
dataset_name = "s1_synthetic"
(   
    expt_id,
    timestep_microsec,
    smooth,
    distortion_func,
    n_times,
    distortion_amp,
    radius,
    n_wiggles,
    embedding_dim,
    noise_var,
    synthetic_rotation,
) = [None for _ in range(11)]


if dataset_name == "experimental":
    expt_id = "34"  # hd: with head direction
    timestep_microsec = int(1e6)
    smooth = False
elif dataset_name == "s1_synthetic":
    distortion_func = "bump"
    n_times = 2000
    distortion_amp = 0.4
    radius = 1
    n_wiggles = 3
    embedding_dim = 2
    noise_var = 1e-3
    synthetic_rotation = SpecialOrthogonal(n=embedding_dim).random_point()
elif dataset_name == "s2_synthetic":
    # actual number of points is n_times*n_times
    n_times = 80
    radius = 1
    distortion_amp = 0.4
    embedding_dim = 3
    noise_var = 1e-4
    synthetic_rotation = SpecialOrthogonal(n=embedding_dim).random_point()


# Models
model_type = "neural_geom_vae"
encoder_width = 400
decoder_width = encoder_width
encoder_depth = 4
decoder_depth = encoder_depth
if dataset_name in ("s1_synthetic", "experimental"):
    latent_dim = 2
elif dataset_name == "s2_synthetic":
    latent_dim = 3
else:
    latent_dim = 2
posterior_type = "hyperspherical"
gen_likelihood_type = "gaussian"

# Results
now = str(datetime.now().replace(second=0, microsecond=0))
results_prefix = f"{dataset_name}_{now}"
trained_model_path = None
