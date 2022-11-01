"""Default configuration for a run."""

import logging
import os
from datetime import datetime

import torch

os.environ["GEOMSTATS_BACKEND"] = "pytorch"
from geomstats.geometry.special_orthogonal import SpecialOrthogonal  # NOQA

run_name = "nina-test-on-harold"  # "testing"

# Can be replaced by logging.DEBUG or logging.WARNING
logging.basicConfig(level=logging.INFO)

# Hardware
device = "cuda" if torch.cuda.is_available() else "cpu"


# Training
batch_size = 128
scheduler = False
log_interval = 20
checkpt_interval = 20
n_epochs = 30  # 240  #
learning_rate = 1e-3
sftbeta = 10
beta = 0.03
gamma = 0

# Dataset
dataset_name = "t2_synthetic"
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
    expt_id = "7"  # hd: with head direction
    timestep_microsec = int(1e6)
    smooth = False
    manifold_dim = 1
elif dataset_name == "s1_synthetic":
    distortion_func = "bump"
    n_times = 500
    distortion_amp = 0.4
    radius = 1
    manifold_dim = 1
    n_wiggles = 3
    embedding_dim = 2
    noise_var = 1e-3
    synthetic_rotation = SpecialOrthogonal(n=embedding_dim).random_point()
elif dataset_name == "s2_synthetic":
    # actual number of points is n_times*n_times
    n_times = 80
    radius = 1
    distortion_amp = 0.4
    manifold_dim = 2
    embedding_dim = 3
    noise_var = 1e-3
    synthetic_rotation = SpecialOrthogonal(n=embedding_dim).random_point()
elif dataset_name == "t2_synthetic":
    # actual number of points is n_times*n_times
    n_times = 80
    radius = 1
    distortion_amp = 0.4
    manifold_dim = 2
    embedding_dim = 3
    noise_var = 1e-3
    synthetic_rotation = SpecialOrthogonal(n=embedding_dim).random_point()


# Models
model_type = "neural_vae"
encoder_width = 400
decoder_width = encoder_width
encoder_depth = 6
decoder_depth = encoder_depth
if dataset_name in ("s1_synthetic", "experimental"):
    latent_dim = 2
elif dataset_name == "s2_synthetic":
    latent_dim = 3
else:
    latent_dim = 2
posterior_type = "toroidal"  # "hyperspherical"
gen_likelihood_type = "gaussian"

# Results
now = str(datetime.now().replace(second=0, microsecond=0))
results_prefix = f"{dataset_name}_{now}"
trained_model_path = None
