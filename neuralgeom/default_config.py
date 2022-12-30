"""Default configuration for a run."""

import logging
import os
from datetime import datetime

import torch

os.environ["GEOMSTATS_BACKEND"] = "pytorch"
from geomstats.geometry.special_orthogonal import SpecialOrthogonal  # NOQA

project = "neural_geom"

# Can be replaced by logging.DEBUG or logging.WARNING
logging.basicConfig(level=logging.INFO)

# Hardware
device = "cuda" if torch.cuda.is_available() else "cpu"


# Dataset
dataset_name = "experimental"
if dataset_name not in ["s1_synthetic", "s2_synthetic", "t2_synthetic", "experimental"]:
    raise ValueError(f"Dataset name {dataset_name} not recognized.")


(
    expt_id,
    timestep_microsec,
    smooth,
    distortion_func,
    n_times,
    distortion_amp,
    radius,
    minor_radius,
    major_radius,
    n_wiggles,
    embedding_dim,
    noise_var,
    synthetic_rotation,
) = [None for _ in range(13)]


if dataset_name == "experimental":
    expt_id = "41"  # hd: with head direction
    timestep_microsec = int(1e6)
    smooth = False
    manifold_dim = 1
    # if there are multiple gains:
    # True selects the first one
    # False selects the second one
    select_first_gain = True
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
    n_times = 60
    radius = 1
    distortion_amp = 0.4
    manifold_dim = 2
    embedding_dim = 3
    noise_var = 1e-3
    synthetic_rotation = SpecialOrthogonal(n=embedding_dim).random_point()
elif dataset_name == "t2_synthetic":
    # actual number of points is n_times*n_times
    n_times = 100
    major_radius = 2
    minor_radius = 1
    distortion_amp = 0.4
    manifold_dim = 2
    embedding_dim = 3
    noise_var = 1e-3
    # synthetic_rotation = SpecialOrthogonal(n=embedding_dim).random_point()
    synthetic_rotation = torch.eye(n=embedding_dim)


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

# Models
model_type = "neural_vae"
encoder_width = 600
decoder_width = encoder_width
encoder_depth = 4
decoder_depth = encoder_depth
if dataset_name in ("s1_synthetic", "experimental"):
    latent_dim = 2
    posterior_type = "hyperspherical"
elif dataset_name == "s2_synthetic":
    latent_dim = 3
    posterior_type = "hyperspherical"
elif dataset_name == "t2_synthetic":
    latent_dim = 2
    posterior_type = "toroidal"


gen_likelihood_type = "gaussian"


# Results
now = str(datetime.now().replace(second=0, microsecond=0))
results_prefix = f"{dataset_name}_{now}"
trained_model_path = None

run_name = f"{now}_{dataset_name}"
if dataset_name == "experimental":
    if select_first_gain:
        run_name += f"_{expt_id}_first_gain"
    else:
        run_name += f"_{expt_id}_second_gain"
