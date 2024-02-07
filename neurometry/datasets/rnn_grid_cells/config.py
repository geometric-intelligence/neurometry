import torch
import argparse
from .utils import generate_run_ID

class Config:
    save_dir = 'models/' #directory to save trained models
    n_epochs = 1 # number of training epochs
    n_steps = 1 # batches per epoch
    batch_size = 1000 # number of trajectories per batch
    sequence_length = 20 # number of steps in trajectory
    learning_rate = 1e-4  # gradient descent learning rate
    Np = 512 # number of place cells
    Ng = 4096 # number of grid cells
    place_cell_rf = 0.12 # width of place cell center tuning curve (m)
    DoG = True # use difference of gaussians tuning curves
    surround_scale = 2 # if DoG, ratio of sigma2^2 to sigma1^2
    RNN_type = 'RNN' # RNN or LSTM
    activation = 'relu' # recurrent nonlinearity
    weight_decay = 1e-6 # strength of weight decay on recurrent weights
    periodic = False # trajectories with periodic boundary conditions
    box_width = 2.2 # width of training environment
    box_height = 2.2 # height of training environment
    device = 'cuda' if torch.cuda.is_available() else 'cpu' # device to use for training
    n_avg = 1 # number of trajectories to average over for rate maps

# If you need to access the configuration as a dictionary
config = Config.__dict__


def create_parser(config):
    parser = argparse.ArgumentParser()

    for attr, value in config.items():
        if not attr.startswith("__"): 
            parser.add_argument(f'--{attr}', type=type(value), default=value, help=f'default: {value}')

    return parser

parser = create_parser(config)

