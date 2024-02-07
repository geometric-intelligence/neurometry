import numpy as np
import torch

from utils import generate_run_ID
import random
from visualize import compute_ratemaps
from tqdm import tqdm
from scores import GridScorer
from place_cells_dual_path_integration import PlaceCells
from trajectory_generator_dual_path_integration import TrajectoryGenerator
from model_dual_path_integration import RNN
from config import parser

import os



parent_dir = os.getcwd() + '/'
model_folder = 'Dual agent path integration disjoint PCs/Seed 1 weight decay 1e-06/'
model_parameters = 'steps_20_batch_200_RNN_4096_relu_rf_012_DoG_True_periodic_False_lr_00001_weight_decay_1e-06/' 

def main(options, epoch="final", res=20):
    print('Evaluating DUAL agent model.')


    place_cells = PlaceCells(options)
    if options.RNN_type == 'RNN':
        model = RNN(options, place_cells)
    elif options.RNN_type == 'LSTM':
        raise NotImplementedError


    print('Creating trajectory generator...')

    trajectory_generator = TrajectoryGenerator(options, place_cells)


    print('Loading dual agent model...')

    model_dual_agent = model.to(options.device)

    if epoch == "final":
        model_name = 'final_model.pth'
    else:
        model_name = f'epoch_{epoch}.pth'
    saved_model_dual_agent = torch.load(parent_dir + model_folder + model_parameters + model_name)
    model_dual_agent.load_state_dict(saved_model_dual_agent)

    print('Computing ratemaps and activations...')

    Ng = options.Ng
    n_avg = options.n_avg
    activations_dual_agent, rate_map_dual_agent, _, _ = compute_ratemaps(model_dual_agent, trajectory_generator, options, res=res, n_avg=n_avg, Ng=Ng, all_activations_flag = True) 

    activations_dir = parent_dir + model_folder + model_parameters + 'activations/'

    np.save(activations_dir + f'activations_dual_agent_epoch_{epoch}.npy', activations_dual_agent)
    np.save(activations_dir + f'rate_map_dual_agent_epoch_{epoch}.npy', rate_map_dual_agent)

    # #   activations is in the shape [number of grid cells (Ng) x res x res x n_avg]
    # #   ratemap is in the shape [Ng x res^2]

    return activations_dual_agent, rate_map_dual_agent


def compute_grid_scores(res,rate_map_dual_agent,scorer):
    print('Computing grid scores...')
    score_60_dual_agent, _, _, _, _, _ = zip(
      *[scorer.get_scores(rm.reshape(res, res)) for rm in tqdm(rate_map_dual_agent)])
    score_60_dual_agent = np.array(score_60_dual_agent)

    return score_60_dual_agent


def compute_border_scores(box_width,res,rate_map_dual_agent,scorer):
    print('Computing border scores...')
    border_scores_dual_agent = []
    for rm in tqdm(rate_map_dual_agent):
        bs, _, _ = scorer.border_score(rm.reshape(res, res), res, box_width)
        border_scores_dual_agent.append(bs)

    return border_scores_dual_agent

def compute_band_scores(box_width,res,rate_map_dual_agent,scorer):
    print('Computing band scores...')

    band_scores_dual_agent = []
    for rm in tqdm(rate_map_dual_agent):
        bs = scorer.band_score(rm.reshape(res, res), res, box_width)
        band_scores_dual_agent.append(bs)

    return band_scores_dual_agent

def compute_all_scores(options,res,rate_map_single_agent):
    starts = [0.2] * 10
    ends = np.linspace(0.4, 1.0, num=10)
    box_width=options.box_width
    box_height=options.box_height
    coord_range=((-box_width/2, box_width/2), (-box_height/2, box_height/2))
    masks_parameters = zip(starts, ends.tolist())
    scorer = GridScorer(res, coord_range, masks_parameters)
    score_60_single_agent = compute_grid_scores(res,rate_map_single_agent,scorer)
    border_scores_single_agent = compute_border_scores(box_width,res,rate_map_single_agent,scorer)
    band_scores_single_agent = compute_band_scores(box_width,res,rate_map_single_agent,scorer)

    scores_dir = parent_dir + model_folder + model_parameters + 'scores/'
    np.save(scores_dir + f'score_60_dual_agent_epoch_{epoch}.npy', score_60_dual_agent)
    np.save(scores_dir + f'border_scores_dual_agent_epoch_{epoch}.npy', border_scores_dual_agent)
    np.save(scores_dir  + f'band_scores_dual_agent_epoch_{epoch}.npy', band_scores_dual_agent)

    return score_60_single_agent, border_scores_single_agent, band_scores_single_agent


if __name__ == "__main__":
    # Setting defaults for network parameters so that the models can be evaluated
    options = parser.parse_args()
    options.run_ID = generate_run_ID(options)
    random.seed(0)

    # resolution of ratemaps
    res = 20

    epochs = list(range(0,100,5))
    epochs.append("final")

    for epoch in epochs:
        print(f"Loading dual agent activations for epoch {epoch} ...")
        activations_dual_agent, rate_map_dual_agent = main(options, epoch, res)

        print("Computing dual agent scores for epoch {epoch} ...")
        score_60_dual_agent, border_scores_dual_agent, band_scores_dual_agent = compute_all_scores(options, res, rate_map_dual_agent)
