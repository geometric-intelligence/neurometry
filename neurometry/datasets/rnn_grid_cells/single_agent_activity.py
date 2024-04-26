# from neurometry.datasets.rnn_grid_cells.utils import generate_run_ID

import numpy as np
import torch
from tqdm import tqdm
import os

from .config import parser
from .model import RNN
from .place_cells import PlaceCells
from .scores import GridScorer
from .trajectory_generator import TrajectoryGenerator
from .utils import generate_run_ID
from .visualize import compute_ratemaps


def main(options, file_path, epoch="final", res=20):
    print("Evaluating SINGLE agent model.")

    place_cells = PlaceCells(options)
    if options.RNN_type == "RNN":
        model = RNN(options, place_cells)
    elif options.RNN_type == "LSTM":
        raise NotImplementedError

    print("Creating trajectory generator...")

    trajectory_generator = TrajectoryGenerator(options, place_cells)

    print("Loading single agent model...")

    model_single_agent = model.to(options.device)

    model_name = "final_model.pth" if epoch == "final" else f"epoch_{epoch}.pth"
    model_path = os.path.join(file_path, model_name)
    saved_model_single_agent = torch.load(model_path)
    model_single_agent.load_state_dict(saved_model_single_agent)

    print("Computing ratemaps and activations...")

    Ng = options.Ng
    n_avg = options.n_avg

    (
        activations_single_agent,
        rate_map_single_agent,
        g_single_agent,
        positions_single_agent,
    ) = compute_ratemaps(
        model_single_agent,
        trajectory_generator,
        options,
        res=res,
        n_avg=n_avg,
        Ng=Ng,
        all_activations_flag=True,
    )

    activations_dir = os.path.join(file_path, "activations")

    np.save(
        os.path.join(activations_dir, f"activations_single_agent_epoch_{epoch}.npy"),
        activations_single_agent,
    )
    np.save(
        os.path.join(activations_dir, f"rate_map_single_agent_epoch_{epoch}.npy"),
        rate_map_single_agent,
    )

    np.save(
        os.path.join(activations_dir, f"positions_single_agent_epoch_{epoch}.npy"),
        positions_single_agent,
    )
    # #   activations is in the shape [number of grid cells (Ng) x res x res x n_avg]
    # #   ratemap is in the shape [Ng x res^2]

    return (
        activations_single_agent,
        rate_map_single_agent,
        g_single_agent,
        positions_single_agent,
    )


def compute_grid_scores(res, rate_map_single_agent, scorer):
    print("Computing grid scores...")
    score_60_single_agent, _, _, _, _, _ = zip(
        *[
            scorer.get_scores(rm.reshape(res, res))
            for rm in tqdm(rate_map_single_agent)
        ],
        strict=False,
    )
    return np.array(score_60_single_agent)


def compute_border_scores(box_width, res, rate_map_single_agent, scorer):
    print("Computing border scores...")
    border_scores_single_agent = []
    for rm in tqdm(rate_map_single_agent):
        bs, _, _ = scorer.border_score(rm.reshape(res, res), res, box_width)
        border_scores_single_agent.append(bs)

    return border_scores_single_agent


def compute_band_scores(box_width, res, rate_map_single_agent, scorer):
    print("Computing band scores...")

    band_scores_single_agent = []
    for rm in tqdm(rate_map_single_agent):
        bs = scorer.band_score(rm.reshape(res, res), res, box_width)
        band_scores_single_agent.append(bs)

    return band_scores_single_agent


def compute_all_scores(options, file_path, res, rate_map_single_agent):
    starts = [0.2] * 10
    ends = np.linspace(0.4, 1.0, num=10)
    box_width = options.box_width
    box_height = options.box_height
    coord_range = ((-box_width / 2, box_width / 2), (-box_height / 2, box_height / 2))
    masks_parameters = zip(starts, ends.tolist(), strict=False)
    scorer = GridScorer(res, coord_range, masks_parameters)
    score_60_single_agent = compute_grid_scores(res, rate_map_single_agent, scorer)
    border_scores_single_agent = compute_border_scores(
        box_width, res, rate_map_single_agent, scorer
    )
    band_scores_single_agent = compute_band_scores(
        box_width, res, rate_map_single_agent, scorer
    )

    scores_dir = os.path.join(file_path, "scores")
    np.save(
        scores_dir + f"score_60_single_agent_epoch_{epoch}.npy", score_60_single_agent
    )
    np.save(
        scores_dir + f"border_scores_single_agent_epoch_{epoch}.npy",
        border_scores_single_agent,
    )
    np.save(
        scores_dir + f"band_scores_single_agent_epoch_{epoch}.npy",
        band_scores_single_agent,
    )

    return score_60_single_agent, border_scores_single_agent, band_scores_single_agent


if __name__ == "__main__":
    #   Setting defaults for network parameters so that the models can be evaluated
    options = parser.parse_args()
    options.run_ID = generate_run_ID(options)

    # resolution of ratemaps
    res = 20

    epochs = list(range(20))
    epochs.append("final")

    file_path = "path"

    for epoch in epochs:
        print(f"Loading single agent activations for epoch {epoch} ...")
        (
            activations_single_agent,
            rate_map_single_agent,
            g_single_agent,
            positions_single_agent,
        ) = main(options, file_path, epoch=epoch, res=res)

        print(f"Computing single agent scores for epoch {epoch} ...")
        score_60_single_agent, border_scores_single_agent, band_scores_single_agent = (
            compute_all_scores(options, file_path, res, rate_map_single_agent)
        )

