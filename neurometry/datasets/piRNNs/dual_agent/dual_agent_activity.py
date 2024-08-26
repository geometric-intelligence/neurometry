import os
import random

import numpy as np
import torch
from tqdm import tqdm

from neurometry.datasets.piRNNs.scores import GridScorer

from .config import parser
from .model_dual_path_integration import RNN
from .place_cells_dual_path_integration import PlaceCells
from .trajectory_generator_dual_path_integration import TrajectoryGenerator
from .utils import generate_run_ID
from .visualize import compute_ratemaps


def main(options, file_path, epoch="final", res=20):
    print("Evaluating DUAL agent model.")

    place_cells = PlaceCells(options)
    if options.RNN_type == "RNN":
        model = RNN(options, place_cells)
    elif options.RNN_type == "LSTM":
        raise NotImplementedError

    print("Creating trajectory generator...")

    trajectory_generator = TrajectoryGenerator(options, place_cells)

    print("Loading dual agent model...")

    model_dual_agent = model.to(options.device)

    model_name = "final_model.pth" if epoch == "final" else f"epoch_{epoch}.pth"
    model_path = os.path.join(file_path, model_name)
    saved_model_dual_agent = torch.load(model_path)
    model_dual_agent.load_state_dict(saved_model_dual_agent)

    print("Computing ratemaps and activations...")

    Ng = options.Ng
    n_avg = options.n_avg
    activations_dual_agent, rate_map_dual_agent, g_dual_agent, positions_dual_agent = (
        compute_ratemaps(
            model_dual_agent,
            trajectory_generator,
            options,
            res=res,
            n_avg=n_avg,
            Ng=Ng,
            all_activations_flag=True,
        )
    )

    activations_dir = os.path.join(file_path, "activations")

    np.save(
        os.path.join(activations_dir, f"activations_dual_agent_epoch_{epoch}.npy"),
        activations_dual_agent,
    )
    np.save(
        os.path.join(activations_dir, f"rate_map_dual_agent_epoch_{epoch}.npy"),
        rate_map_dual_agent,
    )
    np.save(
        os.path.join(activations_dir, f"positions_dual_agent_epoch_{epoch}.npy"),
        positions_dual_agent,
    )

    return (
        activations_dual_agent,
        rate_map_dual_agent,
        g_dual_agent,
        positions_dual_agent,
    )


def compute_grid_scores(res, rate_map_dual_agent, scorer):
    print("Computing grid scores...")
    score_60_dual_agent, _, _, _, _, _ = zip(
        *[scorer.get_scores(rm.reshape(res, res)) for rm in tqdm(rate_map_dual_agent)],
        strict=False,
    )
    return np.array(score_60_dual_agent)


def compute_border_scores(box_width, res, rate_map_dual_agent, scorer):
    print("Computing border scores...")
    border_scores_dual_agent = []
    for rm in tqdm(rate_map_dual_agent):
        bs, _, _ = scorer.border_score(rm.reshape(res, res), res, box_width)
        border_scores_dual_agent.append(bs)

    return border_scores_dual_agent


def compute_band_scores(box_width, res, rate_map_dual_agent, scorer):
    print("Computing band scores...")

    band_scores_dual_agent = []
    for rm in tqdm(rate_map_dual_agent):
        bs = scorer.band_score(rm.reshape(res, res), res, box_width)
        band_scores_dual_agent.append(bs)

    return band_scores_dual_agent


def compute_all_scores(options, file_path, res, rate_map_dual_agent):
    starts = [0.2] * 10
    ends = np.linspace(0.4, 1.0, num=10)
    box_width = options.box_width
    box_height = options.box_height
    coord_range = ((-box_width / 2, box_width / 2), (-box_height / 2, box_height / 2))
    masks_parameters = zip(starts, ends.tolist(), strict=False)
    scorer = GridScorer(res, coord_range, masks_parameters)
    score_60_dual_agent = compute_grid_scores(res, rate_map_dual_agent, scorer)
    border_scores_dual_agent = compute_border_scores(
        box_width, res, rate_map_dual_agent, scorer
    )
    band_scores_dual_agent = compute_band_scores(
        box_width, res, rate_map_dual_agent, scorer
    )

    scores_dir = os.path.join(file_path, "scores")
    np.save(
        scores_dir + f"score_60_dual_agent_epoch_{epoch}.npy", score_60_dual_agent
    )
    np.save(
        scores_dir + f"border_scores_dual_agent_epoch_{epoch}.npy",
        border_scores_dual_agent,
    )
    np.save(
        scores_dir + f"band_scores_dual_agent_epoch_{epoch}.npy",
        band_scores_dual_agent,
    )

    return score_60_dual_agent, border_scores_dual_agent, band_scores_dual_agent


if __name__ == "__main__":
    # Setting defaults for network parameters so that the models can be evaluated
    options = parser.parse_args()
    options.run_ID = generate_run_ID(options)
    random.seed(0)

    # resolution of ratemaps
    res = 20

    epochs = list(range(0, 100, 5))
    epochs.append("final")

    for epoch in epochs:
        print(f"Loading dual agent activations for epoch {epoch} ...")
        (
            activations_dual_agent,
            rate_map_dual_agent,
            g_dual_agent,
            positions_dual_agent,
        ) = main(options, epoch, res)

        print(f"Computing dual agent scores for epoch {epoch} ...")
        score_60_dual_agent, border_scores_dual_agent, band_scores_dual_agent = (
            compute_all_scores(options, res, rate_map_dual_agent)
        )
