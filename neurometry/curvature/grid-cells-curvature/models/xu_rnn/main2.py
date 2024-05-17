import os
import json
import pickle

import experiment
import numpy as np
import utils
from absl import app, flags
from ml_collections import config_flags
import itertools
import logging
from ray import air, tune
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.schedulers import AsyncHyperBandScheduler
import wandb
import default_config
import torch


def main():
    """ Launch all experiments."""
    rng = np.random.default_rng(0)

    # Generate experiment parameter combinations
    for (s_0, sigma_saliency, x_saliency) in itertools.product(default_config.s_0,default_config.sigma_saliency,default_config.x_saliency):
        sweep_name = f"s_0={s_0}_sigma_saliency={sigma_saliency}_x_saliency={x_saliency}"
        logging.info(f"\n---> START training for ray sweep: {sweep_name}.")
        main_sweep(sweep_name=sweep_name, s_0=s_0, sigma_saliency=sigma_saliency, x_saliency=x_saliency,rng=rng)


def main_sweep(sweep_name, s_0, sigma_saliency, x_saliency, rng):
    """Launch a single experiment."""
    sweep_config = {
        "lr": tune.choice(default_config.lr),
        "w_trans": tune.choice(default_config.w_trans),
        "rnn_step": tune.choice(default_config.rnn_step),
        "rnn_inte_step": tune.choice(default_config.n_inte_step),
    }

    fixed_config = {
        #parameters that vary across experiments
        "sweep_name": sweep_name,
        "s_0": s_0,
        "sigma_saliency": sigma_saliency,
        "x_saliency": x_saliency,
        #parameters that are fixed across experiments
        # training parameters
        "load_pretrain": default_config.load_pretrain,
        "pretrain_dir": default_config.pretrain_dir,
        "num_steps_train": default_config.num_steps_train,
        "lr_decay_from": default_config.lr_decay_from,
        "steps_per_logging": default_config.steps_per_logging,
        "steps_per_large_logging": default_config.steps_per_large_logging,
        "steps_per_integration": default_config.steps_per_integration,
        "norm_v": default_config.norm_v,
        "positive_v": default_config.positive_v,
        "positive_u": default_config.positive_u,
        "optimizer_type": default_config.optimizer_type,
        # simulated data parameters
        "max_dr_trans": default_config.max_dr_trans,
        "max_dr_isometry": default_config.max_dr_isometry,
        "batch_size": default_config.batch_size,
        "sigma_data": default_config.sigma_data,
        "add_dx_0": default_config.add_dx_0,
        "small_int": default_config.small_int,
        # model parameters
        "trans_type": default_config.trans_type,
        "num_grid": default_config.num_grid,
        "num_neurons": default_config.num_neurons,
        "block_size": default_config.block_size,
        "sigma": default_config.sigma,
        "w_kernel": default_config.w_kernel,
        "w_isometry": default_config.w_isometry,
        "w_reg_u": default_config.w_reg_u,
        "reg_decay_until": default_config.reg_decay_until,
        "adaptive_dr": default_config.adaptive_dr,
        "reward_step": default_config.reward_step,
        "saliency_type": default_config.saliency_type,
        # path integration parameters
        "n_inte_step": default_config.n_inte_step,
        "n_traj": default_config.n_traj,
        "n_inte_step_vis": default_config.n_inte_step_vis,
        "n_traj_vis": default_config.n_traj_vis,
        # device
        "device": default_config.device,
    }

    def main_run(sweep_config):
        wandb.init(project="grid-cell-rnns", entity="bioshape-lab")
        wandb_config = wandb.config
        wandb_config.update(fixed_config)
        wandb_config.update(sweep_config)

        run_name = "run_" + wandb.run.id + "_" + sweep_name
        wandb.run.name = run_name
        wandb_config.update({"run_name": run_name})
        wandb_config_path = os.path.join(default_config.configs_dir, run_name + ".json")
        with open(wandb_config_path, "w") as config_file:
            json.dump(dict(wandb_config), config_file)

        ###TODO: IMPLEMENT _CONVERT_CONFIG 
        expt_config = _convert_config(wandb_config)

        expt = experiment.Experiment(rng, expt_config, wandb_config.device)
        expt.train_and_evaluate()

        logging.info(f"Done: training for {run_name}")

        _training_plot_log()

        logging.info(f"Done: training's plot & log for {run_name}")

        logging.info(f"\n------> COMPLETED run: {run_name}\n")

        wandb.finish()

        ### PASS SWEEP METRIC
        return {"test_loss": 0.0}

    ### DEFINE SWEEP METRIC HERE??
    sweep_search = HyperOptSearch(metric=default_config.sweep_metric, mode="min")

    sweep_scheduler = AsyncHyperBandScheduler(
        time_attr="training_iteration",
        metric=default_config.sweep_metric,
        brackets=1,
        reduction_factor=8,
        mode="min",
    )

    tuner = tune.Tuner(
        trainable=tune.with_resources(main_run, {"cpu": 4, "gpu": 1}),
        param_space=sweep_config,
        tune_config=tune.TuneConfig(
            search_alg=sweep_search,
            scheduler=sweep_scheduler,
            num_samples=default_config.num_samples,
        ),
        run_config=air.RunConfig(
            name=sweep_name, local_dir=default_config.ray_sweep_dir
        ),
    )
    tuner.fit()
    
    logging.info(f"\n------> COMPLETED RAY SWEEP: {sweep_name}.\n")




def _convert_config():
    raise NotImplementedError

def _training_plot_log(wandb_config, model):
    arch = type(model).__name__
    state = {
        "arch": arch,
        "state_dict": model.state_dict(),
    }
    model_filename = os.path.join(
        default_config.trained_models_dir, f"{wandb_config.run_name}_model.pt"
    )
    torch.save(state, model_filename)
    wandb.save(model_filename)

    activations_filename = os.path.join(
        default_config.activations_dir, f"{wandb_config.run_name}_activations.pkl"
    )
    activations = {
        "v": model.encoder.v.data.cpu().detach().numpy(),
        "u": model.decoder.u.data.cpu().detach().numpy(),
    }
    with open(activations_filename, "wb") as f:
        pickle.dump(activations, f)

    raise NotImplementedError


if __name__ == "__main__":
    main()

