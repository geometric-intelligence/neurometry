import itertools
import json
import logging
import os
import pickle


import default_config
import eval
import experiment
import ml_collections
import numpy as np
import torch
import wandb
import ray

# Initialize Ray
ray.init()

@ray.remote(num_gpus=1)
def run_experiment(sweep_name, s_0, sigma_saliency, x_saliency, plot=True):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(ray.get_gpu_ids()[0])  # Automatically manage GPU assignment with Ray

    sweep_config = {
        "lr": np.random.choice(default_config.lr),
        "w_trans": np.random.choice(default_config.w_trans),
        "rnn_step": np.random.choice(default_config.rnn_step),
        "n_inte_step": np.random.choice(default_config.n_inte_step),
    }

    fixed_config = {
        "sweep_name": sweep_name,
        "s_0": s_0,
        "sigma_saliency": sigma_saliency,
        "x_saliency": x_saliency,
        "load_pretrain": default_config.load_pretrain,
        "pretrain_path": default_config.pretrain_path,
        "num_steps_train": default_config.num_steps_train,
        "lr_decay_from": default_config.lr_decay_from,
        "steps_per_logging": default_config.steps_per_logging,
        "steps_per_large_logging": default_config.steps_per_large_logging,
        "steps_per_integration": default_config.steps_per_integration,
        "norm_v": default_config.norm_v,
        "positive_v": default_config.positive_v,
        "positive_u": default_config.positive_u,
        "optimizer_type": default_config.optimizer_type,
        "max_dr_trans": default_config.max_dr_trans,
        "max_dr_isometry": default_config.max_dr_isometry,
        "batch_size": default_config.batch_size,
        "sigma_data": default_config.sigma_data,
        "add_dx_0": default_config.add_dx_0,
        "small_int": default_config.small_int,
        "freeze_decoder": default_config.freeze_decoder,
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
        "n_traj": default_config.n_traj,
        "n_inte_step_vis": default_config.n_inte_step_vis,
        "n_traj_vis": default_config.n_traj_vis,
        "device": 'cuda',
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

        expt_config = _convert_config(wandb_config)

        rng = np.random.default_rng()

        expt = experiment.Experiment(rng, expt_config, wandb_config.device)
        errors, model = expt.train_and_evaluate()
        error_reencode = errors[-1]["err_reencode"]

        logging.info(f"Done: training for {run_name}")
        if plot:
            _training_plot_log(wandb_config, model)
            logging.info(f"Done: training's plot & log for {run_name}")

        logging.info(f"\n------> COMPLETED run: {run_name}\n")

        wandb.finish()

        return {"error_reencode": error_reencode}

    main_run(sweep_config)

def main():
    """Launch all experiments."""
    param_combinations = list(itertools.product(default_config.s_0, default_config.sigma_saliency, default_config.x_saliency))

    # Create tasks
    tasks = [(f"s_0={s_0}_sigma_saliency={sigma_saliency}_x_saliency={x_saliency}", s_0, sigma_saliency, x_saliency)
             for s_0, sigma_saliency, x_saliency in param_combinations]

    # Run tasks in parallel using Ray
    results = ray.get([run_experiment.remote(*task) for task in tasks])




def _d(**kwargs):
    """Helper of creating a config dict."""
    return ml_collections.ConfigDict(initial_dictionary=kwargs)


def _convert_config(wandb_config):
    """Get the hyperparameters for the model"""
    config = ml_collections.ConfigDict()

    # training config
    config.train = _d(
        load_pretrain=wandb_config.load_pretrain,
        pretrain_path=wandb_config.pretrain_path,
        num_steps_train=wandb_config.num_steps_train,
        lr=wandb_config.lr,
        lr_decay_from=wandb_config.lr_decay_from,
        steps_per_logging=wandb_config.steps_per_logging,
        steps_per_large_logging=wandb_config.steps_per_large_logging,
        steps_per_integration=wandb_config.steps_per_integration,
        norm_v=wandb_config.norm_v,
        positive_v=wandb_config.positive_v,
        positive_u=wandb_config.positive_u,
        optimizer_type=wandb_config.optimizer_type,
    )

    # simulated data
    config.data = _d(
        max_dr_trans=wandb_config.max_dr_trans,
        max_dr_isometry=wandb_config.max_dr_isometry,
        batch_size=wandb_config.batch_size,
        sigma_data=wandb_config.sigma_data,
        add_dx_0=wandb_config.add_dx_0,
        small_int=wandb_config.small_int,
    )

    # model parameter
    config.model = _d(
        freeze_decoder=wandb_config.freeze_decoder,
        trans_type=wandb_config.trans_type,
        rnn_step=wandb_config.rnn_step,
        num_grid=wandb_config.num_grid,
        num_neurons=wandb_config.num_neurons,
        block_size=wandb_config.block_size,
        sigma=wandb_config.sigma,
        w_kernel=wandb_config.w_kernel,
        w_trans=wandb_config.w_trans,
        w_isometry=wandb_config.w_isometry,
        w_reg_u=wandb_config.w_reg_u,
        reg_decay_until=wandb_config.reg_decay_until,
        adaptive_dr=wandb_config.adaptive_dr,
        s_0=wandb_config.s_0,
        x_saliency=wandb_config.x_saliency,
        sigma_saliency=wandb_config.sigma_saliency,
        reward_step=wandb_config.reward_step,
        saliency_type=wandb_config.saliency_type,
    )

    # path integration
    config.integration = _d(
        n_inte_step=wandb_config.n_inte_step,
        n_traj=wandb_config.n_traj,
        n_inte_step_vis=wandb_config.n_inte_step_vis,
        n_traj_vis=wandb_config.n_traj_vis,
    )

    return config


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

    activations_filename = os.path.join(
        default_config.activations_dir, f"{wandb_config.run_name}_activations.pkl"
    )
    activations = {
        "v": model.encoder.v.data.cpu().detach().numpy(),
        "u": model.decoder.u.data.cpu().detach().numpy(),
    }
    with open(activations_filename, "wb") as f:
        pickle.dump(activations, f)

    figs_dir = default_config.figs_dir
    eval.plot_experiment(wandb_config.run_name, figs_dir)

if __name__ == "__main__":
    main()

