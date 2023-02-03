"""Main script."""

import itertools
import json
import logging
import os
import tempfile
import time
import traceback

os.environ["GEOMSTATS_BACKEND"] = "pytorch"  # NOQA
import datasets.utils
import default_config
import evaluate
import geomstats.backend as gs
import matplotlib
import matplotlib.pyplot as plt
import models.neural_vae
import models.toroidal_vae
import numpy as np
import pandas as pd
import torch
import train
import viz
import wandb
from ray import tune
from ray.tune.integration.wandb import wandb_mixin
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.search.hyperopt import HyperOptSearch

# Required to make matplotlib figures in threads:
matplotlib.use("Agg")


def main():
    """Parse the default_config file and launch all experiments.

    This launches experiments with wandb with different config parameters.

    For each set of experiment parameters:
    - it runs a ray tune sweep that optimize on the hyperparameters.
    """
    for dataset_name in default_config.dataset_name:
        if dataset_name == "experimental":
            # Variable experiments parameters (experimental datasets):
            for (
                expt_id,
                timestep_microsec,
                smooth,
                select_gain_1,
            ) in itertools.product(
                default_config.expt_id,
                default_config.timestep_microsec,
                default_config.smooth,
                default_config.select_gain_1,
            ):
                sweep_name = f"{dataset_name}_{expt_id}"
                if select_gain_1:
                    sweep_name += f"_gain_1"
                else:
                    sweep_name += f"_other_gain"

                logging.info(f"\n---> START training for ray sweep: {sweep_name}.")
                main_sweep(
                    sweep_name=sweep_name,
                    dataset_name=dataset_name,
                    expt_id=expt_id,
                    timestep_microsec=timestep_microsec,
                    smooth=smooth,
                    select_gain_1=select_gain_1,
                )
        else:
            # Variable experiments parameters (synthetic datasets):
            for n_times, embedding_dim, distortion_amp, noise_var in itertools.product(
                default_config.n_times,
                default_config.embedding_dim,
                default_config.distortion_amp,
                default_config.noise_var,
            ):
                if (
                    dataset_name in ["s2_synthetic", "t2_synthetic"]
                    and embedding_dim <= 2
                ):
                    continue
                sweep_name = f"{dataset_name}_noise_var_{noise_var}_embedding_dim_{embedding_dim}"
                logging.info(f"\n---> START training for ray sweep: {sweep_name}.")
                main_sweep(
                    sweep_name=sweep_name,
                    dataset_name=dataset_name,
                    n_times=n_times,
                    embedding_dim=embedding_dim,
                    distortion_amp=distortion_amp,
                    noise_var=noise_var,
                )


def main_sweep(
    sweep_name,
    dataset_name,
    expt_id=None,
    timestep_microsec=None,
    smooth=None,
    select_gain_1=None,
    n_times=None,
    embedding_dim=None,
    distortion_amp=None,
    noise_var=None,
):
    """Run a single experiment, possibly with a ray tune sweep.

    Parameters
    ----------
    sweep_name : str
        Name of the sweep that will launches several runs.
    dataset_name : str
        Name of the dataset.
    expt_id : str (optional, only for experimental)
        ID of the experiment.
    timestep_microsec : float (optional, only for experimental)
        Timestep of the experiment.
    smooth : bool (optional, only for experimental)
        Whether to smooth the data or not.
    select_gain_1 : bool (optional, only for experimental)
        Whether to select the first gain or not.
    n_times : int (optional, only for synthetic)
        Number of times.
    embedding_dim : int (optional, only for synthetic)
        Dimension of the embedding space.
    distortion_amp : float (optional, only for synthetic)
        Amplitude of the distortion.
    noise_var : float (optional, only for synthetic)
        Variance of the noise.
    """
    sweep_config = {
        "lr": tune.loguniform(default_config.lr_min, default_config.lr_max),
        "batch_size": tune.choice(default_config.batch_size),
        "encoder_width": tune.choice(default_config.encoder_width),
        "encoder_depth": tune.choice(default_config.encoder_depth),
        "decoder_width": tune.choice(default_config.decoder_width),
        "decoder_depth": tune.choice(default_config.decoder_depth),
        "wandb": {
            "project": default_config.project,
            "api_key": default_config.api_key,
        },
    }

    fixed_config = {
        # Parameters constant across runs of the sweep (unique value):
        "dataset_name": dataset_name,
        "sweep_name": sweep_name,
        "expt_id": expt_id,
        "timestep_microsec": timestep_microsec,
        "smooth": smooth,
        "select_gain_1": select_gain_1,
        "n_times": n_times,
        "embedding_dim": embedding_dim,
        "distortion_amp": distortion_amp,
        "noise_var": noise_var,
        # Parameters fixed across runs and sweeps (unique value depending on dataset_name):
        "manifold_dim": default_config.manifold_dim[dataset_name],
        "latent_dim": default_config.latent_dim[dataset_name],
        "posterior_type": default_config.posterior_type[dataset_name],
        "distortion_func": default_config.distortion_func[dataset_name],
        "n_wiggles": default_config.n_wiggles[dataset_name],
        "radius": default_config.radius[dataset_name],
        "major_radius": default_config.major_radius[dataset_name],
        "minor_radius": default_config.minor_radius[dataset_name],
        "synthetic_rotation": default_config.synthetic_rotation[dataset_name],
        # Else:
        "device": default_config.device,
        "log_interval": default_config.log_interval,
        "checkpt_interval": default_config.checkpt_interval,
        "scheduler": default_config.scheduler,
        "n_epochs": default_config.n_epochs,
        "alpha": default_config.alpha,
        "beta": default_config.beta,
        "gamma": default_config.gamma,
        "sftbeta": default_config.sftbeta,
        "gen_likelihood_type": default_config.gen_likelihood_type,
    }

    @wandb_mixin
    def main_run(sweep_config):
        wandb.init(dir=tempfile.gettempdir())
        wandb_config = wandb.config
        wandb_config.update(fixed_config)
        run_name = "run_" + wandb.run.id + "_" + sweep_name
        wandb.run.name = run_name

        # Load data, labels
        dataset, labels, train_loader, test_loader = datasets.utils.load(wandb_config)
        data_n_times, data_dim = dataset.shape
        wandb_config.update(
            {
                "run_name": run_name,
                "results_prefix": run_name,
                "data_n_times": data_n_times,
                "data_dim": data_dim,
            }
        )
        # Update wandb config with ray tune's config
        wandb_config.update(sweep_config)

        # Save config for easy access from notebooks
        wandb_config_path = os.path.join(default_config.configs_dir, run_name + ".json")
        with open(wandb_config_path, "w") as config_file:
            json.dump(dict(wandb_config), config_file)

        # Note: loaders put data on GPU during each epoch
        dataset = dataset.to(wandb_config.device)
        train_losses, test_losses, model = create_model_and_train_test(
            wandb_config, train_loader, test_loader
        )
        logging.info(f"Done: training for {run_name}")

        training_plot_log(
            wandb_config, dataset, labels, train_losses, test_losses, model
        )
        logging.info(f"Done: training's plot & log for {run_name}")

        curvature_compute_plot_log(wandb_config, dataset, labels, model)
        logging.info(f"Done: curvature's compute, plot & log for {run_name}")
        logging.info(f"\n------> COMPLETED run: {run_name}\n")

        # Wandb records a run as finished even if it has failed.
        wandb.finish()

        # Returns metrics to log into ray tune sweep
        return {"test_loss": np.min(test_losses)}

    sweep_search = HyperOptSearch(
        sweep_config, metric=default_config.sweep_metric, mode="min"
    )

    sweep_scheduler = AsyncHyperBandScheduler(
        time_attr="training_iteration",
        metric=default_config.sweep_metric,
        brackets=1,
        reduction_factor=8,
        mode="min",
    )

    analysis = tune.run(
        main_run,
        name=sweep_name,
        local_dir=default_config.ray_sweep_dir,
        raise_on_failed_trial=False,
        num_samples=default_config.num_samples,
        scheduler=sweep_scheduler,
        search_alg=sweep_search,
        resources_per_trial={"cpu": 4, "gpu": 1},
    )

    logging.info(f"\n------> COMPLETED RAY SWEEP: {sweep_name}.\n")


def create_model_and_train_test(config, train_loader, test_loader):
    """Create model and train and test it.

    Note: train_loader and test_loader have a dataset attribute.

    The dataset attribute is a list of [data_point, label]'s.

    The data_point variable is a tensor of shape (embedding_dim,)
    corresponding to a single data point.
    """
    data_dim = tuple(train_loader.dataset[0][0].data.shape)[0]
    # Create model
    if config.posterior_type in ("gaussian", "hyperspherical"):
        model = models.neural_vae.NeuralVAE(
            data_dim=data_dim,
            latent_dim=config.latent_dim,
            sftbeta=config.sftbeta,
            encoder_width=config.encoder_width,
            encoder_depth=config.encoder_depth,
            decoder_width=config.decoder_width,
            decoder_depth=config.decoder_depth,
            posterior_type=config.posterior_type,
        ).to(config.device)
    elif config.posterior_type == "toroidal":
        model = models.toroidal_vae.ToroidalVAE(
            data_dim=data_dim,
            latent_dim=config.latent_dim,
            sftbeta=config.sftbeta,
            encoder_width=config.encoder_width,
            encoder_depth=config.encoder_depth,
            decoder_width=config.decoder_width,
            decoder_depth=config.decoder_depth,
            posterior_type=config.posterior_type,
        ).to(config.device)

    # Create optimizer, scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, amsgrad=True)
    scheduler = None
    if config.scheduler is True:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5
        )

    # Train test model
    train_losses, test_losses, best_model = train.train_test(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
    )
    return train_losses, test_losses, best_model


def training_plot_log(config, dataset, labels, train_losses, test_losses, model):
    """Plot and log training results."""
    # Plot
    fig_loss = viz.plot_loss(train_losses, test_losses, config)
    fig_latent = viz.plot_latent_space(model, dataset, labels, config)
    fig_recon_per_angle = viz.plot_recon_per_positional_angle(
        model, dataset, labels, config
    )
    fig_recon_per_time = viz.plot_recon_per_time(model, dataset, labels, config)

    # Log
    model_path = os.path.join(
        default_config.trained_models_dir, f"{config.results_prefix}_model.pt"
    )
    torch.save(model, model_path)
    wandb.log(
        {
            "fig_loss": wandb.Image(fig_loss),
            "fig_latent": wandb.Image(fig_latent),
            "fig_recon": wandb.Image(fig_recon_per_angle),
            "fig_recon_per_time": wandb.Image(fig_recon_per_time),
        }
    )
    plt.close("all")


def curvature_compute_plot_log(config, dataset, labels, model):
    """Compute, plot and log curvature results."""
    # Compute
    print("Computing learned curvature...")
    start_time = time.time()
    z_grid, _, curv_norms_learned = evaluate.compute_curvature_learned(
        model, config, dataset.shape[0], dataset.shape[1]
    )
    print("Saving + logging learned curvature profile...")
    curv_norm_learned_profile = pd.DataFrame(
        {"z_grid": z_grid, "curv_norm_learned": curv_norms_learned}
    )
    mean_velocities = []
    median_velocities = []
    std_velocities = []
    min_velocities = []
    max_velocities = []
    for one_z_grid in curv_norm_learned_profile["z_grid"]:
        selected_labels = labels[
            np.abs((one_z_grid - labels["angles"]) % 2 * np.pi) < 0.2
        ]
        mean_velocities.append(np.nanmean(selected_labels["velocities"]))
        median_velocities.append(np.nanmedian(selected_labels["velocities"]))
        std_velocities.append(np.nanstd(selected_labels["velocities"]))
        if len(selected_labels) == 0:
            min_velocities.append(-1)
            max_velocities.append(-1)
        else:
            min_velocities.append(np.nanmin(selected_labels["velocities"]))
            max_velocities.append(np.nanmax(selected_labels["velocities"]))

    curv_norm_learned_profile["mean_velocities"] = mean_velocities
    curv_norm_learned_profile["median_velocities"] = median_velocities
    curv_norm_learned_profile["std_velocities"] = std_velocities
    curv_norm_learned_profile["min_velocities"] = min_velocities
    curv_norm_learned_profile["max_velocities"] = max_velocities

    curv_norm_learned_profile.to_csv(
        os.path.join(
            default_config.curvature_profiles_dir,
            f"{config.results_prefix}_curv_norm_learned_profile.csv",
        )
    )
    wandb.log({"curv_norm_learned_profile": curv_norm_learned_profile})

    comp_time_learned = time.time() - start_time

    norm_val = None
    if config.dataset_name in ("s1_synthetic", "s2_synthetic", "t2_synthetic"):
        print("Computing true curvature for synthetic data...")
        start_time = time.time()
        z_grid, _, curv_norms_true = evaluate.compute_curvature_true(config)
        comp_time_true = time.time() - start_time
        print("Computing curvature error for synthetic data...")

        curvature_error = evaluate.compute_curvature_error(
            z_grid, curv_norms_learned, curv_norms_true, config
        )
        norm_val = max(curv_norms_true)

    # Plot
    fig_curv_norms_learned = viz.plot_curvature_norms(
        angles=z_grid,
        curvature_norms=curv_norms_learned,
        config=config,
        norm_val=norm_val,
        profile_type="learned",
    )
    if config.dataset_name in ("s1_synthetic", "s2_synthetic", "t2_synthetic"):
        fig_curv_norms_true = viz.plot_curvature_norms(
            angles=z_grid,
            curvature_norms=curv_norms_true,
            config=config,
            norm_val=None,
            profile_type="true",
        )
    elif config.dataset_name == "experimental":
        # HACK ALERT: Remove large curvatures
        # Note that the full curvature profile is saved in csv
        # The large curvatures are only removed for the plot
        median = curv_norm_learned_profile["curv_norm_learned"].median()
        filtered = curv_norm_learned_profile[
            curv_norm_learned_profile["curv_norm_learned"] < 8 * median
        ]
        fig_curv_norms_learned_velocities = viz.plot_curvature_velocities(
            curv_norm_learned_profile=filtered, config=config, labels=labels
        )
    # Log
    wandb.log(
        {
            "comp_time_curv_learned": comp_time_learned,
            "average_curv_norms_learned": gs.mean(curv_norms_learned),
            "std_curv_norms_learned": gs.std(curv_norms_learned),
            "fig_curv_norms_learned": wandb.Image(fig_curv_norms_learned),
        }
    )
    if config.dataset_name in ("s1_synthetic", "s2_synthetic", "t2_synthetic"):
        wandb.log(
            {
                "comp_time_curv_true": comp_time_true,
                "average_curv_norms_true": gs.mean(curv_norms_true),
                "std_curv_norms_true": gs.std(curv_norms_true),
                "curvature_error": curvature_error,
                "fig_curv_norms_true": wandb.Image(fig_curv_norms_true),
            }
        )
    elif config.dataset_name == "experimental":
        wandb.log(
            {
                "fig_curv_norms_learned_velocities": wandb.Image(
                    fig_curv_norms_learned_velocities
                ),
            }
        )
    plt.close("all")


main()
