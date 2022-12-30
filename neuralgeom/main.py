"""Main script."""

import itertools
import logging
import os
import tempfile

os.environ["GEOMSTATS_BACKEND"] = "pytorch"
import time

import datasets.utils
import default_config
import geomstats.backend as gs
import models.neural_vae
import models.toroidal_vae
import torch
import train
import wandb
from evaluate import (
    compute_error,
    compute_mean_curvature_learned,
    compute_mean_curvature_true,
)
from viz import plot_curv, plot_latent_space, plot_loss, plot_recon

TRAINED_MODELS = "results/trained_models/"
if not os.path.exists(TRAINED_MODELS):
    os.makedirs(TRAINED_MODELS)


def main():
    """Parse the default_config file and runs all experiments.
    This launches experiments with wandb with different config parameters.

    For each set of experiment parameters:
    - it runs a wandb sweep that optimize on the hyperparameters.
    """
    for dataset_name in default_config.dataset_name:
        if dataset_name == "experimental":
            for (
                expt_id,
                timestep_microsec,
                smooth,
                select_first_gain,
            ) in itertools.product(
                default_config.expt_id,
                default_config.timestep_microsec,
                default_config.smooth,
                default_config.select_first_gain,
            ):
                run_name = f"{default_config.now}_{dataset_name}"
                if select_first_gain:
                    run_name += f"_{expt_id}_first_gain"
                else:
                    run_name += f"_{expt_id}_second_gain"

                logging.info(f"\n---> START training for run: {run_name}.")
                main_run(
                    run_name=run_name,
                    dataset_name=dataset_name,
                    expt_id=expt_id,
                    timestep_microsec=timestep_microsec,
                    smooth=smooth,
                    select_first_gain=select_first_gain,
                )
        else:
            for embedding_dim, distortion_amp, noise_var in itertools.product(
                default_config.embedding_dim,
                default_config.distortion_amp,
                default_config.noise_var,
            ):
                if (
                    dataset_name in ["s2_synthetic", "t2_synthetic"]
                    and embedding_dim <= 2
                ):
                    continue
                run_name = f"{default_config.now}_{dataset_name}"
                run_name += f"_embedding_dim_{embedding_dim}"
                run_name += f"_distortion_amp_{distortion_amp}_noise_var_{noise_var}"
                logging.info(f"\n---> START training for run: {run_name}.")
                main_run(
                    run_name=run_name,
                    dataset_name=dataset_name,
                    embedding_dim=embedding_dim,
                    distortion_amp=distortion_amp,
                    noise_var=noise_var,
                )


def main_run(
    run_name,
    dataset_name,
    expt_id=None,
    select_first_gain=None,
    embedding_dim=None,
    distortion_amp=None,
    noise_var=None,
):
    """Run a single experiment, possibly with a wandb sweep.

    Parameters
    ----------
    run_name : str
        Name of the run.
    dataset_name : str
        Name of the dataset.
    expt_id : str
        ID of the experiment.
    select_first_gain : bool
        Whether to select the first gain or not.
    embedding_dim : int
        Dimension of the embedding space.
    distortion_amp : float
        Amplitude of the distortion.
    noise_var : float
        Variance of the noise.
    """
    wandb.init(
        project=default_config.project,
        dir=tempfile.gettempdir(),
        config={
            "run_name": run_name,
            "dataset_name": dataset_name,
            "expt_id": expt_id,
            "select_first_gain": select_first_gain,
            "embedding_dim": embedding_dim,
            "distortion_amp": distortion_amp,
            "noise_var": noise_var,
            "device": default_config.device,
            "log_interval": default_config.log_interval,
            "checkpt_interval": default_config.checkpt_interval,
            "timestep_microsec": default_config.timestep_microsec,
            "smooth": default_config.smooth,
            "distortion_func": default_config.distortion_func,
            "n_times": default_config.n_times,
            "radius": default_config.radius,
            "major_radius": default_config.major_radius,
            "minor_radius": default_config.minor_radius,
            "n_wiggles": default_config.n_wiggles,
            "synthetic_rotation": default_config.synthetic_rotation,
            "batch_size": default_config.batch_size,
            "scheduler": default_config.scheduler,
            "n_epochs": default_config.n_epochs,
            "learning_rate": default_config.learning_rate,
            "beta": default_config.beta,
            "sftbeta": default_config.sftbeta,
            "encoder_width": default_config.encoder_width,
            "encoder_depth": default_config.encoder_depth,
            "decoder_depth": default_config.decoder_depth,
            "decoder_width": default_config.decoder_width,
            "latent_dim": default_config.latent_dim,
            "posterior_type": default_config.posterior_type,
            "manifold_dim": default_config.manifold_dim,
            "gen_likelihood_type": default_config.gen_likelihood_type,
            "results_prefix": default_config.results_prefix,
            "gamma": default_config.gamma,
        },
    )

    config = wandb.config

    wandb.run.name = config.run_name

    train_losses, test_losses, model = create_model_and_train_test(config)

    plot_and_log(train_losses, test_losses, model)

    evaluate_curvature(model)

    wandb.finish()


def create_model_and_train_test(config):
    # Load data, labels
    testing = datasets.utils.load(config)
    dataset_torch, labels, train_loader, test_loader = datasets.utils.load(config)

    dataset_torch = dataset_torch.to(config.device)
    _, data_dim = dataset_torch.shape

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
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, amsgrad=True
    )
    if config.scheduler is True:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5
        )
    else:
        scheduler = None

    # Train model
    train_losses, test_losses, best_model = train.train_test(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
    )

    print("Done Training!")
    return train_losses, test_losses, best_model


def plot_and_log(config, dataset_torch, labels, train_losses, test_losses, model):
    # FIXME: the labels here might not be aligned in time.
    # Plot the loss
    fig_loss = plot_loss(train_losses, test_losses, config)

    # Plot the latent space
    fig_latent = plot_latent_space(model, dataset_torch, labels, config)

    # Plot original data and reconstruction
    fig_recon = plot_recon(model, dataset_torch, labels, config)

    torch.save(model, os.path.join(TRAINED_MODELS, f"{config.results_prefix}_model.pt"))

    wandb.log(
        {
            "fig_loss": wandb.Image(fig_loss),
            "fig_latent": wandb.Image(fig_latent),
            "fig_recon": wandb.Image(fig_recon),
        }
    )
    print("Done Plotting!")


def evaluate_curvature(config, dataset_torch, model):

    if config.dataset_name in ("s1_synthetic", "s2_synthetic", "t2_synthetic"):
        print("Computing true curvature from synthetic data...")
        # start_time = time.time()
        z_grid, _, curv_norms_true = compute_mean_curvature_true(config)
        # error = compute_error(z_grid, curv_norms_learned, curv_norms_true, config)
        # end_time = time.time()
        # print("Computation time: " + "%.3f" % (end_time - start_time) + " seconds.")
        fig_curv_norms_true = plot_curv(z_grid, curv_norms_true, config, None, "true")

        wandb.log({"fig_curv_norms_true": wandb.Image(fig_curv_norms_true)})

    print("Computing learned curvature...")
    # start_time = time.time()
    z_grid, _, curv_norms_learned = compute_mean_curvature_learned(
        model, config, dataset_torch.shape[0], dataset_torch.shape[1]
    )
    # end_time = time.time()
    # print("Computation time: " + "%.3f" % (end_time - start_time) + " seconds.")
    norm_val = None
    if config.dataset_name in ("s1_synthetic", "s2_synthetic", "t2_synthetic"):
        norm_val = max(curv_norms_true)
        curvature_error = compute_error(
            z_grid, curv_norms_learned, curv_norms_true, config
        )
        wandb.log({"curvature_error": curvature_error})

    fig_curv_norms_learned = plot_curv(
        z_grid, curv_norms_learned, config, norm_val, "learned"
    )

    wandb.log({"fig_curv_norms_learned": wandb.Image(fig_curv_norms_learned)})
