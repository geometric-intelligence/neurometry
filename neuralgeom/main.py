"""Main script."""
import os

os.environ["GEOMSTATS_BACKEND"] = "pytorch"
import time

import datasets.utils
import default_config
import geomstats.backend as gs
import models.neural_vae
import torch
import train
import wandb
from evaluate import (
    compute_error,
    compute_mean_curvature_learned,
    compute_mean_curvature_true,
)
from plots import plot_curv, plot_latent_space, plot_loss, plot_recon

# Initialize WandB
wandb.init(
    project="hippocampus",
    config={
        "run_name": default_config.run_name,
        "device": default_config.device,
        "log_interval": default_config.log_interval,
        "checkpt_interval": default_config.checkpt_interval,
        "dataset_name": default_config.dataset_name,
        "expt_id": default_config.expt_id,
        "timestep_microsec": default_config.timestep_microsec,
        "smooth": default_config.smooth,
        "distortion_func": default_config.distortion_func,
        "n_times": default_config.n_times,
        "radius": default_config.radius,
        "distortion_amp": default_config.distortion_amp,
        "n_wiggles": default_config.n_wiggles,
        "embedding_dim": default_config.embedding_dim,
        "synthetic_rotation": default_config.synthetic_rotation,
        "noise_var": default_config.noise_var,
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

# Load data, labels
dataset_torch, labels, train_loader, test_loader = datasets.utils.load(config)
dataset_torch.to(config.device)
_, data_dim = dataset_torch.shape


def train_test_model():
    # Create model
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

    # Create optimizer, scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
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


def plot_and_log(train_losses, test_losses, model):
    # Plot the loss
    fig_loss = plot_loss(train_losses, test_losses, config)

    # Plot the latent space
    fig_latent = plot_latent_space(model, dataset_torch, labels, config)

    # Plot original data and reconstruction
    fig_recon = plot_recon(model, dataset_torch, labels, config)

    torch.save(best_model, f"results/trained_models/{config.results_prefix}_model.pt")

    wandb.log(
        {
            "fig_loss": wandb.Image(fig_loss),
            "fig_latent": wandb.Image(fig_latent),
            "fig_recon": wandb.Image(fig_recon),
        }
    )
    print("Done Plotting!")


def evaluate_curvature(model):

    print("Computing learned curvature...")
    z_grid, _, curv_norms_learned = compute_mean_curvature_learned(model, config)
    fig_curv_norms_learned = plot_curv(z_grid, curv_norms_learned, config, "learned")
    wandb.log({"fig_curv_norms_learned": wandb.Image(fig_curv_norms_learned)})

    print("Computing true curvature from synthetic data...")
    if config.dataset_name in ("s1_synthetic", "s2_synthetic"):
        z_grid, _, curv_norms_true = compute_mean_curvature_true(config)
        error = compute_error(z_grid, curv_norms_learned, curv_norms_true, config)

        fig_curv_norms_true = plot_curv(z_grid, curv_norms_true, config, "true")

        wandb.log(
            {"error": error, "fig_curv_norms_true": wandb.Image(fig_curv_norms_true)}
        )


train_losses, test_losses, best_model = train_test_model()
plot_and_log(train_losses, test_losses, best_model)

# start_time = time.time()

evaluate_curvature(best_model)

# end_time = time.time()


# print("Computation time: " + "%.3f" % (end_time - start_time) + " seconds.")


wandb.finish()
