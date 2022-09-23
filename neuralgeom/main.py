"""Main script."""
import os

os.environ["GEOMSTATS_BACKEND"] = "pytorch"
import geomstats.backend as gs
import datasets.utils
import default_config
import models.neural_geom_vae
import torch
import train
from main_eval import get_mean_curvature
from main_eval import get_mean_curvature_analytic
from main_eval import get_cross_corr
from main_eval import get_difference
from plots import create_plots
import time

import wandb


def main():

    # Initialize WandB
    wandb.init(
        project="mars",
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

    # Create model
    model = models.neural_geom_vae.VAE(
        data_dim=data_dim,
        latent_dim=config.latent_dim,
        sftbeta=config.sftbeta,
        encoder_width=config.encoder_width,
        encoder_depth=config.encoder_depth,
        decoder_width=config.decoder_width,
        decoder_depth=config.decoder_depth,
        posterior_type=config.posterior_type,
    ).to(config.device)

    # Create optimier, scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    if config.scheduler == True:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5
        )
    else:
        scheduler = None

    # Train model
    train_losses, test_losses, best_model= train.train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
    )

    if config.dataset_name in ("s1_synthetic", "experimental"):
        angles = torch.linspace(0, 2 * gs.pi, dataset_torch.shape[0])
    elif config.dataset_name == "s2_synthetic":
        thetas = gs.linspace(0,gs.pi,config.n_times)
        phis = gs.linspace(0,2*gs.pi,config.n_times)
        angles = torch.cartesian_prod(thetas,phis)



    # print("Computing curvature...")

    # start_time = time.time()

    # # compute model extrinsic curvature
    # mean_curvature, mean_curvature_norms = get_mean_curvature(
    #     best_model, angles, config, dataset_torch.shape[1]
    # )

    # end_time = time.time()

    # print("Computation time: " + "%.3f" %(end_time - start_time) + " seconds.")

    if config.dataset_name == "s1_synthetic":

        print("Computing analytic curvature...")

        mean_curvature_synth, mean_curvature_norms_analytic = get_mean_curvature_analytic(
            angles, config
        )

        s1, s2, correlation = get_cross_corr(
            mean_curvature_norms, mean_curvature_norms_analytic
        )

        error = get_difference(
            angles, mean_curvature_norms_analytic, mean_curvature_norms
        )

        print("Generating plots...")

        fig_loss, fig_recon, fig_latent, fig_curv, fig_curv_analytic, fig_comparison = create_plots(train_losses,test_losses,model,dataset_torch,labels,angles,mean_curvature_norms,mean_curvature_norms_analytic, error, config)

        wandb.log(
            {
                "error": error,
                "correlation": max(correlation),
                "fig_loss": wandb.Image(fig_loss),
                "fig_recon": wandb.Image(fig_recon),
                "fig_latent": wandb.Image(fig_latent),
                "fig_curv": wandb.Image(fig_curv),
                "fig_curv_analytic": wandb.Image(fig_curv_analytic),
                "fig_comparison": wandb.Image(fig_comparison),
                
            }
        )
    elif config.dataset_name == "s2_synthetic":
        print("Generating plots...")

        fig_loss, fig_recon, fig_latent, fig_curv, fig_curv_analytic, fig_comparison = create_plots(train_losses,test_losses,model,dataset_torch,labels,angles,None,None, None, config)

        wandb.log(
            {
            "fig_loss": wandb.Image(fig_loss),
            "fig_recon": wandb.Image(fig_recon),
            "fig_latent": wandb.Image(fig_latent),
            }
        )
    elif config.dataset_name == "experimental":
        print("Generating plots...")

        fig_loss, fig_recon, fig_latent, fig_curv, fig_curv_analytic, fig_comparison = create_plots(train_losses,test_losses,model,dataset_torch,labels,angles,mean_curvature_norms,None, None, config)
        wandb.log(
            {
                "fig_loss": wandb.Image(fig_loss),
                "fig_recon": wandb.Image(fig_recon),
                "fig_latent": wandb.Image(fig_latent),       
            }
        )
        
    print("Done! Saving model...")

    torch.save(
        best_model.state_dict(),
        f"results/trained_models/{config.results_prefix}_model_state_dict.pt",
    )

    torch.save(best_model, f"results/trained_models/{config.results_prefix}_model.pt")

    wandb.finish()




main()