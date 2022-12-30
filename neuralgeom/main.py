"""Main script."""

import itertools
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
import torch
import train
import viz
import wandb

# Note: this is required to make matplotlib figures in threads.
matplotlib.use("Agg")

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
                run_name = f"{default_config.now}_{dataset_name}"
                if select_gain_1:
                    run_name += f"_{expt_id}_first_gain"
                else:
                    run_name += f"_{expt_id}_second_gain"

                logging.info(f"\n---> START training for run: {run_name}.")
                main_sweep(
                    run_name=run_name,
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
                run_name = f"{default_config.now}_{dataset_name}"
                run_name += f"_embedding_dim_{embedding_dim}"
                run_name += f"_distortion_amp_{distortion_amp}_noise_var_{noise_var}"
                logging.info(f"\n---> START training for run: {run_name}.")
                main_sweep(
                    run_name=run_name,
                    dataset_name=dataset_name,
                    n_times=n_times,
                    embedding_dim=embedding_dim,
                    distortion_amp=distortion_amp,
                    noise_var=noise_var,
                )


def main_sweep(
    run_name,
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
    """Run a single experiment, possibly with a wandb sweep.

    Parameters
    ----------
    run_name : str
        Name of the run.
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
    CONFIG = {
        # Parameters specific to this run (unique value):
        "dataset_name": dataset_name,
        "run_name": run_name,
        "results_prefix": run_name,
        "expt_id": expt_id,
        "timestep_microsec": timestep_microsec,
        "smooth": smooth,
        "select_gain_1": select_gain_1,
        "n_times": n_times,
        "embedding_dim": embedding_dim,
        "distortion_amp": distortion_amp,
        "noise_var": noise_var,
        # Parameters fixed across runs (unique value depending on dataset_name):
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
        "beta": default_config.beta,
        "gamma": default_config.gamma,
        "sftbeta": default_config.sftbeta,
        "gen_likelihood_type": default_config.gen_likelihood_type,
    }

    sweep_config = {
        "method": "bayes",
        "name": "sweep",
        "metric": {"goal": "minimize", "name": "test_loss"},
        "early_terminate": {"type": "hyperband", "min_iter": 3},
        "parameters": {
            "lr": {
                "min": default_config.lr_min,
                "max": default_config.lr_max,
            },
            "batch_size": {"values": default_config.batch_size},
            "encoder_width": {
                "values": default_config.encoder_width,
            },
            "encoder_depth": {
                "values": default_config.encoder_depth,
            },
            "decoder_width": {
                "values": default_config.decoder_width,
            },
            "decoder_depth": {
                "values": default_config.decoder_depth,
            },
        },
    }

    # The try/except syntax allows continuing experiments even if one run fails
    # try:
    sweep_id = wandb.sweep(sweep=sweep_config, project=default_config.project)

    def _main_run(config):
        # Load data, labels
        dataset, labels, train_loader, test_loader = datasets.utils.load(config)
        data_n_times, data_dim = dataset.shape
        config.update(
            {
                "data_n_times": data_n_times,
                "data_dim": data_dim,
            }
        )
        # FIXME: loaders might not go on GPUs
        dataset = dataset.to(config.device)  # dataset.to(config.device)

        train_losses, test_losses, model = create_model_and_train_test(
            config, train_loader, test_loader
        )
        logging.info(f"---> Done training for run: {config.run_name}.")

        training_plot_and_log(config, dataset, labels, train_losses, test_losses, model)
        logging.info(f"---> Done training's plots and logs for run: {config.run_name}.")

        curvature_compute_plot_and_log(config, dataset, model)
        logging.info(
            f"---> Done curvature's computations, plots and logs for run: {config.run_name}."
        )
        logging.info(f"\n------> COMPLETED run: {config.run_name}.\n")

    def main_run():
        with wandb.init(
            project=default_config.project, config=CONFIG, dir=tempfile.gettempdir()
        ):
            _main_run(wandb.config)

    wandb.agent(
        sweep_id=sweep_id,
        project=default_config.project,
        function=main_run,
        count=default_config.n_runs_per_sweep,
    )

    # except Exception:
    #     # Note: print() might not print within the try/except syntax
    #     logging.info(f"\n------> FAILED run: {config.run_name}.\n")
    #     traceback.print_exc()
    #     # Note: exit_code different from 0 marks run as failed
    #     wandb.finish(exit_code=1)
    #     pass

    #
    logging.info(f"\n------> COMPLETED SWEEP: {sweep_id}.\n")


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


def training_plot_and_log(config, dataset, labels, train_losses, test_losses, model):
    """Plot and log training results."""
    # Plot
    fig_loss = viz.plot_loss(train_losses, test_losses, config)
    fig_latent = viz.plot_latent_space(model, dataset, labels, config)
    fig_recon = viz.plot_recon(model, dataset, labels, config)

    # Log
    torch.save(model, os.path.join(TRAINED_MODELS, f"{config.results_prefix}_model.pt"))
    wandb.log(
        {
            "fig_loss": wandb.Image(fig_loss),
            "fig_latent": wandb.Image(fig_latent),
            "fig_recon": wandb.Image(fig_recon),
        }
    )
    plt.close("all")


def curvature_compute_plot_and_log(config, dataset, model):
    """Compute, plot and log curvature results."""
    # Compute
    print("Computing learned curvature...")
    start_time = time.time()
    z_grid, _, curv_norms_learned = evaluate.compute_curvature_learned(
        model, config, dataset.shape[0], dataset.shape[1]
    )
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

    plt.close("all")


main()
