"""Main script."""
import os

os.environ["GEOMSTATS_BACKEND"] = "pytorch"
import datasets.utils
import default_config
import matplotlib.pyplot as plt
import models.fc_vae
import models.regressor
import torch
import train

import wandb


wandb.init(
    project="neural_shapes",
    config={
        "run_name": default_config.run_name,
        "device": default_config.device,
        "log_interval": default_config.log_interval,
        "checkpt_interval": default_config.checkpt_interval,
        "dataset_name": default_config.dataset_name,
        "expt_id": default_config.expt_id,
        "timestep_microsec": default_config.timestep_microsec,
        "synth_radius": default_config.synth_radius,
        "amp_wiggles": default_config.amp_wiggles,
        "embedding_dim": default_config.embedding_dim,
        "noise_var": default_config.noise_var,
        "batch_size": default_config.batch_size,
        "n_epochs": default_config.n_epochs,
        "learning_rate": default_config.learning_rate,
        "beta": default_config.beta,
        "encoder_width": default_config.encoder_width,
        "encoder_depth": default_config.encoder_depth,
        "decoder_depth": default_config.decoder_depth,
        "decoder_width": default_config.decoder_width,
        "latent_dim": default_config.latent_dim,
        "posterior_type": default_config.posterior_type,
        "gen_likelihood_type": default_config.gen_likelihood_type,
        "with_regressor": default_config.with_regressor,
        "results_prefix": default_config.results_prefix,
        "alpha": default_config.alpha,
        "gamma": default_config.gamma,
    },
)

config = wandb.config

wandb.run.name = config.run_name

results_prefix = config.results_prefix

dataset_torch, labels, train_loader, test_loader = datasets.utils.load(default_config)

_, data_dim = dataset_torch.shape

if default_config.model_type == "fc_vae":
    model = models.fc_vae.VAE(
        data_dim=data_dim,
        latent_dim=config.latent_dim,
        encoder_width=config.encoder_width,
        encoder_depth=config.encoder_depth,
        decoder_width=config.decoder_width,
        decoder_depth=config.decoder_depth,
        posterior_type=config.posterior_type,
        gen_likelihood_type=config.gen_likelihood_type,
    ).to(config.device)

regressor = None
if config.with_regressor:
    regressor = models.regressor.Regressor(
        input_dim=2, h_dim=default_config.h_dim_regressor, output_dim=2
    )

# wandb.watch(models = model, criterion = None, log="all", log_freq = 100)

optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)


losses = train.train_model(
    model=model,
    dataset_torch=dataset_torch,
    labels=labels,
    train_loader=train_loader,
    test_loader=test_loader,
    optimizer=optimizer,
    config=config,
)

train_losses, test_losses = losses


for data, labs in test_loader:
    data, labs = data.to(config.device), labs.to(config.device)

torch.onnx.export(
    model, data, f"results/trained_models/{config.results_prefix}_model.onnx"
)
wandb.save("/results/trained_models")


plt.figure()
plt.plot(train_losses, label="train")
plt.plot(test_losses, label="test")
plt.legend()
plt.savefig(f"results/figures/{config.results_prefix}_losses.png")
plt.close()

torch.save(
    model.state_dict(),
    f"results/trained_models/{config.results_prefix}_model_state_dict.pt",
)

torch.save(model, f"results/trained_models/{config.results_prefix}_model.pt")
wandb.finish()
