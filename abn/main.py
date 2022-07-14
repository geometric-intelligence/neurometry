"""Main script."""

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
        "batch_size": default_config.batch_size,
        "n_epochs": default_config.n_epochs,
        "learning_rate": default_config.learning_rate,
        "beta": default_config.beta,
        "latent_dim": default_config.latent_dim,
        "posterior_type": default_config.posterior_type,
        "gen_likelihood_type": default_config.gen_likelihood_type,
        "with_regressor": default_config.with_regressor,
        "results_prefix": default_config.results_prefix,
        "alpha": default_config.alpha,
        "gamma": default_config.gamma
    }
)

config = wandb.config

wandb.run.name = config.run_name



dataset_torch, labels, train_loader, test_loader= datasets.utils.load(default_config)

_, data_dim = dataset_torch.shape

if default_config.model_type == "fc_vae":
    model = models.fc_vae.VAE(data_dim=data_dim, latent_dim=config.latent_dim, 
    posterior_type=config.posterior_type,gen_likelihood_type=config.gen_likelihood_type).to(
    config.device)

regressor = None
if config.with_regressor:
    regressor = models.regressor.Regressor(
        input_dim=2, h_dim=default_config.h_dim_regressor, output_dim=2
    )

#wandb.watch(models = model, criterion = None, log="all", log_freq = 100)

optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)


losses = train.train_model(
            model = model,
            dataset_torch = dataset_torch,
            labels = labels,
            train_loader = train_loader,
            test_loader = test_loader,
            optimizer = optimizer,
            config = config
        )

train_losses, test_losses = losses



for data, labs in test_loader:
    data, labs = data.to(config.device), labs.to(config.device)

torch.onnx.export(model,data,"model.onnx")
wandb.save("model.onnx")


plt.figure()
plt.plot(train_losses, label="train")
plt.plot(test_losses, label="test")
plt.legend()
plt.savefig(f"{default_config.results_prefix}_losses.png")
plt.close()
torch.save(
    model.state_dict(),
    f"{default_config.results_prefix}_model_latent{config.latent_dim}.pt",
)

wandb.finish()
