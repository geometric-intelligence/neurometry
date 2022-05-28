"""Main script."""

import datasets
import default_config
import evaluate.latent
import matplotlib.pyplot as plt
import models
import numpy as np
import torch
import train

dataset_torch, labels, train_loader, test_loader = datasets.utils.load(default_config)

model = models.fc_vae.VAE(
    data_dim=default_config.data_dim, latent_dim=default_config.latent_dim
).to(default_config.device)
regressor = models.regressor.Regressor(
    input_dim=2, h_dim=default_config.h_dim_regressor, output_dim=2
)
optimizer = torch.optim.Adam(model.parameters(), lr=default_config.learning_rate)

train_losses = []
test_losses = []
for epoch in range(1, default_config.n_epochs + 1):
    train_losses.append(
        train.train(
            epoch=epoch,
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            config=default_config,
            regressor=regressor,
        )
    )
    test_losses.append(
        train.test(
            epoch=epoch,
            model=model,
            test_loader=test_loader,
            config=default_config,
            regressor=None,
        )
    )

    if epoch % default_config.checkpt_interval == 0:
        mu_torch, logvar_torch = model.encode(dataset_torch)
        mu = mu_torch.cpu().detach().numpy()
        logvar = logvar_torch.cpu().detach().numpy()
        var = np.sum(np.exp(logvar), axis=-1)
        labels["var"] = var
        mu_masked = mu[labels["var"] < 0.8]
        labels_masked = labels[labels["var"] < 0.8]
        assert len(mu) == len(labels)
        evaluate.latent.plot_save_latent_space(
            f"{default_config.results_prefix}_latent_epoch{epoch}.png",
            mu,
            labels,
        )

plt.figure()
plt.plot(train_losses, label="train")
plt.plot(test_losses, label="test")
plt.legend()
plt.savefig(f"{default_config.results_prefix}_losses.png")
plt.close()
torch.save(
    model,
    f"{default_config.results_prefix}_model_latent{default_config.latent_dim}.pt",
)
