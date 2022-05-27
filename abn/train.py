"""Fit a VAE to place cells."""

import analyze
import datasets
import datasets.experimental
import datasets.synthetic
import default_config
import losses
import matplotlib.pyplot as plt
import models.fc_vae
import models.regressor
import numpy as np
import torch
from torch.nn import functional as F

if default_config.dataset == "experimental":
    dataset, labels = datasets.experimental.load_place_cells(
        expt_id="15_hd", timestep_ns=1000000
    )
    print(labels)
    dataset = dataset[labels["velocities"] > 1]
    labels = labels[labels["velocities"] > 1]
    dataset = np.log(dataset.astype(np.float32) + 1)
    # dataset = dataset[:, :-2]  # last column is weird
    dataset = (dataset - np.min(dataset)) / (np.max(dataset) - np.min(dataset))
elif default_config.dataset == "synthetic":
    dataset, labels = datasets.synthetic.load_place_cells(n_times=10000)
    dataset = np.log(dataset.astype(np.float32) + 1)
    dataset = (dataset - np.min(dataset)) / (np.max(dataset) - np.min(dataset))
elif default_config.dataset == "images":
    dataset, labels = datasets.synthetic.load_images(img_size=64)
    dataset = (dataset - np.min(dataset)) / (np.max(dataset) - np.min(dataset))
    height, width = dataset.shape[1:]
    dataset = dataset.reshape((-1, height * width))
elif default_config.dataset == "projections":
    dataset, labels = datasets.synthetic.load_projections(img_size=128)
    dataset = (dataset - np.min(dataset)) / (np.max(dataset) - np.min(dataset))


print(f"Dataset shape: {dataset.shape}.")
data_dim = dataset.shape[-1]
dataset_torch = torch.tensor(dataset)

seventy_perc = int(round(len(dataset) * 0.7))
train_dataset = dataset[:seventy_perc]
train_labels = labels[:seventy_perc]
test_dataset = dataset[seventy_perc:]
test_labels = labels[seventy_perc:]

train = []
for d, l in zip(train_dataset, train_labels["angles"]):
    train.append([d, float(l)])
test = []
for d, l in zip(test_dataset, test_labels["angles"]):
    test.append([d, float(l)])

train_loader = torch.utils.data.DataLoader(train, batch_size=default_config.batch_size)
test_loader = torch.utils.data.DataLoader(test, batch_size=default_config.batch_size)

model = models.fc_vae.VAE(data_dim=data_dim, latent_dim=default_config.latent_dim).to(
    default_config.device
)
regressor = models.regressor.Regressor(input_dim=2, h_dim=20, output_dim=2)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


def train(epoch):
    """Run one epoch on the train set.

    Parameters
    ----------
    epoch : int
        Index of current epoch.

    Returns
    -------
    train_loss : float
        Train loss on epoch.
    """
    model.train()
    train_loss = 0
    for batch_idx, batch_data in enumerate(train_loader):
        data, lab = batch_data
        lab = lab.float()
        data = data.to(default_config.device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        elbo_loss = losses.elbo(recon_batch, data, mu, logvar)

        pred_loss = 0.0
        if default_config.with_regressor:
            norm = torch.unsqueeze(torch.linalg.norm(mu, dim=1), dim=1)
            angle_latent = mu / norm
            angle_pred = regressor(angle_latent)
            angle_true = torch.stack([torch.cos(lab), torch.sin(lab)], axis=1)
            pred_loss = F.mse_loss(angle_pred, angle_true, reduction="mean")

        loss = default_config.weight_regressor * pred_loss + elbo_loss
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % default_config.log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item() / len(data),
                )
            )
            print(f"Regression loss: {pred_loss}")

    train_loss = train_loss / len(train_loader.dataset)

    print("====> Epoch: {} Average loss: {:.4f}".format(epoch, train_loss))
    return train_loss


def test(epoch):
    """Run one epoch on the test set.

    The loss is computed on the whole test set.

    Parameters
    ----------
    epoch : int
        Index of current epoch.

    Returns
    -------
    test_loss : float
        Test loss on epoch.
    """
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, batch_data in enumerate(test_loader):
            data, lab = batch_data
            data = data.to(default_config.device)
            lab = lab.float()
            recon_batch, mu, logvar = model(data)

            pred_loss = 0.0
            if default_config.with_regressor:
                norm = torch.unsqueeze(torch.linalg.norm(mu, dim=1), dim=1)
                angle_latent = mu / norm
                angle_pred = regressor(angle_latent)
                angle_true = torch.stack([torch.cos(lab), torch.sin(lab)], axis=1)
                pred_loss = F.mse_loss(angle_pred, angle_true)

            test_loss += default_config.weight_regressor * pred_loss
            test_loss += losses.elbo(recon_batch, data, mu, logvar).item()

            if i == 0 and epoch % default_config.checkpt_interval == 0:
                _, axs = plt.subplots(2)
                if default_config.dataset == "images":
                    axs[0].imshow(data[0].reshape((height, width)).cpu())
                    axs[1].imshow(recon_batch[0].reshape((height, width)).cpu())
                else:
                    axs[0].imshow(data.cpu())
                    axs[1].imshow(recon_batch.cpu())
                plt.savefig(f"{default_config.results_prefix}_recon_epoch{epoch}.png")

    test_loss /= len(test_loader.dataset)
    print("====> Test set loss: {:.4f}".format(test_loss))
    print("====> Test regression loss: {:.4f}".format(pred_loss))
    return test_loss


if __name__ == "__main__":
    train_losses = []
    test_losses = []
    for epoch in range(1, default_config.n_epochs + 1):
        train_losses.append(train(epoch))
        test_losses.append(test(epoch))

        if epoch % default_config.checkpt_interval == 0:
            mu_torch, logvar_torch = model.encode(dataset_torch)
            mu = mu_torch.cpu().detach().numpy()
            logvar = logvar_torch.cpu().detach().numpy()
            var = np.sum(np.exp(logvar), axis=-1)
            labels["var"] = var
            mu_masked = mu[labels["var"] < 0.8]
            labels_masked = labels[labels["var"] < 0.8]
            assert len(mu) == len(labels)
            analyze.plot_save_latent_space(
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
