"""Fit a VAE to place cells."""

from datetime import datetime

import analyze
import datasets
import losses
import matplotlib.pyplot as plt
import models.fc_vae
import models.regressor
import numpy as np
import torch
from torch.nn import functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128
LOG_INTERVAL = 10
CHECKPT_INTERVAL = 10
N_EPOCHS = 100
DATASET_TYPE = "experimental"
WITH_REGRESSOR = True
WEIGHT_REGRESSOR = 1.0

LATENT_DIM = 2
NOW = str(datetime.now().replace(second=0, microsecond=0))
PREFIX = f"results/{DATASET_TYPE}_{NOW}"

if DATASET_TYPE == "experimental":
    dataset, labels = datasets.load_place_cells(expt_id=34, timestep_ns=1000000)
    dataset = dataset[labels["velocities"] > 1]
    labels = labels[labels["velocities"] > 1]
    dataset = np.log(dataset.astype(np.float32) + 1)
    # dataset = dataset[:, :-2]  # last column is weird
    dataset = (dataset - np.min(dataset)) / (np.max(dataset) - np.min(dataset))
elif DATASET_TYPE == "synthetic":
    dataset, labels = datasets.load_synthetic_place_cells(n_times=10000)
    dataset = np.log(dataset.astype(np.float32) + 1)
    dataset = (dataset - np.min(dataset)) / (np.max(dataset) - np.min(dataset))
elif DATASET_TYPE == "images":
    dataset, labels = datasets.load_synthetic_images(img_size=64)
    dataset = (dataset - np.min(dataset)) / (np.max(dataset) - np.min(dataset))
    height, width = dataset.shape[1:]
    dataset = dataset.reshape((-1, height * width))
elif DATASET_TYPE == "projections":
    dataset, labels = datasets.load_synthetic_projections(img_size=128)
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

train_loader = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE)
test_loader = torch.utils.data.DataLoader(test, batch_size=BATCH_SIZE)

model = models.fc_vae.VAE(data_dim=data_dim, latent_dim=LATENT_DIM).to(DEVICE)
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
        data = data.to(DEVICE)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        elbo_loss = losses.elbo(recon_batch, data, mu, logvar)

        pred_loss = 0.0
        if WITH_REGRESSOR:
            norm = torch.unsqueeze(torch.linalg.norm(mu, dim=1), dim=1)
            angle_latent = mu / norm
            angle_pred = regressor(angle_latent)
            angle_true = torch.stack([torch.cos(lab), torch.sin(lab)], axis=1)
            pred_loss = F.mse_loss(angle_pred, angle_true, reduction="mean")

        loss = WEIGHT_REGRESSOR * pred_loss + elbo_loss
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
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
            data = data.to(DEVICE)
            lab = lab.float()
            recon_batch, mu, logvar = model(data)

            pred_loss = 0.0
            if WITH_REGRESSOR:
                norm = torch.unsqueeze(torch.linalg.norm(mu, dim=1), dim=1)
                angle_latent = mu / norm
                angle_pred = regressor(angle_latent)
                angle_true = torch.stack([torch.cos(lab), torch.sin(lab)], axis=1)
                pred_loss = F.mse_loss(angle_pred, angle_true)

            test_loss += WEIGHT_REGRESSOR * pred_loss
            test_loss += losses.elbo(recon_batch, data, mu, logvar).item()

            if i == 0 and epoch % CHECKPT_INTERVAL == 0:
                _, axs = plt.subplots(2)
                if DATASET_TYPE == "images":
                    axs[0].imshow(data[0].reshape((height, width)).cpu())
                    axs[1].imshow(recon_batch[0].reshape((height, width)).cpu())
                else:
                    axs[0].imshow(data.cpu())
                    axs[1].imshow(recon_batch.cpu())
                plt.savefig(f"{PREFIX}_recon_epoch{epoch}.png")

    test_loss /= len(test_loader.dataset)
    print("====> Test set loss: {:.4f}".format(test_loss))
    print("====> Test regression loss: {:.4f}".format(pred_loss))
    return test_loss


if __name__ == "__main__":
    train_losses = []
    test_losses = []
    for epoch in range(1, N_EPOCHS + 1):
        train_losses.append(train(epoch))
        test_losses.append(test(epoch))

        if epoch % CHECKPT_INTERVAL == 0:
            mu_torch, logvar_torch = model.encode(dataset_torch)
            mu = mu_torch.cpu().detach().numpy()
            logvar = logvar_torch.cpu().detach().numpy()
            var = np.sum(np.exp(logvar), axis=-1)
            labels["var"] = var
            mu_masked = mu[labels["var"] < 0.8]
            labels_masked = labels[labels["var"] < 0.8]
            assert len(mu) == len(labels)
            analyze.plot_save_latent_space(
                f"{PREFIX}_latent_epoch{epoch}.png",
                mu,
                labels,
            )

    plt.figure()
    plt.plot(train_losses, label="train")
    plt.plot(test_losses, label="test")
    plt.legend()
    plt.savefig(f"{PREFIX}_losses.png")
    plt.close()
    torch.save(model, f"{PREFIX}_model_latent{LATENT_DIM}.pt")
