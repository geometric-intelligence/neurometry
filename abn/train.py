"""Fit a VAE to place cells."""

from datetime import datetime

import analyze
import datasets
import matplotlib.pyplot as plt
import models
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
LOG_INTERVAL = 10
CHECKPT_INTERVAL = 10
N_EPOCHS = 100
DATASET_TYPE = "projections"

LATENT_DIM = 2
NOW = str(datetime.now())

if DATASET_TYPE == "experimental":
    dataset = np.load("data/place_cells_expt34_timestep1000000.npy")
    labels = pd.read_csv("data/place_cells_labels_expt34_timestep1000000.txt")
    dataset = dataset[labels["velocities"] > 1]
    labels = labels["angles"][labels["velocities"] > 1]
    dataset = np.log(dataset.astype(np.float32) + 1)
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
    labels = labels["angles"]
elif DATASET_TYPE == "projections":
    dataset, labels = datasets.load_synthetic_projections(img_size=128)
    dataset = (dataset - np.min(dataset)) / (np.max(dataset) - np.min(dataset))
    labels = labels["angles"]


print(f"Dataset shape: {dataset.shape}.")
data_dim = dataset.shape[-1]

seventy_perc = int(round(len(dataset) * 0.7))
train = dataset[:seventy_perc]
test = dataset[seventy_perc:]

train_loader = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE)
test_loader = torch.utils.data.DataLoader(test, batch_size=BATCH_SIZE)

model = models.VAE(data_dim=data_dim, latent_dim=LATENT_DIM).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


def loss_function(recon_x, x, mu, logvar):
    """Compute VAE loss function.

    The VAE loss is defined as:
    = reconstruction loss + Kl divergence
    over all elements and batch

    Notes
    -----
    see Appendix B from VAE paper:
    Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    https://arxiv.org/abs/1312.6114
    0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)

    Parameters
    ----------
    recon_x : array-like, shape=[batch_size, data_dim]
        Reconstructed data corresponding to input data x.
    x : array-like, shape=[batch_size, data_dim]
        Input data.
    mu : array-like, shape=[batch_size, latent_dim]
        Mean of multivariate Gaussian in latent space.
    logvar : array-like, shape=[batch_size, latent_dim]
        Vector representing the diagonal covariance of the
        multivariate Gaussian in latent space.

    Returns
    -------
    _ : array-like, shape=[batch_size,]
        Loss function on each batch element.
    """
    BCE = F.binary_cross_entropy(recon_x, x, reduction="sum")

    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


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
    for batch_idx, data in enumerate(train_loader):
        data = data.to(DEVICE)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
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

    train_loss = train_loss / len(train_loader.dataset)

    print("====> Epoch: {} Average loss: {:.4f}".format(epoch, train_loss))
    return train_loss


def test(epoch):
    """Run one epoch on the test set.

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
        for i, data in enumerate(test_loader):
            data = data.to(DEVICE)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0 and epoch % CHECKPT_INTERVAL == 0:
                _, axs = plt.subplots(2)
                if DATASET_TYPE == "images":
                    axs[0].imshow(data[0].reshape((height, width)).cpu())
                    axs[1].imshow(recon_batch[0].reshape((height, width)).cpu())
                else:
                    axs[0].imshow(data.cpu())
                    axs[1].imshow(recon_batch.cpu())
                plt.savefig(
                    f"results/{DATASET_TYPE}_{NOW}_test_data_recon_epoch{epoch}.png"
                )

                analyze.save_latent_space(
                    f"results/{DATASET_TYPE}_{NOW}_latent_space_epoch{epoch}.png",
                    model,
                    dataset,
                    labels,
                )

    test_loss /= len(test_loader.dataset)
    print("====> Test set loss: {:.4f}".format(test_loss))
    return test_loss


if __name__ == "__main__":
    train_losses = []
    test_losses = []
    for epoch in range(1, N_EPOCHS + 1):
        train_losses.append(train(epoch))
        test_losses.append(test(epoch))

    plt.figure()
    plt.plot(train_losses, label="train")
    plt.plot(test_losses, label="test")
    plt.legend()
    plt.savefig(f"results/{DATASET_TYPE}_{NOW}_losses.png")
    plt.close()
    torch.save(model, f"results/{DATASET_TYPE}_{NOW}_model_latent{LATENT_DIM}.pt")
