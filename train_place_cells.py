"""Fit a VAE to place cells."""

from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import optim
from torch.nn import functional as F

from models import VAE

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
LOG_INTERVAL = 10
N_EPOCHS = 80

LATENT_DIM = 3
NOW = str(datetime.now())

dataset = np.load("data/place_cells_expt34_b.npy")
data_dim = dataset.shape[-1]

dataset = np.log(dataset.astype(np.float32) + 1)
dataset = (dataset - np.min(dataset)) / (np.max(dataset) - np.min(dataset))

seventy_perc = int(round(len(dataset) * 0.7))
train = dataset[:seventy_perc]
test = dataset[seventy_perc:]

train_loader = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE)
test_loader = torch.utils.data.DataLoader(test, batch_size=BATCH_SIZE)

model = VAE(data_dim=data_dim, latent_dim=LATENT_DIM).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    """Compute VAE loss function."""
    BCE = F.binary_cross_entropy(recon_x, x, reduction="sum")

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


def train(epoch):
    """Run one epoch on the train set."""
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
    """Run one epoch on the test set."""
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            data = data.to(DEVICE)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0 and epoch % 25 == 0:
                _, axs = plt.subplots(2)
                axs[0].imshow(data.cpu())
                axs[1].imshow(recon_batch.cpu())
                plt.savefig(f"results/{NOW}test_data_recon_epoch{epoch}.png")

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
    plt.savefig(f"results/{NOW}_losses.png")
    torch.save(model, f"results/{NOW}_model_latent{LATENT_DIM}.pt")

    print(len(dataset))
