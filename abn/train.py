"""Fit a VAE to place cells."""


import losses
import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F


def train(epoch, model, train_loader, optimizer, config, regressor=None):
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
        data = data.to(config.device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        elbo_loss = losses.elbo(recon_batch, data, mu, logvar)

        pred_loss = 0.0
        if config.with_regressor:
            norm = torch.unsqueeze(torch.linalg.norm(mu, dim=1), dim=1)
            angle_latent = mu / norm
            angle_pred = regressor(angle_latent)
            angle_true = torch.stack([torch.cos(lab), torch.sin(lab)], axis=1)
            pred_loss = F.mse_loss(angle_pred, angle_true, reduction="mean")
            pred_loss = config.weight_regressor * pred_loss

        loss = pred_loss + elbo_loss
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % config.log_interval == 0:
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


def test(epoch, model, test_loader, config, regressor=None):
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
            data = data.to(config.device)
            lab = lab.float()
            recon_batch, mu, logvar = model(data)

            pred_loss = 0.0
            if config.with_regressor:
                norm = torch.unsqueeze(torch.linalg.norm(mu, dim=1), dim=1)
                angle_latent = mu / norm
                angle_pred = regressor(angle_latent)
                angle_true = torch.stack([torch.cos(lab), torch.sin(lab)], axis=1)
                pred_loss = F.mse_loss(angle_pred, angle_true)
                pred_loss = config.weight_regressor * pred_loss

            test_loss += pred_loss
            test_loss += losses.elbo(recon_batch, data, mu, logvar).item()

            if i == 0 and epoch % config.checkpt_interval == 0:
                _, axs = plt.subplots(2)
                if config.dataset_name == "images":
                    axs[0].imshow(
                        data[0].reshape((config.img_size, config.img_size)).cpu()
                    )
                    axs[1].imshow(
                        recon_batch[0].reshape((config.img_size, config.img_size)).cpu()
                    )
                else:
                    axs[0].imshow(data.cpu())
                    axs[1].imshow(recon_batch.cpu())
                axs[0].set_title("original", fontsize=10)
                axs[1].set_title("reconstruction", fontsize=10)
                plt.savefig(f"{config.results_prefix}_recon_epoch{epoch}.png")

    test_loss /= len(test_loader.dataset)
    print("====> Test set loss: {:.4f}".format(test_loss))
    print("====> Test regression loss: {:.4f}".format(pred_loss))
    return test_loss
