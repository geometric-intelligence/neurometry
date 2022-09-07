"""Fit a VAE to place cells."""


from pyparsing import null_debug_action
import losses
import matplotlib.pyplot as plt
import torch
import torch.distributions as distributions
import numpy as np
import evaluate.latent
import wandb


def train_model(
    model, dataset_torch, labels, train_loader, test_loader, optimizer, config
):
    train_losses = []
    test_losses = []
    for epoch in range(1, config.n_epochs + 1):

        train_loss = train(
            epoch=epoch,
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            config=config,
        )

        train_losses.append(train_loss)

        test_loss = test(
            epoch=epoch, model=model, test_loader=test_loader, config=config
        )

        test_losses.append(test_loss)

        #wandb.log({"train_loss": train_loss, "test_loss": test_loss}, step=epoch)

        if epoch % config.checkpt_interval == 0:

            posterior_params, (q_z, p_z), z, x_rec = model(dataset_torch)

            z = z.cpu().detach().numpy()

            if model.latent_geometry == "normal":
                var = posterior_params["var"].cpu().detach().numpy()
                var = np.sum(var, axis=-1)
                labels["var"] = var
                mu_masked = z[labels["var"] < 0.8]
                labels_masked = labels[labels["var"] < 0.8]
            elif model.latent_geometry == "hyperspherical":
                labels_kappa = posterior_params["kappa"].cpu().detach().numpy()
                labels_kappa = np.squeeze(labels_kappa)
                labels["kappa"] = labels_kappa
            else:
                raise NotImplementedError

            # default_config_str = config.results_prefix
            # evaluate.latent.plot_save_latent_space(
            #     f"results/figures/{default_config_str}_latent_epoch{epoch}.png",
            #     z,
            #     labels,
            # )

    return train_losses, test_losses


def train(epoch, model, train_loader, optimizer, config):
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
        x, labels = batch_data
        labels = labels.float()
        x = x.to(config.device)
        optimizer.zero_grad()

        posterior_params, (q_z, p_z), z, x_rec = model(x)

        loss = losses.elbo(x, x_rec, q_z, p_z, config)

        loss.backward()

        train_loss += loss.item()
        optimizer.step()

        # if batch_idx % config.log_interval == 0:
        #     print(
        #         "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
        #             epoch,
        #             batch_idx * len(x),
        #             len(train_loader.dataset),
        #             100.0 * batch_idx / len(train_loader),
        #             loss.item() / len(x),
        #         )
        #     )

    train_loss = train_loss / len(train_loader.dataset)

    print("====> Epoch: {} Average loss: {:.4f}".format(epoch, train_loss))
    return train_loss


def test(epoch, model, test_loader, config):
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
            x, labels = batch_data
            x = x.to(config.device)
            labels = labels.float()

            posterior_params, (q_z, p_z), z, x_rec = model(x)

            test_loss += losses.elbo(x, x_rec, q_z, p_z, config)

            #recon_batch = model(x)[-1]

            # if i == 0 and epoch % config.checkpt_interval == 0:
            #     fig = plt.figure()
            #     if config.dataset_name == "experimental":
            #         ax1 = fig.add_subplot(1, 2, 1)

            #         ax2 = fig.add_subplot(1, 2, 1)

            #     if config.dataset_name == "points":
            #         ax = fig.add_subplot(1, 2, 1, projection="3d")
            #         sc = ax.scatter(
            #             x[:, 0],
            #             x[:, 1],
            #             x[:, 2],
            #             s=10,
            #             # c=lab[label_name],
            #             # cmap=CMAP[label_name],
            #         )
            #         plt.xlim(-1.5, 1.5)
            #         plt.ylim(-1.5, 1.5)
            #         ax.set_zlim(0, 1.5)
            #         ax.set_title("original")
            #         ax2 = fig.add_subplot(1, 2, 2, projection="3d")
            #         sc2 = ax2.scatter(
            #             recon_batch[:, 0],
            #             recon_batch[:, 1],
            #             recon_batch[:, 2],
            #             s=10,
            #             # c=lab[label_name],
            #             # cmap=CMAP[label_name],
            #         )
            #         plt.xlim(-1.5, 1.5)
            #         plt.ylim(-1.5, 1.5)
            #         ax2.set_zlim(0, 1.5)
            #         ax2.set_title("reconstruction")
            #     elif config.dataset_name == "images":
            #         _, axs = plt.subplots(2)
            #         axs[0].imshow(
            #             x[0].reshape((config.img_size, config.img_size)).cpu()
            #         )
            #         axs[1].imshow(
            #             recon_batch[0].reshape((config.img_size, config.img_size)).cpu()
            #         )
            #     else:
            #         _, axs = plt.subplots(2)
            #         axs[0].imshow(x.cpu())
            #         axs[1].imshow(recon_batch.cpu())
            #     # axs[0].set_title("original", fontsize=10)
            #     # axs[1].set_title("reconstruction", fontsize=10)
            #     plt.savefig(
            #         f"results/figures/{config.results_prefix}_recon_epoch{epoch}.png"
            #     )

    test_loss /= len(test_loader.dataset)
    print("====> Test set loss: {:.4f}".format(test_loss))
    return test_loss
