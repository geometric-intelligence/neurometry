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


        wandb.log({"train_loss": train_loss, "test_loss": test_loss}, step=epoch)

        # if epoch % config.checkpt_interval == 0:
        #     posterior_params = model.encode(dataset_torch)
        #     if config.posterior_type == "gaussian":
        #         z_mu_torch, z_logvar_torch = posterior_params
        #         z_mu = z_mu_torch.cpu().detach().numpy()
        #         z_logvar = z_logvar_torch.cpu().detach().numpy()
        #         z_var = np.sum(np.exp(z_logvar), axis=-1)

        #     labels["var"] = z_var
        #     mu_masked = z_mu[labels["var"] < 0.8]
        #     labels_masked = labels[labels["var"] < 0.8]
        #     assert len(z_mu) == len(labels)
        #     default_config_str = config.results_prefix
        #     evaluate.latent.plot_save_latent_space(
        #         f"results/figures/{default_config_str}_latent_epoch{epoch}.png",
        #         z_mu,
        #         labels,
        #     )

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
        data, labels = batch_data
        labels = labels.float()
        data = data.to(config.device)
        labels = labels.to(config.device)
        optimizer.zero_grad()
        x_mu_batch, posterior_params = model(data)
        z, _, _ = model.reparameterize(posterior_params)

        loss = losses.elbo(data, x_mu_batch, posterior_params, z, labels, config)

        # pred_loss = 0.0
        # TODO: replace mu with gen_likelihood_params
        # if config.with_regressor:
        #     norm = torch.unsqueeze(torch.linalg.norm(mu, dim=1), dim=1)
        #     angle_latent = mu / norm  # cos(theta), sin(theta)
        #     angle_pred = regressor(angle_latent)  # call to forward method of regressor
        #     angle_true = torch.stack([torch.cos(lab), torch.sin(lab)], axis=1)
        #     pred_loss = F.mse_loss(angle_pred, angle_true, reduction="mean")
        #     pred_loss = config.weight_regressor * pred_loss

        # loss += pred_loss
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
        # print(f"Regression loss: {pred_loss}")

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
            data, labels = batch_data
            data = data.to(config.device)
            labels = labels.float()
            labels = labels.to(config.device)
            x_mu_batch, posterior_params = model(data)
            posterior_type = model.posterior_type
            z, _, _ = model.reparameterize(posterior_params)

            # pred_loss = 0.0
            # TODO: replace mu with gen_likelihood_params
            # if config.with_regressor:
            #     norm = torch.unsqueeze(torch.linalg.norm(mu, dim=1), dim=1)
            #     angle_latent = mu / norm
            #     angle_pred = regressor(angle_latent)
            #     angle_true = torch.stack([torch.cos(lab), torch.sin(lab)], axis=1)
            #     pred_loss = F.mse_loss(angle_pred, angle_true)
            #     pred_loss = config.weight_regressor * pred_loss

            # test_loss += pred_loss
            test_loss += losses.elbo(
                data, x_mu_batch, posterior_params, z, labels, config
            ).item()
            # batch_size = data.shape[0]
            # data_dim = data.shape[1]
            # recon_batch = torch.empty([batch_size, data_dim])
            # for j in range(batch_size):
            #     recon_params = ()
            #     if type(gen_likelihood_params_batch) == tuple:
            #         for param_vec in gen_likelihood_params_batch:
            #             recon_params = recon_params + (param_vec[j],)
            #     else:
            #         recon_params = gen_likelihood_params_batch
            #     recon_batch[j] = sample(recon_params, gen_likelihood_type)

            # if i == 0 and epoch % config.checkpt_interval == 0:
            #     fig = plt.figure()
            #     if config.dataset_name == "experimental":
            #         ax1 = fig.add_subplot(1, 2, 1)

            #         ax2 = fig.add_subplot(1, 2, 1)

            #     if config.dataset_name == "points":
            #         ax = fig.add_subplot(1, 2, 1, projection="3d")
            #         sc = ax.scatter(
            #             data[:, 0],
            #             data[:, 1],
            #             data[:, 2],
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
            #             data[0].reshape((config.img_size, config.img_size)).cpu()
            #         )
            #         axs[1].imshow(
            #             recon_batch[0].reshape((config.img_size, config.img_size)).cpu()
            #         )
            #     else:
            #         _, axs = plt.subplots(2)
            #         axs[0].imshow(data.cpu())
            #         axs[1].imshow(recon_batch.cpu())
            #     # axs[0].set_title("original", fontsize=10)
            #     # axs[1].set_title("reconstruction", fontsize=10)
            #     plt.savefig(
            #         f"results/figures/{config.results_prefix}_recon_epoch{epoch}.png"
            #     )

    test_loss /= len(test_loader.dataset)
    print("====> Test set loss: {:.4f}".format(test_loss))
    # print("====> Test regression loss: {:.4f}".format(pred_loss))
    return test_loss
