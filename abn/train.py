"""Fit a VAE to place cells."""


import losses
import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F
import torch.distributions as distributions




def sample(params, distribution):
    """Produce sample drawn from a given probability distribution. 
    Currently used to sample from learned generative model to obtain reconstruction.

    Parameters
    ----------
    distribution : string
        Specifies the type of distribution being sampled
    params : tuple
        Tuple of distirbutional parameters (e.g., (mu,logvar) for Gaussian distribution.)

    Returns
    -------
    samp : torch tensor 
        Sample from given distribution
    """
    
    if distribution == "Gaussian":
        mu, logvar = params
        var = torch.exp(logvar)
        # manually setting x_var = 0.001 (temporarily)
        var = torch.zeros(var.shape) + 1e-3
        covar_matrix = torch.diag(var)
        m = distributions.multivariate_normal.MultivariateNormal(mu,covar_matrix)
    elif distribution == "Poisson":
        lambd = params
        m = distributions.poisson.Poisson(lambd)

    samp = m.sample()
    
    return samp    



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
        data, lab = batch_data  # lab = label
        lab = lab.float()
        data = data.to(config.device)
        optimizer.zero_grad()
        gen_likelihood_params_batch, posterior_params = model(data)
        gen_likelihood_type = model.gen_likelihood_type
        posterior_type = model.posterior_type
        elbo_loss = losses.elbo(data,gen_likelihood_type, posterior_type,
        gen_likelihood_params_batch, posterior_params, beta=config.beta)
        print(type(elbo_loss))
        #print(elbo_loss)
        print(elbo_loss.shape)
        

        pred_loss = 0.0
        #TODO: replace mu with gen_likelihood_params
        # if config.with_regressor:
        #     norm = torch.unsqueeze(torch.linalg.norm(mu, dim=1), dim=1)
        #     angle_latent = mu / norm  # cos(theta), sin(theta)
        #     angle_pred = regressor(angle_latent)  # call to forward method of regressor
        #     angle_true = torch.stack([torch.cos(lab), torch.sin(lab)], axis=1)
        #     pred_loss = F.mse_loss(angle_pred, angle_true, reduction="mean")
        #     pred_loss = config.weight_regressor * pred_loss

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
            gen_likelihood_params_batch, posterior_params = model(data)
            gen_likelihood_type = model.gen_likelihood_type
            posterior_type = model.posterior_type

            pred_loss = 0.0
            #TODO: replace mu with gen_likelihood_params
            # if config.with_regressor:
            #     norm = torch.unsqueeze(torch.linalg.norm(mu, dim=1), dim=1)
            #     angle_latent = mu / norm
            #     angle_pred = regressor(angle_latent)
            #     angle_true = torch.stack([torch.cos(lab), torch.sin(lab)], axis=1)
            #     pred_loss = F.mse_loss(angle_pred, angle_true)
            #     pred_loss = config.weight_regressor * pred_loss

            test_loss += pred_loss
            test_loss += losses.elbo(data, gen_likelihood_type, posterior_type, 
            gen_likelihood_params_batch,posterior_params, beta=config.beta).item()
            batch_size = data.shape[0]
            data_dim = data.shape[1]
            recon_batch = torch.empty([batch_size,data_dim])
            for j in range(batch_size):
                recon_params = ()
                for param_vec in gen_likelihood_params_batch:
                    recon_params = recon_params + (param_vec[j],)
                recon_batch[j] = sample(recon_params,gen_likelihood_type)

            if i == 0 and epoch % config.checkpt_interval == 0:
                fig = plt.figure()
                if config.dataset_name == "points":
                    ax = fig.add_subplot(1,2,1,projection="3d")
                    sc = ax.scatter(
                        data[:, 0],
                        data[:, 1],
                        data[:, 2],
                        s=10,
                        #c=lab[label_name],
                        #cmap=CMAP[label_name],
                    )
                    plt.xlim(-1.5, 1.5)
                    plt.ylim(-1.5, 1.5)
                    ax.set_zlim(0,1.5)
                    ax.set_title("original")
                    ax2 = fig.add_subplot(1,2,2,projection="3d")
                    sc2 = ax2.scatter(
                        recon_batch[:, 0],
                        recon_batch[:, 1],
                        recon_batch[:, 2],
                        s=10,
                        #c=lab[label_name],
                        #cmap=CMAP[label_name],
                    )
                    plt.xlim(-1.5, 1.5)
                    plt.ylim(-1.5, 1.5)
                    ax2.set_zlim(0,1.5)
                    ax2.set_title("reconstruction")
                elif config.dataset_name == "images":
                    _, axs = plt.subplots(2)
                    axs[0].imshow(
                        data[0].reshape((config.img_size, config.img_size)).cpu()
                    )
                    axs[1].imshow(
                        recon_batch[0].reshape((config.img_size, config.img_size)).cpu()
                    )
                else:
                    axs[0].imshow(data.cpu())
                    axs[1].imshow(recon_batch.cpu())
                #axs[0].set_title("original", fontsize=10)
                #axs[1].set_title("reconstruction", fontsize=10)
                plt.savefig(f"{config.results_prefix}_recon_epoch{epoch}.png")

    test_loss /= len(test_loader.dataset)
    print("====> Test set loss: {:.4f}".format(test_loss))
    print("====> Test regression loss: {:.4f}".format(pred_loss))
    return test_loss
            
"""             if i == 0 and epoch % config.checkpt_interval == 0:
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
                plt.savefig(f"{config.results_prefix}_recon_epoch{epoch}.png") """


