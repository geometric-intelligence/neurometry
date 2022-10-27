"""Fit a VAE to place cells."""


import copy

import losses
import torch
import wandb


def train_test(model, train_loader, test_loader, optimizer, scheduler, config):
    train_losses = []
    test_losses = []
    lowest_test_loss = 1000
    for epoch in range(1, config.n_epochs + 1):

        train_loss = train_one_epoch(
            epoch=epoch,
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            config=config,
        )

        train_losses.append(train_loss)

        test_loss = test_one_epoch(model=model, test_loader=test_loader, config=config)

        if config.scheduler == "True":
            scheduler.step(test_loss)

        test_losses.append(test_loss)

        if test_loss < lowest_test_loss:
            lowest_test_loss = test_loss
            best_model = copy.deepcopy(model)

        wandb.log({"train_loss": train_loss, "test_loss": test_loss}, step=epoch)

    return train_losses, test_losses, best_model


def train_one_epoch(epoch, model, train_loader, optimizer, config):
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

        labels = labels  # .float()
        data = data.to(config.device)
        labels = labels.to(config.device)
        optimizer.zero_grad()
        x_mu_batch, posterior_params = model(data)
        z, _, _ = model.reparameterize(posterior_params)

        loss = losses.elbo(data, x_mu_batch, posterior_params, z, labels, config)

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

    train_loss = train_loss / len(train_loader.dataset)

    print("====> Epoch: {} Average loss: {:.4f}".format(epoch, train_loss))
    return train_loss


def test_one_epoch(model, test_loader, config):
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
            z, _, _ = model.reparameterize(posterior_params)

            test_loss += losses.elbo(
                data, x_mu_batch, posterior_params, z, labels, config
            ).item()

    test_loss /= len(test_loader.dataset)
    print("====> Test set loss: {:.4f}".format(test_loss))
    return test_loss
