"""Provide summary plots with the results.

This file provides plots that can be generated during training
to evaluate the efficiency of the methods.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch


def save_latent_space(fname, model, dataset, labels):
    """Plot the data projected in the latent space.

    Parameters
    ----------
    fname : str
        Filename where to save the plot.
    model : torch.nn.Module
        VAE whose encoder projects in latent space.
    dataset : np.array, shape=[n_times, n_neurons]
        Dataset of firing neurons.
    labels : np.array, shape=[n_times,]
        Labels used to color the plotted points.
        Examples: Angles of the animal's position.
    """
    assert len(dataset) == len(labels)

    dataset = np.log(dataset.astype(np.float32) + 1)
    dataset = (dataset - np.min(dataset)) / (np.max(dataset) - np.min(dataset))
    dataset_torch = torch.tensor(dataset)

    latents_torch, _ = model.encode(dataset_torch)
    latents = latents_torch.cpu().detach().numpy()

    fig = plt.figure(figsize=(8, 8))

    if model.latent_dim == 2:
        ax = fig.add_subplot()
        ax.scatter(latents[:, 0], latents[:, 1], s=20, c=labels, cmap="twilight")
    elif model.latent_dim == 3:
        ax = fig.add_subplot(projection="3d")
        ax.scatter(
            latents[:, 0], latents[:, 1], latents[:, 2], s=20, c=labels, cmap="twilight"
        )
    plt.savefig(fname)
    plt.close()
