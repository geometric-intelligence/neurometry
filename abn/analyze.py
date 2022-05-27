"""Provide summary plots with the results.

This file provides plots that can be generated during training
to evaluate the efficiency of the methods.
"""

import matplotlib.pyplot as plt

CMAP = {
    "angles": "twilight",
    "var": "magma",
    "velocities": "viridis",
    "times": "winter",
    "gains": "cool",
    "radius2": "viridis",
    "z": "viridis",
    "qz": "twilight",
    "angle_tracked": "twilight",
    "angle_head": "twilight",
    "success": "cool",
}
plt.rcParams.update({"figure.max_open_warning": 0})


def plot_save_latent_space(fname, points, labels):
    """Plot the data projected in the latent space.

    Parameters
    ----------
    fname : str
        Filename where to save the plot.
    points : array-like, shape=[n_samples, latent_dim]
        Points, typically corresponding to the means mus of the
        latent variables.
    labels : pd.DataFrame, shape=[n_samples, n_cols]
        Labels used to color the plotted points.
        The columns are the different labels
    """
    assert len(points) == len(labels)

    latent_dim = points.shape[-1]
    label_names = list(labels.columns)
    if "Unnamed: 0" in label_names:
        label_names.remove("Unnamed: 0")
    n_labels = len(label_names)
    nrows = 2
    ncols = n_labels // 2 + 1

    if latent_dim == 2:
        fig, axs = plt.subplots(
            ncols=ncols, nrows=nrows, figsize=(5 * ncols, 4 * nrows)
        )
        for i, label_name in enumerate(label_names):
            sc = axs[i % 2, i // 2].scatter(
                points[:, 0],
                points[:, 1],
                s=5,
                c=labels[label_name],
                cmap=CMAP[label_name],
            )
            axs[i % 2, i // 2].set_title(label_name, fontsize=14)
            fig.colorbar(sc, ax=axs[i % 2, i // 2])

    elif latent_dim == 3:
        fig = plt.figure(figsize=(4 * ncols, 4 * nrows))
        for i, label_name in enumerate(label_names):
            ax = fig.add_subplot(ncols, nrows, i + 1, projection="3d")
            sc = ax.scatter(
                points[:, 0],
                points[:, 1],
                points[:, 2],
                s=5,
                c=labels[label_name],
                cmap=CMAP[label_name],
            )
            ax.set_title(label_name, fontsize=14)
            fig.colorbar(sc, ax=ax)

    plt.tight_layout()
    plt.savefig(fname)
    plt.close()
