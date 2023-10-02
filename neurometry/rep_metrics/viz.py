import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
from PIL import Image
import cortex



def plot_pca_spectrum(region, slope, y_intercept, log_ranks, log_eigvals, knee_x, knee_y):
    # Create a scatter plot of the log-ranks and log-eigenvalues
    plt.scatter(log_ranks, log_eigvals, label="Data", alpha=0.5)

    # Plot the line corresponding to the computed power law index
    plt.plot(
        log_ranks,
        slope * log_ranks + y_intercept,
        color="red",
        label=f"Power law index: {slope:.2f}",
    )

    plt.xlabel("Log Rank")
    plt.ylabel("Log Eigenvalue")
    plt.title(f"Log-Log Plot of Eigenvalues vs. Ranks in region {region}")

    # Show point corresponding to end of linear (power law) regime
    if knee_x is not None and knee_y is not None:
        plt.plot(knee_x, knee_y, "ro", markersize=10, label="Linear regime")

    plt.legend()
    plt.show()


def plot_pairwise_dis_matrix(pairwise_dis_matrix, functional_rois, method):
    plt.imshow(pairwise_dis_matrix)
    plt.xticks(ticks=np.arange(len(functional_rois)), labels=functional_rois, rotation=90)
    plt.yticks(ticks=np.arange(len(functional_rois)), labels=functional_rois)
    if method == "RSA": 
        matrix_type = "Dissimilarity"
    if method == "Shape":
        matrix_type = "Distance"
    plt.title(f"{method} Pairwise {matrix_type} Matrix")
    cbar = plt.colorbar()
    cbar.set_label(f"{method}-based {matrix_type}")


def plot_pca_distortions(corrs, stresses, corrs_dim=None, stresses_dim=None):
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    if corrs_dim is not None:
        axs[0].set_xlim(-1,corrs_dim+5)
        axs[0].axvline(x=corrs_dim, color="red",linestyle="--")
    axs[0].plot(corrs.T)
    axs[0].grid()
    axs[0].set_xlabel("Dimensions kept")
    axs[0].set_ylabel("Pairwise Distances Correlation \n(wrt original distances)")
    axs[0].set_title("Correlation")

    if stresses_dim is not None:
        axs[1].set_xlim(-1,stresses_dim+5)
        axs[1].axvline(x=stresses_dim, color="red",linestyle="--")
    axs[1].plot(stresses.T)
    axs[1].grid()
    axs[1].set_xlabel("Dimensions kept")
    axs[1].set_ylabel("Pairwise Distances Stress \n(wrt original distances)")
    axs[1].set_title("Stress")
    plt.tight_layout()
    plt.show()


def plot_ROI_flatmap(
    voxel_data,
    subj,
    mapper="nearest",
    rois=False,
    roi_labels=False,
    title=None,
    vmin=None,
    vmax=None,
    cmap="inferno",
    figsize=(20, 10),
    dpi=100,
    colorbar=True,
):
    voxel_data = np.swapaxes(voxel_data, 0, 2)
    plot_data = cortex.Volume(
        voxel_data,
        subj,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        xfmname="func1pt8_to_anat0pt8_autoFSbbr",
    )
    fig = plt.figure(figsize=figsize)

    cortex_handle = cortex.quickshow(
        plot_data,
        fig=fig,
        dpi=dpi,
        recache=False,
        with_rois=rois,
        with_labels=roi_labels,
        with_curvature=True,
        curvature_contrast=0.1,
        curvature_brightness=0.9,
        with_colorbar=colorbar,
    )

    if title:
        plt.title(title)

    return plot_data, fig

def convert_figure_to_image(fig, dpi=300):
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi)
    buffer.seek(0)

    # Load image from buffer; convert to RGB
    return Image.open(buffer).convert("RGB")