import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE, Isomap
from umap import UMAP

def plot_2d_manifold_projections(
    data1, data2, dataset_name_1="Dataset 1", dataset_name_2="Dataset 2", n_components=2
):
    models = {
        "Isomap": Isomap(n_components=n_components),
        "UMAP": UMAP(n_components=n_components),
        "MDS": MDS(n_components=n_components),
        "t-SNE": TSNE(n_components=n_components, learning_rate="auto", init="random"),
    }

    fig, axes = plt.subplots(4, 2, figsize=(10, 20))

    for i, (name, model) in enumerate(models.items()):
        data1_2d = model.fit_transform(data1)
        data2_2d = model.fit_transform(data2)

        combined_data = np.vstack((data1_2d, data2_2d))
        x_min, x_max = combined_data[:, 0].min(), combined_data[:, 0].max()
        y_min, y_max = combined_data[:, 1].min(), combined_data[:, 1].max()

        def plot_density(ax, data, cmap):
            x = data[:, 0]
            y = data[:, 1]
            xy = np.vstack([x, y])
            kde = gaussian_kde(xy)
            density = kde(xy)
            sc = ax.scatter(x, y, c=density, s=50, cmap=cmap)
            ax.set_xlim(x_min-2, x_max+2)
            ax.set_ylim(y_min-2, y_max+2)
            return sc

        plot_density(axes[i, 0], data1_2d, cmap="Blues")
        axes[i, 0].set_title(f"{name} - {dataset_name_1}")

        plot_density(axes[i, 1], data2_2d, cmap="Reds")
        axes[i, 1].set_title(f"{name} - {dataset_name_2}")

    plt.tight_layout()
    plt.show()

    return fig


def plot_pca_projections(X1, X2, dataset_name_1, dataset_name_2, K):
    pca1 = PCA(n_components=K)
    pca1.fit(X1)
    X1_pca = pca1.transform(X1)
    ev1 = pca1.explained_variance_ratio_

    pca2 = PCA(n_components=K)
    pca2.fit(X2)
    X2_pca = pca2.transform(X2)
    ev2 = pca2.explained_variance_ratio_

    fig, axes = plt.subplots(
        K, K * 2, figsize=(8 * K, 4 * K), sharex="col", sharey="row", dpi=100
    )

    def plot_pca_comparison(X_pca, ev, axes_row, cmap):
        for i in range(K):
            for j in range(K):
                ax = axes[i, axes_row + j]
                if i <= j:
                    ax.axis("off")
                else:
                    x = X_pca[:, j]
                    y = X_pca[:, i]
                    xy = np.vstack([x, y])
                    kde = gaussian_kde(xy)
                    density = kde(xy)
                    ax.scatter(x, y, c=density, s=50, cmap=cmap)
                    ax.set_xlabel(f"PC {j+1}", fontsize=16)
                    ax.set_ylabel(f"PC {i+1}", fontsize=16)
                    ax.set_title(f"{100*(ev[i]+ev[j]):.1f}% exp. var.", fontsize=24)
                    ax.set_aspect("equal")

    plot_pca_comparison(X1_pca, ev1, 0, cmap="Blues")
    plot_pca_comparison(X2_pca, ev2, K, cmap="Reds")

    x_min = np.min([X1_pca[:, :K].min(), X2_pca[:, :K].min()])
    x_max = np.max([X1_pca[:, :K].max(), X2_pca[:, :K].max()])
    y_min = np.min([X1_pca[:, :K].min(), X2_pca[:, :K].min()])
    y_max = np.max([X1_pca[:, :K].max(), X2_pca[:, :K].max()])

    for i in range(K):
        for j in range(K * 2):
            if i > j % K:
                axes[i, j].set_xlim(x_min-1, x_max+1)
                axes[i, j].set_ylim(y_min-1, y_max+1)

    fig.subplots_adjust(top=1)
    fig.suptitle(
        f"PC projections - {dataset_name_1} (left) vs {dataset_name_2} (right)",
        fontsize=30,
        fontweight="bold",
        verticalalignment="top",
    )

    plt.show()

    print(
        f"The {K} top PCs in {dataset_name_1} explain {100*np.cumsum(ev1)[-1]:.2f}% of the variance"
    )
    print(
        f"The {K} top PCs in {dataset_name_2} explain {100*np.cumsum(ev2)[-1]:.2f}% of the variance"
    )

    return fig


