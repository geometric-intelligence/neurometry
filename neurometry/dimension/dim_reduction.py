import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


def plot_pca_projections(X, K, title):
    pca = PCA(n_components=K)
    pca.fit(X)
    X_pca = pca.transform(X)
    ev = pca.explained_variance_ratio_

    # Create plot with KxK subplots
    fig, axes = plt.subplots(K, K, figsize=(3 * K, 3 * K))

    for i in range(K):
        for j in range(K):
            if i <= j:
                axes[i, j].axis("off")  # Turn off plot for upper triangle
            else:
                if i != j:
                    axes[i, j].scatter(X_pca[:, i], X_pca[:, j], alpha=0.5)
                    axes[i, j].set_xlabel(f"PC {i+1}")
                    axes[i, j].set_ylabel(f"PC {j+1}")
                    axes[i, j].set_title(
                        f"PCs {i+1} & {j+1}; {100*(ev[i]+ev[j]):.1f}% exp. var."
                    )

    plt.tight_layout()

    fig.suptitle(title, fontsize=30, fontweight="bold", verticalalignment="top")
    plt.show()

    print(f"The {K} top PCs explain {100*np.cumsum(ev)[-1]:.2f}% of the variance")

    return fig


