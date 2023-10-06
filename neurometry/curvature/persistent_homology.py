from decorators import timer
import gph
import pandas as pd


@timer
def compute_persistence_diagrams(point_cloud, maxdim=2, n_threads=-1):
    """Compute persistence diagrams for a point cloud.

    Parameters
    ----------
    point_cloud : array-like, shape = [n_samples, n_features]
        Point cloud to compute persistence diagrams for.
    maxdim : int, optional, default: 2
        Maximum homology dimension computed.
    n_threads : int, optional, default: -1
        Number of threads to use. If -1, all available threads are used.

    Returns
    -------
    diagrams_df : pandas.DataFrame
        Dataframe containing persistence diagrams for each homology dimension.
    """

    pers = gph.ripser_parallel(X=point_cloud, maxdim=maxdim, n_threads=n_threads)
    diagrams = pers["dgms"]

    dfs = []
    for i, diagram in enumerate(diagrams):
        df = pd.DataFrame(diagram, columns=["Birth", "Death"])
        df["Dimension"] = i
        dfs.append(df)

    diagrams_df = pd.concat(dfs, ignore_index=True)

    return diagrams_df
