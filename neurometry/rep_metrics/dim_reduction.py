import torch
import numpy as np

from concurrent.futures import ProcessPoolExecutor
from sklearn.metrics import pairwise_distances
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from kneed import KneeLocator
from tqdm.auto import tqdm

from .structural import TorchSKLearn


class TorchPCA(TorchSKLearn):
    def __init__(self, n_components=None, ev_threshold=None, device="cpu"):
        super().__init__(device=device)
        self.n_components = n_components
        self.ev_threshold = ev_threshold

    def fit(self, X):
        X = self.parse_input_data(X)
        self.mean = X.mean(dim=0)
        X = X - self.mean
        _, S, V = torch.svd(X)
        if self.n_components is None:
            self.n_components = X.shape[1]
        self.components_ = V[:, : self.n_components]

        ev = S**2 / (X.size(0) - 1)
        ev_ratio = ev / ev[: self.n_components].sum()
        cumulative_ev_ratio = torch.cumsum(ev_ratio, dim=0)

        if self.ev_threshold is not None:
            self.n_components = (cumulative_ev_ratio <= 0.9).sum().item()

        self.ev_ = ev[: self.n_components]
        self.ev_ratio_ = self.ev_ / ev.sum()
        self.cumulative_ev_ = torch.cumsum(self.ev_, dim=0)
        self.cumulative_ev_ratio_ = torch.cumsum(self.ev_ratio_, dim=0)
        self.total_ev_ = self.cumulative_ev_[-1].item()
        self.total_ev_ratio_ = self.cumulative_ev_ratio_[-1].item()

        if self.replace_acronyms:
            self._replace_acronyms("ev_", "explained_variance_")

        return self

    def transform(self, X):
        X = self.parse_input_data(X)
        X = X - self.mean
        return torch.mm(X, self.components_)

    def fit_transform(self, X):
        X = self.parse_input_data(X)
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        X = self.parse_input_data(X)
        return torch.mm(X, self.components_.T) + self.mean

    def get_top_n_components(self, n_components="auto", ev_threshold=0.9):
        ev_name = "explained_variance" if self.replace_acronyms else "ev"
        ev_var = f"cumulative_{ev_name}_ratio_"
        if n_components == "auto":
            n_components = (self.__dict__[ev_var] <= ev_threshold).sum().item()
        return self.components_[:, :n_components]

    def transform_by_top_n_components(self, X, n_components="auto", ev_threshold=0.9):
        X = self.parse_input_data(X)
        X = X - self.mean
        top_n_components = self.get_top_n_components(n_components, ev_threshold)
        return torch.mm(X, top_n_components)

    def to_pandas(
        self, kind="explained_variance", feature_names=None, use_acronyms=True
    ):
        if not isinstance(kind, list):
            kind = [kind]
        pandas_dataframes = []

        PCIDs = range(self.components_.shape[1])

        if "explained_variance" in kind:
            data_keys = [
                e.replace("ev", "explained_variance")
                for e in ["ev", "ev_ratio", "cumulative_ev", "cumulative_ev_ratio"]
            ]
            data_dict = {f"{key}": self.__dict__[key + "_"].cpu() for key in data_keys}
            dataframe = pd.DataFrame(data_dict)
            dataframe.insert(0, "pc_id", PCIDs)
            if use_acronyms:
                dataframe.columns = [
                    col.replace("explained_variance_", "ev_")
                    for col in dataframe.columns
                ]

            pandas_dataframes.append(dataframe)

        if "loadings" in kind:
            dataframe = pd.DataFrame(self.components_)
            dataframe.columns = [f"PC{i}" for i in PCIDs]
            if feature_names is not None:
                dataframe.index = feature_names

            pandas_dataframes.append(dataframe)

        outputs = pandas_dataframes

        return outputs[0] if len(outputs) == 1 else tuple(outputs)


def compute_power_law_index(activations, device="cuda"):
    # Perform PCA on the activations
    pca = TorchPCA(device=device).fit(activations)
    eigvals = pca.explained_variance_
    ranks = torch.arange(1, len(eigvals) + 1)

    log_ranks = torch.log(ranks.float())[:-1]
    log_eigvals = torch.log(eigvals.float())[:-1]

    knee_locator = KneeLocator(
        log_ranks.cpu(),
        log_eigvals.cpu(),
        curve="concave",
        direction="decreasing",
        online=True,
    )
    knee_x = knee_locator.knee
    knee_y = knee_locator.knee_y

    # Filter the data based on the knee point's position
    linear_region_indices = torch.where(log_ranks <= knee_x)
    log_ranks_linear = log_ranks[linear_region_indices]
    log_eigvals_linear = log_eigvals[linear_region_indices]

    # Create the design matrix and move it to the GPU
    X = torch.vstack((log_ranks_linear, torch.ones_like(log_ranks_linear))).T
    X = X.to(device)
    Y = log_eigvals_linear.view(-1, 1).to(device)

    # Solve the normal equations
    result = torch.linalg.lstsq(X, Y)

    # Extract the slope which is our alpha (power law index) and intercept
    slope, y_intercept = result.solution.cpu().squeeze().numpy()

    return slope, y_intercept, log_ranks.cpu(), log_eigvals.cpu(), knee_x, knee_y


def _state_space_pairwise_distances(X):
    distances = pairwise_distances(X, metric="sqeuclidean")
    return distances.flatten()


def _compute_stress(d0, d1):
    """Compute the stress between two distance matrices."""
    stress = np.sqrt(np.sum((d0 - d1) ** 2) / np.sum(d0**2))
    return stress


def _compute_one_dimension(args):
    X_reduced, d0, k = args
    d1 = _state_space_pairwise_distances(X_reduced[:, : k + 1])

    pearson_val = pearsonr(d0, d1)[0]
    stress_val = _compute_stress(d0, d1)

    return pearson_val, stress_val


def compute_distance_preservation(data_matrices):
    num_datasets = len(data_matrices)
    max_dim = min([X.shape[1] for X in data_matrices])

    all_d0, all_us = [], []
    for X in data_matrices:
        d0 = _state_space_pairwise_distances(X)
        all_d0.append(d0.copy())
        u, s, _ = np.linalg.svd(X, full_matrices=False)
        all_us.append(u * s)

    corrs = np.zeros((num_datasets, max_dim))
    stresses = np.zeros((num_datasets, max_dim))

    # Parallel execution
    with ProcessPoolExecutor() as executor:
        for i, X_reduced in enumerate(all_us):
            tasks = [(X_reduced, all_d0[i], k) for k in range(max_dim)]
            results = list(
                tqdm(executor.map(_compute_one_dimension, tasks), total=max_dim)
            )
            corrs[i, :], stresses[i, :] = zip(*results)

    return corrs, stresses, all_us


def get_minimal_embedding(
    corrs, stresses, all_us, plot=False, corr_threshold=0.99, stress_threshold=0.1
):
    try:
        ind_min_dim_corr = np.where(np.all(corrs > corr_threshold, axis=0))[0][0]
        print(f"correlation >= {corr_threshold} at dim {ind_min_dim_corr}")
    except:
        ind_min_dim_corr = None
        print(f"correlation >= {corr_threshold} not attainable")
    try:
        ind_min_dim_stress = np.where(np.all(stresses < stress_threshold, axis=0))[0][0]
        print(f"stress <= {stress_threshold} at dim {ind_min_dim_stress}")
    except:
        ind_min_dim_stress = None
        print(f"stress <= {stress_threshold} not attainable")

    return [us[:, :ind_min_dim_corr] for us in all_us], [
        us[:, :ind_min_dim_stress] for us in all_us
    ]
