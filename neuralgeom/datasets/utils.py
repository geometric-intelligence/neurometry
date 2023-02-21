"""Utils to import data from matlab."""

import datasets.experimental
import datasets.synthetic
import mat73
import numpy as np
import scipy.io
import torch
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA


def load(config):
    """Load dataset according to configuration in config.

    Parameters
    ----------
    config

    Returns
    -------
    dataset_torch : torch.Tensor
        Dataset (without labels) as a torch tensor.
        Each row represents one data point.
    labels : pd.DataFrame
        Dataframe of labels corresponding to dataset_torch
    train_loader : torch.DataLoader
        Loader that yields minibatches (data and labels) from the
        train dataset.
    test_loader : torch.DataLoader
        Loader that yields minibatches (data and labels) from the
        test dataset.
    """
    if config.dataset_name == "experimental":
        dataset, labels = datasets.experimental.load_place_cells(
            expt_id=config.expt_id, timestep_microsec=config.timestep_microsec
        )
        dataset = dataset[labels["velocities"] > 5]
        labels = labels[labels["velocities"] > 5]
        dataset = np.log(dataset.astype(np.float32) + 1)

        if config.smooth == True:
            dataset_smooth = np.zeros_like(dataset)
            for _ in range(dataset.shape[1]):
                dataset_smooth[:, _] = savgol_filter(
                    dataset[:, _], window_length=40, polyorder=2
                )
            dataset = dataset_smooth
        dataset = (dataset - np.min(dataset)) / (np.max(dataset) - np.min(dataset))

        gain_counts = labels["gains"].value_counts()
        gain_1 = 1
        one_gain = labels["gains"].value_counts().is_unique
        if one_gain:
            print(f"The dataset contains only one gain value: {gain_counts.index[0]}")
            gain = gain_1
        else:
            print(
                "The dataset transitions between two gains:"
                f" {gain_counts.index[0]:4f} and {gain_counts.index[1]:4f}."
            )
            if config.select_gain_1:
                gain = gain_1
                print(f"We select gain 1: gain = {gain_1}.")
            else:
                other_gain = gain_counts.index[0]
                if other_gain == gain_1:
                    other_gain = gain_counts.index[1]
                gain = other_gain
                print(f"We select the other gain: gain = {other_gain:4f}.")
        config.update({"gain": gain})
        dataset = dataset[labels["gains"] == gain]
        labels = labels[labels["gains"] == gain]

    elif config.dataset_name == "synthetic":
        dataset, labels = datasets.synthetic.load_place_cells()
        dataset = np.log(dataset.astype(np.float32) + 1)
        dataset = (dataset - np.min(dataset)) / (np.max(dataset) - np.min(dataset))
    elif config.dataset_name == "images":
        dataset, labels = datasets.synthetic.load_images(img_size=config.img_size)
        dataset = (dataset - np.min(dataset)) / (np.max(dataset) - np.min(dataset))
        height, width = dataset.shape[1:3]
        dataset = dataset.reshape((-1, height * width))
    elif config.dataset_name == "projected_images":
        dataset, labels = datasets.synthetic.load_projected_images(
            img_size=config.img_size
        )
        dataset = (dataset - np.min(dataset)) / (np.max(dataset) - np.min(dataset))
    elif config.dataset_name == "points":
        dataset, labels = datasets.synthetic.load_points()
        dataset = dataset.astype(np.float32)
    elif config.dataset_name == "s1_synthetic":
        dataset, labels = datasets.synthetic.load_s1_synthetic(
            synthetic_rotation=config.synthetic_rotation,
            n_times=config.n_times,
            radius=config.radius,
            n_wiggles=config.n_wiggles,
            distortion_amp=config.distortion_amp,
            embedding_dim=config.embedding_dim,
            noise_var=config.noise_var,
            distortion_func=config.distortion_func,
        )
    elif config.dataset_name == "s2_synthetic":
        dataset, labels = datasets.synthetic.load_s2_synthetic(
            synthetic_rotation=config.synthetic_rotation,
            n_times=config.n_times,
            radius=config.radius,
            distortion_amp=config.distortion_amp,
            embedding_dim=config.embedding_dim,
            noise_var=config.noise_var,
        )
    elif config.dataset_name == "t2_synthetic":
        dataset, labels = datasets.synthetic.load_t2_synthetic(
            synthetic_rotation=config.synthetic_rotation,
            n_times=config.n_times,
            major_radius=config.major_radius,
            minor_radius=config.minor_radius,
            distortion_amp=config.distortion_amp,
            embedding_dim=config.embedding_dim,
            noise_var=config.noise_var,
        )
    elif config.dataset_name == "grid_cells":
        dataset, labels = datasets.gridcells.load_grid_cells_synthetic(
            grid_scale=config.grid_scale,
            arena_dims=config.arena_dims,
            n_cells=config.n_cells,
            grid_orientation_mean=config.grid_orientation_mean,
            grid_orientation_std=config.grid_orientation_std,
            field_width=config.field_width,
            resolution=config.resolution,
        )
    print(f"Dataset shape: {dataset.shape}.")
    if type(dataset) == np.ndarray:
        dataset_torch = torch.from_numpy(dataset)
    else:
        dataset_torch = dataset

    # dataset_torch = dataset_torch - torch.mean(dataset_torch, dim=0)

    train_num = int(round(0.7 * len(dataset)))  # 70% training
    indices = np.arange(len(dataset))

    # Note: this breaks the temporal ordering.
    train_indices = np.random.choice(indices, train_num, replace=False)
    test_indices = np.delete(indices, train_indices)

    train_dataset = dataset[train_indices]
    train_labels = labels.iloc[train_indices]

    test_dataset = dataset[test_indices]
    test_labels = labels.iloc[test_indices]

    train_dataset = dataset[train_indices]
    train_labels = labels.iloc[train_indices]
    test_dataset = dataset[test_indices]
    test_labels = labels.iloc[test_indices]

    # The angles are positional angles in the lab frame
    if config.dataset_name in ("experimental", "s1_synthetic"):
        train = []
        for d, l in zip(train_dataset, train_labels["angles"]):
            train.append([d, float(l)])
        test = []
        for d, l in zip(test_dataset, test_labels["angles"]):
            test.append([d, float(l)])
    elif config.dataset_name in ("s2_synthetic", "t2_synthetic"):
        train = []
        for d, t, p in zip(train_dataset, train_labels["thetas"], train_labels["phis"]):
            train.append([d, torch.tensor([float(t), float(p)])])
        test = []
        for d, t, p in zip(test_dataset, test_labels["thetas"], test_labels["phis"]):
            test.append([d, torch.tensor([float(t), float(p)])])

    train_loader = torch.utils.data.DataLoader(train, batch_size=config.batch_size)
    test_loader = torch.utils.data.DataLoader(test, batch_size=config.batch_size)
    return dataset_torch, labels, train_loader, test_loader


def loadmat(filename):
    """Improved loadmat (replacement for scipy.io.loadmat).

    Ensures correct loading of python dictionaries from mat files.

    Inspired by: https://stackoverflow.com/a/29126361/572908.

    Parameters
    ----------
    filename : str
        Name of the file containing matlab data.
        Example: expt34.mat
    """

    def _has_struct(elem):
        """Check if elem is an array & if its first item is a struct."""
        return (
            isinstance(elem, np.ndarray)
            and (elem.size > 0)
            and isinstance(elem[0], scipy.io.matlab.mio5_params.mat_struct)
        )

    def _check_keys(d):
        """Check if entries in dictionary are mat-objects.

        If they are mat-objects, then todict is called to change
        them to nested dictionaries.
        """
        for key in d:
            elem = d[key]
            if isinstance(elem, scipy.io.matlab.mio5_params.mat_struct):
                d[key] = _todict(elem)
            elif _has_struct(elem):
                d[key] = _tolist(elem)
        return d

    def _todict(matobj):
        """Build nested dictionaries from mat-objects.

        This is a recursive function.
        """
        d = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, scipy.io.matlab.mio5_params.mat_struct):
                d[strg] = _todict(elem)
            elif _has_struct(elem):
                d[strg] = _tolist(elem)
            else:
                d[strg] = elem
        return d

    def _tolist(ndarray):
        """Build lists from cellarays.

        This is a recursive function that constructs lists from
        cellarrays (which are loaded as numpy ndarrays).

        It is recursing into the elements if they contain matobjects.
        """
        elem_list = []
        for sub_elem in ndarray:
            if isinstance(sub_elem, scipy.io.matlab.mio5_params.mat_struct):
                elem_list.append(_todict(sub_elem))
            elif _has_struct(sub_elem):
                elem_list.append(_tolist(sub_elem))
            else:
                elem_list.append(sub_elem)
        return elem_list

    try:
        data = scipy.io.loadmat(filename, struct_as_record=False, squeeze_me=True)
    except Exception:
        data = mat73.loadmat(filename)
    return _check_keys(data)
