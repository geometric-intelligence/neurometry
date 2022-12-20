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
        print(labels)
        dataset = dataset[labels["velocities"] > 5]
        labels = labels[labels["velocities"] > 5]
        dataset = np.log(dataset.astype(np.float32) + 1)


        if labels["gains"].value_counts().is_unique:
            two_gains = False
            print("the dataset contains only one gain value")
        else:
            two_gains = True
            print("the dataset contains more than one gain value")
            gain1 = 1
            gain2 = labels["gains"].value_counts().index[0]
            if gain2 == gain1:
                gain2 = labels["gains"].value_counts().index[1]
            dataset_gain1 = dataset[labels["gains"] == gain1]
            labels_gain1 = labels[labels["gains"] == gain1]
            dataset_gain2 = dataset[labels["gains"] == gain2]
            labels_gain2 = labels[labels["gains"] == gain2]

        if config.smooth == True:
            dataset_smooth = np.zeros_like(dataset)
            for _ in range(dataset.shape[1]):
                dataset_smooth[:, _] = savgol_filter(
                    dataset[:, _], window_length=40, polyorder=2
                )
            dataset = dataset_smooth
        dataset = (dataset - np.min(dataset)) / (np.max(dataset) - np.min(dataset))
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
            n_times=config.n_times,
            radius=config.radius,
            n_wiggles=config.n_wiggles,
            distortion_amp=config.distortion_amp,
            embedding_dim=config.embedding_dim,
            noise_var=config.noise_var,
            distortion_func=config.distortion_func,
            rot=config.synthetic_rotation,
        )
    elif config.dataset_name == "s2_synthetic":
        dataset, labels = datasets.synthetic.load_s2_synthetic(
            rot=config.synthetic_rotation,
            n_times=config.n_times,
            radius=config.radius,
            distortion_amp=config.distortion_amp,
            embedding_dim=config.embedding_dim,
            noise_var=config.noise_var,
        )
    elif config.dataset_name == "t2_synthetic":
        dataset, labels = datasets.synthetic.load_t2_synthetic(
            rot=config.synthetic_rotation,
            n_times=config.n_times,
            major_radius=config.major_radius,
            minor_radius=config.minor_radius,
            distortion_amp=config.distortion_amp,
            embedding_dim=config.embedding_dim,
            noise_var=config.noise_var,
        )

    print(f"Dataset shape: {dataset.shape}.")
    if type(dataset) == np.ndarray:
        dataset_torch = torch.from_numpy(dataset)
    else:
        dataset_torch = dataset

    # dataset_torch = dataset_torch - torch.mean(dataset_torch, dim=0)

    train_num = int(round(0.7 * len(dataset)))  # 70% training
    indeces = np.arange(len(dataset))
    train_indeces = np.random.choice(indeces, train_num, replace=False)
    test_indeces = np.delete(indeces, train_indeces)

    train_dataset = dataset[train_indeces]
    train_labels = labels.iloc[train_indeces]

    test_dataset = dataset[test_indeces]
    test_labels = labels.iloc[test_indeces]
    if config.dataset_name == "experimental" and two_gains:
        train_num_gain1 = int(round(0.8 * len(dataset_gain1)))
        train_num_gain2 = int(round(0.8 * len(dataset_gain2)))
        train_dataset_gain1 = dataset_gain1[:train_num_gain1]
        train_labels_gain1 = labels_gain1.iloc[:train_num_gain1]
        test_dataset_gain1 = dataset_gain1[train_num_gain1:]
        test_labels_gain1 = labels_gain1.iloc[train_num_gain1:]
        train_dataset_gain2 = dataset_gain2[:train_num_gain2]
        train_labels_gain2 = labels_gain2.iloc[:train_num_gain2]
        test_dataset_gain2 = dataset_gain2[train_num_gain2:]
        test_labels_gain2 = labels_gain2.iloc[train_num_gain2:]
    else:
        train_dataset = dataset[train_indeces]
        train_labels = labels.iloc[train_indeces]
        test_dataset = dataset[test_indeces]
        test_labels = labels.iloc[test_indeces]
        # train_dataset = dataset[0:round(0.7*len(dataset))]
        # train_labels = labels.iloc[0:round(0.7*len(dataset))]
        # test_dataset = dataset[round(0.7*len(dataset)):]
        # test_labels = labels.iloc[round(0.7*len(dataset)):]
    if config.dataset_name == "experimental":
        if two_gains:
            train_gain1 = []
            for d, l in zip(
                train_dataset_gain1, train_labels_gain1["angles"]
            ):  # angles : positional angles
                train_gain1.append([d, float(l)])
            test_gain1 = []
            for d, l in zip(test_dataset_gain1, test_labels_gain1["angles"]):
                test_gain1.append([d, float(l)])
            train_gain2 = []
            for d, l in zip(
                train_dataset_gain2, train_labels_gain2["angles"]
            ):
                train_gain2.append([d, float(l)])
            test_gain2 = []
            for d, l in zip(test_dataset_gain2, test_labels_gain2["angles"]):
                test_gain2.append([d, float(l)])
    
            train_loader_gain1 = torch.utils.data.DataLoader(train_gain1, batch_size=config.batch_size)
            test_loader_gain1 = torch.utils.data.DataLoader(test_gain1, batch_size=config.batch_size)
            train_loader_gain2 = torch.utils.data.DataLoader(train_gain2, batch_size=config.batch_size)
            test_loader_gain2 = torch.utils.data.DataLoader(test_gain2, batch_size=config.batch_size)
            return dataset_gain1, labels_gain1, dataset_gain2, labels_gain2, train_loader_gain1, test_loader_gain1, train_loader_gain2, test_loader_gain2
        else:
            train = []
            for d, l in zip(train_dataset, train_labels["angles"]):
                train.append([d, float(l)])
            test = []
            for d, l in zip(test_dataset, test_labels["angles"]):
                test.append([d, float(l)])
    elif config.dataset_name =="s1_synthetic":
        train = []
        for d, l in zip(
            train_dataset, train_labels["angles"]
        ):  # angles : positional angles
            train.append([d, float(l)])
        test = []
        for d, l in zip(test_dataset, test_labels["angles"]):
            test.append([d, float(l)])
    elif config.dataset_name in ("s2_synthetic", "t2_synthetic"):
        train = []
        for d, t, p in zip(
            train_dataset, train_labels["thetas"], train_labels["phis"]
        ):  # angles : positional angles
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
