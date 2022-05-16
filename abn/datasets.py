"""Load synthetic or real datasets."""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io
import skimage


def load_synthetic_images(n_scalars=10, n_thetas=100):
    """Load a dataset of images.

    The actions are:
    - action of SO(2): rotation
    - action of R^_+: blur

    Parameters
    ----------
    n_scalars : int
        Number of scalar used for action of scalings.
    n_thetas : int
        Number of thetas used for action of SO(2).

    Returns
    -------
    images : array-like, shape=[n_scalars * n_thetas, 128, 128]]
        Images with different orientations and blurs.
    labels : array-like, shape=[n_scalars * n_thetas, 2]]
        Labels of 2D rotation angle and blur-level sigma.
    """
    image = skimage.data.camera()
    image = skimage.transform.resize(image, (128, 128), anti_aliasing=True)

    images = []
    labels = []
    for i_theta in range(n_thetas):
        theta = 2 * np.pi * i_theta / n_thetas
        rot_image = skimage.transform.rotate(image, theta)
        for i_scalar in range(n_scalars):
            scalar = 1 + 2 * i_scalar / n_scalars
            images.append(skimage.filters.gaussian(rot_image, sigma=scalar))
            labels.append(np.array([theta, scalar]))
    return np.array(images), np.array(labels)


def load_synthetic_points(n_scalars=10, n_thetas=100):
    """Load a dataset of points in R^3.

    The actions are:
    - action of SO(2): along z-axis
    - action of R^_+

    Parameters
    ----------
    n_scalars : int
        Number of scalar used for action of scalings.
    n_thetas : int
        Number of thetas used for action of SO(2).

    Returns
    -------
    points : array-like, shape=[n_scalars * n_thetas, 3]
        Points sampled on a cone.
    labels : array-like, shape=[n_scalars * n_thetas, 2]
        Values of angles (thetas) and scaling (scalars)
        corresponding to the points.
    """
    points = []
    labels = []
    point = np.array([1, 1, 1])
    for i_theta in range(n_thetas):
        theta = 2 * np.pi * i_theta / n_thetas
        rotmat = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1.0],
            ]
        )
        rot_point = rotmat @ point
        for i_scalar in range(n_scalars):
            scalar = 1 + i_scalar
            points.append(scalar * rot_point)
            labels.append(np.array([theta, scalar]))

    return np.array(points), np.array(labels)


def load_synthetic_place_cells(n_times=10000, n_cells=40):
    """Load synthetic place cells.

    This is a dataset of synthetic place cell firings, that
    simulates a rat walking in a circle.

    Each place cell activated (2 firings) also activates
    its neighbors (1 firing each) to simulate the circular
    relationship.

    Parameters
    ----------
    n_times : int
        Number of times.
    n_cells : int
        Number of place cells.

    Returns
    -------
    place_cells : array-like, shape=[n_times, n_cells]
        Number of firings per time step and per cell.
    labels : list, length = n_times
        Angle of the rat.
    """
    n_firing_per_cell = int(n_times / n_cells)
    place_cells = []
    labels = []
    for _ in range(n_firing_per_cell):
        for i_cell in range(n_cells):
            cell_firings = np.zeros(n_cells)

            if i_cell == 0:
                cell_firings[-2] = np.random.poisson(1.0)
                cell_firings[-1] = np.random.poisson(2.0)
                cell_firings[0] = np.random.poisson(4.0)
                cell_firings[1] = np.random.poisson(2.0)
                cell_firings[2] = np.random.poisson(1.0)
            elif i_cell == 1:
                cell_firings[-1] = np.random.poisson(1.0)
                cell_firings[0] = np.random.poisson(2.0)
                cell_firings[1] = np.random.poisson(4.0)
                cell_firings[2] = np.random.poisson(2.0)
                cell_firings[3] = np.random.poisson(1.0)
            elif i_cell == n_cells - 2:
                cell_firings[-4] = np.random.poisson(1.0)
                cell_firings[-3] = np.random.poisson(2.0)
                cell_firings[-2] = np.random.poisson(4.0)
                cell_firings[-1] = np.random.poisson(2.0)
                cell_firings[0] = np.random.poisson(1.0)
            elif i_cell == n_cells - 1:
                cell_firings[-3] = np.random.poisson(1.0)
                cell_firings[-2] = np.random.poisson(2.0)
                cell_firings[-1] = np.random.poisson(4.0)
                cell_firings[0] = np.random.poisson(2.0)
                cell_firings[1] = np.random.poisson(1.0)
            else:
                cell_firings[i_cell - 2] = np.random.poisson(1.0)
                cell_firings[i_cell - 1] = np.random.poisson(2.0)
                cell_firings[i_cell] = np.random.poisson(4.0)
                cell_firings[i_cell + 1] = np.random.poisson(2.0)
                cell_firings[i_cell - 3] = np.random.poisson(1.0)
            place_cells.append(cell_firings)
            labels.append(i_cell / n_cells * 360)

    return np.array(place_cells), labels


def load_place_cells(expt_id=34, timestep_ns=1000000):
    """Load pre-processed experimental place cells firings.

    Parameters
    ----------
    expt_id : int
        Index of the experiment, as conducted by Manu Madhav in 2017.
    timestep_ns : int
        Length of time-step in nanoseconds.
        The preprocessing counts the number of firings in each time-window
        of length timestep_ns.

    Returns
    -------
    place_cells : array-like, shape=[n_timesteps, n_cells]
        Number of firings at each time-steps, for each place cell.
    """
    data_path = f"data/place_cells_expt{expt_id}_timestep{timestep_ns}.npy"
    labels_path = f"data/place_cells_labels_expt{expt_id}_timestep{timestep_ns}.txt"

    if not os.path.exists(data_path) or not os.path.exists(labels_path):
        print(f"Loading experiment {expt_id}...")
        expt = loadmat(f"data/expt{expt_id}.mat")
        expt = expt[f"expt{expt_id}"]

        firing_times = _extract_firing_times(expt)
        times = np.arange(
            start=firing_times[0], stop=firing_times[-1], step=timestep_ns
        )

    if os.path.exists(data_path):
        print(f"Found file at {data_path}!")
        place_cells = np.load(data_path)

    else:
        n_timesteps = len(times) - 1
        n_cells = len(expt["clust"])
        print(f"Number of cells: {n_cells}")
        place_cells = np.zeros((n_timesteps, n_cells))
        for i_cell, cell in enumerate(expt["clust"]):
            print(f"Counting firings per time-step in cell {i_cell}...")
            counts, bins, _ = plt.hist(cell["ts"], bins=times)
            assert sum(bins != times) == 0
            assert len(counts) == n_timesteps
            place_cells[:, i_cell] = counts

        print(f"Saving to {data_path}...")
        np.save(data_path, place_cells)

    if os.path.exists(labels_path):
        print(f"Found file at {labels_path}!")
        labels = pd.read_csv(labels_path)

    else:
        expt = loadmat(f"data/expt{expt_id}.mat")
        expt = expt[f"expt{expt_id}"]

        enc_times = expt["rosdata"]["encTimes"]
        enc_angles = expt["rosdata"]["encAngle"]
        vel = expt["rosdata"]["vel"]
        gain = expt["rosdata"]["gain"]

        print("Averaging variables angle, velocity and gain per time-step...")

        angles = _average_in_timestep(enc_angles, enc_times, times)
        angles = [angle % 360 for angle in angles]
        velocities = _average_in_timestep(vel, enc_times, times)
        gains = _average_in_timestep(gain, enc_times, times)

        labels = pd.DataFrame(
            {
                "times": times[:-1],
                "angles": angles,
                "velocities": velocities,
                "gains": gains,
            }
        )

        print(f"Saving to {labels_path}...")
        labels.to_csv(labels_path)

    return place_cells, labels


def _extract_firing_times(expt):
    """Extract firing times for all cells in the experiment.

    Parameters
    ----------
    expt : dict
        Dictionnary summarizing the experiment, with:
            key: clust
    """
    times = []
    for cell in expt["clust"]:
        times.extend(cell["ts"])

    times = sorted(times)
    n_times = len(times)
    print(f"Nb of firing times (all cells) before deleting duplicates: {n_times}.")
    aux = []
    for time in times:
        if time not in aux:
            aux.append(time)
    n_times = len(aux)
    print(f"Nb of firing times  (all cells) after deleting duplicates: {n_times}.")
    return aux


def _average_in_timestep(variable_to_average, variable_times, times):
    """Average values of recorded variables for each time-step.

    Parameters
    ----------
    variable_to_average : array-like, shape=[n_times,]
        Values of the variable to average.
    variable_times : array-like, shape=[n_times,]
        Times at which the variable has been recorded.
    times : array-like, shape=[n_new_times,]
        Times at which the variable is resampled.

    Returns
    -------
    variable_averaged : array-like, shape=[n_new_times]
        Values of the variable at times.
    """
    counts, bins, _ = plt.hist(variable_times, bins=times)
    assert len(counts) == len(times) - 1, len(counts)
    assert len(bins) == len(times), len(bins)

    variable_averaged = []
    cum_count = 0
    for count in counts:
        averaged = np.mean(variable_to_average[cum_count : cum_count + int(count)])
        variable_averaged.append(averaged)
        cum_count += int(count)
    assert len(variable_averaged) == len(times) - 1
    return variable_averaged


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

    data = scipy.io.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)
