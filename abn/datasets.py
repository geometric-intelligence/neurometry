"""Load synthetic or real datasets."""

import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import utils


def load_place_cells(expt_id=34, time_step_ns=1000000):
    """Load pre-processed experimental place cells firings.

    Parameters
    ----------
    expt_id : int
        Index of the experiment, as conducted by Manu Madhav in 2017.
    time_step_ns : int
        Length of time-step in nanoseconds.
        The preprocessing counts the number of firings in each time-window
        of length time_step_ns.

    Returns
    -------
    place_cells : array-like, shape=[n_time_steps, n_cells]
        Number of firings at each time-steps, for each place cell.
    """
    data_path = f"data/place_cells_expt{expt_id}_{time_step_ns}.npy"
    labels_path = f"data/place_cells_labels_expt{expt_id}_{time_step_ns}.txt"

    if os.path.exists(data_path):
        place_cells = np.load(data_path)

    else:
        expt = utils.loadmat(f"data/expt{expt_id}.mat")
        expt = expt[f"expt{expt_id}"]

        firing_times = _extract_firing_times(expt)
        times = np.arange(
            start=firing_times[0], stop=firing_times[-1], step=time_step_ns
        )

        n_time_steps = len(times) - 1
        n_cells = len(expt["clust"])
        place_cells = np.zeros((n_time_steps, n_cells))
        for i_cell, cell in enumerate(expt["clust"]):
            print(f"Counting firings per time-step in cell {i_cell}...")
            counts, bins, _ = plt.hist(cell["ts"], bins=times)
            assert sum(bins != times) == 0
            assert len(counts) == n_time_steps
            place_cells[:, i_cell] = counts

        print(f"Saving to {data_path}...")
        np.save(data_path, place_cells)

    if os.path.exists(labels_path):
        labels = np.loadtxt(labels_path, skiprows=1)

    else:
        enc_times = expt["rosdata"]["encTimes"]
        enc_angles = expt["rosdata"]["encAngle"]
        vel = expt["rosdata"]["vel"]
        gain = expt["rosdata"]["gain"]

        print("Averaging variables angle, velocity and gain per time-step...")

        angles = _average_in_time_step(enc_angles, enc_times, times)
        angles = [angle % 360 for angle in angles]
        velocities = _average_in_time_step(vel, enc_times, times)
        gains = _average_in_time_step(gain, enc_times, times)

        labels = np.vstack([angles, velocities, gains]).T
        assert len(labels) == len(times)

        print(f"Saving to {labels_path}...")
        np.savetxt(labels_path, labels, header="angles,velocities,gain")

    return place_cells, labels


def _extract_firing_times(expt):
    times = []
    for cell in expt["clust"]:
        times.extend(cell["ts"])

    times = sorted(times)
    n_times = len(times)
    print(f"Nb of firing times (all units) before deleting duplicates: {n_times}.")
    aux = []
    for time in times:
        if time not in aux:
            aux.append(time)
    n_times = len(aux)
    print(f"Nb of firing times  (all units) after deleting duplicates: {n_times}.")
    return aux


def _average_in_time_step(variable_to_average, variable_times, times):
    counts, bins, _ = plt.hist(variable_times, bins=times)
    assert len(bins) == len(times) == len(counts)

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
