"""Load synthetic or real datasets."""

import logging
import os

import mat73
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io
import skimage
from scipy.spatial.transform import Rotation as R


def load_synthetic_projections(n_scalars=5, n_angles=1000, img_size=128):
    """Load a dataset of 2D images projected into 1D projections.

    The actions are:
    - action of SO(2): rotation
    - action of R^_+: blur

    Parameters
    ----------
    n_scalars : int
        Number of scalar used for action of scalings.
    n_angles : int
        Number of angles used for action of SO(2).

    Returns
    -------
    projections : array-like, shape=[n_scalars * n_angles, img_size]]
        Projections with different orientations and blurs.
    labels : pd.DataFrame, shape=[n_scalars * n_angles, 2]
        Labels organized in 2 columns: angles, and scalars.
    """
    images, labels = load_synthetic_images(
        n_scalars=n_scalars, n_angles=n_angles, img_size=img_size
    )

    projections = np.sum(images, axis=-1)
    return projections, labels


def load_synthetic_images(n_scalars=4, n_angles=2000, img_size=128):
    """Load a dataset of images.

    The actions are:
    - action of SO(2): rotation
    - action of R^_+: blur

    Parameters
    ----------
    n_scalars : int
        Number of scalar used for action of scalings.
    n_angles : int
        Number of angles used for action of SO(2).

    Returns
    -------
    images : array-like, shape=[n_scalars * n_angles, img_size, img_size]]
        Images with different orientations and blurs.
    labels : pd.DataFrame, shape=[n_scalars * n_angles, 2]
        Labels organized in 2 columns: angles, and scalars.
    """
    logging.info("Generating dataset of synthetic images.")
    image = skimage.data.camera()
    image = skimage.transform.resize(image, (img_size, img_size), anti_aliasing=True)

    images = []
    angles = []
    scalars = []
    for i_angle in range(n_angles):
        angle = 360 * i_angle / n_angles
        rot_image = skimage.transform.rotate(image, angle)
        for i_scalar in range(n_scalars):
            scalar = 1 + 0.2 * i_scalar
            blur_image = skimage.filters.gaussian(rot_image, sigma=scalar)
            noise = np.random.normal(loc=0.0, scale=0.05, size=blur_image.shape)
            images.append((blur_image + noise).astype(np.float32))
            angles.append(angle)
            scalars.append(scalar)

    labels = pd.DataFrame(
        {
            "angles": angles,
            "scalars": scalars,
        }
    )
    return np.array(images), labels


def load_synthetic_points(n_scalars=10, n_angles=100):
    """Load a dataset of points in R^3.

    The actions are:
    - action of SO(2): along z-axis
    - action of R^_+

    Parameters
    ----------
    n_scalars : int
        Number of scalar used for action of scalings.
    n_angles : int
        Number of angles used for action of SO(2).

    Returns
    -------
    points : array-like, shape=[n_scalars * n_angles, 3]
        Points sampled on a cone.
    labels : pd.DataFrame, shape=[n_scalars * n_angles, 2]
        Labels organized in 2 columns: angles, and scalars.
    """
    points = []
    angles = []
    scalars = []
    point = np.array([1, 1, 1])
    for i_angle in range(n_angles):
        angle = 2 * np.pi * i_angle / n_angles
        rotmat = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1.0],
            ]
        )
        rot_point = rotmat @ point
        for i_scalar in range(n_scalars):
            scalar = 1 + i_scalar
            points.append(scalar * rot_point)

            angles.append(angle)
            scalars.append(scalar)

    labels = pd.DataFrame(
        {
            "angles": angles,
            "scalars": scalars,
        }
    )

    return np.array(points), labels


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
    labels : pd.DataFrame, shape=[n_timess, 1]
        Labels organized in 1 column: angles.
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

    return np.array(place_cells), pd.DataFrame({"angles": labels})


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
    data_path = f"data/expt{expt_id}_place_cells_timestep{timestep_ns}.npy"
    labels_path = f"data/expt{expt_id}_labels_timestep{timestep_ns}.txt"
    times_path = f"data/expt{expt_id}_times_timestep{timestep_ns}.txt"

    if os.path.exists(times_path):
        logging.info(f"# - Found file at {times_path}! Loading...")
        times = np.loadtxt(times_path)
    else:
        logging.info(f"# - No file at {times_path}. Preprocessing needed:")
        logging.info(f"Loading experiment {expt_id} to bin firing times into times...")
        expt = loadmat(f"data/expt{expt_id}.mat")
        expt = expt["x"]

        firing_times = _extract_firing_times(expt)
        times = np.arange(
            start=firing_times[0], stop=firing_times[-1], step=timestep_ns
        )

        logging.info(f"# - Saving times of shape {times.shape} to {times_path}...")
        np.savetxt(fname=times_path, X=times)

    if os.path.exists(data_path):
        logging.info(f"# - Found file at {data_path}! Loading...")
        place_cells = np.load(data_path)

    else:
        logging.info(f"# - No file at {data_path}. Preprocessing needed:")
        n_timesteps = len(times) - 1
        n_cells = len(expt["clust"])
        logging.info(f"Number of cells: {n_cells}")
        place_cells = np.zeros((n_timesteps, n_cells))
        for i_cell, cell in enumerate(expt["clust"]):
            logging.info(f"Counting firings per time-step in cell {i_cell}...")
            counts, bins, _ = plt.hist(cell["ts"], bins=times)
            assert sum(bins != times) == 0
            assert len(counts) == n_timesteps
            place_cells[:, i_cell] = counts

        logging.info(
            f"# - Saving place_cells of shape {place_cells.shape} to {data_path}..."
        )
        np.save(data_path, place_cells)

    if os.path.exists(labels_path):
        logging.info(f"# - Found file at {labels_path}! Loading...")
        labels = pd.read_csv(labels_path)
        logging.debug(f"Labels:\n {labels}")

    else:
        logging.info(f"# - No file at {labels_path}. Preprocessing needed:")
        expt = loadmat(f"data/expt{expt_id}.mat")
        expt = expt["x"]

        enc_times = expt["rosdata"]["encTimes"]
        enc_angles = expt["rosdata"]["encAngle"]
        vel = expt["rosdata"]["vel"]
        gain = expt["rosdata"]["gain"]

        rat = expt["rat"]
        day = expt["day"]

        tracking_path = f"data/{rat}-{day}_trackingResults.mat"
        if os.path.exists(tracking_path):
            logging.info(f"Found file at {data_path}! Loading...")
            tracking = loadmat(tracking_path)
            tracking = tracking["tracked_results"]
            tracked_times, x, y, z = (
                tracking["t"],
                tracking["x"],
                tracking["y"],
                tracking["z"],
            )
            qx, qy, qz, qw = (
                tracking["qx"],
                tracking["qy"],
                tracking["qz"],
                tracking["qw"],
            )
            success = tracking["success"]

            logging.info("Averaging tracking variables per time-step...")
            x = _average_in_timestep(x, tracked_times, times)
            y = _average_in_timestep(y, tracked_times, times)
            z = _average_in_timestep(z, tracked_times, times)
            qx = _average_in_timestep(qx, tracked_times, times)
            qy = _average_in_timestep(qy, tracked_times, times)
            qz = _average_in_timestep(qz, tracked_times, times)
            success = _average_in_timestep(success, tracked_times, times)

            radius2 = x**2 + y**2
            angles_tracked = np.arctan2(y, x)
            quat_head = np.array([qx, qy, qz, qw]).T  # scalar-last format
            rotvec_head = R.from_quat(quat_head).as_rotvec()
            angles_head = np.linalg.norm(rotvec_head, axis=-1)
            angles_head = [angle % 360 for angle in angles_head]
            print("SHAPE")
            print(rotvec_head.shape)
            rx_head, ry_head, rz_head = (
                rotvec_head[:, 0],
                rotvec_head[:, 1],
                rx_head,
            ) = rotvec_head[:, 2]

            logging.warning("(Min, Max) of tracked_times:")
            logging.warning(f"{np.min(tracked_times):.3e}, {np.max(tracked_times):.3e}")
            logging.warning("(Min, Max) of enc_times:")
            logging.warning(f"{np.min(enc_times):.3e}, {np.max(enc_times):.3e}")
            logging.warning("(Min, Max) of (firing) times:")
            logging.warning(f"{np.min(times):.3e}, {np.max(times):.3e}")

        else:
            logging.info(f"No file at {data_path}! Skipping...")

        logging.info("Averaging variables angle, velocity and gain per time-step...")

        angles = _average_in_timestep(enc_angles, enc_times, times)
        angles = [angle % 360 for angle in angles]
        velocities = _average_in_timestep(vel, enc_times, times)
        gains = _average_in_timestep(gain, enc_times, times)

        if os.path.exists(tracking_path):
            labels = pd.DataFrame(
                {
                    "times": times[:-1],
                    "angles": angles,
                    "velocities": velocities,
                    "gains": gains,
                    "radius2": radius2,
                    "x": x,
                    "y": y,
                    "z": z,
                    "rx_head": rx_head,
                    "ry_head": ry_head,
                    "rz_head": rz_head,
                    "angles_tracked": angles_tracked,
                    "angles_head": angles_head,
                    "success": success,
                }
            )
        else:
            labels = pd.DataFrame(
                {
                    "times": times[:-1],
                    "angles": angles,
                    "velocities": velocities,
                    "gains": gains,
                }
            )

        logging.info(
            f"# - Saving DataFrame with labels {labels.columns} to {labels_path}..."
        )
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
    # TODO: Speed this up (joblib - parallelism?)
    times = []
    for cell in expt["clust"]:
        times.extend(cell["ts"])

    times = sorted(times)
    n_times = len(times)
    logging.info(
        f"Nb of firing times (all cells) before deleting duplicates: {n_times}."
    )
    aux = []
    for time in times:
        if time not in aux:
            aux.append(time)
    n_times = len(aux)
    logging.info(
        f"Nb of firing times  (all cells) after deleting duplicates: {n_times}."
    )
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

    try:
        data = scipy.io.loadmat(filename, struct_as_record=False, squeeze_me=True)
    except Exception:
        data = mat73.loadmat(filename)
    return _check_keys(data)
