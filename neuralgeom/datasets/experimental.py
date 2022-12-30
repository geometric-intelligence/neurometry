"""Preprocess and load experimental datasets."""


import logging
import os

import datasets.utils as utils
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

RAW_DIR = "data/raw"
BINNED_DIR = "data/binned"


def load_place_cells(expt_id=34, timestep_microsec=1000000):
    """Load pre-processed experimental place cells firings.

    Parameters
    ----------
    expt_id : int
        Index of the experiment, as conducted by Manu Madhav in 2017.
    timestep_microsec : int
        Length of time-step in microseconds.
        The preprocessing counts the number of firings in each time-window
        of length timestep_microsec.

    Returns
    -------
    place_cells : array-like, shape=[n_timesteps, n_cells]
        Number of firings at each time-steps, for each place cell.
    labels : pandas.DataFrame, shape= [n_timesteps, n_labels]
        Ground truth variables.
        Example: positional angle.
    """
    data_path = os.path.join(
        BINNED_DIR, f"expt{expt_id}_place_cells_timestep{timestep_microsec}.npy"
    )
    labels_path = os.path.join(
        BINNED_DIR, f"expt{expt_id}_labels_timestep{timestep_microsec}.txt"
    )
    times_path = os.path.join(
        BINNED_DIR, f"expt{expt_id}_times_timestep{timestep_microsec}.txt"
    )
    if os.path.exists(times_path):
        logging.info(f"# - Found file at {times_path}! Loading...")
        times = np.loadtxt(times_path)
    else:
        logging.info(f"# - No file at {times_path}. Preprocessing needed:")
        logging.info(f"Loading experiment {expt_id} to bin firing times into times...")
        expt = utils.loadmat(os.path.join(RAW_DIR, f"expt{expt_id}.mat"))
        expt = expt["x"]

        firing_times = _extract_firing_times(expt)
        times = np.arange(
            start=firing_times[0], stop=firing_times[-1], step=timestep_microsec
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
        expt = utils.loadmat(os.path.join(RAW_DIR, f"expt{expt_id}.mat"))
        expt = expt["x"]

        enc_times = expt["rosdata"]["encTimes"]
        enc_angles = expt["rosdata"]["encAngle"]
        vel = expt["rosdata"]["vel"]
        gain = expt["rosdata"]["gain"]

        rat = expt["rat"]
        day = expt["day"]

        tracking_path = os.path.join(RAW_DIR, f"{rat}-{day}_trackingResults.mat")
        if os.path.exists(tracking_path):
            logging.info(f"Found file at {data_path}! Loading...")
            tracking = utils.loadmat(tracking_path)
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
            qw = _average_in_timestep(qw, tracked_times, times)
            success = _average_in_timestep(success, tracked_times, times)

            radius2 = [xx**2 + yy**2 for xx, yy in zip(x, y)]
            angles_tracked = np.arctan2(y, x)

            quat_head = np.stack([qx, qy, qz, qw], axis=1)  # scalar-last format
            assert quat_head.shape == (len(times) - 1, 4), quat_head.shape
            rotvec_head = R.from_quat(quat_head).as_rotvec()
            assert rotvec_head.shape == (len(times) - 1, 3), rotvec_head.shape
            angles_head = np.linalg.norm(rotvec_head, axis=-1)
            angles_head = [angle % 360 for angle in angles_head]
            rx_head, ry_head, rz_head = (
                rotvec_head[:, 0],
                rotvec_head[:, 1],
                rotvec_head[:, 2],
            )

            if np.min(tracked_times) > np.max(times) or np.min(times) > np.max(
                tracked_times
            ):
                raise ValueError("Tracking times and firing times do not match.")

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
    return np.array(variable_averaged)
