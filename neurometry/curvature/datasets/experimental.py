"""Preprocess and load experimental datasets."""


import logging
import os

import datasets.utils as utils
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

WORK_DIR = os.getcwd()
RAW_DIR = os.path.join(WORK_DIR, "data/raw")
BINNED_DIR = os.path.join(WORK_DIR, "data/binned")


def load_neural_activity(expt_id=34, vel_threshold=5, timestep_microsec=int(1e5)):
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
    neural_activity : array-like, shape=[n_timesteps, n_cells]
        Number of firings at each time-steps, for each place cell.
    labels : pandas.DataFrame, shape= [n_timesteps, n_labels]
        Ground truth variables.
        Example: positional angle.
    """
    neural_data_path = os.path.join(
        BINNED_DIR,
        f"expt{expt_id}_neural_activity_timestep{timestep_microsec}_velthreshold_{vel_threshold}.npy",
    )
    labels_path = os.path.join(
        BINNED_DIR,
        f"expt{expt_id}_labels_timestep{timestep_microsec}_velthreshold_{vel_threshold}.txt",
    )
    times_path = os.path.join(
        BINNED_DIR,
        f"expt{expt_id}_times_timestep{timestep_microsec}_velthreshold_{vel_threshold}.txt",
    )

    if not (
        os.path.exists(neural_data_path)
        and os.path.exists(labels_path)
        and os.path.exists(times_path)
    ):
        expt = utils.loadmat(os.path.join(RAW_DIR, f"expt{expt_id}.mat"))
        period_start_times, period_end_times, _ = _apply_velocity_threshold(
            expt, vel_threshold
        )
        sampling_times = _get_sampling_times(
            period_start_times, period_end_times, timestep_microsec
        )

    if os.path.exists(times_path):
        logging.info(f"# - Found file at {times_path}! Loading...")
        all_times = np.loadtxt(times_path)
    else:
        logging.info(f"# - No file at {times_path}. Preprocessing needed:")
        logging.info(f"Loading experiment {expt_id} to bin firing times into times...")

        recorded_times = expt["x"]["rosdata"]["encTimes"]

        all_times = _average_variable(recorded_times, recorded_times, sampling_times)

        logging.info(f"# - Saving times of shape {all_times.shape} to {times_path}...")
        np.savetxt(fname=times_path, X=all_times)

    if os.path.exists(neural_data_path):
        logging.info(f"# - Found file at {neural_data_path}! Loading...")
        neural_activity = np.load(neural_data_path)

    else:
        logging.info(f"# - No file at {neural_data_path}. Preprocessing needed:")

        neural_activity = []

        for _, neuron in enumerate(expt["x"]["clust"]):
            neuron_i_activity = []
            for times in sampling_times:
                spike_count, _ = np.histogram(neuron["ts"], bins=times)
                neuron_i_activity.extend(spike_count)

            neural_activity.append(neuron_i_activity)

        neural_activity = np.array(neural_activity).T
        neural_activity = neural_activity / (timestep_microsec / 1e6)

        logging.info(
            f"# - Saving neural_activity of shape {neural_activity.shape} to {neural_data_path}..."
        )
        np.save(neural_data_path, neural_activity)

    if os.path.exists(labels_path):
        logging.info(f"# - Found file at {labels_path}! Loading...")
        labels = pd.read_csv(labels_path)
        logging.debug(f"Labels:\n {labels}")

    else:
        logging.info(f"# - No file at {labels_path}. Preprocessing needed:")

        recorded_times = expt["x"]["rosdata"]["encTimes"]
        recorded_angles = expt["x"]["rosdata"]["encAngle"]
        recorded_vel = expt["x"]["rosdata"]["vel"]
        recorded_gain = expt["x"]["rosdata"]["gain"]

        rat = expt["x"]["rat"]
        day = expt["x"]["day"]

        tracking_path = os.path.join(RAW_DIR, f"{rat}-{day}_trackingResults.mat")
        if os.path.exists(tracking_path):
            logging.info(f"Found file at {neural_data_path}! Loading...")
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

            x = _average_variable(x, tracked_times, sampling_times)
            y = _average_variable(y, tracked_times, sampling_times)
            z = _average_variable(z, tracked_times, sampling_times)
            qx = _average_variable(qx, tracked_times, sampling_times)
            qy = _average_variable(qy, tracked_times, sampling_times)
            qz = _average_variable(qz, tracked_times, sampling_times)
            qw = _average_variable(qw, tracked_times, sampling_times)
            success = _average_variable(success, tracked_times, sampling_times)

            radius2 = [xx**2 + yy**2 for xx, yy in zip(x, y)]
            angles_tracked = np.arctan2(y, x)

            quat_head = np.stack([qx, qy, qz, qw], axis=1)  # scalar-last format
            assert quat_head.shape == (len(all_times), 4), quat_head.shape
            rotvec_head = R.from_quat(quat_head).as_rotvec()
            assert rotvec_head.shape == (len(all_times), 3), rotvec_head.shape
            angles_head = np.linalg.norm(rotvec_head, axis=-1)
            angles_head = angles_head % 360
            rx_head, ry_head, rz_head = (
                rotvec_head[:, 0],
                rotvec_head[:, 1],
                rotvec_head[:, 2],
            )

            if np.min(tracked_times) > np.max(all_times) or np.min(all_times) > np.max(
                tracked_times
            ):
                raise ValueError("Tracking times and firing times do not match.")

        else:
            logging.info(f"No file at {neural_data_path}! Skipping...")

        logging.info("Averaging variables angle, velocity and gain per time-step...")

        angles = _average_variable(recorded_angles, recorded_times, sampling_times)
        lap = angles // 360
        angles = angles % 360
        velocities = _average_variable(recorded_vel, recorded_times, sampling_times)
        gains = _average_variable(recorded_gain, recorded_times, sampling_times)

        if os.path.exists(tracking_path):
            labels = pd.DataFrame(
                {
                    "times": all_times,
                    "lap": lap,
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
                    "times": all_times,
                    "lap": lap,
                    "angles": angles,
                    "velocities": velocities,
                    "gains": gains,
                }
            )

        logging.info(
            f"# - Saving DataFrame with labels {labels.columns} to {labels_path}..."
        )
        labels.to_csv(labels_path)

    return neural_activity, labels


def _apply_velocity_threshold(expt, threshold=5):
    """Apply a velocity threshold to the data.

    This function finds the start and end times of all the contiguous periods where the velocity is above a threshold.

    Parameters
    ----------
    expt : dict
        Dictionary containing the experimental data.
    threshold : float
        Threshold value for velocity.

    Returns
    -------
    period_start_times : array-like, shape=[number of contiguous periods above threshold]
        List of times, each denoting the start of a period where velocity is above threshold.
    period_end_times : list
        List of times, each denoting the end of a period where velocity is above threshold.
    """
    df = pd.DataFrame(
        {
            k: pd.Series(v)
            for k, v in expt["x"]["rosdata"].items()
            if isinstance(v, np.ndarray)
        }
    )

    # create a new column 'above_threshold' to indicate whether the value is above the threshold
    df["above_threshold"] = df["vel"] > threshold

    # find the periods where the value rises above and then dips below the threshold
    df["rise_event"] = np.ceil(df["above_threshold"].diff()[1:].ne(0).cumsum() / 2)

    # Compute the lengths of the rises
    rise_lengths_microseconds = (
        df[df["above_threshold"]]
        .groupby("rise_event")
        .apply(lambda x: x["encTimes"].max() - x["encTimes"].min())
    )

    df["above_threshold"] = df["above_threshold"].astype(int)

    # Convert lengths from microseconds to seconds
    rise_lengths_in_seconds = rise_lengths_microseconds / 1_000_000

    # then you can analyze the lengths, for example, plot a histogram
    counts, bins, _ = plt.hist(rise_lengths_in_seconds, bins=300)
    plt.xlabel("Length of rise events (seconds)")
    plt.ylabel("Count")

    print(f"Mean rise length: {rise_lengths_in_seconds.mean()} seconds")
    print(f"Median rise length: {rise_lengths_in_seconds.median()} seconds")
    print(f"Total number of rise events: {len(rise_lengths_in_seconds)}")

    period_start_times = []
    period_end_times = []

    difference = df["above_threshold"].diff()

    if df["above_threshold"][0] == 1:
        period_start_times.append(df["encTimes"][0])

    for i in range(1, len(difference)):
        if difference[i] == 0:
            pass
        elif difference[i] == 1:
            start_time = df["encTimes"][i]
            period_start_times.append(start_time)
        elif difference[i] == -1:
            end_time = df["encTimes"][i]
            period_end_times.append(end_time)

    if df["above_threshold"][len(df["above_threshold"]) - 1] == 1:
        period_end_times.append(df["encTimes"][len(df["above_threshold"]) - 1])

    period_start_times = np.array(period_start_times)
    period_end_times = np.array(period_end_times)

    return period_start_times, period_end_times, df


def _get_sampling_times(period_start_times, period_end_times, timestep_microseconds):
    """Get the sampling times for the data.

    This function finds the sampling times for the data, given the start and end times of the periods where velocity is above a threshold and the timestep.

    Parameters
    ----------
    period_start_times : array-like, shape=[number of contiguous periods above threshold]
        List of times, each denoting the start of a period where velocity is above threshold.
    period_end_times : array_like, shape=[number of contiguous periods above threshold]
        List of times, each denoting the end of a period where velocity is above threshold.
    timestep_microseconds : int
        The timestep in microseconds.

    Returns
    -------
    sampling_times : array_like, shape=[number of valid sampling points]

    """
    sampling_times = []
    for i in range(len(period_start_times)):
        if period_end_times[i] - period_start_times[i] > timestep_microseconds:
            times = []
            t = period_start_times[i]
            while t < period_end_times[i]:
                times.append(t)
                t += timestep_microseconds
            sampling_times.append(times)

    return sampling_times


def _average_variable(variable_to_average, recorded_times, sampling_times):
    """Average values of a variable for each valid timestep.

    Parameters
    ----------
    variable_to_average : array-like, shape=[len(recorded_times)]
        List of all recorded values of the variable to be averaged.
    recorded_times : array-like, shape=[len(recorded_times)]
        List of all times at which the variable to be averaged was recorded.
    sampling_times : nested list, shape = (number of contiguous periods above threshold, number of sampling points in each period)
        List of lists of times, each denoting the sampling times for a period where velocity is above threshold.

    """
    variable_averaged = []
    for times in sampling_times:
        recordings_count, _ = np.histogram(recorded_times, bins=times)
        cum_count = np.searchsorted(recorded_times, times[0])
        for count in recordings_count:
            averaged = np.mean(variable_to_average[cum_count : cum_count + int(count)])
            variable_averaged.append(averaged)
            cum_count += int(count)

    variable_averaged = np.array(variable_averaged)

    return variable_averaged


def get_place_field_centers(neural_activity, task_variable):
    """Get the center of mass of the place fields of a list of neurons.

    Parameters
    ----------
    neural_activity : array-like, shape=[number of sampling points, number of neurons]
        The neural activity of the neurons.
    task_variable : array-like, shape=[number of sampling points]
        The task variable (e.g. position) for which the place field centers are to be found.

    Returns
    -------
    center_of_mass : array-like, shape=[number of neurons]
    """

    # convert task_variable from degrees to radians
    task_variable_rad = np.deg2rad(task_variable)

    weights = neural_activity.T
    points_sin = np.tile(np.sin(task_variable_rad), (weights.shape[0], 1))
    points_cos = np.tile(np.cos(task_variable_rad), (weights.shape[0], 1))

    # Compute weighted sum of sines and cosines
    weighted_sum_sin = np.average(points_sin, weights=weights, axis=1)
    weighted_sum_cos = np.average(points_cos, weights=weights, axis=1)

    # Compute average angle, taking care to handle the quadrant correctly
    weighted_center_of_mass_rad = np.arctan2(weighted_sum_sin, weighted_sum_cos)

    # Convert from radians back to degrees
    weighted_center_of_mass = np.rad2deg(weighted_center_of_mass_rad)

    # handle the negative angles from arctan2
    weighted_center_of_mass = (weighted_center_of_mass + 360) % 360

    center_of_mass_indices = np.argmax(weights, axis=1)

    return weighted_center_of_mass, center_of_mass_indices
