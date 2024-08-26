import os
import pickle

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import torch
import umap
import yaml
from sklearn.cluster import DBSCAN

from neurometry.datasets.piRNNs.scores import GridScorer

# sys.path.append(str(Path(__file__).parent.parent))
from .dual_agent import config, dual_agent_activity, single_agent_activity, utils


def load_rate_maps(run_id, step):
    #XU_RNN
    model_dir = os.path.join(os.getcwd(), "curvature/grid-cells-curvature/models/xu_rnn")
    run_dir = os.path.join(model_dir, f"logs/rnn_isometry/{run_id}")
    activations_file = os.path.join(run_dir, f"ckpt/activations/activations-step{step}.pkl")
    with open(activations_file, "rb") as f:
        return pickle.load(f)

def load_config(run_id):
    model_dir = os.path.join(os.getcwd(), "curvature/grid-cells-curvature/models/xu_rnn")
    run_dir = os.path.join(model_dir, f"logs/rnn_isometry/{run_id}")
    config_file = os.path.join(run_dir, "config.txt")

    with open(config_file) as file:
        return yaml.safe_load(file)





def extract_tensor_events(event_file, verbose=True):
    #XU_RNN
    records = []
    losses = []
    try:
        for e in tf.compat.v1.train.summary_iterator(event_file):
            if verbose:
                print(f"Found event at step {e.step} with wall time {e.wall_time}")
            for v in e.summary.value:
                if verbose:
                    print(f"Found value with tag: {v.tag}")
                if v.HasField("tensor"):
                    tensor = tf.make_ndarray(v.tensor)
                    record = {
                        "step": e.step,
                        "wall_time": e.wall_time,
                        "tag": v.tag,
                        "tensor": tensor,
                    }
                    records.append(record)
                    if v.tag == "loss":
                        loss = {"step": e.step, "loss": tensor}
                        losses.append(loss)
                else:
                    if verbose:
                        print(f"No 'tensor' found for tag {v.tag}")
    except Exception as e:
        print(f"An error occurred: {e}")
    return records, losses



def _compute_scores(activations, config):
    block_size = config["model"]["block_size"]
    num_neurons = config["model"]["num_neurons"]
    num_block = num_neurons // block_size

    starts = [0.1] * 20
    ends = np.linspace(0.2, 1.4, num=20)
    masks_parameters = zip(starts, ends.tolist(), strict=False)

    scorer = GridScorer(40, ((0, 1), (0, 1)), masks_parameters)

    score_list = np.zeros(shape=[len(activations["v"])], dtype=np.float32)
    scale_list = np.zeros(shape=[len(activations["v"])], dtype=np.float32)
    #orientation_list = np.zeros(shape=[len(weights)], dtype=np.float32)
    sac_list = []

    for i in range(len(activations["v"])):
        rate_map = activations["v"][i]
        rate_map = (rate_map - rate_map.min()) / (rate_map.max() - rate_map.min())

        score_60, score_90, max_60_mask, max_90_mask, sac, _ = scorer.get_scores(
            activations["v"][i])
        sac_list.append(sac)

        score_list[i] = score_60
        # scale_list[i] = scale
        scale_list[i] = max_60_mask[1]
        # orientation_list[i] = orientation


    scale_tensor = torch.from_numpy(scale_list)
    score_tensor = torch.from_numpy(score_list)
    max_scale = torch.max(scale_tensor[score_list > 0.37])

    scale_tensor = scale_tensor.reshape((num_block, block_size))
    scale_tensor = torch.mean(scale_tensor, dim=1)

    # score_tensor = score_tensor.reshape((num_block, block_size))
    score_tensor = torch.mean(score_tensor)
    sac_array = np.array(sac_list)

    return {"sac":sac_array, "scale":scale_tensor, "score": score_tensor, "max_scale": max_scale}




def get_scores(run_dir, activations, config):
    scores_dir = os.path.join(run_dir, "ckpt/scores")
    if not os.path.exists(scores_dir):
        os.makedirs(scores_dir)
    scores_file = os.path.join(scores_dir, "scores.pkl")
    if os.path.exists(scores_file):
        with open(scores_file, "rb") as f:
            scores = pickle.load(f)
    else:
        scores = _compute_scores(activations, config)
        with open(scores_file, "wb") as f:
            pickle.dump(scores, f)

    return scores


def load_activations(epochs, file_path, version="single", verbose=True, save=True):
    #SORSCHER RNN
    activations = []
    rate_maps = []
    state_points = []
    # positions = []
    # g_s = []

    activations_dir = os.path.join(file_path, "activations")

    for epoch in epochs:
        activations_epoch_path = os.path.join(
            activations_dir, f"activations_{version}_agent_epoch_{epoch}.npy"
        )
        rate_map_epoch_path = os.path.join(
            activations_dir, f"rate_map_{version}_agent_epoch_{epoch}.npy"
        )
        # positions_epoch_path = os.path.join(
        #     activations_dir, f"positions_{version}_agent_epoch_{epoch}.npy"
        # )
        # gs_epoch_path = os.path.join(
        #     activations_dir, f"g_{version}_agent_epoch_{epoch}.npy"
        # )

        if (
            os.path.exists(activations_epoch_path)
            and os.path.exists(rate_map_epoch_path)
        ):
            activations.append(np.load(activations_epoch_path))
            rate_maps.append(np.load(rate_map_epoch_path))
            #positions.append(np.load(positions_epoch_path))
            #g_s.append(np.load(gs_epoch_path))
            if verbose:
                print(f"Epoch {epoch} found.")
        else:
            print(f"Epoch {epoch} not found. Loading ...")
            parser = config.parser
            options, _ = parser.parse_known_args()
            options.run_ID = utils.generate_run_ID(options)
            if version == "single":
                (
                    activations_single_agent,
                    rate_map_single_agent,
                    g_single_agent,
                    positions_single_agent,
                ) = single_agent_activity.main(options, file_path, epoch=epoch)
                activations.append(activations_single_agent)
                rate_maps.append(rate_map_single_agent)
                #positions.append(positions_single_agent)
                #g_s.append(g_single_agent)
            elif version == "dual":
                (
                    activations_dual_agent,
                    rate_map_dual_agent,
                    g_dual_agent,
                    positions_dual_agent,
                ) = dual_agent_activity.main(options, file_path, epoch=epoch)
                activations.append(activations_dual_agent)
                rate_maps.append(rate_map_dual_agent)
                #positions.append(positions_dual_agent)
                #g_s.append(g_dual_agent)

            if save:
                np.save(activations_epoch_path, activations[-1])
                np.save(rate_map_epoch_path, rate_maps[-1])
                #np.save(positions_epoch_path, positions[-1])
                #np.save(gs_epoch_path, g_s[-1])

        state_points_epoch = activations[-1].reshape(activations[-1].shape[0], -1)
        state_points.append(state_points_epoch)

    if verbose:
        print(f"Loaded epochs {epochs} of {version} agent model.")
        print(
            f"activations has shape {activations[0].shape}. There are {activations[0].shape[0]} grid cells with {activations[0].shape[1]} x {activations[0].shape[2]} environment resolution, averaged over {activations[0].shape[3]} trajectories."
        )
        print(
            f"state_points has shape {state_points[0].shape}. There are {state_points[0].shape[1]} data points in the {state_points[0].shape[0]}-dimensional state space."
        )
        print(
            f"rate_maps has shape {rate_maps[0].shape}. There are {rate_maps[0].shape[1]} data points averaged over {activations[0].shape[3]} trajectories in the {rate_maps[0].shape[0]}-dimensional state space."
        )
        #print(f"positions has shape {positions[0].shape}.")

    return activations, rate_maps, state_points#, positions, g_s


def plot_rate_map(indices, num_plots, activations, title, seed=None):
    rng = np.random.default_rng(seed=seed)
    if indices is None:
        idxs = rng.integers(0, activations.shape[0] - 1, num_plots)
    else:
        idxs = indices
        num_plots = len(indices)

    rows = 4
    cols = num_plots // rows + (num_plots % rows > 0)

    plt.rcParams["text.usetex"] = False

    fig, axes = plt.subplots(rows, cols, figsize=(20, 8))

    for i in range(rows):
        for j in range(cols):
            if i * cols + j < num_plots:
                if len(activations.shape) == 4:
                    gc = np.mean(activations[idxs[i * cols + j]], axis=2)
                else:
                    gc = activations[idxs[i * cols + j]]

                ax = axes[i, j] if axes.ndim > 1 else axes[i * cols + j]
                ax.imshow(gc, interpolation="nearest", cmap="viridis", aspect="auto")
                ax.set_title(f"grid cell id: {idxs[i * cols + j]}", fontsize=10)
                ax.axis("off")
            else:
                if axes.ndim > 1:
                    axes[i, j].axis("off")
                else:
                    axes[i * cols + j].axis("off")

    fig.suptitle(title, fontsize=30)
    #plt.tight_layout()
    plt.show()



def draw_heatmap(activations, title):
    # activations should a 4-D tensor: [M, N, H, W]
    nrow, ncol = activations.shape[0], activations.shape[1]
    fig = plt.figure(figsize=(ncol, nrow))

    for i in range(nrow):
        for j in range(ncol):
            plt.subplot(nrow, ncol, i * ncol + j + 1)
            weight = activations[i, j]
            vmin, vmax = weight.min() - 0.01, weight.max()

            cmap = cm.get_cmap("jet", 1000)
            cmap.set_under("w")

            plt.imshow(
                weight,
                interpolation="nearest",
                cmap=cmap,
                aspect="auto",
                vmin=vmin,
                vmax=vmax,
            )
            plt.axis("off")

    fig.canvas.draw()
    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_from_plot = image_from_plot.reshape(
        fig.canvas.get_width_height()[::-1] + (3,)
    )

    fig.suptitle(title, fontsize=20, fontweight="bold", verticalalignment="top")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    plt.close(fig)

    return np.expand_dims(image_from_plot, axis=0)




def _z_standardize(matrix):
    return (matrix - np.mean(matrix, axis=0)) / np.std(matrix, axis=0)


def _vectorized_spatial_autocorrelation_matrix(spatial_autocorrelation):
    num_cells = spatial_autocorrelation.shape[0]
    num_bins = spatial_autocorrelation.shape[1] * spatial_autocorrelation.shape[2]

    spatial_autocorrelation_matrix = np.zeros((num_bins, num_cells))

    for i in range(num_cells):
        vector = spatial_autocorrelation[i].flatten()

        spatial_autocorrelation_matrix[:, i] = vector

    return _z_standardize(spatial_autocorrelation_matrix)


def umap_dbscan(activations, run_dir, config, sac_array=None, plot=True):
    if sac_array is None:
        sac_array = get_scores(run_dir, activations, config)["sac"]

    spatial_autocorrelation_matrix = _vectorized_spatial_autocorrelation_matrix(sac_array)

    umap_reducer_2d = umap.UMAP(n_components=2, random_state=10)
    umap_embedding = umap_reducer_2d.fit_transform(spatial_autocorrelation_matrix.T)

    # Clustering with DBSCAN
    umap_dbscan = DBSCAN(eps=0.5, min_samples=5).fit(umap_embedding)

    # Plot each cluster
    unique_labels = np.unique(umap_dbscan.labels_)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for k, col in zip(unique_labels, colors, strict=False):
        if k == -1:
            # Black used for noise.
            # col = [0, 0, 0, 1]
            continue

        class_member_mask = (umap_dbscan.labels_ == k)

        xy = umap_embedding[class_member_mask]
        if plot:
            axes[0].plot(xy[:, 0], xy[:, 1], "o", markerfacecolor=tuple(col), markeredgecolor="none", markersize=5, label=f"Cluster {k}")

        umap_cluster_labels = umap_dbscan.fit_predict(umap_embedding)
        clusters = {}
        for i in np.unique(umap_cluster_labels):
            #cluster = _get_data_from_cluster(activations,i, umap_cluster_labels)
            cluster = activations[umap_cluster_labels == i]
            clusters[i] = cluster

    if plot:
        axes[0].set_xlabel("UMAP 1")
        axes[0].set_ylabel("UMAP 2")
        axes[0].set_title("UMAP embedding of spatial autocorrelation")
        axes[0].legend(title="Cluster IDs", loc="center left", bbox_to_anchor=(1, 0.5))

        axes[1].hist(umap_cluster_labels, bins=len(np.unique(umap_cluster_labels)))
        axes[1].set_xlabel("Cluster ID")
        axes[1].set_ylabel("Number of units")
        axes[1].set_yscale("log")
        plt.tight_layout()
        plt.show()
    return clusters, umap_cluster_labels

