import os
import pickle
import default_config
import yaml
from neurometry.datasets.load_rnn_grid_cells import get_scores
from neurometry.datasets.load_rnn_grid_cells import umap_dbscan


pretrained_run_id = "20240418-180712"
pretrained_run_dir = os.path.join(
    os.getcwd(),
    f"curvature/grid-cells-curvature/models/xu_rnn/logs/rnn_isometry/{pretrained_run_id}",
)

pretrained_config_file = os.path.join(pretrained_run_dir, "config.txt")
with open(pretrained_config_file) as f:
    pretrained_config = yaml.safe_load(f)

pretrained_activations_file = os.path.join(pretrained_run_dir, "ckpt/activations/activations-step25000.pkl")
with open(pretrained_activations_file, "rb") as f:
    pretrained_activations = pickle.load(f)

scores = get_scores(pretrained_run_dir, pretrained_activations, pretrained_config)

clusters_before, umap_cluster_labels = umap_dbscan(
    pretrained_activations["v"], pretrained_run_dir, pretrained_config, sac_array=None, plot=False
)









def load_expt_rate_maps(run_name):
    activations_dir = default_config.activations_dir
    activations_file = os.path.join(activations_dir, f"{run_name}_activations.pkl")
    with open(activations_file, "rb") as f:
        return pickle.load(f)

def load_expt_config(run_name):
    configs_dir = default_config.configs_dir
    config_file = os.path.join(configs_dir, f"{run_name}_config.txt")

    with open(config_file) as file:
        return yaml.safe_load(file)
    
def plot_experiment(run_name):
    activations = load_expt_rate_maps(run_name)["v"]
    config = load_expt_config(run_name)



