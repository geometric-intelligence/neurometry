
import matplotlib.pyplot as plt
import numpy as np
from gtda.diagrams import PairwiseDistance
from gtda.homology import VietorisRipsPersistence, WeightedRipsPersistence
from matplotlib.collections import LineCollection
from dreimac import ToroidalCoords
import plotly.graph_objects as go


def compute_persistence_diagrams(
    representations,
    homology_dimensions=(0, 1, 2),
    coeff=2,
    metric="euclidean",
    weighted=False,
    n_jobs=-1
):
    if weighted:
        WR = WeightedRipsPersistence(
            metric=metric, homology_dimensions=homology_dimensions, coeff=coeff
        )
        diagrams = WR.fit_transform(representations)
    else:
        VR = VietorisRipsPersistence(
            metric=metric, homology_dimensions=homology_dimensions, coeff=coeff, n_jobs=n_jobs)
        diagrams = VR.fit_transform(representations)
    return diagrams


def compute_pairwise_distances(diagrams, metric="bottleneck"):
    PD = PairwiseDistance(metric=metric)
    return PD.fit_transform(diagrams)


def compare_representation_to_references(
    representation, reference_topologies, metric="bottleneck"
):
    raise NotImplementedError



def cohomological_toroidal_coordinates(data):
    n_landmarks = data.shape[0]
    tc = ToroidalCoords(data, n_landmarks=n_landmarks)
    cohomology_classes = [0,1]
    toroidal_coords = tc.get_coordinates(cocycle_idxs=cohomology_classes,standard_range=False)
    return toroidal_coords.T

def plot_activity_on_torus(neural_activations, toroidal_coords, neuron_id):
    phis = toroidal_coords[:,0]
    thetas = toroidal_coords[:,1]

    r = 1
    R = 2

    xs = (R+r*np.cos(thetas)) * np.cos(phis)
    ys = (R+r*np.cos(thetas)) * np.sin(phis)
    zs = r*np.sin(thetas)

    color = neural_activations[:,neuron_id-1]

    fig = go.Figure(data=[go.Scatter3d(x=xs, y=ys, z=zs, mode='markers', marker=dict(size=5, color=color, colorscale='Viridis', opacity=0.8))])

    title = f"Neural activations on the torus for neuron {neuron_id}"

    fig.update_layout(title=title, autosize=False, width=400, height=400)

    fig.show()



def plot_lifetimes(diagram):
    diagram_0 = np.array([dgm[:-1] for dgm in diagram if dgm[2] == 0])
    diagram_1 = np.array([dgm[:-1] for dgm in diagram if dgm[2] == 1])
    diagram_2 = np.array([dgm[:-1] for dgm in diagram if dgm[2] == 2])

    births_0 = diagram_0[:,0]
    deaths_0 = diagram_0[:,1]
    lifetimes_0 = deaths_0 - births_0

    births_1 = diagram_1[:,0]
    deaths_1 = diagram_1[:,1]
    lifetimes_1 = deaths_1 - births_1

    births_2 = diagram_2[:,0]
    deaths_2 = diagram_2[:,1]
    lifetimes_2 = deaths_2 - births_2

    f, (a0, a1, a2) = plt.subplots(1, 3, width_ratios=[1, 1, 1], figsize=(12, 3))

    a0.scatter(births_0, lifetimes_0, c="red", label="H0")
    a0.set_xlabel("Birth")
    a0.set_ylabel("Lifetime")
    a0.set_title("Feature lifetimes, homology group H0")

    a1.scatter(births_1, lifetimes_1, c="blue", label="H1")
    a1.set_xlabel("Birth")
    a1.set_ylabel("Lifetime")
    a1.set_title("Feature lifetimes, homology group H1")

    a2.scatter(births_2, lifetimes_2, c="green", label="H2")
    a2.set_xlabel("Birth")
    a2.set_ylabel("Lifetime")
    a2.set_title("Feature lifetimes, homology group H2")
    plt.legend();
    plt.tight_layout()


def plot_barcode(topology_result, maxdim):
    fig, axs = plt.subplots(maxdim+1, 1, sharex=True, figsize=(7, 8))
    axs[0].set_xlim(0,2)
    cocycle = ["Points", "Loops", "Voids"]
    for k in range(maxdim+1):
        bars = topology_result["dgms"][k]
        bars[bars == np.inf] = 10
        lc = (
            np.vstack(
                [
                    bars[:, 0],
                    np.arange(len(bars), dtype=int) * 6,
                    bars[:, 1],
                    np.arange(len(bars), dtype=int) * 6,
                ]
            )
            .swapaxes(1, 0)
            .reshape(-1, 2, 2)
        )
        line_segments = LineCollection(lc, linewidth=5, color="gray", alpha=0.5)
        axs[k].set_ylabel(cocycle[k], fontsize=20)
        if k == 0:
            axs[k].set_ylim(len(bars) * 6 - 120, len(bars) * 6)
        elif k == 1:
            axs[k].set_ylim(0, len(bars) * 1 - 30)
        elif k == 2:
            axs[k].set_ylim(0, len(bars) * 6 + 10)
        axs[k].add_collection(line_segments)
        axs[k].set_yticks([])
        if k == 2:
            axs[k].set_xticks(np.linspace(0, 2, 3), np.linspace(0, 2, 3), fontsize=15)
            axs[k].set_xlabel("Lifespan", fontsize=20)

    return fig
