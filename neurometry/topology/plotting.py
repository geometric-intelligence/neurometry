import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go



def _plot_bars_from_diagrams(ax, diagrams, dim, **kwargs):
    birth = diagrams[:, 0]
    death = diagrams[:, 1]
    lifespan = death - birth
    indices = np.argsort(-lifespan)[:20]  # Adjust the number of bars if necessary

    birth = birth[indices]
    death = death[indices]
    finite_bars = death[death != np.inf]
    inf_end = 2 * max(finite_bars) if len(finite_bars) > 0 else 2
    death[death == np.inf] = inf_end

    offset = kwargs.get("bar_offset", 1.0)
    linewidth = kwargs.get("linewidth", 5)

    # Plotting each bar
    for i, (b, d) in enumerate(zip(birth, death)):
        ax.plot(
            [0, d - b],
            [i * offset, i * offset],
            linewidth=linewidth,
            **{k: v for k, v in kwargs.items() if k not in ["linewidth", "bar_offset"]},
        )

    # Dynamically adjust the y-axis limits to fit the bars
    if len(birth) > 0:
        ax.set_ylim(-1, len(birth) * offset + 1)


def plot_all_barcodes_with_null(diagrams, **kwargs):
    original_diagram = diagrams[0]
    shuffled_diagrams = diagrams[1:]

    dims = np.unique(original_diagram[:, 2]).astype(int)
    num_dims = len(dims)
    colors = plt.cm.Greens(np.linspace(0.5, 1, num_dims))
    fig, axs = plt.subplots(
        num_dims, 1, figsize=kwargs.get("figsize", (10, 5 * num_dims)), sharex=True
    )

    if num_dims == 1:
        axs = [axs]  # Make it iterable if there's only one subplot

    for ax, dim, color in zip(axs, dims, colors):
        diag_dim = original_diagram[original_diagram[:, 2] == dim]
        null_diag_dim = shuffled_diagrams[:, :, 2] == dim
        null_diag = shuffled_diagrams[null_diag_dim]

        _plot_bars_from_diagrams(
            ax,
            null_diag,
            dim,
            color="lightgrey",
            linewidth=10,
            alpha=0.9,
            bar_offset=kwargs.get("bar_offset", 0.2),
        )
        _plot_bars_from_diagrams(
            ax,
            diag_dim,
            dim,
            color=color,
            linewidth=6,
            bar_offset=kwargs.get("bar_offset", 0.2),
        )

        ax.set_ylabel(f"Homology {dim}", fontsize=20)
        ax.set_yticks([])

    plt.xlabel(kwargs.get("xlabel", "Filtration Value"), fontsize=20)
    plt.tight_layout()
    plt.show()



def plot_activity_on_torus(neural_activations, toroidal_coords, neuron_id, neuron_id2=None):

    phis = toroidal_coords[:,0]
    thetas = toroidal_coords[:,1]

    r = 1
    R = 2

    xs = (R+r*np.cos(thetas)) * np.cos(phis)
    ys = (R+r*np.cos(thetas)) * np.sin(phis)
    zs = r*np.sin(thetas)

    fig = go.Figure()

    if neuron_id2 is None:
        activations = neural_activations[:, neuron_id]
        colors = activations
        fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="markers", marker=dict(size=5, color=colors, opacity=1), name=f"Neuron {neuron_id}"))
        title = f"Neural activations on the torus for neuron {neuron_id}"
    else:
        activations1 = neural_activations[:, neuron_id]
        activations2 = neural_activations[:, neuron_id2]
        threshold1 = np.percentile(activations1, 95)
        threshold2 = np.percentile(activations2, 95)
        colors1 = []
        colors2 = []
        for i in range(len(xs)):
            if activations1[i] > threshold1:
                alpha = 1#min(1, activations1[i] / threshold1)
                colors1.append(f"rgba(255, 0, 0, {alpha})")
                colors2.append("rgba(128, 128, 128, 0)")
            elif activations2[i] > threshold2:
                alpha = 1#min(1, activations2[i] / threshold2)
                colors1.append("rgba(128, 128, 128, 0)")
                colors2.append(f"rgba(255, 255, 0, {alpha})")
            else:
                colors1.append("rgba(5, 0, 15, 0.1)")
                colors2.append("rgba(5, 0, 15, 0.1)")

        # Populate the figure with data for both neurons
        fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="markers", marker=dict(size=5, color=colors1), name=f"Neuron {neuron_id}"))
        fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="markers", marker=dict(size=5, color=colors2), name=f"Neuron {neuron_id2}"))

        title = f"Neural activations on the torus for neurons {neuron_id} (Red) and {neuron_id2} (Yellow)"

    fig.update_layout(title=title, autosize=False, width=800, height=500)
    fig.show()
    return fig