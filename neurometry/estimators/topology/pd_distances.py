import numpy as np
from gtda.diagrams import PairwiseDistance


def compute_pairwise_distances(diagrams, metric="bottleneck"):
    PD = PairwiseDistance(metric=metric)
    return PD.fit_transform(diagrams)


def compare_representations_to_references(
    diagrams, reference_diagram, metric="bottleneck"
):
    distances = []
    for diagram in diagrams:
        distance_matrix = compute_pairwise_distances([diagram, reference_diagram], metric=metric)
        distances.append(
            np.sum(distance_matrix)/2
        )
    return distances

