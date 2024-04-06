import numpy as np
import persim


def distance(diagrams1, diagrams2, dist_type="bottleneck"):
    maxdim1 = len(diagrams1)
    maxdim2 = len(diagrams2)
    assert maxdim1 == maxdim2

    total_distance = 0

    for dim in range(maxdim1):
        if dist_type == "bottleneck":
            bottleneck_dist_dim = persim.bottleneck(diagrams1[dim], diagrams2[dim])
            total_distance += bottleneck_dist_dim
        elif dist_type == "sliced_wasserstein":
            sliced_wasserstein_dist_dim = persim.sliced_wasserstein(
                diagrams1[dim], diagrams2[dim]
            )
            total_distance += sliced_wasserstein_dist_dim
        else:
            raise ValueError("Invalid distance type")

    return total_distance


def distances_to_reference(diagrams_list, reference_diagrams, dist_type="bottleneck"):
    num_diagrams = len(diagrams_list)
    distances = np.zeros(num_diagrams)

    for i in range(num_diagrams):
        distances[i] = distance(diagrams_list[i], reference_diagrams, dist_type)

    return distances


def pairwise_distances(diagrams_list):
    # TODO: Parallelize this
    num_diagrams = len(diagrams_list)
    distances = np.zeros((num_diagrams, num_diagrams))

    for i in range(num_diagrams):
        for j in range(i + 1, num_diagrams):
            distances[i][j] = bottleneck_distance(diagrams_list[i], diagrams_list[j])
            distances[j][i] = distances[i][j]

    return distances
