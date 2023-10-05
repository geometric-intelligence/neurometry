import numpy as np
import itertools
from neurometry.rep_metrics.dissimilarity import compute_rsa_pairwise_dissimilarities
from neurometry.rep_metrics.dissimilarity import compute_pairwise_distances
from collections import defaultdict


def nested_dict():
    return defaultdict(nested_dict)


def all_rsa_geometry_matrices(neural_data, rdm_compute_methods, rdm_compare_methods):
    rsa_methods = list(itertools.product(rdm_compute_methods, rdm_compare_methods))
    subject_ids = np.array(list(neural_data.keys()))

    rsa_geometry_matrices = defaultdict(nested_dict)

    for rsa_method in rsa_methods:
        rdm_compute_method, rdm_compare_method = rsa_method
        for subject_id in subject_ids:
            matrix = compute_rsa_pairwise_dissimilarities(
                neural_data[subject_id], rdm_compute_method, rdm_compare_method
            )
            rsa_geometry_matrices[subject_id][rdm_compute_method][
                rdm_compare_method
            ] = matrix

    return rsa_geometry_matrices


def all_shapemetrics_geometry_matrices(pca_reduced_neural_data, alphas):
    subject_ids = np.array(list(pca_reduced_neural_data.keys()))

    shapemetric_geometry_matrices = defaultdict(nested_dict)

    for alpha in alphas:
        for subject_id in subject_ids:
            matrix = compute_pairwise_distances(
                pca_reduced_neural_data[subject_id], alpha
            )
            shapemetric_geometry_matrices[subject_id][alpha] = matrix

    return shapemetric_geometry_matrices


def get_voxels_coordinates(voxels_list):
    coordinates = []

    for coord_str in voxels_list:
        split_str = coord_str.split("-")
        x = int(split_str[1])
        y = int(split_str[2])
        z = int(split_str[3])
        coordinates.append((x, y, z))

    coordinates = np.array(coordinates)

    centroid = np.mean(coordinates, axis=0)

    return coordinates, centroid
