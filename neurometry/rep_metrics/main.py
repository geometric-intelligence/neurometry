import config
import itertools
import pandas as pd
from collections import defaultdict
from .load_nsd import load_nsd, get_neural_data
from .dissimilarity import (
    compute_rsa_pairwise_dissimilarities,
    shape_preprocess,
    compute_shape_pairwise_distances,
)
from .anatomy import (
    get_subjects_rois,
    get_roi_list_intersection,
    compute_cortex_pairwise_geodesic_dist,
)


# NSD data
target_regions = config.target_regions
subjects = config.subjects
subject_ids = [int(s.split("subj")[1]) for s in subjects]
response_data, voxel_metadata, stimulus_data, benchmark_rois = load_nsd(target_regions)

# Anatomical data
anatomical_rois = get_subjects_rois(subjects)

# Consider only common ROIs (TODO: fix pycortex roi hand-drawn definitions)
rois = {}
for subject in subjects:
    rois[subject] = get_roi_list_intersection(benchmark_rois, anatomical_rois[subject])

neural_data = get_neural_data(subjects, rois, voxel_metadata, response_data)


# RSA parameters
rdm_compute_types = config.rdm_compute_types
rdm_compare_types = config.rdm_compare_types

# Shape Metrics parameters
alphas = config.alphas

# CKA parameters
# TODO


def _nested_dict():
    return defaultdict(str)


def anatomical_distance_matrices():
    anatomy_final_matrices = defaultdict(_nested_dict)
    for subject in subjects:
        print(f"Computing anatomical pairwise distance for subject {subject}...")
        cortex_pairwise_dists = compute_cortex_pairwise_geodesic_dist(
            subject, rois[subject]
        )
        anatomy_final_matrices[subject] = cortex_pairwise_dists
        print("done!")
    return anatomy_final_matrices


def rsa_pairwise_matrices():
    rsa_types = list(itertools.product(rdm_compute_types, rdm_compare_types))
    rsa_final_matrices = defaultdict(_nested_dict)
    for i, subject in enumerate(subjects):
        for rsa_type in rsa_types:
            rsa_type_name = "_".join(rsa_type)
            print(
                f"Computing RSA {rsa_type_name} pairwise dissimilarities for subject {subject}..."
            )
            rsa_pairwise_dissimilarity = compute_rsa_pairwise_dissimilarities(
                neural_data[subject_ids[i]], rois[subject], rsa_type[0], rsa_type[1]
            )
            rsa_final_matrices[subject][rsa_type_name] = rsa_pairwise_dissimilarity
            print("done!")
    return rsa_final_matrices


preprocessed_shape_neural_data, pcas = shape_preprocess(neural_data, subjects, rois)


def shape_pairwise_matrices():
    shape_final_matrices = defaultdict(_nested_dict)
    for i, subject in enumerate(subjects):
        for alpha in alphas:
            print(
                f"Computing Shape Metric (alpha={alpha}) pairwise dissimilarities for subject {subject}..."
            )
            shape_pairwise_distance = compute_shape_pairwise_distances(
                preprocessed_shape_neural_data[subject_ids[i]],
                rois[subject],
                stimulus_data,
                alpha,
            )
            shape_final_matrices[subject][alpha] = shape_pairwise_distance
        print("done!")

    return shape_final_matrices


# TODO
def cka_pairwise_matrices():
    return NotImplementedError


# anatomy_final_matrices = anatomical_distance_matrices()
# rsa_final_matrices = rsa_pairwise_matrices()
# shape_final_matrices = shape_pairwise_matrices()
