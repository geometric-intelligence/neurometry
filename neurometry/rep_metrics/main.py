import config
import itertools
import numpy as np
from collections import defaultdict
from .load_nsd import load_nsd, get_neural_data
from .dissimilarity import (
    compute_rsa_pairwise_dissimilarities,
    compute_cka_pairwise_dissimilarities,
    shape_preprocess,
    compute_shape_pairwise_distances,
)
from .anatomy import (
    get_subjects_rois,
    get_roi_list_intersection,
    compute_cortex_pairwise_geodesic_dist,
)
from .dim_reduction import compute_stress


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

rois["subj02"].remove(
    "OPA"
)  # (sometimes) gives nans in many geodesic computations, not sure why

neural_data = get_neural_data(subjects, rois, voxel_metadata, response_data)


# RSA parameters
rdm_compute_types = config.rdm_compute_types
rdm_compare_types = config.rdm_compare_types

# Shape Metrics parameters
alphas = config.alphas

# CKA parameters
cka_types = config.cka_types


def _nested_dict_2():
    return defaultdict(str)


def _nested_dict_3():
    return defaultdict(dict)


def _normalize_by_mean(matrix):
    mean_val = np.mean(matrix)
    return matrix / mean_val


def anatomical_distance_matrices():
    anatomy_final_matrices = defaultdict(_nested_dict_2)
    empty_rois_dict = defaultdict(_nested_dict_2)
    for subject in subjects:
        print(f"Computing anatomical pairwise distance for subject {subject}...")
        cortex_pairwise_dists, empty_rois = compute_cortex_pairwise_geodesic_dist(
            subject, rois[subject]
        )
        anatomy_final_matrices[subject] = _normalize_by_mean(cortex_pairwise_dists)
        empty_rois_dict[subject] = empty_rois

        print("done!")
    print(f"-----finished anatomical computations for all subjects-----")
    return anatomy_final_matrices, empty_rois_dict


anatomy_final_matrices, empty_rois_dict = anatomical_distance_matrices()

# for subj01, pycortex returns empty vertices for some rois that should not be empty - why?
nonempty_rois = {}
for subject in subjects:
    subject_rois = rois[subject]
    subject_rois = [roi for roi in subject_rois if roi not in empty_rois_dict[subject]]
    nonempty_rois[subject] = subject_rois

rois = nonempty_rois


def rsa_pairwise_matrices():
    rsa_types = list(itertools.product(rdm_compute_types, rdm_compare_types))
    rsa_types.remove(("spearman", "spearman"))  # WHY IS THIS ONE SO SLOW ???
    rsa_final_matrices = defaultdict(_nested_dict_3)
    for i, subject in enumerate(subjects):
        for rsa_type in rsa_types:
            rsa_type_name = "_".join(rsa_type)
            print(
                f"Computing RSA {rsa_type_name} pairwise dissimilarities for subject {subject}..."
            )
            rsa_pairwise_dissimilarity = compute_rsa_pairwise_dissimilarities(
                neural_data[subject_ids[i]], rois[subject], rsa_type[0], rsa_type[1]
            )
            matrix = _normalize_by_mean(rsa_pairwise_dissimilarity)
            stress = compute_stress(matrix, anatomy_final_matrices[subject])
            rsa_final_matrices[subject][rsa_type_name]["matrix"] = matrix
            rsa_final_matrices[subject][rsa_type_name]["stress"] = stress
            print("done!")
        print(f"finished RSA for subject{subject}.")
    print(f"-----finished RSA computations for all subjects-----")
    return rsa_final_matrices


def cka_pairwise_matrices():
    cka_final_matrices = defaultdict(_nested_dict_3)
    for i, subject in enumerate(subjects):
        for cka_type in cka_types:
            print(
                f"Computing RSA {cka_type} pairwise dissimilarities for subject {subject}..."
            )
            cka_pairwise_dissimilarity = compute_cka_pairwise_dissimilarities(
                neural_data[subject_ids[i]], rois[subject], cka_type
            )
            matrix = _normalize_by_mean(cka_pairwise_dissimilarity)
            stress = compute_stress(matrix, anatomy_final_matrices[subject])
            cka_final_matrices[subject][cka_type]["matrix"] = matrix
            cka_final_matrices[subject][cka_type]["stress"] = stress
            print("done!")
        print(f"finished CKA for subject{subject}.")
    print(f"-----finished CKA computations for all subjects-----")
    return cka_final_matrices


preprocessed_shape_neural_data, pcas = shape_preprocess(neural_data, subjects, rois)


def shape_pairwise_matrices():
    shape_final_matrices = defaultdict(_nested_dict_3)
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
            matrix = _normalize_by_mean(shape_pairwise_distance)
            stress = compute_stress(matrix, anatomy_final_matrices[subject])
            shape_final_matrices[subject][f"shape_alpha={alpha}"]["matrix"] = matrix
            shape_final_matrices[subject][f"shape_alpha={alpha}"]["stress"] = stress
            print("done!")
        print(f"finished shape metrics for subject {subject}.")
    print(f"-----finished shape metrics computations for all subjects-----")

    return shape_final_matrices


if __name__ == "__main__":
    print("This script is being run directly.")
    anatomy_final_matrices = anatomical_distance_matrices()
    rsa_final_matrices = rsa_pairwise_matrices()
    cka_final_matrices = cka_pairwise_matrices()
    shape_final_matrices = shape_pairwise_matrices()
