import config
import itertools
from collections import defaultdict
from .load_nsd import load_nsd, get_neural_data
from .dissimilarity import compute_rsa_pairwise_dissimilarities, compute_shape_pairwise_distances
from .anatomy import get_subjects_rois, get_roi_list_intersection, compute_cortex_pairwise_geodesic_dist

# NSD data
target_regions = config.target_regions
subjects = config.subjects
response_data, voxel_metadata, stimulus_data, functional_rois = load_nsd(target_regions)
neural_data = get_neural_data(subjects,functional_rois, voxel_metadata, response_data)


# RSA
rdm_compute_types = config.rdm_compute_types
rdm_compare_types = config.rdm_compare_types

#Shape metrics
alphas = config.alphas

def _nested_dict():
    return defaultdict(str)

def rsa_pairwise_matrices():
    rsa_types = list(itertools.product(rdm_compute_types, rdm_compare_types))
    rsa_final_matrices = defaultdict(_nested_dict)
    for subject in subjects:
        for rsa_type in rsa_types:
            rsa_type_name = "_".join(rsa_type)
            rsa_pairwise_dissimilarity = compute_rsa_pairwise_dissimilarities(neural_data, rsa_type[0], rsa_type[1])
            rsa_final_matrices[subject][rsa_type_name] = rsa_pairwise_dissimilarity
    return rsa_final_matrices


def shape_pairwise_matrices():
    shape_final_matrices = defaultdict(_nested_dict)
    for subject in subjects:
        for alpha in alphas:
            shape_pairwise_distance = compute_shape_pairwise_distances(neural_data,stimulus_data,alpha)
            shape_final_matrices[subject][alpha] = shape_pairwise_distance

    return shape_final_matrices

def cka_pairwise_matrices():
    return NotImplementedError


def anatomical_distance_matrices():
    anatomical_rois = get_subjects_rois(subjects)
    rois = get_roi_list_intersection(functional_rois, anatomical_rois)
    anatomy_final_matrices = defaultdict(_nested_dict)
    for subject in subjects:
        cortex_pairwise_dists = compute_cortex_pairwise_geodesic_dist(subject, rois)
        anatomy_final_matrices[subject] = cortex_pairwise_dists
    return anatomy_final_matrices

