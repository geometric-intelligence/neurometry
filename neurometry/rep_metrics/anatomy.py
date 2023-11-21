import cortex
import multiprocessing
from multiprocessing.pool import ThreadPool
import itertools
from tqdm import tqdm
import numpy as np


def _ids_from_hemi_to_roi(hemi_ids, pt_id, hemi="left"):
    return list(hemi_ids).index(pt_id)


def _ids_from_roi_to_hemi(hemi_ids, pt_id, hemi="left"):
    return hemi_ids[pt_id]


def get_roi_vertices(subject, rois):
    left_hemisphere, right_hemisphere = [
        cortex.polyutils.Surface(*d) for d in cortex.db.get_surf(subject, "fiducial")
    ]

    num_l = left_hemisphere.pts.shape[0]

    all_rois_vertices = cortex.get_roi_verts(subject, rois)

    left_rois_vertices = {}
    right_rois_vertices = {}
    for roi in rois:
        roi_vertices = all_rois_vertices[roi]
        left_rois_vertices[roi] = roi_vertices[roi_vertices < num_l]
        right_rois_vertices[roi] = roi_vertices[roi_vertices >= num_l]

    return left_rois_vertices, right_rois_vertices


def get_subjects_anatomical_rois(subjects):
    subject_rois = {}
    for subject in subjects:
        subject_rois[subject] = list(cortex.get_roi_verts(subject).keys())
    return subject_rois


def compute_frechet_mean(roi_surface):
    n_points = len(roi_surface.pts)
    dists = np.zeros((n_points, n_points))
    for point_id in range(n_points):
        row = roi_surface.geodesic_distance(verts=[point_id])
        dists[point_id, :] = row

    squared_dists = np.square(dists)

    sum_squared_dists = np.sum(squared_dists, axis=0)
    if sum_squared_dists.size > 0:
        frechet_mean_id = np.argmin(sum_squared_dists)
    else:
        frechet_mean_id = None
        print("Warning: Empty sequence, cannot compute Frechet mean.")
    return frechet_mean_id


def compute_all_frechet_means(subject, surface, rois):
    rois_frechet_mean_ids = {}
    nonempty_rois = []
    for roi in rois:
        vertex_mask = cortex.get_roi_verts(subject, roi=roi, mask=True)[roi]
        roi_surface = surface.create_subsurface(vertex_mask=vertex_mask)
        roi_pts_ids = cortex.get_roi_verts(subject, roi=roi)[roi]
        print(f"computing Frechet mean of {roi}...")
        frechet_mean_id = compute_frechet_mean(roi_surface)
        if frechet_mean_id is not None:
            print("done.")
            frechet_mean_id = _ids_from_roi_to_hemi(roi_pts_ids, frechet_mean_id)
            rois_frechet_mean_ids[roi] = frechet_mean_id
            nonempty_rois.append(roi)
        else:
            print("skipping...")

    return rois_frechet_mean_ids, nonempty_rois


def _compute_cortex_geodesic_dist(i, j, surface, frechet_mean_id_i, frechet_mean_id_j):
    dists_i = surface.geodesic_distance([frechet_mean_id_i])
    dist_ij = dists_i[frechet_mean_id_j]
    return i, j, dist_ij


def _compute_cortex_geodesic_dist_star(args):
    return _compute_cortex_geodesic_dist(*args)


def compute_cortex_pairwise_geodesic_dist(
    subject, rois, left_frechet_means=None, right_frechet_means=None, processes=None
):
    left_surface, right_surface = (
        cortex.polyutils.Surface(*d) for d in cortex.db.get_surf(subject, "fiducial")
    )

    if left_frechet_means is None:
        left_rois_frechet_mean_ids, left_nonempty_rois = compute_all_frechet_means(
            subject, left_surface, rois
        )
    else:
        left_rois_frechet_mean_ids = left_frechet_means
    
    if right_frechet_means is None:
        right_rois_frechet_mean_ids, right_nonempty_rois = compute_all_frechet_means(
            subject, right_surface, rois
        )
    else:
        right_rois_frechet_mean_ids = right_frechet_means


    left_frechet_mean_ids_list = list(left_rois_frechet_mean_ids.values())
    right_frechet_mean_ids_list = list(right_rois_frechet_mean_ids.values())

    left_n = len(left_nonempty_rois)

    left_n_dists = left_n * (left_n - 1) / 2

    ij = itertools.combinations(range(left_n), 2)
    args = (
        (i, j, left_surface, left_frechet_mean_ids_list[i], left_frechet_mean_ids_list[j]) for i, j in ij
    )

    print(
        f"Parallelizing n(n-1)/2 = {int(left_n_dists)} left hemisphere distance calculations with {multiprocessing.cpu_count() if processes is None else processes} processes."
    )
    pbar = lambda x: tqdm(x, total=left_n_dists, desc="Computing distances on left cortex")

    with ThreadPool(processes=processes) as pool:
        results = []
        for result in pbar(pool.imap(_compute_cortex_geodesic_dist_star, args)):
            results.append(result)

    left_cortex_pairwise_dists = np.zeros((left_n, left_n))

    for i, j, geodesic_dist in results:
        left_cortex_pairwise_dists[i, j], left_cortex_pairwise_dists[j, i] = (
            geodesic_dist,
            geodesic_dist,
        )

    right_n = len(right_nonempty_rois)

    right_n_dists = right_n * (right_n - 1) / 2

    ij = itertools.combinations(range(right_n), 2)
    args = (
        (i, j, right_surface, right_frechet_mean_ids_list[i], right_frechet_mean_ids_list[j]) for i, j in ij
    )

    print(
        f"Parallelizing n(n-1)/2 = {int(right_n_dists)} right hemisphere distance calculations with {multiprocessing.cpu_count() if processes is None else processes} processes."
    )
    pbar = lambda x: tqdm(x, total=right_n_dists, desc="Computing distances on left cortex")

    with ThreadPool(processes=processes) as pool:
        results = []
        for result in pbar(pool.imap(_compute_cortex_geodesic_dist_star, args)):
            results.append(result)

    right_cortex_pairwise_dists = np.zeros((right_n, right_n))

    for i, j, geodesic_dist in results:
        right_cortex_pairwise_dists[i, j], right_cortex_pairwise_dists[j, i] = (
            geodesic_dist,
            geodesic_dist,
        )


    return left_cortex_pairwise_dists, right_cortex_pairwise_dists, left_nonempty_rois, right_nonempty_rois


def get_roi_list_intersection(roi_list1, roi_list2):
    set1 = set(roi_list1)
    set2 = set(roi_list2)

    intersection = set1 & set2

    return list(intersection)
