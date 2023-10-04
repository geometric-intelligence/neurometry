import cortex
import multiprocessing
import itertools
from tqdm import tqdm
import numpy as np


def _ids_from_hemi_to_roi(hemi_ids, pt_id, hemi="left"):
    return list(hemi_ids).index(pt_id)


def _ids_from_roi_to_hemi(hemi_ids, pt_id, hemi="left"):
    return hemi_ids[pt_id]


def get_roi_vertices(subject,rois):
    left_hemisphere, right_hemisphere = [
        cortex.polyutils.Surface(*d) for d in cortex.db.get_surf(subject, "fiducial")
    ]

    num_l = left_hemisphere.pts.shape[0]

    all_rois_vertices = cortex.get_roi_verts(subject,rois)
    
    left_rois_vertices = {}
    right_rois_vertices = {}
    for roi in rois:
        roi_vertices = all_rois_vertices[roi]
        left_rois_vertices[roi] = roi_vertices[roi_vertices<num_l]
        right_rois_vertices[roi] = roi_vertices[roi_vertices>=num_l]

    return left_rois_vertices, right_rois_vertices




def compute_frechet_mean(roi_surface, hemi="left"):
    n_points = len(roi_surface.pts)
    dists = np.zeros((n_points, n_points))
    for point_id in range(n_points):
        row = roi_surface.geodesic_distance(verts=[point_id])
        dists[point_id, :] = row

    squared_dists = np.square(dists)

    sum_squared_dists = np.sum(squared_dists, axis=0)

    frechet_mean_id = np.argmin(sum_squared_dists)

    # roi_pts_ids = cortex.get_roi_verts("subj01", roi="PPA", mask=False)

    # roi_pts_ids = cortex.get_roi_verts("subj01", roi="PPA")["PPA"]

    # left_roi_pts_ids = roi_pts_ids[roi_pts_ids < n_points]

    # frechet_mean_id_left = _ids_from_roi_to_hemi(roi_pts_ids, frechet_mean_id)   

    return frechet_mean_id


def _compute_cortex_geodesic_dist(i, j, roi1, roi2):
    geodesic_dist = None
    return i, j, geodesic_dist


def _compute_cortex_geodesic_dist_star(args):
    return _compute_cortex_geodesic_dist(*args)

def compute_cortex_pairwise_geodesic_dist(subject, functional_rois, processes=None):

    left_rois_vertices, right_rois_vertices = get_roi_vertices(subject, functional_rois)

    left_rois_frechet_means = compute_frechet_mean(left_rois_vertices)
    right_rois_frechet_means = compute_frechet_mean(right_rois_vertices)
    
    n = len(functional_rois)

    n_dists = n*(n-1)/2

    ij = itertools.combinations(range(n),2)
    args = ((i,j, left_rois_frechet_means[i], left_rois_frechet_means[j]) for i, j in ij)

    print(f"Parallelizing n(n-1)/2 = {int(n_dists)} distance calculations with {multiprocessing.cpu_count() if processes is None else processes} processes.")
    pbar = lambda x: tqdm(x, total=n_dists, desc="Computing distances on left cortex")

    with multiprocessing.pool.ThreadPool(processes=processes) as pool:
        results = []
        for result in pbar(pool.imap_unordered(_compute_cortex_geodesic_dist_star, args)):
            results.append(result)



    cortex_pairwise_dists = np.zeros((n,n))

    for i, j, geodesic_dist in results:
        cortex_pairwise_dists[i,j], cortex_pairwise_dists[j,i] = geodesic_dist, geodesic_dist

    return cortex_pairwise_dists






