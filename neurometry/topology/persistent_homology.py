import numpy as np
from dreimac import CircularCoords, ToroidalCoords
from gtda.homology import VietorisRipsPersistence, WeightedRipsPersistence


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
            metric=metric, homology_dimensions=homology_dimensions, coeff=coeff,
        )
        diagrams = WR.fit_transform(representations)
    else:
        VR = VietorisRipsPersistence(
            metric=metric, homology_dimensions=homology_dimensions, coeff=coeff, reduced_homology=False, n_jobs=n_jobs)
        diagrams = VR.fit_transform(representations)
    return diagrams


def _shuffle_entries(data, rng):
    return np.array([rng.permutation(row) for row in data])


def compute_diagrams_shuffle(X, num_shuffles, seed=0, homology_dimensions=(0, 1)):
    rng = np.random.default_rng(seed)
    shuffled_Xs = [_shuffle_entries(X, rng) for _ in range(num_shuffles)]
    return compute_persistence_diagrams(
        [X, *shuffled_Xs], homology_dimensions=homology_dimensions
    )

def cohomological_toroidal_coordinates(data):
    n_landmarks = data.shape[0]
    tc = ToroidalCoords(data, n_landmarks=n_landmarks)
    cohomology_classes = [0,1]
    toroidal_coords = tc.get_coordinates(cocycle_idxs=cohomology_classes,standard_range=False)
    return toroidal_coords.T


def cohomological_circular_coordinates(data):
    n_landmarks = data.shape[0]
    cc = CircularCoords(data, n_landmarks=n_landmarks)
    circular_coords = cc.get_coordinates(standard_range=False)
    return circular_coords.T


