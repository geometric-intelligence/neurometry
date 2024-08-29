import numpy as np
from dreimac import CircularCoords, ToroidalCoords
from gtda.homology import VietorisRipsPersistence, WeightedRipsPersistence
from gtda.diagrams import PersistenceEntropy
import neurometry.datasets.synthetic as synthetic


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import PCA
import os
os.environ["GEOMSTATS_BACKEND"] = "pytorch"
import geomstats.backend as gs


class TopologicalClassifier(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        num_samples,
        poisson_multiplier,
        homology_dimensions=[0, 1, 2],
        reduce_dim=False,
    ):
        self.num_samples = num_samples
        self.poisson_multiplier = poisson_multiplier
        self.homology_dimensions = homology_dimensions
        self.reduce_dim = reduce_dim
        self.classifier = RandomForestClassifier()

    def _generate_ref_data(self, input_data):
        num_points = input_data.shape[0]
        encoding_dim = input_data.shape[1]
        circle_task_points = synthetic.hypersphere(1, num_points)
        circle_point_clouds = []
        for i in range(self.num_samples):
            circle_noisy_points, _ = synthetic.synthetic_neural_manifold(
                points=circle_task_points,
                encoding_dim=encoding_dim,
                nonlinearity="sigmoid",
                scales=gs.ones(encoding_dim),
                poisson_multiplier=self.poisson_multiplier,
            )
            circle_point_clouds.append(circle_noisy_points)

        sphere_task_points = synthetic.hypersphere(2, num_points)
        sphere_point_clouds = []
        for i in range(self.num_samples):
            sphere_noisy_points, _ = synthetic.synthetic_neural_manifold(
                points=sphere_task_points,
                encoding_dim=encoding_dim,
                nonlinearity="sigmoid",
                scales=gs.ones(encoding_dim),
                poisson_multiplier=self.poisson_multiplier,
            )
            sphere_point_clouds.append(sphere_noisy_points)

        torus_task_points = synthetic.hypertorus(2, num_points)
        torus_point_clouds = []
        for i in range(self.num_samples):
            torus_noisy_points, _ = synthetic.synthetic_neural_manifold(
                points=torus_task_points,
                encoding_dim=encoding_dim,
                nonlinearity="sigmoid",
                scales=gs.ones(encoding_dim),
                poisson_multiplier=self.poisson_multiplier,
            )
            torus_point_clouds.append(torus_noisy_points)

        circle_labels = np.zeros(self.num_samples)
        sphere_labels = np.ones(self.num_samples)
        torus_labels = 2 * np.ones(self.num_samples)
        ref_labels = np.concatenate(
            [
                circle_labels,
                sphere_labels,
                torus_labels,
            ]
        )

        ref_point_clouds = [
            *circle_point_clouds,
            *sphere_point_clouds,
            *torus_point_clouds,
        ]

        return ref_point_clouds, ref_labels

    def _compute_topo_features(self, diagrams):
        PE = PersistenceEntropy()
        features = PE.fit_transform(diagrams)
        return features

    def fit(self, X, y=None):
        ref_point_clouds, ref_labels = self._generate_ref_data(X)
        if self.reduce_dim:
            pca = PCA(n_components=10)
            ref_point_clouds = [
                pca.fit_transform(point_cloud) for point_cloud in ref_point_clouds
            ]
        ref_diagrams = compute_persistence_diagrams(
            ref_point_clouds, homology_dimensions=self.homology_dimensions
        )
        ref_features = self._compute_topo_features(ref_diagrams)
        X_ref_train, X_ref_valid, y_ref_train, y_ref_valid = train_test_split(
            ref_features, ref_labels
        )
        self.classifier.fit(X_ref_train, y_ref_train)
        print(f"Classifier score: {self.classifier.score(X_ref_valid, y_ref_valid)}")
        return self

    def predict(self, X):
        if self.reduce_dim:
            pca = PCA(n_components=10)
            X = pca.fit_transform(X)
        diagram = compute_persistence_diagrams(
            [X], homology_dimensions=self.homology_dimensions
        )
        features = self._compute_topo_features(diagram)
        return self.classifier.predict(features)





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


