from gtda.diagrams import PairwiseDistance


def compute_pairwise_distances(diagrams, metric="bottleneck"):
    PD = PairwiseDistance(metric=metric)
    return PD.fit_transform(diagrams)


def compare_representation_to_references(
    representation, reference_topologies, metric="bottleneck"
):
    raise NotImplementedError
