from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics.pairwise import cosine_similarity

from facesorter.clustering import cluster_embeddings


def at_angle(degrees: float) -> np.ndarray:
    """2-D unit vector at the given angle; cosine distance = 1 - cos(angle difference)."""
    rad = np.radians(degrees)
    return np.array([np.cos(rad), np.sin(rad)])


def partition(labels) -> set[frozenset[int]]:
    """Label-independent view of a clustering: the set of index groups."""
    groups: dict[int, set[int]] = {}
    for index, label in enumerate(labels):
        groups.setdefault(int(label), set()).add(index)
    return {frozenset(g) for g in groups.values()}


def test_empty_input_gives_empty_int_array():
    labels = cluster_embeddings([], 0.6)
    assert labels.shape == (0,)
    assert labels.dtype.kind == "i"


def test_single_embedding_is_its_own_cluster():
    assert list(cluster_embeddings([np.array([1.0, 2.0, 3.0])], 0.6)) == [0]


def test_two_separated_identities_make_two_clusters():
    rng = np.random.default_rng(0)
    alice = [at_angle(0 + d) for d in rng.uniform(-5, 5, 4)]
    bob = [at_angle(90 + d) for d in rng.uniform(-5, 5, 4)]
    labels = cluster_embeddings(alice + bob, 0.3)
    assert partition(labels) == {frozenset(range(4)), frozenset(range(4, 8))}


def test_every_face_gets_a_cluster_never_noise():
    labels = cluster_embeddings([at_angle(a) for a in (0, 90, 180, 270)], 0.1)
    assert sorted(labels) == [0, 1, 2, 3]


def test_labels_follow_first_seen_order():
    labels = cluster_embeddings([at_angle(90), at_angle(0), at_angle(90)], 0.1)
    assert list(labels) == [0, 1, 0]


@pytest.mark.parametrize(
    ("threshold", "expected_clusters"),
    [(0.1, 3), (0.2, 3), (0.3, 1), (0.9, 1)],
)
def test_threshold_controls_granularity_and_chaining(threshold, expected_clusters):
    # 0, 40 and 80 degrees: neighbours are ~0.234 apart but the ends ~0.826 apart, so a
    # threshold in between joins all three through the middle face (single-linkage chaining).
    labels = cluster_embeddings([at_angle(0), at_angle(40), at_angle(80)], threshold)
    assert len(set(labels)) == expected_clusters


def test_threshold_boundary_is_inclusive():
    pair = [np.array([3.0, 4.0]), np.array([1.0, 0.0])]
    distance = float(np.clip(1 - cosine_similarity(pair)[0, 1], 0, None))
    assert len(set(cluster_embeddings(pair, distance))) == 1  # DBSCAN joins at distance <= eps
    assert len(set(cluster_embeddings(pair, float(np.nextafter(distance, 0))))) == 2


def test_vector_length_does_not_matter():
    labels = cluster_embeddings([at_angle(0) * 100, at_angle(1) * 0.01], 0.05)
    assert list(labels) == [0, 0]


def test_zero_threshold_is_usable():
    """Config accepts 0.0, but DBSCAN needs eps > 0: it must not crash."""
    vec = np.random.default_rng(1).normal(size=64)
    assert list(cluster_embeddings([vec, vec.copy()], 0.0)) == [0, 0]


def test_tiny_threshold_clusters_exact_duplicates():
    vec = np.random.default_rng(1).normal(size=64)
    assert list(cluster_embeddings([vec, vec.copy()], 1e-9)) == [0, 0]


@pytest.mark.parametrize("seed", range(5))
def test_result_does_not_depend_on_input_order(seed):
    rng = np.random.default_rng(seed)
    centres = np.eye(6)[:3]
    points = np.vstack([c + rng.normal(scale=0.02, size=6) for c in centres for _ in range(5)])
    order = rng.permutation(len(points))
    original = partition(cluster_embeddings(list(points), 0.2))
    shuffled = cluster_embeddings(list(points[order]), 0.2)
    unshuffled = {frozenset(int(order[i]) for i in group) for group in partition(shuffled)}
    assert unshuffled == original
    assert len(original) == 3


@pytest.mark.parametrize("seed", range(5))
def test_larger_threshold_never_creates_more_clusters(seed):
    points = np.random.default_rng(seed).normal(size=(30, 5))
    counts = [len(set(cluster_embeddings(list(points), t))) for t in (0.2, 0.6, 1.0, 1.5, 2.0)]
    assert counts == sorted(counts, reverse=True)
    assert counts[-1] == 1  # distance never exceeds 2
