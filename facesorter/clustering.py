"""
facesorter/clustering.py
Clustering logic for face embeddings.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity

_MIN_EPS = 1e-12


def cluster_embeddings(embeddings: Sequence[np.ndarray], threshold: float) -> np.ndarray:
    """
    Cluster face embeddings using DBSCAN with cosine distance.

    Note on ``min_samples=1``: every face is a core point, so no face is ever labelled
    noise (-1) and each face ends up in some cluster (possibly a singleton). The cost is
    single-linkage chaining: if A is within ``threshold`` of B and B is within
    ``threshold`` of C, all three share a cluster even when A and C are far apart. One
    ambiguous face (blurry, profile view, a look-alike) can therefore bridge and merge
    two different people. Lowering ``threshold`` reduces chaining at the price of
    splitting one person across several clusters. This is a deliberate simplicity
    tradeoff; a larger ``min_samples`` or a centroid-based method would be stricter.

    Args:
        embeddings: Face embedding vectors.
        threshold: Cosine distance threshold (lower = stricter).

    Returns:
        np.ndarray: One cluster label per embedding (empty array for empty input).
    """
    if len(embeddings) == 0:
        return np.array([], dtype=int)
    dist_matrix = np.clip(1 - cosine_similarity(embeddings), 0, None)
    # DBSCAN requires eps > 0, but a threshold of 0 ("only identical faces") is a valid
    # setting, so floor it at a value far below any real embedding distance.
    clustering = DBSCAN(eps=max(threshold, _MIN_EPS), min_samples=1, metric="precomputed")
    return clustering.fit_predict(dist_matrix)
