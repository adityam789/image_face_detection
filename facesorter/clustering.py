"""
facesorter/clustering.py
Clustering logic for face embeddings.
"""
from typing import List
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity

def cluster_embeddings(embeddings: List[np.ndarray], threshold: float) -> np.ndarray:
    """
    Cluster face embeddings using DBSCAN with cosine distance.
    
    Args:
        embeddings: List of face embedding vectors
        threshold: Distance threshold (lower = stricter, range 0-1 matches cosine dist)
        
    Returns:
        np.ndarray: Cluster labels (-1 means noise/unclustered)
    """
    if len(embeddings) == 0:
        return np.array([])
    sim_matrix = cosine_similarity(embeddings)
    dist_matrix = 1 - sim_matrix
    dist_matrix = np.clip(dist_matrix, 0, None)
    clustering = DBSCAN(eps=threshold, min_samples=1, metric='precomputed')
    labels = clustering.fit_predict(dist_matrix)
    return labels
