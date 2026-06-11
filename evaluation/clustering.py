import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from typing import Tuple


def compute_cosine_similarity(embeddings: np.ndarray) -> np.ndarray:
    """
    Computes pairwise cosine similarity matrix.

    Args:
        embeddings: numpy array of shape (N, embedding_dim).

    Returns:
        similarity_matrix: numpy array of shape (N, N).
    """
    return cosine_similarity(embeddings)


def run_kmeans(
    embeddings: np.ndarray,
    n_clusters: int = 5,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Runs KMeans clustering on embeddings.
    Since there are no labels, clusters are used purely for visual grouping.

    Args:
        embeddings: numpy array of shape (N, embedding_dim).
        n_clusters: number of clusters (default 5 — adjust based on visual inspection).
        seed:       random seed for reproducibility.

    Returns:
        labels:   cluster label per image, shape (N,).
        centers:  cluster centroids, shape (n_clusters, embedding_dim).
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init="auto")
    labels = kmeans.fit_predict(embeddings)
    centers = kmeans.cluster_centers_
    return labels, centers


def get_nearest_neighbors(
    embeddings: np.ndarray,
    k: int = 5
) -> np.ndarray:
    """
    For each image, finds the k most similar images by cosine similarity.

    Args:
        embeddings: numpy array of shape (N, embedding_dim).
        k:          number of nearest neighbors.

    Returns:
        neighbors: numpy array of shape (N, k) — indices of nearest neighbors.
    """
    sim_matrix = cosine_similarity(embeddings)
    np.fill_diagonal(sim_matrix, -1)  # exclude self
    neighbors = np.argsort(sim_matrix, axis=1)[:, -k:][:, ::-1]
    return neighbors
