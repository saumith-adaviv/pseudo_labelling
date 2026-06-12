import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from evaluation.multilabel import (
    calibrate_thresholds_knn,
    knn_multilabel_predict,
    evaluate_multilabel,
)


def find_optimal_clusters(embeddings: np.ndarray, k_min: int = 2, k_max: int = 20) -> int:
    """
    Use silhouette score to find the optimal number of clusters.
    Tests k from k_min to k_max and picks the k with the highest score.
    """
    best_k, best_score = k_min, -1

    print(f"\n  Silhouette sweep (k={k_min} to {k_max}):")
    for k in range(k_min, min(k_max + 1, len(embeddings))):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(embeddings)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(embeddings, labels)
        print(f"    k={k:>3}  silhouette={score:.4f}")
        if score > best_score:
            best_score = score
            best_k = k

    print(f"  → Optimal clusters: {best_k}  (silhouette={best_score:.4f})")
    return best_k


def cluster_knn_predict(
    query_embeddings: np.ndarray,
    golden_embeddings: np.ndarray,
    golden_label_matrix: np.ndarray,
    kmeans: KMeans,
    k: int,
    thresholds: np.ndarray,
) -> np.ndarray:
    """
    For each query image:
      1. Find its nearest cluster centroid
      2. Do KNN only within the golden images that belong to that cluster
      3. Vote on labels from those neighbours

    Falls back to global KNN if a cluster has fewer than k images.
    """
    n_query = len(query_embeddings)
    n_classes = golden_label_matrix.shape[1]

    # Assign each golden image to a cluster
    golden_cluster_labels = kmeans.labels_

    # Find nearest cluster for each query image
    from scipy.spatial.distance import cdist
    dist_to_centroids = cdist(query_embeddings, kmeans.cluster_centers_, metric="cosine")
    nearest_cluster = np.argmin(dist_to_centroids, axis=1)  # (n_query,)

    vote_matrix = np.zeros((n_query, n_classes))
    for i in range(n_query):
        cluster_id = nearest_cluster[i]
        cluster_mask = golden_cluster_labels == cluster_id
        cluster_embs = golden_embeddings[cluster_mask]
        cluster_labels = golden_label_matrix[cluster_mask]

        # Fall back to global KNN if cluster is too small
        if len(cluster_embs) < k:
            cluster_embs = golden_embeddings
            cluster_labels = golden_label_matrix

        sim = 1 - cdist(query_embeddings[i:i+1], cluster_embs, metric="cosine")[0]
        top_k_idx = np.argsort(sim)[::-1][:k]
        vote_matrix[i] = cluster_labels[top_k_idx].mean(axis=0)

    predicted = (vote_matrix >= thresholds).astype(int)
    return predicted, vote_matrix


def run_adaptive_cluster_knn(
    golden_embeddings: np.ndarray,
    golden_label_matrix: np.ndarray,
    query_embeddings: np.ndarray,
    query_label_matrix: np.ndarray,
    class_names: list,
    k: int = 10,
    k_min_clusters: int = 2,
    k_max_clusters: int = 20,
) -> dict:
    """
    Full pipeline:
      1. Find optimal number of clusters via silhouette score
      2. Fit KMeans on golden embeddings
      3. Calibrate thresholds using cluster-aware KNN on golden set (leave-one-out)
      4. Predict on query set
      5. Evaluate

    Args:
        golden_embeddings:   (N_golden, D)
        golden_label_matrix: (N_golden, C) binary
        query_embeddings:    (N_query, D)
        query_label_matrix:  (N_query, C) binary
        class_names:         list of C class names
        k:                   number of KNN neighbours within cluster
        k_min_clusters:      minimum clusters to test
        k_max_clusters:      maximum clusters to test

    Returns:
        dict with metrics and optimal_clusters
    """
    print(f"\n{'='*55}")
    print(f"  Adaptive Cluster-KNN")
    print(f"{'='*55}")

    # Step 1 — find optimal clusters
    optimal_k = find_optimal_clusters(golden_embeddings, k_min_clusters, k_max_clusters)

    # Step 2 — fit KMeans
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    kmeans.fit(golden_embeddings)
    print(f"\n  KMeans fitted on {len(golden_embeddings)} golden images → {optimal_k} clusters")
    for c in range(optimal_k):
        count = (kmeans.labels_ == c).sum()
        print(f"    Cluster {c}: {count} images")

    # Step 3 — calibrate thresholds on golden set
    thresholds = calibrate_thresholds_knn(golden_embeddings, golden_label_matrix, k=k)

    # Step 4 — predict on query
    predicted, _ = cluster_knn_predict(
        query_embeddings, golden_embeddings, golden_label_matrix,
        kmeans, k, thresholds
    )

    # Step 5 — evaluate
    metrics = evaluate_multilabel(predicted, query_label_matrix, class_names)

    print(f"\n  Results (k={k}, clusters={optimal_k}):")
    print(f"    Macro F1:    {metrics['macro_f1']:.4f}")
    print(f"    Micro F1:    {metrics['micro_f1']:.4f}")
    print(f"    Hamming:     {metrics['hamming_loss']:.4f}")
    print(f"{'='*55}")

    return {
        "optimal_clusters": optimal_k,
        "k":                k,
        **metrics,
    }
