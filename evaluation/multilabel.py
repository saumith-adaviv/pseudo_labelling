"""
Multi-Label Classification Methods
====================================
Three methods implemented on top of frozen embeddings:

Method 1 — KNN Thresholding (from Adaptive Global-Local paper)
    For each query image, find K nearest golden set images.
    Vote on labels from neighbors. Per-class adaptive threshold.

Method 2 — Partial Aggregation Prototypes (Pattern Recognition 2025)
    When building centroid for class C, only use images that have label C.
    Fixes centroid pollution from multi-label images.

Method 3 — Adaptive Per-Class Threshold Calibration
    Calibrate one threshold per class on golden set leave-one-out.
    Threshold = value that maximizes F1 for that class.
"""

import numpy as np
from sklearn.metrics import f1_score, classification_report
from scipy.spatial.distance import cdist


# ── Method 1: KNN Thresholding ────────────────────────────────────────────────

def knn_multilabel_predict(
    query_embeddings: np.ndarray,
    golden_embeddings: np.ndarray,
    golden_label_matrix: np.ndarray,
    k: int = 10,
    thresholds: np.ndarray = None,
) -> np.ndarray:
    """
    For each query image:
      1. Find K nearest golden images by cosine similarity
      2. Count how many neighbors have each label
      3. Label fires if vote_fraction > threshold_for_class

    Args:
        query_embeddings  : (N_query, dim)
        golden_embeddings : (N_golden, dim)
        golden_label_matrix: (N_golden, n_classes) binary matrix
        k                 : number of neighbors
        thresholds        : (n_classes,) per-class threshold, default 0.5 for all

    Returns:
        predicted_matrix  : (N_query, n_classes) binary predictions
        vote_matrix       : (N_query, n_classes) raw vote fractions (confidence)
    """
    n_classes = golden_label_matrix.shape[1]
    if thresholds is None:
        thresholds = np.full(n_classes, 0.5)

    # Cosine similarity = 1 - cosine distance (embeddings already L2-normalised)
    sim_matrix = 1 - cdist(query_embeddings, golden_embeddings, metric="cosine")  # (N_query, N_golden)

    vote_matrix = np.zeros((len(query_embeddings), n_classes))
    for i in range(len(query_embeddings)):
        top_k_idx = np.argsort(sim_matrix[i])[::-1][:k]
        neighbor_labels = golden_label_matrix[top_k_idx]          # (k, n_classes)
        vote_matrix[i] = neighbor_labels.mean(axis=0)             # fraction per class

    predicted_matrix = (vote_matrix >= thresholds).astype(int)
    return predicted_matrix, vote_matrix


def calibrate_thresholds_knn(
    golden_embeddings: np.ndarray,
    golden_label_matrix: np.ndarray,
    k: int = 10,
    n_thresholds: int = 20,
) -> np.ndarray:
    """
    Leave-one-out calibration on the golden set.
    For each class, find the threshold that maximises F1.

    Returns:
        thresholds: (n_classes,) optimal threshold per class
    """
    n_golden, n_classes = golden_label_matrix.shape
    thresholds = np.full(n_classes, 0.5)

    sim_matrix = 1 - cdist(golden_embeddings, golden_embeddings, metric="cosine")
    np.fill_diagonal(sim_matrix, -1)   # exclude self

    vote_matrix = np.zeros((n_golden, n_classes))
    for i in range(n_golden):
        top_k_idx = np.argsort(sim_matrix[i])[::-1][:k]
        vote_matrix[i] = golden_label_matrix[top_k_idx].mean(axis=0)

    candidates = np.linspace(0.1, 0.9, n_thresholds)
    for c in range(n_classes):
        true_c = golden_label_matrix[:, c]
        if true_c.sum() == 0:
            continue
        best_f1, best_t = 0, 0.5
        for t in candidates:
            pred_c = (vote_matrix[:, c] >= t).astype(int)
            f1 = f1_score(true_c, pred_c, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thresholds[c] = best_t

    return thresholds


# ── Method 2: Partial Aggregation Prototypes ──────────────────────────────────

def partial_aggregation_centroids(
    golden_embeddings: np.ndarray,
    golden_label_matrix: np.ndarray,
    n_classes: int,
) -> np.ndarray:
    """
    Compute one centroid per class using ONLY golden images that carry that label.
    Fixes centroid pollution when an image has multiple defects.

    Args:
        golden_embeddings  : (N_golden, dim)
        golden_label_matrix: (N_golden, n_classes) binary
        n_classes          : number of classes

    Returns:
        centroids          : (n_classes, dim) L2-normalised
    """
    dim = golden_embeddings.shape[1]
    centroids = np.zeros((n_classes, dim))

    for c in range(n_classes):
        mask = golden_label_matrix[:, c].astype(bool)
        if mask.sum() == 0:
            print(f"  ⚠️  Class {c} has no golden images — centroid will be zeros")
            continue
        centroid = golden_embeddings[mask].mean(axis=0)
        norm = np.linalg.norm(centroid)
        centroids[c] = centroid / norm if norm > 0 else centroid

    return centroids


def centroid_multilabel_predict(
    query_embeddings: np.ndarray,
    centroids: np.ndarray,
    thresholds: np.ndarray = None,
) -> np.ndarray:
    """
    Cosine similarity of each query to every centroid.
    Label fires if similarity > threshold_for_class.

    Args:
        query_embeddings : (N_query, dim)
        centroids        : (n_classes, dim)
        thresholds       : (n_classes,) per-class, default 0.5

    Returns:
        predicted_matrix : (N_query, n_classes) binary
        sim_matrix       : (N_query, n_classes) raw similarities
    """
    n_classes = centroids.shape[0]
    if thresholds is None:
        thresholds = np.full(n_classes, 0.5)

    sim_matrix = 1 - cdist(query_embeddings, centroids, metric="cosine")  # (N_query, n_classes)
    predicted_matrix = (sim_matrix >= thresholds).astype(int)
    return predicted_matrix, sim_matrix


def calibrate_thresholds_centroid(
    golden_embeddings: np.ndarray,
    golden_label_matrix: np.ndarray,
    centroids: np.ndarray,
    n_thresholds: int = 20,
) -> np.ndarray:
    """
    Find per-class threshold on golden set that maximises F1.
    Uses centroid similarity scores (not KNN votes).
    """
    n_classes = centroids.shape[0]
    sim_matrix = 1 - cdist(golden_embeddings, centroids, metric="cosine")
    candidates = np.linspace(0.1, 0.9, n_thresholds)
    thresholds = np.full(n_classes, 0.5)

    for c in range(n_classes):
        true_c = golden_label_matrix[:, c]
        if true_c.sum() == 0:
            continue
        best_f1, best_t = 0, 0.5
        for t in candidates:
            pred_c = (sim_matrix[:, c] >= t).astype(int)
            f1 = f1_score(true_c, pred_c, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thresholds[c] = best_t

    return thresholds


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_multilabel(
    predicted_matrix: np.ndarray,
    true_matrix: np.ndarray,
    class_names: list,
) -> dict:
    """
    Evaluate multi-label predictions.
    Returns dict with macro/micro F1, hamming loss, per-class metrics.
    """
    from sklearn.metrics import hamming_loss, multilabel_confusion_matrix

    macro_f1  = f1_score(true_matrix, predicted_matrix, average="macro",  zero_division=0)
    micro_f1  = f1_score(true_matrix, predicted_matrix, average="micro",  zero_division=0)
    sample_f1 = f1_score(true_matrix, predicted_matrix, average="samples", zero_division=0)
    h_loss    = hamming_loss(true_matrix, predicted_matrix)

    per_class = {}
    for i, cls in enumerate(class_names):
        t = true_matrix[:, i]
        p = predicted_matrix[:, i]
        per_class[cls] = {
            "f1":        round(float(f1_score(t, p, zero_division=0)), 4),
            "precision": round(float(__import__("sklearn.metrics", fromlist=["precision_score"]).precision_score(t, p, zero_division=0)), 4),
            "recall":    round(float(__import__("sklearn.metrics", fromlist=["recall_score"]).recall_score(t, p, zero_division=0)), 4),
            "support":   int(t.sum()),
        }

    return {
        "macro_f1":   round(float(macro_f1), 4),
        "micro_f1":   round(float(micro_f1), 4),
        "sample_f1":  round(float(sample_f1), 4),
        "hamming_loss": round(float(h_loss), 4),
        "per_class":  per_class,
    }


def single_label_to_matrix(labels: np.ndarray, n_classes: int) -> np.ndarray:
    """
    Convert single-label array [0,1,2,...] to binary matrix (N, n_classes).
    Used when multi-label audit finds no overlap — falls back to single-label mode.
    """
    matrix = np.zeros((len(labels), n_classes), dtype=int)
    for i, lbl in enumerate(labels):
        matrix[i, int(lbl)] = 1
    return matrix
