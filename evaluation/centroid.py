import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Tuple


def compute_centroids(
    embeddings: np.ndarray,
    labels: List[int],
    n_classes: int
) -> np.ndarray:
    """
    Computes one centroid per class by averaging all golden set
    embeddings belonging to that class.

    Args:
        embeddings: numpy array (N_golden, embedding_dim)
        labels:     list of class indices for each golden set image
        n_classes:  total number of classes (8)

    Returns:
        centroids: numpy array (n_classes, embedding_dim)
    """
    embedding_dim = embeddings.shape[1]
    centroids = np.zeros((n_classes, embedding_dim))

    for class_idx in range(n_classes):
        class_mask = [i for i, l in enumerate(labels) if l == class_idx]
        if len(class_mask) == 0:
            print(f"[centroid] Warning: no golden set images for class {class_idx}")
            continue
        class_embeddings = embeddings[class_mask]
        centroid = class_embeddings.mean(axis=0)
        # Normalize centroid
        centroid = centroid / np.linalg.norm(centroid)
        centroids[class_idx] = centroid

    return centroids


def match_query_to_centroid(
    query_embeddings: np.ndarray,
    centroids: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    For each query image finds the nearest centroid and assigns that label.

    Args:
        query_embeddings: numpy array (N_query, embedding_dim)
        centroids:        numpy array (n_classes, embedding_dim)

    Returns:
        predicted_labels:      numpy array (N_query,) — predicted class index
        confidence_scores:     numpy array (N_query,) — cosine similarity to nearest centroid
    """
    # Similarity between every query image and every centroid
    sim_matrix = cosine_similarity(query_embeddings, centroids)  # (N_query, n_classes)

    predicted_labels  = np.argmax(sim_matrix, axis=1)
    confidence_scores = np.max(sim_matrix, axis=1)

    return predicted_labels, confidence_scores


def evaluate(
    predicted_labels: np.ndarray,
    true_labels: List[int],
    class_names: List[str],
    confidence_scores: np.ndarray
) -> Dict:
    """
    Computes overall accuracy and per-class accuracy.

    Args:
        predicted_labels:  numpy array (N_query,)
        true_labels:       list of ground truth class indices
        class_names:       list of class name strings
        confidence_scores: numpy array (N_query,)

    Returns:
        results: dict with overall accuracy, per-class accuracy, avg confidence
    """
    true_labels = np.array(true_labels)
    correct = predicted_labels == true_labels
    overall_accuracy = correct.mean() * 100

    per_class = {}
    for class_idx, class_name in enumerate(class_names):
        class_mask = true_labels == class_idx
        if class_mask.sum() == 0:
            continue
        class_accuracy = correct[class_mask].mean() * 100
        avg_confidence = confidence_scores[class_mask].mean()
        per_class[class_name] = {
            "accuracy":       round(class_accuracy, 2),
            "avg_confidence": round(float(avg_confidence), 4),
            "n_query":        int(class_mask.sum())
        }

    return {
        "overall_accuracy": round(overall_accuracy, 2),
        "avg_confidence":   round(float(confidence_scores.mean()), 4),
        "per_class":        per_class
    }
