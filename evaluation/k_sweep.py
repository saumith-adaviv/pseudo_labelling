import numpy as np
from evaluation.multilabel import (
    calibrate_thresholds_knn,
    knn_multilabel_predict,
    evaluate_multilabel,
)


K_VALUES = [5, 10, 15, 20, 30]


def run_k_sweep(
    golden_embeddings: np.ndarray,
    golden_label_matrix: np.ndarray,
    query_embeddings: np.ndarray,
    query_label_matrix: np.ndarray,
    class_names: list,
    k_values: list = K_VALUES,
) -> dict:
    """
    Sweeps over k values for KNN multi-label classification.
    Returns results dict and the best k by macro F1.

    Args:
        golden_embeddings:   (N_golden, D) float32
        golden_label_matrix: (N_golden, C) binary int
        query_embeddings:    (N_query, D) float32
        query_label_matrix:  (N_query, C) binary int
        class_names:         list of C class name strings
        k_values:            list of k values to test

    Returns:
        {
            "results": [ {k, macro_f1, micro_f1, hamming_loss, thresholds}, ... ],
            "best_k":  int,
            "best_macro_f1": float,
        }
    """
    results = []

    print(f"\n{'='*55}")
    print(f"  K Sweep — testing k = {k_values}")
    print(f"{'='*55}")
    print(f"{'k':>5} | {'Macro F1':>10} | {'Micro F1':>10} | {'Hamming':>10}")
    print(f"{'-'*5}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")

    for k in k_values:
        # Calibrate thresholds using leave-one-out on golden set
        thresholds = calibrate_thresholds_knn(
            golden_embeddings, golden_label_matrix, k=k
        )

        # Predict on query set
        predicted = knn_multilabel_predict(
            query_embeddings, golden_embeddings, golden_label_matrix,
            k=k, thresholds=thresholds
        )

        # Evaluate
        metrics = evaluate_multilabel(predicted, query_label_matrix, class_names)

        results.append({
            "k":           k,
            "macro_f1":    metrics["macro_f1"],
            "micro_f1":    metrics["micro_f1"],
            "hamming_loss": metrics["hamming_loss"],
            "thresholds":  thresholds.tolist(),
        })

        print(f"{k:>5} | {metrics['macro_f1']:>10.4f} | {metrics['micro_f1']:>10.4f} | {metrics['hamming_loss']:>10.4f}")

    print(f"{'='*55}")

    # Pick best k by macro F1
    best = max(results, key=lambda r: r["macro_f1"])
    print(f"\n  Best k = {best['k']}  →  Macro F1 = {best['macro_f1']:.4f}\n")

    return {
        "results":       results,
        "best_k":        best["k"],
        "best_macro_f1": best["macro_f1"],
    }
