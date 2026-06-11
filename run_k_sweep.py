"""
Run k sweep across all models to find optimal k for KNN multi-label classification.
Usage:
    python run_k_sweep.py --data /content/drive/MyDrive/preprocessed
    python run_k_sweep.py --data /content/drive/MyDrive/HQ_tomato_multi_defect.v4-v0.multiclass
"""
import argparse
import json
import os
import numpy as np
import torch

from data.image_loader import load_dataset
from data.audit_multilabel import build_label_matrix
from embeddings.extractor import extract_embeddings
from evaluation.k_sweep import run_k_sweep
from models import MODEL_REGISTRY

# ── Config ────────────────────────────────────────────────────────────────────
MODELS_TO_RUN = ["siglip2", "dinov2", "bioclip2", "inat_eva02", "evaclip"]
K_VALUES      = [5, 10, 15, 20, 30]
OUTPUT_FILE   = "results/k_sweep_results.json"
# ─────────────────────────────────────────────────────────────────────────────


def main(data_path: str):
    os.makedirs("results", exist_ok=True)

    print(f"\n[k_sweep] Dataset: {data_path}")

    # Load dataset
    golden, query = load_dataset(data_path)
    class_names   = golden["class_names"]
    n_classes     = len(class_names)
    print(f"[k_sweep] Classes ({n_classes}): {class_names}")

    # Build label matrices
    audit_csv = os.path.join(os.path.dirname(data_path), "multilabel_audit.csv")
    if os.path.exists(audit_csv):
        print(f"[k_sweep] Using multi-label audit: {audit_csv}")
        golden_lm = build_label_matrix(golden["filenames"], audit_csv, n_classes)
        query_lm  = build_label_matrix(query["filenames"],  audit_csv, n_classes)
    else:
        print("[k_sweep] No audit CSV found — using single-label fallback")
        golden_lm = np.eye(n_classes, dtype=int)[golden["labels"]]
        query_lm  = np.eye(n_classes, dtype=int)[query["labels"]]

    all_results = {}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[k_sweep] Using device: {device}")

    for model_key in MODELS_TO_RUN:
        if model_key not in MODEL_REGISTRY:
            print(f"[k_sweep] Skipping unknown model: {model_key}")
            continue

        print(f"\n{'#'*55}")
        print(f"  Model: {model_key}")
        print(f"{'#'*55}")

        model = MODEL_REGISTRY[model_key](device=device)
        model.load_model()

        golden_emb = extract_embeddings(model, golden["images"])
        query_emb  = extract_embeddings(model, query["images"])

        sweep = run_k_sweep(
            golden_emb, golden_lm,
            query_emb,  query_lm,
            class_names, k_values=K_VALUES
        )

        all_results[model_key] = sweep

    # Save results
    with open(OUTPUT_FILE, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[k_sweep] Results saved to {OUTPUT_FILE}")

    # Final summary
    print(f"\n{'='*55}")
    print("  FINAL SUMMARY — Best k per model")
    print(f"{'='*55}")
    print(f"{'Model':<15} | {'Best k':>6} | {'Macro F1':>10}")
    print(f"{'-'*15}-+-{'-'*6}-+-{'-'*10}")
    for model_key, sweep in all_results.items():
        print(f"{model_key:<15} | {sweep['best_k']:>6} | {sweep['best_macro_f1']:>10.4f}")
    print(f"{'='*55}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to dataset folder")
    args = parser.parse_args()
    main(args.data)
