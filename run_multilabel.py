"""
Multi-Label Classification Pipeline
=====================================
Extends the single-label centroid pipeline to multi-label.

Steps:
  1. Load multi-label audit CSV (from data/audit_multilabel.py)
     OR fall back to single-label mode if no multi-label data found
  2. Load images from disk matching the audit CSV
  3. Extract embeddings with each model
  4. Method A: KNN Thresholding with calibrated per-class thresholds
  5. Method B: Partial Aggregation Prototypes + centroid threshold calibration
  6. Evaluate with macro F1, micro F1, hamming loss
  7. Plot prediction grids (YOLO-style)
  8. Save results CSV + JSON

Run:
    cd /Users/saumithdeversetty/Embedding_models/embedding_comparison
    python run_multilabel.py
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

from embeddings.extractor import extract_embeddings
from evaluation.multilabel import (
    knn_multilabel_predict,
    calibrate_thresholds_knn,
    partial_aggregation_centroids,
    centroid_multilabel_predict,
    calibrate_thresholds_centroid,
    evaluate_multilabel,
    single_label_to_matrix,
)
from visualization.plot import plot_predictions_grid
from models import MODEL_REGISTRY

# ── Config ────────────────────────────────────────────────────────────────────
IMAGE_FOLDER    = "/Users/saumithdeversetty/Embedding_models/preprocessed"
AUDIT_CSV       = str(Path(__file__).parent / "results" / "multilabel_audit.csv")
RESULTS_DIR     = str(Path(__file__).parent / "results" / "multilabel")
N_GOLDEN        = 30
KNN_K           = 10
SEED            = 42
DEVICE          = "mps"

MODELS_TO_RUN = [
    "siglip2",
    "dinov2",
    "inat_eva02",
    "plantclef",
    "bioclip2",
]
# ─────────────────────────────────────────────────────────────────────────────

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def load_image_safe(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def load_multilabel_dataset(audit_csv: str, image_folder: str, class_names: list, n_golden: int, seed: int):
    """
    Load images and their multi-label matrix from the audit CSV.
    Splits into golden (n_golden per class) and query sets.

    Returns:
        golden: dict with images, label_matrix, filenames
        query:  dict with images, label_matrix, filenames
        class_names: list of class names (from audit CSV columns)
    """
    df = pd.read_csv(audit_csv)
    # class columns = all columns except hash, filename, found_in_folders, n_labels
    meta_cols = {"hash", "filename", "found_in_folders", "n_labels"}
    class_cols = [c for c in df.columns if c not in meta_cols]

    print(f"Audit CSV: {len(df)} unique images, classes: {class_cols}")

    # Build a filename → full path map
    root = Path(image_folder)
    path_map = {}
    for folder in root.iterdir():
        if folder.is_dir():
            for f in folder.iterdir():
                if f.suffix.lower() in EXTENSIONS:
                    if f.name not in path_map:
                        path_map[f.name] = f

    rng = np.random.default_rng(seed)

    golden_rows = []
    query_rows  = []

    for cls_col in class_cols:
        cls_df = df[df[cls_col] == 1].copy()
        total  = len(cls_df)
        # Adaptive split — same logic as image_loader.py
        if total >= n_golden * 2:
            actual_golden = n_golden
        else:
            actual_golden = max(5, min(n_golden, int(total * 0.6)))
            actual_golden = min(actual_golden, total - 5)
        if actual_golden < n_golden:
            print(f"  ⚠️  {cls_col}: only {total} images → {actual_golden} golden | {total - actual_golden} query")
        idx = rng.permutation(total)
        golden_idx = idx[:actual_golden]
        query_idx  = idx[actual_golden:]
        golden_rows.append(cls_df.iloc[golden_idx])
        query_rows.append(cls_df.iloc[query_idx])

    golden_df = pd.concat(golden_rows).drop_duplicates(subset="hash").reset_index(drop=True)
    query_df  = pd.concat(query_rows).drop_duplicates(subset="hash").reset_index(drop=True)

    def load_split(split_df):
        images, label_matrices, filenames = [], [], []
        missing = 0
        for _, row in split_df.iterrows():
            fname = row["filename"]
            if fname not in path_map:
                missing += 1
                continue
            try:
                img = load_image_safe(path_map[fname])
                images.append(img)
                filenames.append(fname)
                label_matrices.append([int(row[c]) for c in class_cols])
            except Exception as e:
                missing += 1
        if missing:
            print(f"  ⚠️  {missing} images could not be loaded")
        return {
            "images": images,
            "label_matrix": np.array(label_matrices, dtype=int),
            "filenames": filenames,
        }

    print("Loading golden set...")
    golden = load_split(golden_df)
    print(f"  Golden: {len(golden['images'])} images")

    print("Loading query set...")
    query = load_split(query_df)
    print(f"  Query:  {len(query['images'])} images")

    return golden, query, class_cols


def load_single_label_dataset():
    """
    Fallback: load dataset using original single-label loader,
    convert labels to binary matrix.
    """
    from data.image_loader import load_dataset
    golden, query = load_dataset(IMAGE_FOLDER, n_golden=N_GOLDEN, seed=SEED)
    class_names = golden["class_names"]
    n_classes   = len(class_names)

    golden["label_matrix"] = single_label_to_matrix(np.array(golden["labels"]), n_classes)
    query["label_matrix"]  = single_label_to_matrix(np.array(query["labels"]),  n_classes)
    return golden, query, class_names


def main():
    print("=" * 60)
    print("Multi-Label Classification Pipeline")
    print("=" * 60)

    # Load dataset
    audit_path = Path(AUDIT_CSV)
    if audit_path.exists():
        # Check if there are actually multi-label images
        df_check = pd.read_csv(audit_path)
        has_multilabel = (df_check["n_labels"] > 1).any()
        if has_multilabel:
            n_multi = (df_check["n_labels"] > 1).sum()
            print(f"\n✅ Audit CSV found — {n_multi} multi-label images detected")
            meta_cols = {"hash", "filename", "found_in_folders", "n_labels"}
            class_names = [c for c in df_check.columns if c not in meta_cols]
            golden, query, class_names = load_multilabel_dataset(
                AUDIT_CSV, IMAGE_FOLDER, class_names, N_GOLDEN, SEED
            )
            multilabel_mode = True
        else:
            print("\n⚠️  Audit CSV found but no multi-label images — running single-label mode")
            golden, query, class_names = load_single_label_dataset()
            multilabel_mode = False
    else:
        print(f"\n⚠️  No audit CSV at {AUDIT_CSV}")
        print("   Run:  python data/audit_multilabel.py  first")
        print("   Falling back to single-label mode\n")
        golden, query, class_names = load_single_label_dataset()
        multilabel_mode = False

    n_classes = len(class_names)
    print(f"\nMode: {'MULTI-LABEL' if multilabel_mode else 'SINGLE-LABEL'}")
    print(f"Classes ({n_classes}): {class_names}")
    print(f"Golden: {len(golden['images'])} | Query: {len(query['images'])}\n")

    results_dir = Path(RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for model_key in MODELS_TO_RUN:
        print(f"\n{'─' * 50}")
        print(f"Model: {model_key.upper()}")
        print(f"{'─' * 50}")

        try:
            model_class = MODEL_REGISTRY[model_key]
            model = model_class(device=DEVICE)
            model.load_model()

            print(f"[{model_key}] Extracting golden embeddings...")
            golden_emb = extract_embeddings(model, golden["images"])

            print(f"[{model_key}] Extracting query embeddings...")
            query_emb = extract_embeddings(model, query["images"])

            golden_lm = golden["label_matrix"]
            query_lm  = query["label_matrix"]

            model_results = {}

            # ── Method A: KNN Thresholding ────────────────────────────────
            print(f"\n[{model_key}] Method A: KNN Thresholding (k={KNN_K})")
            knn_thresholds = calibrate_thresholds_knn(golden_emb, golden_lm, k=KNN_K)
            print(f"  Calibrated thresholds: {np.round(knn_thresholds, 3)}")

            knn_pred, knn_votes = knn_multilabel_predict(
                query_emb, golden_emb, golden_lm,
                k=KNN_K, thresholds=knn_thresholds
            )
            knn_eval = evaluate_multilabel(knn_pred, query_lm, class_names)
            model_results["knn"] = knn_eval

            print(f"  Macro F1:    {knn_eval['macro_f1']}")
            print(f"  Micro F1:    {knn_eval['micro_f1']}")
            print(f"  Hamming:     {knn_eval['hamming_loss']}")

            # ── Method B: Partial Aggregation Prototypes ──────────────────
            print(f"\n[{model_key}] Method B: Partial Aggregation Prototypes")
            pa_centroids = partial_aggregation_centroids(golden_emb, golden_lm, n_classes)
            pa_thresholds = calibrate_thresholds_centroid(golden_emb, golden_lm, pa_centroids)
            print(f"  Calibrated thresholds: {np.round(pa_thresholds, 3)}")

            pa_pred, pa_sim = centroid_multilabel_predict(query_emb, pa_centroids, pa_thresholds)
            pa_eval = evaluate_multilabel(pa_pred, query_lm, class_names)
            model_results["partial_aggregation"] = pa_eval

            print(f"  Macro F1:    {pa_eval['macro_f1']}")
            print(f"  Micro F1:    {pa_eval['micro_f1']}")
            print(f"  Hamming:     {pa_eval['hamming_loss']}")

            # ── Best method for visualization ─────────────────────────────
            best_method = "knn" if knn_eval["macro_f1"] >= pa_eval["macro_f1"] else "partial_aggregation"
            best_pred   = knn_pred if best_method == "knn" else pa_pred
            best_conf   = knn_votes if best_method == "knn" else pa_sim

            print(f"\n[{model_key}] Best method: {best_method}")

            # ── Prediction grid visualization ─────────────────────────────
            plot_predictions_grid(
                images=query["images"],
                true_labels=query_lm,
                predicted_labels=best_pred,
                class_names=class_names,
                model_name=f"{model_key}_{best_method}",
                save_dir=str(results_dir),
                confidence=best_conf,
                max_images=60,
            )

            all_results[model_key] = model_results
            del model

        except Exception as e:
            import traceback
            print(f"[{model_key}] ERROR: {e}")
            traceback.print_exc()
            continue

    # Save results
    json_path = results_dir / "multilabel_results.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nDetailed results → {json_path}")

    # Summary table
    rows = []
    for model_key, methods in all_results.items():
        for method, res in methods.items():
            rows.append({
                "model":        model_key,
                "method":       method,
                "macro_f1":     res["macro_f1"],
                "micro_f1":     res["micro_f1"],
                "hamming_loss": res["hamming_loss"],
            })
    df = pd.DataFrame(rows)
    csv_path = results_dir / "multilabel_summary.csv"
    df.to_csv(csv_path, index=False)

    print(f"\n{'=' * 60}")
    print("MULTI-LABEL RESULTS SUMMARY")
    print("=" * 60)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
