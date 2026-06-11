import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from data.image_loader import load_dataset
from embeddings.extractor import extract_embeddings
from evaluation.centroid import compute_centroids, match_query_to_centroid, evaluate
from evaluation.clustering import compute_cosine_similarity
from visualization.plot import (
    plot_umap_thumbnails,
    plot_accuracy_summary,
    plot_per_class_accuracy,
    plot_similarity_heatmap
)
from models import MODEL_REGISTRY


# ── Configuration ─────────────────────────────────────────────────────────────

IMAGE_FOLDER = "/Users/saumithdeversetty/Embedding_models/preprocessed"
N_GOLDEN     = 30        # images per class for golden set
N_CLUSTERS   = 8         # known defect classes — flexible, change as needed
SEED         = 42
RESULTS_DIR  = str(Path(__file__).parent / "results")
DEVICE       = "mps"     # change to "cuda" if GPU available

MODELS_TO_RUN = [
    "bioclip2",
    "dinov2",
    "siglip2",
    "inat_eva02",
    "plantclef",
]

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Embedding Comparison Pipeline — Centroid Matching")
    print("=" * 60)

    # Step 1: Load dataset — golden set + query set, same split every run
    golden, query = load_dataset(IMAGE_FOLDER, n_golden=N_GOLDEN, seed=SEED)
    class_names   = golden["class_names"]
    n_classes     = len(class_names)

    print(f"\nClasses ({n_classes}): {class_names}")
    print(f"Running {len(MODELS_TO_RUN)} models.\n")

    all_results = {}

    for model_key in MODELS_TO_RUN:
        print(f"\n{'─' * 50}")
        print(f"Model: {model_key.upper()}")
        print(f"{'─' * 50}")

        try:
            # Step 2: Load model
            model_class = MODEL_REGISTRY[model_key]
            model       = model_class(device=DEVICE)
            model.load_model()

            # Step 3: Extract embeddings for golden set
            print(f"[{model_key}] Extracting golden set embeddings...")
            golden_embeddings = extract_embeddings(model, golden["images"])

            # Step 4: Extract embeddings for query set
            print(f"[{model_key}] Extracting query set embeddings...")
            query_embeddings = extract_embeddings(model, query["images"])

            # Step 5: Compute centroids from golden set
            centroids = compute_centroids(
                golden_embeddings,
                golden["labels"],
                n_classes
            )

            # Step 6: Match query images to nearest centroid
            predicted_labels, confidence_scores = match_query_to_centroid(
                query_embeddings,
                centroids
            )

            # Step 7: Evaluate accuracy
            eval_results = evaluate(
                predicted_labels,
                query["labels"],
                class_names,
                confidence_scores
            )
            all_results[model_key] = eval_results

            print(f"\n[{model_key}] Overall accuracy: {eval_results['overall_accuracy']}%")
            print(f"[{model_key}] Avg confidence:   {eval_results['avg_confidence']}")
            print(f"[{model_key}] Per-class accuracy:")
            for cls, stats in eval_results["per_class"].items():
                print(f"   {cls}: {stats['accuracy']}% (n={stats['n_query']}, conf={stats['avg_confidence']})")

            # Step 8: UMAP — golden set coloured by actual label
            plot_umap_thumbnails(
                embeddings=golden_embeddings,
                images=golden["images"],
                labels=golden["labels"],
                class_names=class_names,
                model_name=model_key,
                save_dir=RESULTS_DIR,
                seed=SEED
            )

            # Step 9: Per-class accuracy bar chart
            plot_per_class_accuracy(
                model_name=model_key,
                eval_results=eval_results,
                save_dir=RESULTS_DIR
            )

            # Step 10: Similarity heatmap on golden set
            sim_matrix = compute_cosine_similarity(golden_embeddings)
            plot_similarity_heatmap(
                similarity_matrix=sim_matrix,
                filenames=golden["filenames"],
                model_name=model_key,
                save_dir=RESULTS_DIR
            )

            # Free memory
            del model

        except Exception as e:
            import traceback
            print(f"[{model_key}] ERROR: {e}")
            traceback.print_exc()
            continue

    # Step 11: Save accuracy summary chart
    if all_results:
        plot_accuracy_summary(all_results, save_dir=RESULTS_DIR)

    # Step 12: Save summary CSV
    rows = []
    for model_key, res in all_results.items():
        rows.append({
            "model":            model_key,
            "overall_accuracy": res["overall_accuracy"],
            "avg_confidence":   res["avg_confidence"],
        })
    df = pd.DataFrame(rows)
    summary_path = Path(RESULTS_DIR) / "summary.csv"
    df.to_csv(summary_path, index=False)

    # Step 13: Save detailed results as JSON
    json_path = Path(RESULTS_DIR) / "detailed_results.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"All done. Results saved to: {RESULTS_DIR}")
    print(f"{'=' * 60}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
