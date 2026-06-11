import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
from typing import List, Optional
from pathlib import Path
import umap

THUMBNAIL_SIZE = (80, 80)
BORDER_WIDTH   = 4
CLUSTER_COLORS = [
    "#e6194b",  # red
    "#3cb44b",  # green
    "#4363d8",  # blue
    "#f58231",  # orange
    "#911eb4",  # purple
    "#42d4f4",  # cyan
    "#f032e6",  # magenta
    "#bfef45",  # lime
    "#fabed4",  # pink
    "#469990",  # teal
]


def _pil_to_array(img: Image.Image, size: tuple = THUMBNAIL_SIZE) -> np.ndarray:
    return np.array(img.resize(size))


def _add_colour_border(img_array: np.ndarray, colour_hex: str, border: int = BORDER_WIDTH) -> np.ndarray:
    h, w, c = img_array.shape
    r = int(colour_hex[1:3], 16)
    g = int(colour_hex[3:5], 16)
    b = int(colour_hex[5:7], 16)
    bordered = np.full((h + 2 * border, w + 2 * border, c), [r, g, b], dtype=np.uint8)
    bordered[border:border + h, border:border + w] = img_array
    return bordered


def plot_umap_thumbnails(
    embeddings: np.ndarray,
    images: List[Image.Image],
    labels: List[int],
    class_names: List[str],
    model_name: str,
    save_dir: str = "results",
    seed: int = 42,
    subset_size: int = 200
):
    """
    Plots UMAP with image thumbnails coloured by actual defect label.
    Uses a subset for speed if dataset is large.

    Args:
        embeddings:   numpy array (N, embedding_dim)
        images:       list of PIL Images
        labels:       list of class indices (ground truth)
        class_names:  list of class name strings
        model_name:   name of the model
        save_dir:     where to save the plot
        seed:         UMAP random seed
        subset_size:  max images to plot (for readability)
    """
    print(f"[plot] Running UMAP for {model_name}...")

    # Subset for readability — too many thumbnails overlap
    n = len(embeddings)
    if n > subset_size:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, subset_size, replace=False)
        embeddings_plot = embeddings[idx]
        images_plot     = [images[i] for i in idx]
        labels_plot     = [labels[i] for i in idx]
    else:
        embeddings_plot = embeddings
        images_plot     = images
        labels_plot     = labels

    reducer    = umap.UMAP(n_components=2, random_state=seed, n_neighbors=15, min_dist=0.3)
    coords_2d  = reducer.fit_transform(embeddings_plot)

    fig, ax = plt.subplots(figsize=(22, 16))
    ax.set_title(f"UMAP — {model_name}\n(coloured by defect class)", fontsize=16, fontweight="bold", pad=20)
    ax.set_xlabel("UMAP Dimension 1", fontsize=12)
    ax.set_ylabel("UMAP Dimension 2", fontsize=12)
    ax.set_facecolor("#f8f8f8")
    ax.grid(True, linestyle="--", alpha=0.4, color="gray")

    for i, (img, coord) in enumerate(zip(images_plot, coords_2d)):
        colour    = CLUSTER_COLORS[labels_plot[i] % len(CLUSTER_COLORS)]
        thumbnail = _pil_to_array(img)
        thumbnail = _add_colour_border(thumbnail, colour)
        imagebox  = OffsetImage(thumbnail, zoom=1.0)
        ab        = AnnotationBbox(imagebox, coord, frameon=False, pad=0)
        ax.add_artist(ab)

    x_pad = (coords_2d[:, 0].max() - coords_2d[:, 0].min()) * 0.12
    y_pad = (coords_2d[:, 1].max() - coords_2d[:, 1].min()) * 0.12
    ax.set_xlim(coords_2d[:, 0].min() - x_pad, coords_2d[:, 0].max() + x_pad)
    ax.set_ylim(coords_2d[:, 1].min() - y_pad, coords_2d[:, 1].max() + y_pad)

    # Legend — one entry per actual defect class
    handles = [
        mpatches.Patch(
            color=CLUSTER_COLORS[i % len(CLUSTER_COLORS)],
            label=class_names[i] if i < len(class_names) else f"Class {i}"
        )
        for i in range(len(class_names))
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=9, framealpha=0.9)

    plt.tight_layout()
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_path = Path(save_dir) / f"umap_{model_name}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] Saved → {save_path}")


def plot_accuracy_summary(
    results: dict,
    save_dir: str = "results"
):
    """
    Bar chart showing overall accuracy per model and per class breakdown.
    """
    models   = list(results.keys())
    accuracy = [results[m]["overall_accuracy"] for m in models]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(models, accuracy, color=CLUSTER_COLORS[:len(models)], edgecolor="black", linewidth=0.8)
    ax.set_title("Overall Accuracy — Centroid Matching per Model", fontsize=14, fontweight="bold")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 105)
    ax.axhline(y=80, color="red", linestyle="--", linewidth=1.5, label="80% target")
    ax.legend()

    for bar, acc in zip(bars, accuracy):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{acc:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")

    plt.tight_layout()
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_path = Path(save_dir) / "accuracy_summary.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] Saved → {save_path}")


def plot_per_class_accuracy(
    model_name: str,
    eval_results: dict,
    save_dir: str = "results"
):
    """
    Bar chart showing accuracy per defect class for one model.
    """
    per_class   = eval_results["per_class"]
    class_names = list(per_class.keys())
    accuracies  = [per_class[c]["accuracy"] for c in class_names]

    fig, ax = plt.subplots(figsize=(14, 6))
    bars = ax.bar(class_names, accuracies, color=CLUSTER_COLORS[:len(class_names)], edgecolor="black")
    ax.set_title(f"Per-Class Accuracy — {model_name}", fontsize=13, fontweight="bold")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 105)
    ax.axhline(y=80, color="red", linestyle="--", linewidth=1.5, label="80% target")
    ax.legend()
    plt.xticks(rotation=30, ha="right", fontsize=9)

    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{acc:.1f}%", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    save_path = Path(save_dir) / f"per_class_{model_name}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] Saved → {save_path}")


def plot_predictions_grid(
    images: list,
    true_labels: np.ndarray,
    predicted_labels,   # can be 1D array (single-label) or 2D matrix (multi-label)
    class_names: list,
    model_name: str,
    save_dir: str,
    confidence=None,    # 1D array or 2D matrix matching predicted_labels
    max_images: int = 60,
    ncols: int = 6,
    seed: int = 42,
):
    """
    Renders test images in a grid with YOLO-style label tags.
    Green border = correct prediction. Red border = wrong.
    Label tag shown at top-left of each image like the annotation UI.

    Works for both single-label (1D arrays) and multi-label (2D matrices).
    """
    random.seed(seed)

    is_multilabel = (isinstance(predicted_labels, np.ndarray) and predicted_labels.ndim == 2)

    # Sample up to max_images
    indices = list(range(len(images)))
    if len(indices) > max_images:
        indices = random.sample(indices, max_images)
    indices.sort()

    n = len(indices)
    nrows = (n + ncols - 1) // ncols

    THUMB = 160
    TAG_H = 28
    BORDER = 4

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.2, nrows * 2.4))
    axes = np.array(axes).reshape(-1)

    for ax_i, idx in enumerate(indices):
        ax = axes[ax_i]
        img = images[idx].copy().convert("RGB")
        img = img.resize((THUMB, THUMB), Image.LANCZOS)
        img_arr = np.array(img)

        if is_multilabel:
            pred = predicted_labels[idx]   # binary vector
            true = true_labels[idx]        # binary vector
            correct = np.array_equal(pred, true)
            # Build label string from predicted classes
            pred_class_names = [class_names[c] for c in range(len(class_names)) if pred[c] == 1]
            label_str = ", ".join(pred_class_names) if pred_class_names else "none"
            conf_val = float(confidence[idx].max()) if confidence is not None else None
        else:
            pred = int(predicted_labels[idx])
            true = int(true_labels[idx])
            correct = (pred == true)
            label_str = class_names[pred]
            conf_val = float(confidence[idx]) if confidence is not None else None

        # Border color
        border_color = (34, 197, 94) if correct else (239, 68, 68)   # green / red

        # Add colored border
        bordered = np.ones((THUMB + 2*BORDER, THUMB + 2*BORDER, 3), dtype=np.uint8)
        bordered[:, :] = border_color
        bordered[BORDER:-BORDER, BORDER:-BORDER] = img_arr

        ax.imshow(bordered)
        ax.axis("off")

        # Label tag at top-left (YOLO style)
        tag = label_str if conf_val is None else f"{label_str}  {conf_val:.2f}"
        ax.text(
            0.01, 0.97, tag,
            transform=ax.transAxes,
            fontsize=6.5,
            color="white",
            fontweight="bold",
            va="top", ha="left",
            bbox=dict(
                boxstyle="square,pad=0.2",
                facecolor=tuple(c/255 for c in border_color),
                edgecolor="none",
                alpha=0.9,
            )
        )

    # Hide unused axes
    for ax_i in range(n, len(axes)):
        axes[ax_i].axis("off")

    mode = "multilabel" if is_multilabel else "singlelabel"
    fig.suptitle(f"{model_name.upper()} — Predictions ({mode})\nGreen=correct  Red=wrong", fontsize=11)
    plt.tight_layout()

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    out = save_dir / f"{model_name}_predictions_grid.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Predictions grid saved → {out}")


def plot_similarity_heatmap(
    similarity_matrix: np.ndarray,
    filenames: List[str],
    model_name: str,
    save_dir: str = "results"
):
    """
    Cosine similarity heatmap.
    Yellow = similar | Purple = different | Dark row = outlier image.
    """
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(similarity_matrix, cmap="viridis", vmin=0, vmax=1)
    ax.set_title(
        f"Cosine Similarity Heatmap — {model_name}\n"
        f"Yellow = similar | Purple = different",
        fontsize=13, fontweight="bold"
    )
    plt.colorbar(im, ax=ax, label="Cosine Similarity")
    ax.set_xlabel("Image index")
    ax.set_ylabel("Image index")
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_path = Path(save_dir) / f"heatmap_{model_name}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] Saved → {save_path}")
