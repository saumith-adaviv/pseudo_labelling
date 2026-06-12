"""
Centroid-based Synthetic Augmentation in Embedding Space.

For classes with fewer than `min_samples` images:
  1. Compute the class centroid (average of all class embeddings)
  2. Generate synthetic embeddings by interpolating between
     existing samples and the centroid
  3. Add synthetic samples to the training set

This is SMOTE adapted for embedding space — keeps synthetic
samples close to the real distribution rather than random noise.
"""

import numpy as np


def augment_small_classes(
    embeddings: np.ndarray,
    label_matrix: np.ndarray,
    min_samples: int = 30,
    seed: int = 42,
) -> tuple:
    """
    Augments underrepresented classes to have at least min_samples each.

    Args:
        embeddings:   (N, D) L2-normalised float32
        label_matrix: (N, C) binary int
        min_samples:  target minimum per class
        seed:         random seed for reproducibility

    Returns:
        aug_embeddings:   (N + synthetic, D)
        aug_label_matrix: (N + synthetic, C)
        stats: dict with per-class augmentation counts
    """
    rng = np.random.RandomState(seed)
    n_classes = label_matrix.shape[1]

    aug_embs   = list(embeddings)
    aug_labels = list(label_matrix)
    stats      = {}

    for c in range(n_classes):
        class_mask = label_matrix[:, c].astype(bool)
        class_embs = embeddings[class_mask]
        n = len(class_embs)

        if n == 0:
            stats[c] = {"original": 0, "synthetic": 0}
            continue

        if n >= min_samples:
            stats[c] = {"original": n, "synthetic": 0}
            continue

        needed  = min_samples - n
        centroid = class_embs.mean(axis=0)
        norm     = np.linalg.norm(centroid)
        centroid = centroid / norm if norm > 0 else centroid

        class_labels = label_matrix[class_mask]

        for _ in range(needed):
            # Pick a random existing sample from this class
            idx   = rng.randint(0, n)
            alpha = rng.uniform(0.3, 0.7)

            # Interpolate between the sample and the centroid
            synthetic = alpha * class_embs[idx] + (1 - alpha) * centroid

            # Re-normalise to stay on the unit sphere
            norm = np.linalg.norm(synthetic)
            synthetic = synthetic / norm if norm > 0 else synthetic

            # Inherit the parent sample's full label row
            syn_label = class_labels[idx].copy()

            aug_embs.append(synthetic)
            aug_labels.append(syn_label)

        stats[c] = {"original": n, "synthetic": needed}

    aug_embeddings   = np.array(aug_embs,   dtype=np.float32)
    aug_label_matrix = np.array(aug_labels, dtype=np.float32)

    return aug_embeddings, aug_label_matrix, stats


def print_augmentation_stats(stats: dict, class_names: list):
    print("\n  Centroid Augmentation Stats:")
    print(f"  {'Class':<40} | {'Original':>8} | {'Synthetic':>9} | {'Total':>6}")
    print(f"  {'-'*40}-+-{'-'*8}-+-{'-'*9}-+-{'-'*6}")
    for c, s in stats.items():
        name = class_names[c] if c < len(class_names) else f"class_{c}"
        total = s["original"] + s["synthetic"]
        flag  = " ⚠️" if s["synthetic"] > 0 else ""
        print(f"  {name:<40} | {s['original']:>8} | {s['synthetic']:>9} | {total:>6}{flag}")
