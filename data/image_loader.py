import random
from pathlib import Path
from PIL import Image
from typing import List, Tuple, Dict


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def load_dataset(
    folder_path: str,
    n_golden: int = 30,
    seed: int = 42
) -> Tuple[Dict, Dict]:
    """
    Loads images from all subfolders (one subfolder = one defect class).
    Splits each class into a golden set and a query set.

    Args:
        folder_path: Path to the preprocessed folder containing 8 subfolders.
        n_golden:    Number of images per class for the golden set (default 30).
        seed:        Fixed seed for reproducibility.

    Returns:
        golden: dict with keys 'images', 'labels', 'filenames', 'class_names'
        query:  dict with keys 'images', 'labels', 'filenames', 'class_names'
    """
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    # Discover all subfolders — each is one defect class
    subfolders = sorted([f for f in folder.iterdir() if f.is_dir()])
    if not subfolders:
        raise ValueError(f"No subfolders found in: {folder_path}")

    print(f"[image_loader] Found {len(subfolders)} classes:")
    for sf in subfolders:
        print(f"  → {sf.name}")

    golden_images, golden_labels, golden_filenames = [], [], []
    query_images,  query_labels,  query_filenames  = [], [], []
    class_names = []

    random.seed(seed)

    for label_idx, subfolder in enumerate(subfolders):
        class_name = subfolder.name
        class_names.append(class_name)

        # Collect all valid images in this subfolder
        all_files = sorted([
            f for f in subfolder.iterdir()
            if f.suffix.lower() in SUPPORTED_EXTENSIONS
        ])

        if len(all_files) == 0:
            print(f"[image_loader] Warning: no images found in {class_name}, skipping.")
            continue

        # Adaptive golden set size:
        # - Large classes (>= 2x n_golden): use exactly n_golden
        # - Small classes: use 60% for golden, 40% for query (min 5 golden, min 5 query)
        total = len(all_files)
        if total >= n_golden * 2:
            actual_n_golden = n_golden
        else:
            actual_n_golden = max(5, min(n_golden, int(total * 0.6)))
            # Ensure at least 5 images remain for query
            actual_n_golden = min(actual_n_golden, total - 5)
        if actual_n_golden < n_golden:
            print(f"[image_loader] ⚠️  {class_name}: only {total} images → {actual_n_golden} golden | {total - actual_n_golden} query")

        # Fixed random sample for golden set
        golden_files = random.sample(all_files, actual_n_golden)
        golden_set   = set(f.name for f in golden_files)
        query_files  = [f for f in all_files if f.name not in golden_set]

        # Load golden set
        for f in golden_files:
            try:
                img = Image.open(f).convert("RGB")
                golden_images.append(img)
                golden_labels.append(label_idx)
                golden_filenames.append(f.name)
            except Exception as e:
                print(f"[image_loader] Skipping {f.name}: {e}")

        # Load query set
        for f in query_files:
            try:
                img = Image.open(f).convert("RGB")
                query_images.append(img)
                query_labels.append(label_idx)
                query_filenames.append(f.name)
            except Exception as e:
                print(f"[image_loader] Skipping {f.name}: {e}")

        print(f"[image_loader] {class_name}: {actual_n_golden} golden | {len(query_files)} query")

    golden = {
        "images":      golden_images,
        "labels":      golden_labels,
        "filenames":   golden_filenames,
        "class_names": class_names
    }

    query = {
        "images":      query_images,
        "labels":      query_labels,
        "filenames":   query_filenames,
        "class_names": class_names
    }

    print(f"\n[image_loader] Total golden: {len(golden_images)} | Total query: {len(query_images)}")
    return golden, query
