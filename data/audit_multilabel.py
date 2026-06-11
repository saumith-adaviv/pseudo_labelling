"""
Multi-Label Audit Script
========================
Scans all class folders, finds images that appear in multiple folders
(using MD5 hash matching), and builds a multi-label matrix.

Output:
  results/multilabel_audit.csv   — one row per unique image, binary label columns + n_labels
  results/multilabel_summary.txt — statistics
  results/multilabel_audit.png   — distribution charts

Run:
    cd /Users/saumithdeversetty/Embedding_models/embedding_comparison
    python data/audit_multilabel.py
"""

import hashlib
import sys
from pathlib import Path
from collections import defaultdict

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IMAGE_FOLDER = "/Users/saumithdeversetty/Embedding_models/preprocessed"
RESULTS_DIR  = str(Path(__file__).parent.parent / "results")
EXTENSIONS   = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def file_hash(path: Path, chunk_size: int = 65536) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def scan_folders(root: Path):
    root = Path(root)
    subfolders = sorted([d for d in root.iterdir() if d.is_dir()])
    class_names = [d.name for d in subfolders]
    print(f"Found {len(class_names)} class folders: {class_names}\n")

    hash_to_info = defaultdict(list)
    for folder in subfolders:
        images = [f for f in folder.iterdir() if f.suffix.lower() in EXTENSIONS]
        print(f"  {folder.name}: {len(images)} images")
        for img_path in images:
            h = file_hash(img_path)
            hash_to_info[h].append((folder.name, img_path))

    return class_names, hash_to_info


def build_label_matrix(class_names, hash_to_info):
    rows = []
    for h, entries in hash_to_info.items():
        row = {"hash": h}
        row["filename"] = entries[0][1].name
        row["found_in_folders"] = ", ".join(sorted(set(e[0] for e in entries)))
        for cls in class_names:
            row[cls] = int(any(e[0] == cls for e in entries))
        row["n_labels"] = sum(row[cls] for cls in class_names)
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values("n_labels", ascending=False).reset_index(drop=True)
    return df


def print_summary(df, class_names):
    lines = []
    lines.append("=" * 60)
    lines.append("MULTI-LABEL AUDIT SUMMARY")
    lines.append("=" * 60)
    lines.append(f"Total unique images : {len(df)}")
    lines.append(f"Single-label images : {(df['n_labels'] == 1).sum()}")
    lines.append(f"Multi-label images  : {(df['n_labels'] > 1).sum()}")
    lines.append(f"Zero-label images   : {(df['n_labels'] == 0).sum()}")
    lines.append("")
    lines.append("Label distribution:")
    for n, count in df["n_labels"].value_counts().sort_index().items():
        lines.append(f"  {n} label(s): {count} images")
    lines.append("")
    lines.append("Per-class count:")
    for cls in class_names:
        lines.append(f"  {cls}: {df[cls].sum()} images")
    lines.append("")
    lines.append("Most common label combinations (top 15):")
    for combo, count in df["found_in_folders"].value_counts().head(15).items():
        lines.append(f"  [{combo}]: {count} images")
    text = "\n".join(lines)
    print(text)
    return text


def plot_distribution(df, class_names, save_dir):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Multi-Label Audit", fontsize=14, fontweight="bold")

    ax = axes[0]
    dist = df["n_labels"].value_counts().sort_index()
    ax.bar(dist.index, dist.values, color="#4C72B0", edgecolor="white")
    ax.set_xlabel("Number of Labels per Image")
    ax.set_ylabel("Image Count")
    ax.set_title("Labels per Image Distribution")
    ax.set_xticks(dist.index)
    for x, y in zip(dist.index, dist.values):
        ax.text(x, y + 0.5, str(y), ha="center", va="bottom", fontsize=10)

    ax = axes[1]
    counts = [int(df[cls].sum()) for cls in class_names]
    colors = ["#2ca02c" if c == max(counts) else "#4C72B0" for c in counts]
    ax.bar(range(len(class_names)), counts, color=colors, edgecolor="white")
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Image Count")
    ax.set_title("Images per Class")
    for i, c in enumerate(counts):
        ax.text(i, c + 0.5, str(c), ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    out = save_dir / "multilabel_audit.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nPlot saved → {out}")


def main():
    root = Path(IMAGE_FOLDER)
    if not root.exists():
        print(f"ERROR: IMAGE_FOLDER not found: {root}")
        sys.exit(1)

    print(f"Scanning: {root}\n")
    class_names, hash_to_info = scan_folders(root)
    print(f"\nTotal unique images: {len(hash_to_info)}")

    df = build_label_matrix(class_names, hash_to_info)

    results_dir = Path(RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)

    csv_path = results_dir / "multilabel_audit.csv"
    df.to_csv(csv_path, index=False)
    print(f"Label matrix saved → {csv_path}\n")

    summary = print_summary(df, class_names)
    (results_dir / "multilabel_summary.txt").write_text(summary)

    plot_distribution(df, class_names, RESULTS_DIR)

    multi = df[df["n_labels"] > 1][["filename", "found_in_folders", "n_labels"]].head(20)
    if len(multi):
        print(f"\nTop multi-label images (up to 20):")
        print(multi.to_string(index=False))
    else:
        print("\n⚠️  No multi-label images found — all images appear in exactly one folder.")
        print("   You will need manual multi-label annotations.")


if __name__ == "__main__":
    main()
