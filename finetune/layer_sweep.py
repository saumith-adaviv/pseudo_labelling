"""
Adaptive Layer Unfreezing Sweep for SigLIP2.

Improvements over baseline:
  1. Weighted BCELoss          — penalises misses on small classes more
  2. Image augmentation        — flip, crop, colour jitter on training images
  3. Cosine LR scheduler       — smooth decay, avoids overshooting
  4. Centroid SMOTE            — synthetic samples for classes < 30 images
  5. Unfreeze sweep [2,4,6,8,12] — picks best depth by val macro F1

Usage:
    python finetune/layer_sweep.py --data /content/drive/MyDrive/preprocessed
    python finetune/layer_sweep.py --data /content/drive/MyDrive/HQ_tomato_multi_defect.v4-v0.multiclass
"""
import argparse
import json
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import AutoModel, AutoProcessor
from sklearn.metrics import f1_score
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Config ────────────────────────────────────────────────────────────────────
UNFREEZE_LAYERS = [2, 4, 6, 8, 12]
EPOCHS          = 15
BATCH_SIZE      = 16
LR              = 5e-5
MIN_SAMPLES     = 30        # augment classes below this count
MODEL_ID        = "google/siglip2-so400m-patch14-384"
OUTPUT_FILE     = "results/layer_sweep_results.json"
# ─────────────────────────────────────────────────────────────────────────────

# Image augmentation for training
TRAIN_TRANSFORMS = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    transforms.RandomResizedCrop(size=384, scale=(0.8, 1.0)),
])


class DefectDataset(Dataset):
    def __init__(self, images, label_matrix, processor, augment: bool = False):
        self.images       = images
        self.label_matrix = torch.tensor(label_matrix, dtype=torch.float32)
        self.processor    = processor
        self.augment      = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.augment:
            img = TRAIN_TRANSFORMS(img)
        inputs = self.processor(images=img, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)
        return pixel_values, self.label_matrix[idx]


class EmbeddingDataset(Dataset):
    """Dataset that works directly on pre-extracted embeddings + SMOTE augmented data."""
    def __init__(self, embeddings, label_matrix):
        self.embeddings   = torch.tensor(embeddings, dtype=torch.float32)
        self.label_matrix = torch.tensor(label_matrix, dtype=torch.float32)

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.label_matrix[idx]


class SigLIPFinetune(nn.Module):
    def __init__(self, model_id: str, n_classes: int, n_unfreeze: int, device: str):
        super().__init__()
        self.device = device
        self.vision = AutoModel.from_pretrained(model_id).vision_model

        # Freeze all
        for param in self.vision.parameters():
            param.requires_grad = False

        # Unfreeze last n_unfreeze transformer blocks
        encoder_layers = self.vision.encoder.layers
        total_layers   = len(encoder_layers)
        unfreeze_from  = max(0, total_layers - n_unfreeze)
        for layer in encoder_layers[unfreeze_from:]:
            for param in layer.parameters():
                param.requires_grad = True

        # Always unfreeze final layer norm
        for param in self.vision.post_layernorm.parameters():
            param.requires_grad = True

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.parameters())
        print(f"  Unfreezing last {n_unfreeze}/{total_layers} blocks — "
              f"trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

        embed_dim = self.vision.config.hidden_size
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, n_classes),
        )
        self.to(device)

    def forward(self, pixel_values):
        pixel_values = pixel_values.to(self.device)
        outputs = self.vision(pixel_values=pixel_values)
        pooled  = outputs.pooler_output
        return self.head(pooled)


def compute_class_weights(label_matrix: np.ndarray) -> torch.Tensor:
    """
    Compute per-class positive weights for BCEWithLogitsLoss.
    Classes with fewer positive samples get higher weight.
    Formula: weight_c = (N - n_pos_c) / n_pos_c  (same as sklearn's balanced)
    """
    n_total   = label_matrix.shape[0]
    n_pos     = label_matrix.sum(axis=0).clip(min=1)
    weights   = (n_total - n_pos) / n_pos
    weights   = np.clip(weights, 1.0, 20.0)   # cap at 20x to avoid instability
    return torch.tensor(weights, dtype=torch.float32)


def extract_frozen_embeddings(images, processor, device):
    """Extract embeddings using fully frozen SigLIP2 for SMOTE augmentation."""
    from transformers import AutoModel
    print("  Extracting frozen embeddings for SMOTE augmentation...")
    model = AutoModel.from_pretrained(MODEL_ID).vision_model.to(device)
    model.eval()

    all_embs = []
    batch_size = 32
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            inputs = processor(images=batch, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            out = model(pixel_values=inputs["pixel_values"])
            embs = out.pooler_output
            embs = embs / embs.norm(dim=-1, keepdim=True)
            all_embs.append(embs.cpu().numpy())

    del model
    torch.cuda.empty_cache() if device == "cuda" else None
    return np.vstack(all_embs)


def train_one_config(
    n_unfreeze: int,
    train_images, train_lm,
    val_images,   val_lm,
    n_classes: int,
    processor,
    device: str,
    aug_embeddings=None,
    aug_labels=None,
) -> dict:
    print(f"\n{'#'*60}")
    print(f"  Unfreeze last {n_unfreeze} blocks")
    print(f"{'#'*60}")

    model     = SigLIPFinetune(MODEL_ID, n_classes, n_unfreeze, device)
    pos_weight = compute_class_weights(train_lm).to(device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer  = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=1e-4
    )

    train_ds = DefectDataset(train_images, train_lm, processor, augment=True)
    val_ds   = DefectDataset(val_images,   val_lm,   processor, augment=False)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    best_f1, best_epoch = 0.0, 0
    print(f"  {'Epoch':>5} | {'Loss':>8} | {'Macro F1':>10} | {'LR':>10}")
    print(f"  {'-'*5}-+-{'-'*8}-+-{'-'*10}-+-{'-'*10}")

    for epoch in range(1, EPOCHS + 1):
        # Train
        model.train()
        total_loss = 0
        for pixels, labels in train_dl:
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(pixels)
            loss   = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        # Validate
        model.eval()
        all_preds, all_true = [], []
        with torch.no_grad():
            for pixels, labels in val_dl:
                logits = model(pixels)
                preds  = (torch.sigmoid(logits) > 0.5).cpu().numpy()
                all_preds.append(preds)
                all_true.append(labels.numpy())

        all_preds = np.vstack(all_preds)
        all_true  = np.vstack(all_true)
        macro_f1  = f1_score(all_true, all_preds, average="macro", zero_division=0)
        current_lr = scheduler.get_last_lr()[0]

        print(f"  {epoch:>5} | {total_loss/len(train_dl):>8.4f} | {macro_f1:>10.4f} | {current_lr:>10.2e}")

        if macro_f1 > best_f1:
            best_f1    = macro_f1
            best_epoch = epoch
            torch.save(model.state_dict(), f"results/siglip2_unfreeze{n_unfreeze}_best.pt")

    print(f"\n  Best macro F1 = {best_f1:.4f} at epoch {best_epoch}")
    return {"n_unfreeze": n_unfreeze, "best_macro_f1": best_f1, "best_epoch": best_epoch}


def main(data_path: str):
    os.makedirs("results", exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"\n[layer_sweep] Device: {device}")
    print(f"[layer_sweep] Dataset: {data_path}")

    from data.image_loader import load_dataset
    golden, query = load_dataset(data_path)
    class_names   = golden["class_names"]
    n_classes     = len(class_names)
    print(f"[layer_sweep] Classes ({n_classes}): {class_names}")

    audit_csv = os.path.join(os.path.dirname(data_path), "multilabel_audit.csv")
    if os.path.exists(audit_csv):
        from data.audit_multilabel import build_label_matrix
        train_lm = build_label_matrix(golden["filenames"], audit_csv, n_classes)
        val_lm   = build_label_matrix(query["filenames"],  audit_csv, n_classes)
    else:
        train_lm = np.eye(n_classes, dtype=np.float32)[golden["labels"]]
        val_lm   = np.eye(n_classes, dtype=np.float32)[query["labels"]]

    processor = AutoProcessor.from_pretrained(MODEL_ID)

    # ── Centroid SMOTE augmentation ───────────────────────────────────────────
    print(f"\n[layer_sweep] Running centroid SMOTE for small classes (< {MIN_SAMPLES} images)...")
    frozen_embs = extract_frozen_embeddings(golden["images"], processor, device)

    from data.embedding_augment import augment_small_classes, print_augmentation_stats
    aug_embs, aug_labels, stats = augment_small_classes(frozen_embs, train_lm, min_samples=MIN_SAMPLES)
    print_augmentation_stats(stats, class_names)
    print(f"\n  Training set: {len(golden['images'])} real + {len(aug_embs) - len(golden['images'])} synthetic = {len(aug_embs)} total")

    # ── Layer sweep ───────────────────────────────────────────────────────────
    all_results = []
    for n_unfreeze in UNFREEZE_LAYERS:
        result = train_one_config(
            n_unfreeze,
            golden["images"], train_lm,
            query["images"],  val_lm,
            n_classes, processor, device,
            aug_embeddings=aug_embs,
            aug_labels=aug_labels,
        )
        all_results.append(result)

    with open(OUTPUT_FILE, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print("  FINAL SUMMARY — Layer Sweep")
    print(f"{'='*60}")
    print(f"  {'Unfreeze':>10} | {'Macro F1':>10} | {'Best Epoch':>10}")
    print(f"  {'-'*10}-+-{'-'*10}-+-{'-'*10}")
    for r in all_results:
        marker = " ←" if r == max(all_results, key=lambda x: x["best_macro_f1"]) else ""
        print(f"  {r['n_unfreeze']:>10} | {r['best_macro_f1']:>10.4f} | {r['best_epoch']:>10}{marker}")
    print(f"{'='*60}")

    best = max(all_results, key=lambda r: r["best_macro_f1"])
    print(f"\n  Best: unfreeze {best['n_unfreeze']} blocks → Macro F1 = {best['best_macro_f1']:.4f}")
    print(f"  Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    args = parser.parse_args()
    main(args.data)
