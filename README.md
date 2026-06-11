# Embedding Comparison Pipeline
**Adaviv — Harvest Quality Pipeline | Phase 1**

Compares embedding models on preprocessed crop images to identify which model best groups visually similar defects together. No labels required — evaluation is done through visual inspection of UMAP clusters.

---

## Goal
Pick the best embedding model for the pseudo-labelling pipeline. The chosen model will be used to match incoming unlabelled images against a golden set of labelled examples.

---

## Models Tested

| Key | Model | HuggingFace ID | Type |
|---|---|---|---|
| `bioclip2` | BioCLIP 2 | `imageomics/bioclip-2` | Biology-focused CLIP |
| `dinov2` | DINOv2-Large | `facebook/dinov2-large` | Self-supervised ViT |
| `siglip2` | SigLIP 2 | `google/siglip2-so400m-patch14-384` | General CLIP |
| `evaclip` | EVA-CLIP-8B | `BAAI/EVA-CLIP-8B` | Large CLIP |
| `inat_eva02` | iNat EVA-02 | `timm/eva02_large_patch14_clip_336.merged2b_ft_inat21` | iNat fine-tuned |
| `plantclef` | PlantCLEF DINOv2 | `vincent-espitalier/dino-v2-reg4-with-plantclef2024-weights` | Plant fine-tuned DINOv2 |

---

## Project Structure

```
embedding_comparison/
├── models/
│   ├── base.py            # Abstract base class — contract every model must follow
│   ├── bioclip.py         # BioCLIP 2
│   ├── dinov2.py          # DINOv2-Large
│   ├── siglip.py          # SigLIP 2
│   ├── evaclip.py         # EVA-CLIP-8B
│   ├── inat_eva02.py      # iNat EVA-02 (timm)
│   ├── plantclef.py       # PlantCLEF DINOv2
│   └── __init__.py        # Model registry
├── data/
│   └── image_loader.py    # Loads 40 fixed images from preprocessed folder
├── embeddings/
│   └── extractor.py       # Runs any model, returns embedding matrix
├── evaluation/
│   └── clustering.py      # Cosine similarity + KMeans clustering
├── visualization/
│   └── plot.py            # UMAP thumbnail plots + cosine similarity heatmaps
├── results/               # Auto-generated: plots + summary CSV
├── run_comparison.py      # Main entry point — run this
└── requirements.txt
```

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Usage

```bash
cd /Users/saumithdeversetty/Embedding_models/embedding_comparison
python run_comparison.py
```

### To skip a model
In `run_comparison.py`, comment out any model key in `MODELS_TO_RUN`:

```python
MODELS_TO_RUN = [
    "bioclip2",
    "dinov2",
    # "evaclip",   ← skipped
]
```

### To change the image folder
In `run_comparison.py`:

```python
IMAGE_FOLDER = "/your/path/to/preprocessed/images"
```

### To use GPU
In `run_comparison.py`:

```python
DEVICE = "cuda"
```

---

## Outputs

All outputs are saved to `results/`:

| File | Description |
|---|---|
| `umap_{model}.png` | UMAP plot with image thumbnails at cluster coordinates |
| `heatmap_{model}.png` | Cosine similarity heatmap across all 40 images |
| `summary.csv` | Embedding dim, avg cosine similarity per model |

---

## How to Evaluate Results

Open each `umap_{model}.png` and ask:
- Do visually similar fruits (same defect type, same severity) land near each other?
- Are there clear, separated clusters or is everything mixed together?
- Does the model separate healthy from infected images?

The model that produces the most visually coherent clusters is the one to carry forward to Phase 2.

---

## Configuration (run_comparison.py)

| Variable | Default | Description |
|---|---|---|
| `IMAGE_FOLDER` | `01_Hongo podrido_preprocessed` | Path to preprocessed images |
| `N_SAMPLES` | `40` | Number of images to sample |
| `SEED` | `42` | Fixed seed — same 40 images every run |
| `N_CLUSTERS` | `5` | KMeans cluster count |
| `DEVICE` | `cpu` | `cpu` or `cuda` |

---

## Notes
- All models follow the same interface: image in → embedding vector out
- Adding a new model = one new file in `models/` + one line in `__init__.py`
- No labels required — evaluation is purely visual
- EVA-CLIP-8B requires GPU for reasonable speed
- PlantCLEF and iNat models are CC-BY-NC-4.0 (non-commercial)
