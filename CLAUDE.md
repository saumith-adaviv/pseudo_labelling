# CLAUDE.md — Embedding Comparison Pipeline

## What this project is
Phase 1 of the Adaviv Harvest Quality pseudo-labelling pipeline.
The goal is to find which embedding model best groups visually similar crop defect images together, so it can be used for golden-set matching in Phase 2.

No labels. No training. Pure zero-shot embedding comparison on 40 preprocessed cropped fruit images.

---

## Project context
- **Sprint:** HQ Pipeline Sprint Plan v2 — Saumith's work
- **Deadline:** Friday Jun 5
- **Deliverable:** Short memo — models tried, which one picked, why, with example UMAP clusters
- **Next phase:** Use the picked model to convert 200-500 labelled tomato images to vectors and match new images against the golden set

---

## Code principles
- **Modular:** Every model is one file. Same interface in and out — image in, embedding vector out
- **No labels needed:** Evaluation is visual inspection of UMAP clusters, not metrics
- **Fixed seed:** Same 40 images run through every model for fair comparison
- **Swap models easily:** Comment/uncomment in `MODELS_TO_RUN` in `run_comparison.py`
- **No training:** All models are frozen — zero-shot embedding only

---

## How to run
```bash
pip install -r requirements.txt
python run_comparison.py
```

---

## Adding a new model
1. Create `models/yournewmodel.py` — inherit from `BaseEmbeddingModel`
2. Implement `load_model()` and `get_embeddings()`
3. Add one line to `models/__init__.py` MODEL_REGISTRY
4. Add the key to `MODELS_TO_RUN` in `run_comparison.py`

Nothing else needs to change.

---

## Key files
| File | Role |
|---|---|
| `run_comparison.py` | Main entry point — configure and run here |
| `models/base.py` | Contract every model must follow |
| `models/__init__.py` | Model registry — one place to add/remove models |
| `data/image_loader.py` | Loads 40 fixed images from preprocessed folder |
| `embeddings/extractor.py` | Runs any model, returns embedding matrix |
| `evaluation/clustering.py` | Cosine similarity + KMeans |
| `visualization/plot.py` | UMAP thumbnails + heatmaps → saved to results/ |

---

## Image data
- **Source:** Preprocessed cropped individual fruit images (one fruit per image)
- **Current dataset:** `01_Hongo podrido_preprocessed` (fungal infection — tomato)
- **Sampling:** 40 images, fixed seed=42, same across all models
- **No labels** — evaluation is visual only

---

## Models in priority order
1. **MUST TEST:** `bioclip2`, `dinov2`, `inat_eva02`
2. **STRONGLY RECOMMENDED:** `siglip2`, `evaclip`
3. **OPTIONAL:** `plantclef`

---

## What NOT to do
- Do not fine-tune any model — zero-shot only in this phase
- Do not change the random seed between model runs — breaks comparability
- Do not add classification heads — embeddings only
- Do not access the raw images folder — use preprocessed only

---

## Dependencies
See `requirements.txt`. Key libraries:
- `open-clip-torch` — for BioCLIP 2
- `timm` — for iNat EVA-02
- `transformers` — for DINOv2, SigLIP, EVA-CLIP, PlantCLEF
- `umap-learn` — for dimensionality reduction
- `scikit-learn` — for KMeans and cosine similarity
