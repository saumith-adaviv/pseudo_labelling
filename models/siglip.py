import numpy as np
import torch
from PIL import Image
from typing import List
from transformers import AutoModel, AutoProcessor

from models.base import BaseEmbeddingModel


class SigLIPModel(BaseEmbeddingModel):
    """
    SigLIP 2 embedding model (Google).
    Improved CLIP with sigmoid loss. Strong text+image alignment.
    Useful for zero-shot label matching in later pipeline phases.
    HuggingFace: google/siglip2-so400m-patch14-384
    """

    MODEL_ID = "google/siglip2-so400m-patch14-384"

    def load_model(self):
        self.processor = AutoProcessor.from_pretrained(self.MODEL_ID)
        self.model = AutoModel.from_pretrained(self.MODEL_ID)
        self.model = self.model.to(self.device)
        self.model.eval()
        print(f"[SigLIP 2] Loaded on {self.device}")

    def get_embeddings(self, images: List[Image.Image]) -> np.ndarray:
        with torch.no_grad():
            inputs = self.processor(images=images, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model.vision_model(**inputs)
            embeddings = outputs.pooler_output
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        return embeddings.cpu().numpy()
