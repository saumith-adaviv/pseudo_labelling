import numpy as np
import torch
from PIL import Image
from typing import List
from transformers import AutoModel, AutoImageProcessor

from models.base import BaseEmbeddingModel


class DINOv2Model(BaseEmbeddingModel):
    """
    DINOv2-Large embedding model.
    Self-supervised ViT trained on LVD-142M web images.
    Excellent texture, color, and fine-grained visual sensitivity.
    HuggingFace: facebook/dinov2-large
    """

    MODEL_ID = "facebook/dinov2-large"

    def load_model(self):
        self.processor = AutoImageProcessor.from_pretrained(self.MODEL_ID)
        self.model = AutoModel.from_pretrained(self.MODEL_ID)
        self.model = self.model.to(self.device)
        self.model.eval()
        print(f"[DINOv2-Large] Loaded on {self.device}")

    def get_embeddings(self, images: List[Image.Image]) -> np.ndarray:
        with torch.no_grad():
            inputs = self.processor(images=images, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :]
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        return embeddings.cpu().numpy()
