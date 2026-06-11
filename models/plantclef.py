import numpy as np
import torch
from PIL import Image
from typing import List
from transformers import Dinov2Model, AutoImageProcessor

from models.base import BaseEmbeddingModel


class PlantCLEFModel(BaseEmbeddingModel):
    """
    PlantCLEF 2024 DINOv2 embedding model.
    DINOv2-Base fine-tuned on PlantCLEF 2024 competition data (plant species recognition).
    Switched from vincent-espitalier (weights on Zenodo, not HuggingFace) to
    gerald29/plantclef2024 which is properly hosted on HuggingFace with MIT license.
    HuggingFace: gerald29/plantclef2024
    License: MIT
    """

    MODEL_ID     = "gerald29/plantclef2024"
    PROCESSOR_ID = "facebook/dinov2-base"

    def load_model(self):
        # Use standard DINOv2 processor — same architecture
        self.processor = AutoImageProcessor.from_pretrained(self.PROCESSOR_ID)
        # Load as Dinov2Model directly
        self.model = Dinov2Model.from_pretrained(self.MODEL_ID)
        self.model = self.model.to(self.device)
        self.model.eval()
        print(f"[PlantCLEF DINOv2] Loaded on {self.device}")

    def get_embeddings(self, images: List[Image.Image]) -> np.ndarray:
        with torch.no_grad():
            inputs = self.processor(images=images, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :]
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        return embeddings.cpu().numpy()
