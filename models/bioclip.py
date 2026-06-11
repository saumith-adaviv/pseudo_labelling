import numpy as np
import torch
from PIL import Image
from typing import List

from models.base import BaseEmbeddingModel


class BioCLIPModel(BaseEmbeddingModel):
    """
    BioCLIP 2 embedding model.
    Trained on 200M biological organism images (TreeOfLife-200M).
    Uses open_clip library.
    HuggingFace: imageomics/bioclip-2
    """

    MODEL_ID = "hf-hub:imageomics/bioclip-2"

    def load_model(self):
        import open_clip
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.MODEL_ID
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        print(f"[BioCLIP 2] Loaded on {self.device}")

    def get_embeddings(self, images: List[Image.Image]) -> np.ndarray:
        with torch.no_grad():
            batch = torch.stack([self.preprocess(img) for img in images]).to(self.device)
            embeddings = self.model.encode_image(batch)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        return embeddings.cpu().numpy()
