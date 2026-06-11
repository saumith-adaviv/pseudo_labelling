import numpy as np
import torch
from PIL import Image
from typing import List

from models.base import BaseEmbeddingModel


class EVACLIPModel(BaseEmbeddingModel):
    """
    EVA-CLIP-8B embedding model (BAAI).
    Loaded via open_clip which has native EVA-CLIP support.
    HuggingFace: BAAI/EVA-CLIP-8B
    NOTE: Requires GPU for reasonable inference speed.
    """

    # EVA02-E-14 is the correct open_clip name for the EVA-02 Enormous model
    MODEL_NAME = "EVA02-E-14"
    PRETRAINED = "laion2b_s4b_b115k"

    def load_model(self):
        import open_clip
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.MODEL_NAME,
            pretrained=self.PRETRAINED
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        print(f"[EVA-CLIP-8B] Loaded on {self.device}")

    def get_embeddings(self, images: List[Image.Image]) -> np.ndarray:
        embeddings = []
        with torch.no_grad():
            for img in images:
                tensor = self.preprocess(img).unsqueeze(0).to(self.device)
                embedding = self.model.encode_image(tensor)
                embedding = embedding / embedding.norm(dim=-1, keepdim=True)
                embeddings.append(embedding.cpu().float().numpy())
        return np.vstack(embeddings)
