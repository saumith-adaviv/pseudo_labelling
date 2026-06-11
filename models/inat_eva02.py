import numpy as np
import torch
from PIL import Image
from typing import List

from models.base import BaseEmbeddingModel


class iNatEVA02Model(BaseEmbeddingModel):
    """
    EVA-02 Large fine-tuned on iNaturalist 2021 (10,000 species).
    Real-world citizen science plant photos — closest domain to harvest images.
    Uses timm library. forward_features() extracts embeddings before classifier head.
    HuggingFace: timm/eva02_large_patch14_clip_336.merged2b_ft_inat21
    License: CC-BY-NC-4.0
    """

    # Load from HuggingFace hub via timm
    MODEL_ID = "hf_hub:timm/eva02_large_patch14_clip_336.merged2b_ft_inat21"

    def load_model(self):
        import timm
        from timm.data import resolve_data_config
        from timm.data.transforms_factory import create_transform

        # num_classes=0 removes the classification head — gives raw embeddings
        self.model = timm.create_model(
            self.MODEL_ID,
            pretrained=True,
            num_classes=0
        )
        self.model = self.model.to(self.device)
        self.model.eval()

        # Use timm's built-in transform for this model
        config = resolve_data_config({}, model=self.model)
        self.preprocess = create_transform(**config)
        print(f"[iNat EVA-02] Loaded on {self.device}")

    def get_embeddings(self, images: List[Image.Image]) -> np.ndarray:
        with torch.no_grad():
            batch = torch.stack([self.preprocess(img) for img in images]).to(self.device)
            embeddings = self.model.forward_features(batch)
            if embeddings.dim() == 3:
                embeddings = embeddings.mean(dim=1)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        return embeddings.cpu().numpy()
