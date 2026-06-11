from abc import ABC, abstractmethod
import numpy as np
from PIL import Image
from typing import List


class BaseEmbeddingModel(ABC):
    """
    Abstract base class for all embedding models.
    Every model must implement load_model() and get_embeddings().
    The rest of the pipeline only calls these two methods.
    """

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.model = None

    @abstractmethod
    def load_model(self):
        """
        Load the model and preprocessor.
        Called once before any embedding extraction.
        """
        pass

    @abstractmethod
    def get_embeddings(self, images: List[Image.Image]) -> np.ndarray:
        """
        Takes a list of PIL images.
        Returns a numpy array of shape (N, embedding_dim).
        """
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}(device={self.device})"
