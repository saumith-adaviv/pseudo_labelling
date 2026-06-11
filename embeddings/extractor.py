import time
import numpy as np
from PIL import Image
from typing import List
from tqdm import tqdm


def extract_embeddings(
    model,
    images: List[Image.Image],
    batch_size: int = 32
) -> np.ndarray:
    """
    Extracts embeddings for a list of images using any model.
    Processes images in batches with a progress bar and timing.

    Args:
        model:      Any model that implements BaseEmbeddingModel.
        images:     List of PIL Image objects.
        batch_size: Number of images to process at once (default 32).

    Returns:
        embeddings: numpy array of shape (N, embedding_dim).
    """
    total_images  = len(images)
    total_batches = (total_images + batch_size - 1) // batch_size
    all_embeddings = []

    start_time = time.time()

    with tqdm(
        total=total_images,
        desc=f"{model.__class__.__name__}",
        unit="img",
        ncols=80,
        colour="green"
    ) as pbar:
        for i in range(0, total_images, batch_size):
            batch      = images[i:i + batch_size]
            embeddings = model.get_embeddings(batch)
            all_embeddings.append(embeddings)
            pbar.update(len(batch))

    elapsed = time.time() - start_time
    result  = np.vstack(all_embeddings)

    print(f"  ✓ {total_images} images | shape {result.shape} | {elapsed:.1f}s ({total_images/elapsed:.1f} img/s)")
    return result
