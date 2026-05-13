"""
Path Foundation Embedder - Google's histopathology foundation model.
Uses TensorFlow SavedModel format from google/path-foundation.
Produces 384-dim embeddings, same as DINOv2-small.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class PathFoundationEmbedder:
    """Feature extractor using Path Foundation model (384-dim embeddings)."""

    EMBEDDING_DIM = 384
    INPUT_SIZE = 224

    def __init__(self, cache_dir: str = "data/embeddings", batch_size: int = 64):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size
        self._model: Any | None = None
        self._infer: Any | None = None

    def _load_model(self) -> None:
        if self._model is not None:
            return

        import tensorflow as tf
        from huggingface_hub import from_pretrained_keras

        # Configure GPU memory growth to avoid OOM
        gpus = tf.config.experimental.list_physical_devices("GPU")
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"Found {len(gpus)} GPU(s)")

        logger.info("Loading Path Foundation model from Hugging Face...")
        self._model = from_pretrained_keras("google/path-foundation")
        if self._model is None:
            raise RuntimeError("Failed to load Path Foundation model")
        self._infer = self._model.signatures["serving_default"]
        logger.info("Path Foundation model loaded successfully")

    def _preprocess_batch(self, patches: list[np.ndarray]) -> Any:
        """Preprocess a batch of patches for Path Foundation."""
        import tensorflow as tf

        # Stack patches into batch
        batch = np.stack(patches, axis=0)

        # Convert to float32 and normalize to [0, 1]
        batch = batch.astype(np.float32) / 255.0

        return tf.constant(batch)

    def embed(
        self,
        patches: list[np.ndarray],
        cache_key: str | None = None,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Generate embeddings for a list of patches."""
        self._load_model()
        if self._infer is None:
            raise RuntimeError("Path Foundation inference function is not initialized")

        from tqdm import tqdm

        all_embeddings = []

        iterator = range(0, len(patches), self.batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Embedding patches (PathFound)")

        for i in iterator:
            batch_patches = patches[i : i + self.batch_size]

            # Ensure all patches are 224x224
            processed = []
            for p in batch_patches:
                if p.shape[:2] != (self.INPUT_SIZE, self.INPUT_SIZE):
                    from PIL import Image

                    img = Image.fromarray(p).resize((self.INPUT_SIZE, self.INPUT_SIZE))
                    p = np.array(img)
                processed.append(p)

            # Preprocess and run inference
            batch_tensor = self._preprocess_batch(processed)
            outputs = self._infer(batch_tensor)

            # Extract embeddings (output key is 'output_0')
            batch_embeddings = outputs["output_0"].numpy()
            all_embeddings.append(batch_embeddings)

        concatenated_embeddings: np.ndarray = np.concatenate(all_embeddings, axis=0)
        logger.info(f"Generated embeddings: {concatenated_embeddings.shape}")
        return concatenated_embeddings
