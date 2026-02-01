"""
Path Foundation Embedder - Feature extraction from histopathology patches.

Uses the Path Foundation model from Google Health AI:
- Input: 224x224 H&E patches
- Output: 384-dimensional embeddings
- Architecture: ViT-S
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union
import logging
import hashlib

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingConfig:
    """Embedding model configuration."""
    model: str = "path-foundation"
    batch_size: int = 64
    precision: str = "fp16"
    cache_dir: str = "data/embeddings"


class PathFoundationEmbedder:
    """
    Feature extractor using Path Foundation model.
    
    Path Foundation produces 384-dimensional embeddings from 224x224 H&E patches.
    Designed for histopathology tasks with efficient downstream computation.
    """
    
    # Model constants
    EMBEDDING_DIM = 384
    INPUT_SIZE = 224
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self._model = None
        self._processor = None
        self._device = None
        
        # Setup cache directory
        self.cache_dir = Path(config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_model(self) -> None:
        """Load the Path Foundation model."""
        if self._model is not None:
            return
        
        import torch
        from transformers import AutoModel, AutoImageProcessor
        
        # Determine device
        if torch.cuda.is_available():
            self._device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self._device = torch.device("mps")
        else:
            self._device = torch.device("cpu")
        
        logger.info(f"Loading Path Foundation model on {self._device}")
        
        # Load model and processor
        # Note: Path Foundation is available at google/path-foundation on HuggingFace
        model_id = "google/path-foundation"
        
        self._processor = AutoImageProcessor.from_pretrained(model_id)
        self._model = AutoModel.from_pretrained(model_id)
        
        # Move to device and set precision
        self._model = self._model.to(self._device)
        if self.config.precision == "fp16" and self._device.type == "cuda":
            self._model = self._model.half()
        
        self._model.eval()
        logger.info("Path Foundation model loaded successfully")
    
    def _get_cache_key(self, cache_key: str) -> str:
        """Generate a hash-based cache key."""
        return hashlib.md5(cache_key.encode()).hexdigest()
    
    def _load_from_cache(self, cache_key: str) -> Optional[np.ndarray]:
        """Load embeddings from cache if available."""
        key = self._get_cache_key(cache_key)
        cache_path = self.cache_dir / f"{key}.npy"
        
        if cache_path.exists():
            logger.info(f"Loading embeddings from cache: {cache_path}")
            return np.load(cache_path)
        return None
    
    def _save_to_cache(self, embeddings: np.ndarray, cache_key: str) -> None:
        """Save embeddings to cache."""
        key = self._get_cache_key(cache_key)
        cache_path = self.cache_dir / f"{key}.npy"
        
        np.save(cache_path, embeddings)
        logger.info(f"Saved embeddings to cache: {cache_path}")
    
    def embed_single(self, patch: np.ndarray) -> np.ndarray:
        """
        Embed a single patch.
        
        Args:
            patch: RGB image of shape (224, 224, 3)
            
        Returns:
            Embedding of shape (384,)
        """
        self._load_model()
        
        import torch
        from PIL import Image
        
        # Convert to PIL Image if needed
        if isinstance(patch, np.ndarray):
            patch = Image.fromarray(patch)
        
        # Preprocess
        inputs = self._processor(images=patch, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        
        if self.config.precision == "fp16" and self._device.type == "cuda":
            inputs = {k: v.half() if v.dtype == torch.float32 else v for k, v in inputs.items()}
        
        # Forward pass
        with torch.no_grad():
            outputs = self._model(**inputs)
            # Get CLS token embedding
            embedding = outputs.last_hidden_state[:, 0, :].squeeze()
        
        return embedding.cpu().numpy()
    
    def embed(
        self,
        patches: List[np.ndarray],
        cache_key: Optional[str] = None,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Embed a batch of patches.
        
        Args:
            patches: List of RGB images, each (224, 224, 3)
            cache_key: Optional cache key for storing/loading embeddings
            show_progress: Whether to show progress bar
            
        Returns:
            Embeddings of shape (n_patches, 384)
        """
        # Try loading from cache first
        if cache_key is not None:
            cached = self._load_from_cache(cache_key)
            if cached is not None:
                return cached
        
        self._load_model()
        
        import torch
        from PIL import Image
        from tqdm import tqdm
        
        all_embeddings = []
        batch_size = self.config.batch_size
        
        # Process in batches
        iterator = range(0, len(patches), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Embedding patches")
        
        for i in iterator:
            batch = patches[i:i + batch_size]
            
            # Convert to PIL Images
            pil_images = [Image.fromarray(p) if isinstance(p, np.ndarray) else p for p in batch]
            
            # Preprocess batch
            inputs = self._processor(images=pil_images, return_tensors="pt")
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            
            if self.config.precision == "fp16" and self._device.type == "cuda":
                inputs = {k: v.half() if v.dtype == torch.float32 else v for k, v in inputs.items()}
            
            # Forward pass
            with torch.no_grad():
                outputs = self._model(**inputs)
                embeddings = outputs.last_hidden_state[:, 0, :]
            
            all_embeddings.append(embeddings.cpu().numpy())
        
        # Concatenate all embeddings
        embeddings = np.concatenate(all_embeddings, axis=0)
        
        # Convert to FP16 for storage efficiency
        if self.config.precision == "fp16":
            embeddings = embeddings.astype(np.float16)
        
        # Save to cache
        if cache_key is not None:
            self._save_to_cache(embeddings, cache_key)
        
        logger.info(f"Generated embeddings: {embeddings.shape}")
        return embeddings


class MedSigLIPEmbedder:
    """
    Optional embedder using MedSigLIP for text-to-patch retrieval.
    
    MedSigLIP provides dual encoder for medical image + text.
    Useful for semantic evidence search (text query → patch retrieval).
    """
    
    EMBEDDING_DIM = 768
    INPUT_SIZE = 448
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self._model = None
        self._processor = None
        self._device = None
    
    def _load_model(self) -> None:
        """Load MedSigLIP model."""
        if self._model is not None:
            return
        
        import torch
        from transformers import AutoModel, AutoProcessor
        
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Loading MedSigLIP model on {self._device}")
        
        model_id = "google/medsiglip"
        self._processor = AutoProcessor.from_pretrained(model_id)
        self._model = AutoModel.from_pretrained(model_id).to(self._device).eval()
        
        logger.info("MedSigLIP model loaded successfully")
    
    def embed_image(self, image: np.ndarray) -> np.ndarray:
        """Embed a single image."""
        self._load_model()
        
        import torch
        from PIL import Image
        
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        inputs = self._processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        
        with torch.no_grad():
            embedding = self._model.get_image_features(**inputs)
        
        return embedding.cpu().numpy().squeeze()
    
    def embed_text(self, text: str) -> np.ndarray:
        """Embed a text query."""
        self._load_model()
        
        import torch
        
        inputs = self._processor(text=text, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        
        with torch.no_grad():
            embedding = self._model.get_text_features(**inputs)
        
        return embedding.cpu().numpy().squeeze()
