"""
DINOv2 Embedder - Alternative to Path Foundation for histopathology.
Uses facebook/dinov2-small which produces 384-dim embeddings.
"""

from pathlib import Path
from typing import List, Optional
import logging
import hashlib
import numpy as np

logger = logging.getLogger(__name__)


class DINOv2Embedder:
    """Feature extractor using DINOv2-small model (384-dim embeddings)."""
    
    EMBEDDING_DIM = 384
    INPUT_SIZE = 224
    
    def __init__(self, cache_dir: str = "data/embeddings", batch_size: int = 64, precision: str = "fp16"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size
        self.precision = precision
        self._model = None
        self._processor = None
        self._device = None
    
    def _load_model(self) -> None:
        if self._model is not None:
            return
        
        import torch
        from transformers import AutoModel, AutoImageProcessor
        
        if torch.cuda.is_available():
            self._device = torch.device("cuda")
        else:
            self._device = torch.device("cpu")
        
        logger.info(f"Loading DINOv2-small model on {self._device}")
        
        model_id = "facebook/dinov2-small"
        self._processor = AutoImageProcessor.from_pretrained(model_id)
        self._model = AutoModel.from_pretrained(model_id)
        self._model = self._model.to(self._device)
        
        if self.precision == "fp16" and self._device.type == "cuda":
            self._model = self._model.half()
        
        self._model.eval()
        logger.info("DINOv2 model loaded successfully")
    
    def _get_cache_key(self, cache_key: str) -> str:
        return hashlib.md5(cache_key.encode()).hexdigest()
    
    def embed(
        self,
        patches: List[np.ndarray],
        cache_key: Optional[str] = None,
        show_progress: bool = True,
    ) -> np.ndarray:
        self._load_model()
        
        import torch
        from PIL import Image
        from tqdm import tqdm
        
        all_embeddings = []
        
        iterator = range(0, len(patches), self.batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Embedding patches")
        
        for i in iterator:
            batch = patches[i:i + self.batch_size]
            pil_images = [Image.fromarray(p) if isinstance(p, np.ndarray) else p for p in batch]
            
            inputs = self._processor(images=pil_images, return_tensors="pt")
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            
            if self.precision == "fp16" and self._device.type == "cuda":
                inputs = {k: v.half() if v.dtype == torch.float32 else v for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self._model(**inputs)
                embeddings = outputs.last_hidden_state[:, 0, :]
            
            all_embeddings.append(embeddings.cpu().numpy())
        
        embeddings = np.concatenate(all_embeddings, axis=0)
        
        if self.precision == "fp16":
            embeddings = embeddings.astype(np.float16)
        
        logger.info(f"Generated embeddings: {embeddings.shape}")
        return embeddings
