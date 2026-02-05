"""
MedSigLIP Embedder for Semantic Pathology Search.

MedSigLIP is Google's medical vision-language model based on SigLIP architecture,
fine-tuned on medical images including pathology, radiology, dermatology, and ophthalmology.

This module provides:
- Text embedding for pathology queries ("tumor cells", "necrosis", etc.)
- Patch embedding for histopathology image tiles  
- Semantic search: find patches matching text descriptions

Supports:
- google/medsiglip-448 (preferred, medical-specific)
- google/siglip-so400m-patch14-384 (fallback, general-purpose)

Reference: google/medsiglip-448 on HuggingFace
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
import logging
import time
import hashlib

import numpy as np

logger = logging.getLogger(__name__)


def _find_medsiglip_model() -> str:
    """Find MedSigLIP model path, checking common locations."""
    import os
    candidates = [
        os.environ.get("MEDSIGLIP_MODEL_PATH", ""),
        "/app/models/medsiglip",  # Docker path
        os.path.expanduser("~/med-gemma-hackathon/models/medsiglip"),  # Local dev
        "models/medsiglip",  # Relative path
    ]
    for path in candidates:
        if path and Path(path).exists():
            return path
    # Fallback to HuggingFace Hub
    return "google/siglip-so400m-patch14-384"


@dataclass
class MedSigLIPConfig:
    """MedSigLIP configuration."""
    model_id: str = ""  # Auto-detected if empty
    # Fallback: "google/siglip-so400m-patch14-384" for general SigLIP
    batch_size: int = 32
    precision: str = "fp16"  # fp16 or fp32
    cache_dir: str = "data/embeddings/medsiglip_cache"
    device: str = "auto"  # auto, cuda, cpu
    
    def __post_init__(self):
        if not self.model_id:
            self.model_id = _find_medsiglip_model()


class MedSigLIPEmbedder:
    """
    Embedder using MedSigLIP/SigLIP for text-to-patch semantic retrieval in pathology.

    Provides dual encoder for medical image + text.
    Enables "semantic evidence search" - query with text like
    "tumor infiltrating lymphocytes" and find matching patches.
    
    Supports:
    - google/medsiglip-448: 1152-dim, 448x448 input (medical-specific)
    - google/siglip-so400m-patch14-384: 1152-dim, 384x384 input (general)
    """

    # Model constants (updated on load)
    EMBEDDING_DIM = 1152  
    INPUT_SIZE = 384  

    def __init__(self, config: Optional[MedSigLIPConfig] = None):
        self.config = config or MedSigLIPConfig()
        self._model = None
        self._image_processor = None
        self._tokenizer = None
        self._device = None

        # Setup cache directory
        self.cache_dir = Path(self.config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _load_model(self) -> None:
        """Load the SigLIP/MedSigLIP model."""
        if self._model is not None:
            return

        import torch
        from transformers import SiglipImageProcessor, SiglipTokenizer, SiglipModel

        # Determine device
        if self.config.device == "auto":
            if torch.cuda.is_available():
                self._device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self._device = torch.device("mps")
            else:
                self._device = torch.device("cpu")
        else:
            self._device = torch.device(self.config.device)

        logger.info(f"Loading SigLIP model ({self.config.model_id}) on {self._device}")
        start = time.time()

        try:
            self._image_processor = SiglipImageProcessor.from_pretrained(self.config.model_id)
            self._tokenizer = SiglipTokenizer.from_pretrained(self.config.model_id)
            self._model = SiglipModel.from_pretrained(self.config.model_id)

            # Update embedding dim from model config
            if hasattr(self._model.config, 'vision_config'):
                self.EMBEDDING_DIM = self._model.config.vision_config.hidden_size
            if hasattr(self._model.config, 'vision_config') and hasattr(self._model.config.vision_config, 'image_size'):
                self.INPUT_SIZE = self._model.config.vision_config.image_size

            # Move to device and set precision
            self._model = self._model.to(self._device)
            if self.config.precision == "fp16" and self._device.type == "cuda":
                self._model = self._model.half()

            self._model.eval()
            logger.info(f"SigLIP loaded in {time.time()-start:.1f}s: {self.config.model_id} (dim={self.EMBEDDING_DIM}, size={self.INPUT_SIZE})")
            
        except Exception as e:
            logger.error(f"Failed to load SigLIP: {e}")
            raise

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None

    def embed_text(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Embed text query/queries for semantic search.

        Args:
            text: Single query string or list of queries
                  e.g., "tumor cells", "necrotic tissue", "lymphocyte infiltration"

        Returns:
            Normalized embedding(s) of shape (embedding_dim,) or (n_queries, embedding_dim)
        """
        self._load_model()

        import torch

        is_single = isinstance(text, str)
        if is_single:
            text = [text]

        inputs = self._tokenizer(text, padding=True, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model.get_text_features(**inputs)

            # HF SiglipModel.get_text_features() can return a Tensor or a model output.
            if hasattr(outputs, "pooler_output") and isinstance(outputs.pooler_output, torch.Tensor):
                embeddings = outputs.pooler_output
            elif hasattr(outputs, "last_hidden_state") and isinstance(outputs.last_hidden_state, torch.Tensor):
                # Use CLS token representation if pooling isn't provided.
                embeddings = outputs.last_hidden_state[:, 0, :]
            else:
                embeddings = outputs
                if isinstance(embeddings, (tuple, list)) and embeddings and isinstance(embeddings[0], torch.Tensor):
                    embeddings = embeddings[0]

            if not isinstance(embeddings, torch.Tensor):
                raise TypeError(f"Unexpected output from get_text_features: {type(outputs)}")

            # L2 normalize for cosine similarity
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

        embeddings = embeddings.detach().cpu().float().numpy()
        
        if is_single:
            return embeddings.squeeze(0)
        return embeddings

    def embed_image(self, image: np.ndarray) -> np.ndarray:
        """
        Embed a single image.

        Args:
            image: RGB image array (any size, will be resized)

        Returns:
            Normalized embedding of shape (embedding_dim,)
        """
        self._load_model()

        import torch
        from PIL import Image

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        inputs = self._image_processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        if self.config.precision == "fp16" and self._device.type == "cuda":
            inputs = {k: v.half() if v.dtype == torch.float32 else v for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model.get_image_features(**inputs)
            
            # Handle case where model returns BaseModelOutputWithPooling instead of tensor
            if hasattr(outputs, "pooler_output") and isinstance(outputs.pooler_output, torch.Tensor):
                features = outputs.pooler_output
            elif hasattr(outputs, "last_hidden_state") and isinstance(outputs.last_hidden_state, torch.Tensor):
                features = outputs.last_hidden_state[:, 0, :]
            else:
                features = outputs
                if isinstance(features, (tuple, list)) and features and isinstance(features[0], torch.Tensor):
                    features = features[0]
            
            if not isinstance(features, torch.Tensor):
                raise TypeError(f"Unexpected output from get_image_features: {type(outputs)}")
            
            features = features / features.norm(dim=-1, keepdim=True)

        return features.cpu().float().numpy().squeeze()

    def embed_patches(
        self,
        patches: List[np.ndarray],
        cache_key: Optional[str] = None,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Embed a batch of image patches.

        Args:
            patches: List of RGB images (any size, will be resized)
            cache_key: Optional cache key (e.g., slide_id) for persistent storage
            show_progress: Whether to show progress bar

        Returns:
            Embeddings of shape (n_patches, embedding_dim)
        """
        # Check cache first
        if cache_key is not None:
            cache_path = self.cache_dir / f"{cache_key}_siglip.npy"
            if cache_path.exists():
                logger.info(f"Loading SigLIP embeddings from cache: {cache_path}")
                return np.load(cache_path)

        self._load_model()

        import torch
        from PIL import Image
        from tqdm import tqdm

        all_embeddings = []
        batch_size = self.config.batch_size

        # Process in batches
        iterator = range(0, len(patches), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="SigLIP embedding")

        for i in iterator:
            batch = patches[i:i + batch_size]

            # Convert to PIL Images
            pil_images = [
                Image.fromarray(p) if isinstance(p, np.ndarray) else p 
                for p in batch
            ]

            # Preprocess batch
            inputs = self._image_processor(images=pil_images, return_tensors="pt")
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            if self.config.precision == "fp16" and self._device.type == "cuda":
                inputs = {k: v.half() if v.dtype == torch.float32 else v for k, v in inputs.items()}

            # Forward pass
            with torch.no_grad():
                outputs = self._model.get_image_features(**inputs)
                
                # Handle case where model returns BaseModelOutputWithPooling instead of tensor
                if hasattr(outputs, "pooler_output") and isinstance(outputs.pooler_output, torch.Tensor):
                    features = outputs.pooler_output
                elif hasattr(outputs, "last_hidden_state") and isinstance(outputs.last_hidden_state, torch.Tensor):
                    features = outputs.last_hidden_state[:, 0, :]
                else:
                    features = outputs
                    if isinstance(features, (tuple, list)) and features and isinstance(features[0], torch.Tensor):
                        features = features[0]
                
                features = features / features.norm(dim=-1, keepdim=True)

            all_embeddings.append(features.cpu().float().numpy())

        # Concatenate all embeddings
        embeddings = np.concatenate(all_embeddings, axis=0).astype(np.float32)

        # Save to cache
        if cache_key is not None:
            np.save(cache_path, embeddings)
            logger.info(f"Saved SigLIP embeddings to cache: {cache_path}")

        logger.info(f"Generated SigLIP embeddings: {embeddings.shape}")
        return embeddings

    def search(
        self,
        query: str,
        embeddings: np.ndarray,
        metadata: Optional[List[Dict[str, Any]]] = None,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Search for patches matching a text query using cosine similarity.

        Args:
            query: Text query (e.g., "tumor cells", "necrosis", "lymphocytes")
            embeddings: Patch embeddings of shape (n_patches, embedding_dim)
            metadata: Optional metadata for each patch (index, coordinates, etc.)
            top_k: Number of top results to return

        Returns:
            List of dicts with patch_index, similarity_score, and metadata
        """
        # Get text embedding
        query_embedding = self.embed_text(query)
        
        # Ensure embeddings are normalized and correct dtype
        embeddings = embeddings.astype(np.float32)
        query_embedding = query_embedding.astype(np.float32)
        
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # Compute cosine similarities (embeddings should already be normalized)
        similarities = np.dot(embeddings, query_embedding.T).squeeze()

        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        # Build results
        results = []
        for idx in top_indices:
            idx = int(idx)
            result = {
                "patch_index": idx,
                "similarity_score": float(similarities[idx]),
            }
            if metadata is not None and idx < len(metadata):
                result["metadata"] = metadata[idx]
            else:
                result["metadata"] = {"index": idx}
            results.append(result)

        return results

    def compute_similarity(
        self, 
        text: str, 
        images: List[np.ndarray]
    ) -> np.ndarray:
        """
        Compute similarity scores between a text query and multiple images.

        Args:
            text: Text query
            images: List of images

        Returns:
            Array of similarity scores
        """
        text_emb = self.embed_text(text).reshape(1, -1)
        image_embs = self.embed_patches(images, show_progress=False)
        
        return np.dot(image_embs, text_emb.T).squeeze()


# Example pathology queries for semantic search
PATHOLOGY_QUERIES = {
    "tumor": [
        "tumor cells",
        "malignant cells",
        "cancer cells", 
        "neoplastic tissue",
        "carcinoma",
    ],
    "inflammation": [
        "lymphocyte infiltration",
        "inflammatory infiltrate",
        "tumor infiltrating lymphocytes",
        "immune cells",
        "plasma cells",
    ],
    "necrosis": [
        "necrotic tissue",
        "necrosis",
        "cell death",
        "coagulative necrosis",
    ],
    "stroma": [
        "stromal tissue",
        "fibrous stroma",
        "desmoplastic stroma",
        "connective tissue",
    ],
    "mitosis": [
        "mitotic figures",
        "cell division",
        "mitosis",
    ],
    "vessels": [
        "blood vessels",
        "vasculature",
        "angiogenesis",
        "endothelial cells",
    ],
}
