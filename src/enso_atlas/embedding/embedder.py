"""
Embedders for histopathology patches.

Contains:
- PathFoundationEmbedder: Feature extraction using Path Foundation model
- MedSigLIPEmbedder: Text-to-patch semantic search using SigLIP/MedSigLIP
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union
import logging
import hashlib

import numpy as np

from enso_atlas.config import EmbeddingConfig

logger = logging.getLogger(__name__)


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
        """Load the Path Foundation model.
        
        Uses local_files_only=True when the model is already cached locally,
        preventing any network access to HuggingFace Hub.
        """
        if self._model is not None:
            return

        import os
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

        # Detect whether model is already cached locally.
        # Check HF_HOME / TRANSFORMERS_CACHE / default cache for the model directory.
        local_files_only = False
        hf_home = os.environ.get("HF_HOME", os.environ.get(
            "TRANSFORMERS_CACHE",
            os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
        ))
        model_cache_dir = os.path.join(hf_home, "hub", f"models--{model_id.replace('/', '--')}")
        if os.path.isdir(model_cache_dir):
            local_files_only = True
            logger.info(f"Path Foundation model cache found at {model_cache_dir}; loading with local_files_only=True (no network)")

        # Also respect explicit offline env vars
        if os.environ.get("TRANSFORMERS_OFFLINE", "0") == "1" or os.environ.get("HF_HUB_OFFLINE", "0") == "1":
            local_files_only = True
            logger.info("Offline mode enabled via environment variable; loading with local_files_only=True")

        self._processor = AutoImageProcessor.from_pretrained(model_id, local_files_only=local_files_only)
        self._model = AutoModel.from_pretrained(model_id, local_files_only=local_files_only)

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


@dataclass
class MedSigLIPConfig:
    """MedSigLIP configuration."""
    model_id: str = "google/siglip-so400m-patch14-384"  # Base SigLIP model
    batch_size: int = 32
    precision: str = "fp16"
    cache_dir: str = "data/medsiglip_cache"


class MedSigLIPEmbedder:
    """
    Embedder using MedSigLIP/SigLIP for text-to-patch semantic retrieval.

    MedSigLIP provides dual encoder for medical image + text.
    Enables "semantic evidence search" - query with text like
    "tumor infiltrating lymphocytes" and find matching patches.

    Note: Using SigLIP as base since MedSigLIP follows the same architecture.
    For medical-specific fine-tuned model, update model_id when available.
    """

    EMBEDDING_DIM = 1152  # SigLIP SO400M dimension
    INPUT_SIZE = 384

    def __init__(self, config: Optional[MedSigLIPConfig] = None):
        self.config = config or MedSigLIPConfig()
        self._model = None
        self._processor = None
        self._device = None
        self._patch_embeddings = None
        self._patch_metadata = None

        # Setup cache directory
        self.cache_dir = Path(self.config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _load_model(self) -> None:
        """Load the SigLIP/MedSigLIP model.
        
        Uses local_files_only=True when the model is already cached locally.
        """
        if self._model is not None:
            return

        import os
        import torch
        from transformers import AutoModel, AutoProcessor

        # Determine device
        if torch.cuda.is_available():
            self._device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self._device = torch.device("mps")
        else:
            self._device = torch.device("cpu")

        logger.info(f"Loading MedSigLIP model on {self._device}")

        model_id = self.config.model_id

        # Check if model is cached locally
        local_files_only = False
        hf_home = os.environ.get("HF_HOME", os.environ.get(
            "TRANSFORMERS_CACHE",
            os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
        ))
        model_cache_dir = os.path.join(hf_home, "hub", f"models--{model_id.replace('/', '--')}")
        if os.path.isdir(model_cache_dir):
            local_files_only = True
            logger.info(f"MedSigLIP model cache found at {model_cache_dir}; loading with local_files_only=True")
        if os.environ.get("TRANSFORMERS_OFFLINE", "0") == "1" or os.environ.get("HF_HUB_OFFLINE", "0") == "1":
            local_files_only = True

        self._processor = AutoProcessor.from_pretrained(model_id, local_files_only=local_files_only)
        self._model = AutoModel.from_pretrained(model_id, local_files_only=local_files_only)

        # Move to device and set precision
        self._model = self._model.to(self._device)
        if self.config.precision == "fp16" and self._device.type == "cuda":
            self._model = self._model.half()

        self._model.eval()
        logger.info(f"MedSigLIP model loaded successfully: {model_id}")

    def embed_image(self, image: np.ndarray) -> np.ndarray:
        """
        Embed a single image.

        Args:
            image: RGB image array

        Returns:
            Normalized embedding vector
        """
        self._load_model()

        import torch
        from PIL import Image

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        inputs = self._processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        if self.config.precision == "fp16" and self._device.type == "cuda":
            inputs = {k: v.half() if v.dtype == torch.float32 else v for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model.get_image_features(**inputs)
            
            # Handle case where model returns BaseModelOutputWithPooling instead of tensor
            if hasattr(outputs, "pooler_output") and isinstance(outputs.pooler_output, torch.Tensor):
                embedding = outputs.pooler_output
            elif hasattr(outputs, "last_hidden_state") and isinstance(outputs.last_hidden_state, torch.Tensor):
                embedding = outputs.last_hidden_state[:, 0, :]
            else:
                embedding = outputs
                if isinstance(embedding, (tuple, list)) and embedding and isinstance(embedding[0], torch.Tensor):
                    embedding = embedding[0]
            
            # Normalize for cosine similarity
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)

        return embedding.cpu().numpy().squeeze()

    def embed_text(self, text: str) -> np.ndarray:
        """
        Embed a text query.

        Args:
            text: Text query (e.g., "tumor infiltrating lymphocytes")

        Returns:
            Normalized embedding vector
        """
        self._load_model()

        import torch

        inputs = self._processor(text=text, return_tensors="pt", padding=True)
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model.get_text_features(**inputs)
            
            # Handle case where model returns BaseModelOutputWithPooling instead of tensor
            if hasattr(outputs, "pooler_output") and isinstance(outputs.pooler_output, torch.Tensor):
                embedding = outputs.pooler_output
            elif hasattr(outputs, "last_hidden_state") and isinstance(outputs.last_hidden_state, torch.Tensor):
                embedding = outputs.last_hidden_state[:, 0, :]
            else:
                embedding = outputs
                if isinstance(embedding, (tuple, list)) and embedding and isinstance(embedding[0], torch.Tensor):
                    embedding = embedding[0]
            
            # Normalize for cosine similarity
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)

        return embedding.cpu().numpy().squeeze()

    def embed_patches(
        self,
        patches: List[np.ndarray],
        cache_key: Optional[str] = None,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Embed a batch of image patches.

        Args:
            patches: List of RGB images
            cache_key: Optional cache key for storing/loading embeddings
            show_progress: Whether to show progress bar

        Returns:
            Embeddings of shape (n_patches, embedding_dim)
        """
        # Try loading from cache first
        if cache_key is not None:
            cache_path = self.cache_dir / f"{hashlib.md5(cache_key.encode()).hexdigest()}_siglip.npy"
            if cache_path.exists():
                logger.info(f"Loading MedSigLIP embeddings from cache: {cache_path}")
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
            iterator = tqdm(iterator, desc="MedSigLIP embedding patches")

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
                outputs = self._model.get_image_features(**inputs)
                
                # Handle case where model returns BaseModelOutputWithPooling instead of tensor
                if hasattr(outputs, "pooler_output") and isinstance(outputs.pooler_output, torch.Tensor):
                    embeddings = outputs.pooler_output
                elif hasattr(outputs, "last_hidden_state") and isinstance(outputs.last_hidden_state, torch.Tensor):
                    embeddings = outputs.last_hidden_state[:, 0, :]
                else:
                    embeddings = outputs
                    if isinstance(embeddings, (tuple, list)) and embeddings and isinstance(embeddings[0], torch.Tensor):
                        embeddings = embeddings[0]
                
                # Normalize for cosine similarity
                embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

            all_embeddings.append(embeddings.cpu().numpy())

        # Concatenate all embeddings
        embeddings = np.concatenate(all_embeddings, axis=0)

        # Convert to FP16 for storage efficiency
        if self.config.precision == "fp16":
            embeddings = embeddings.astype(np.float16)

        # Save to cache
        if cache_key is not None:
            np.save(cache_path, embeddings)
            logger.info(f"Saved MedSigLIP embeddings to cache: {cache_path}")

        logger.info(f"Generated MedSigLIP embeddings: {embeddings.shape}")
        return embeddings

    def build_search_index(
        self,
        embeddings: np.ndarray,
        metadata: Optional[List[dict]] = None,
    ) -> None:
        """
        Build an in-memory search index from patch embeddings.

        Args:
            embeddings: Patch embeddings of shape (n_patches, embedding_dim)
            metadata: Optional list of metadata dicts for each patch
        """
        self._patch_embeddings = embeddings.astype(np.float32)
        self._patch_metadata = metadata or [{"index": i} for i in range(len(embeddings))]
        logger.info(f"Built search index with {len(embeddings)} patches")

    def search(
        self,
        query: str,
        top_k: int = 10,
        embeddings: Optional[np.ndarray] = None,
        metadata: Optional[List[dict]] = None,
    ) -> List[dict]:
        """
        Search for patches matching a text query using cosine similarity.

        Args:
            query: Text query (e.g., "tumor cells", "necrosis", "lymphocytes")
            top_k: Number of top results to return
            embeddings: Optional embeddings to search (uses built index if None)
            metadata: Optional metadata for the embeddings

        Returns:
            List of dicts with patch_index, similarity_score, and metadata
        """
        # Use provided embeddings or built index
        if embeddings is not None:
            search_embeddings = embeddings.astype(np.float32)
            search_metadata = metadata or [{"index": i} for i in range(len(embeddings))]
        elif self._patch_embeddings is not None:
            search_embeddings = self._patch_embeddings
            search_metadata = self._patch_metadata
        else:
            raise ValueError("No embeddings available. Call build_search_index() first or provide embeddings.")

        # Get text embedding
        query_embedding = self.embed_text(query).astype(np.float32)

        # Ensure embeddings are normalized (should already be, but be safe)
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # Compute cosine similarities
        similarities = np.dot(search_embeddings, query_embedding.T).squeeze()

        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            result = {
                "patch_index": int(idx),
                "similarity_score": float(similarities[idx]),
                "metadata": search_metadata[idx] if idx < len(search_metadata) else {},
            }
            results.append(result)

        return results

    def search_with_faiss(
        self,
        query: str,
        faiss_index,
        metadata: List[dict],
        top_k: int = 10,
    ) -> List[dict]:
        """
        Search using a pre-built FAISS index for large-scale retrieval.

        Args:
            query: Text query
            faiss_index: FAISS index built from patch embeddings
            metadata: Metadata list aligned with FAISS index
            top_k: Number of results

        Returns:
            List of search results with similarity scores
        """
        import faiss

        # Get text embedding
        query_embedding = self.embed_text(query).astype(np.float32)
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # Search FAISS index
        distances, indices = faiss_index.search(query_embedding, top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            result = {
                "patch_index": int(idx),
                "similarity_score": float(1.0 - dist),  # Convert L2 distance to similarity
                "metadata": metadata[idx] if idx < len(metadata) else {},
            }
            results.append(result)

        return results
