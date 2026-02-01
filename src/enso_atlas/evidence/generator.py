"""
Evidence Generator - Heatmaps, patch selection, and similarity search.

This module produces the human-interpretable evidence that makes
Enso Atlas valuable for clinical decision support:
- Attention heatmaps overlaid on slide thumbnails
- Top-K evidence patch selection
- FAISS-based similarity search
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import logging

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EvidenceConfig:
    """Evidence generation configuration."""
    top_k_patches: int = 12
    heatmap_alpha: float = 0.4
    similarity_k: int = 20
    faiss_index_type: str = "IVF1024,Flat"
    colormap: str = "jet"  # Colormap for heatmap


class EvidenceGenerator:
    """
    Generator for interpretable evidence from MIL predictions.
    
    Produces:
    - Heatmap overlays showing model attention
    - Top evidence patches with coordinates
    - Similar patches/cases from reference cohort
    """
    
    def __init__(self, config: EvidenceConfig):
        self.config = config
        self._faiss_index = None
        self._reference_metadata = []  # Store metadata for indexed embeddings
    
    def create_heatmap(
        self,
        attention_weights: np.ndarray,
        coordinates: List[Tuple[int, int]],
        slide_dimensions: Tuple[int, int],
        thumbnail_size: Tuple[int, int] = (1024, 1024),
    ) -> np.ndarray:
        """
        Create an attention heatmap overlay.
        
        Args:
            attention_weights: Per-patch attention weights (n_patches,)
            coordinates: List of (x, y) patch coordinates at level 0
            slide_dimensions: (width, height) of the slide at level 0
            thumbnail_size: Size of the output heatmap
            
        Returns:
            RGBA heatmap array of shape (H, W, 4)
        """
        import cv2
        
        slide_w, slide_h = slide_dimensions
        thumb_w, thumb_h = thumbnail_size
        
        # Create empty heatmap
        heatmap = np.zeros((thumb_h, thumb_w), dtype=np.float32)
        count_map = np.zeros((thumb_h, thumb_w), dtype=np.float32)
        
        # Normalize attention weights
        attention_weights = attention_weights.astype(np.float32)
        if attention_weights.max() > attention_weights.min():
            attention_weights = (attention_weights - attention_weights.min()) / (
                attention_weights.max() - attention_weights.min()
            )
        
        # Map patches to thumbnail coordinates
        scale_x = thumb_w / slide_w
        scale_y = thumb_h / slide_h
        patch_size_thumb = max(1, int(224 * scale_x))  # Approximate patch size in thumbnail
        
        for (x, y), weight in zip(coordinates, attention_weights):
            tx = int(x * scale_x)
            ty = int(y * scale_y)
            
            # Add attention weight to heatmap region
            x1 = max(0, tx)
            y1 = max(0, ty)
            x2 = min(thumb_w, tx + patch_size_thumb)
            y2 = min(thumb_h, ty + patch_size_thumb)
            
            if x2 > x1 and y2 > y1:
                heatmap[y1:y2, x1:x2] += weight
                count_map[y1:y2, x1:x2] += 1
        
        # Average overlapping regions
        count_map[count_map == 0] = 1
        heatmap = heatmap / count_map
        
        # Apply Gaussian blur for smoothness
        heatmap = cv2.GaussianBlur(heatmap, (31, 31), 0)
        
        # Normalize to 0-1
        if heatmap.max() > heatmap.min():
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        
        # Apply colormap
        heatmap_colored = self._apply_colormap(heatmap)
        
        logger.info(f"Created heatmap: {heatmap_colored.shape}")
        return heatmap_colored
    
    def _apply_colormap(self, heatmap: np.ndarray) -> np.ndarray:
        """Apply colormap to grayscale heatmap."""
        import cv2
        
        # Convert to uint8
        heatmap_uint8 = (heatmap * 255).astype(np.uint8)
        
        # Apply colormap
        if self.config.colormap == "jet":
            colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        elif self.config.colormap == "hot":
            colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_HOT)
        else:
            colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        
        # Convert BGR to RGB
        colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
        
        # Add alpha channel based on intensity
        alpha = (heatmap * 255 * self.config.heatmap_alpha).astype(np.uint8)
        heatmap_rgba = np.dstack([colored, alpha])
        
        return heatmap_rgba
    
    def select_top_patches(
        self,
        patches: List[np.ndarray],
        coordinates: List[Tuple[int, int]],
        attention_weights: np.ndarray,
        k: Optional[int] = None,
    ) -> List[Dict]:
        """
        Select top-K patches based on attention weights.
        
        Args:
            patches: List of patch images
            coordinates: List of (x, y) coordinates
            attention_weights: Per-patch attention weights
            k: Number of patches to select (default from config)
            
        Returns:
            List of dicts with patch info
        """
        k = k or self.config.top_k_patches
        
        # Get top-k indices
        top_indices = np.argsort(attention_weights)[-k:][::-1]
        
        evidence_patches = []
        for rank, idx in enumerate(top_indices):
            evidence_patches.append({
                "rank": rank + 1,
                "index": int(idx),
                "coordinates": coordinates[idx],
                "attention_weight": float(attention_weights[idx]),
                "patch": patches[idx] if idx < len(patches) else None,
            })
        
        logger.info(f"Selected top {len(evidence_patches)} evidence patches")
        return evidence_patches
    
    def build_reference_index(
        self,
        embeddings_list: List[np.ndarray],
        metadata_list: List[Dict],
    ) -> None:
        """
        Build FAISS index for similarity search.
        
        Args:
            embeddings_list: List of embedding arrays
            metadata_list: List of metadata dicts for each embedding set
        """
        import faiss
        
        # Concatenate all embeddings
        all_embeddings = []
        all_metadata = []
        
        for embeddings, metadata in zip(embeddings_list, metadata_list):
            for i, emb in enumerate(embeddings):
                all_embeddings.append(emb)
                all_metadata.append({
                    **metadata,
                    "patch_index": i,
                })
        
        all_embeddings = np.array(all_embeddings).astype(np.float32)
        
        # Create FAISS index
        dim = all_embeddings.shape[1]
        
        if len(all_embeddings) < 1024:
            # Use flat index for small datasets
            self._faiss_index = faiss.IndexFlatL2(dim)
        else:
            # Use IVF index for larger datasets
            quantizer = faiss.IndexFlatL2(dim)
            n_clusters = min(1024, len(all_embeddings) // 10)
            self._faiss_index = faiss.IndexIVFFlat(quantizer, dim, n_clusters)
            self._faiss_index.train(all_embeddings)
        
        self._faiss_index.add(all_embeddings)
        self._reference_metadata = all_metadata
        
        logger.info(f"Built FAISS index with {len(all_embeddings)} embeddings")
    
    def find_similar(
        self,
        embeddings: np.ndarray,
        attention_weights: np.ndarray,
        k: Optional[int] = None,
        top_patches: int = 5,
    ) -> List[Dict]:
        """
        Find similar patches/cases from reference cohort.
        
        Args:
            embeddings: Query patch embeddings (n_patches, dim)
            attention_weights: Attention weights for query patches
            k: Number of similar patches per query
            top_patches: Number of top attention patches to use as queries
            
        Returns:
            List of similar case dicts
        """
        k = k or self.config.similarity_k
        
        if self._faiss_index is None:
            logger.warning("No reference index built. Returning empty results.")
            return []
        
        # Get top attention patches as queries
        top_indices = np.argsort(attention_weights)[-top_patches:][::-1]
        query_embeddings = embeddings[top_indices].astype(np.float32)
        
        # Search FAISS index
        distances, indices = self._faiss_index.search(query_embeddings, k)
        
        # Compile results
        similar_cases = []
        for i, (query_idx, dists, idxs) in enumerate(zip(top_indices, distances, indices)):
            for dist, ref_idx in zip(dists, idxs):
                if ref_idx >= 0 and ref_idx < len(self._reference_metadata):
                    similar_cases.append({
                        "query_patch_index": int(query_idx),
                        "query_attention": float(attention_weights[query_idx]),
                        "reference_index": int(ref_idx),
                        "distance": float(dist),
                        "metadata": self._reference_metadata[ref_idx],
                    })
        
        # Sort by distance and deduplicate by reference slide
        similar_cases.sort(key=lambda x: x["distance"])
        
        logger.info(f"Found {len(similar_cases)} similar patches")
        return similar_cases
    
    def save_index(self, path: str | Path) -> None:
        """Save FAISS index to disk."""
        import faiss
        import pickle
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if self._faiss_index is not None:
            faiss.write_index(self._faiss_index, str(path.with_suffix(".faiss")))
            with open(path.with_suffix(".meta.pkl"), "wb") as f:
                pickle.dump(self._reference_metadata, f)
            logger.info(f"Saved FAISS index to {path}")
    
    def load_index(self, path: str | Path) -> None:
        """Load FAISS index from disk."""
        import faiss
        import pickle
        
        path = Path(path)
        faiss_path = path.with_suffix(".faiss")
        meta_path = path.with_suffix(".meta.pkl")
        
        if faiss_path.exists():
            self._faiss_index = faiss.read_index(str(faiss_path))
            if meta_path.exists():
                with open(meta_path, "rb") as f:
                    self._reference_metadata = pickle.load(f)
            logger.info(f"Loaded FAISS index from {path}")
        else:
            raise FileNotFoundError(f"Index not found: {faiss_path}")


def create_overlay_image(
    thumbnail: np.ndarray,
    heatmap: np.ndarray,
) -> np.ndarray:
    """
    Create composite image of thumbnail with heatmap overlay.
    
    Args:
        thumbnail: RGB thumbnail image
        heatmap: RGBA heatmap image
        
    Returns:
        Composite RGB image
    """
    from PIL import Image
    
    # Ensure same size
    thumb_h, thumb_w = thumbnail.shape[:2]
    heat_h, heat_w = heatmap.shape[:2]
    
    if (thumb_w, thumb_h) != (heat_w, heat_h):
        heatmap_pil = Image.fromarray(heatmap).resize((thumb_w, thumb_h))
        heatmap = np.array(heatmap_pil)
    
    # Convert to PIL
    thumb_pil = Image.fromarray(thumbnail).convert("RGBA")
    heat_pil = Image.fromarray(heatmap)
    
    # Composite
    composite = Image.alpha_composite(thumb_pil, heat_pil)
    
    return np.array(composite.convert("RGB"))
