"""
Application state management for Enso Atlas API.

This module manages shared resources like the MIL classifier,
embedder, evidence generator, and MedGemma reporter.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Dict, List

import numpy as np

logger = logging.getLogger(__name__)


class AppState:
    """
    Shared application state for the API.

    Manages lazy-loaded resources to minimize startup time
    while ensuring efficient reuse across requests.
    """

    def __init__(self):
        self._classifier = None
        self._embedder = None
        self._evidence_gen = None
        self._reporter = None
        self._initialized = False

        # Paths from environment
        self.embeddings_dir = Path(
            os.environ.get("EMBEDDINGS_DIR", "data/embeddings")
        )
        self.model_path = Path(
            os.environ.get("MODEL_PATH", "models/demo_clam.pt")
        )

        # Cache for slide data
        self._slide_cache: Dict[str, Dict] = {}
        self._available_slides: List[str] = []

    async def initialize(self) -> None:
        """Initialize shared resources."""
        if self._initialized:
            return

        # Discover available slides
        await self._discover_slides()

        # Build FAISS index for similarity search
        await self._build_faiss_index()

        self._initialized = True
        logger.info("AppState initialized successfully")

    async def _discover_slides(self) -> None:
        """Discover available slides from embeddings directory."""
        self._available_slides = []

        if self.embeddings_dir.exists():
            for f in sorted(self.embeddings_dir.glob("*.npy")):
                if not f.name.endswith("_coords.npy"):
                    slide_id = f.stem
                    self._available_slides.append(slide_id)

        logger.info(f"Discovered {len(self._available_slides)} slides")

    async def _build_faiss_index(self) -> None:
        """Build FAISS index for similarity search."""
        if not self._available_slides:
            logger.warning("No slides available for FAISS index")
            return

        evidence_gen = self.evidence_generator

        all_embeddings = []
        all_metadata = []

        for slide_id in self._available_slides:
            emb_path = self.embeddings_dir / f"{slide_id}.npy"
            if emb_path.exists():
                embs = np.load(emb_path)
                all_embeddings.append(embs)
                all_metadata.append({
                    "slide_id": slide_id,
                    "n_patches": len(embs),
                })

        if all_embeddings:
            evidence_gen.build_reference_index(all_embeddings, all_metadata)
            logger.info(f"Built FAISS index with {len(all_embeddings)} slides")

    @property
    def classifier(self):
        """Lazy-load the MIL classifier."""
        if self._classifier is None:
            from enso_atlas.config import MILConfig
            from enso_atlas.mil.clam import CLAMClassifier

            config = MILConfig(input_dim=384, hidden_dim=128)
            self._classifier = CLAMClassifier(config)

            if self.model_path.exists():
                self._classifier.load(self.model_path)
                logger.info(f"Loaded CLAM model from {self.model_path}")
            else:
                logger.warning(f"Model not found at {self.model_path}, using random weights")

        return self._classifier

    @property
    def embedder(self):
        """Lazy-load the Path Foundation embedder."""
        if self._embedder is None:
            from enso_atlas.config import EmbeddingConfig
            from enso_atlas.embedding.embedder import PathFoundationEmbedder

            config = EmbeddingConfig()
            self._embedder = PathFoundationEmbedder(config)
            logger.info("Initialized Path Foundation embedder")

        return self._embedder

    @property
    def evidence_generator(self):
        """Lazy-load the evidence generator."""
        if self._evidence_gen is None:
            from enso_atlas.config import EvidenceConfig
            from enso_atlas.evidence.generator import EvidenceGenerator

            config = EvidenceConfig()
            self._evidence_gen = EvidenceGenerator(config)
            logger.info("Initialized evidence generator")

        return self._evidence_gen

    @property
    def reporter(self):
        """Lazy-load the MedGemma reporter."""
        if self._reporter is None:
            from enso_atlas.reporting.medgemma import MedGemmaReporter, ReportingConfig

            config = ReportingConfig()
            self._reporter = MedGemmaReporter(config)
            logger.info("Initialized MedGemma reporter")

        return self._reporter

    @property
    def available_slides(self) -> List[str]:
        """Get list of available slide IDs."""
        return self._available_slides

    def get_slide_embeddings(self, slide_id: str) -> Optional[np.ndarray]:
        """Load embeddings for a slide."""
        emb_path = self.embeddings_dir / f"{slide_id}.npy"
        if emb_path.exists():
            return np.load(emb_path)
        return None

    def get_slide_coords(self, slide_id: str) -> Optional[np.ndarray]:
        """Load coordinates for a slide."""
        coord_path = self.embeddings_dir / f"{slide_id}_coords.npy"
        if coord_path.exists():
            return np.load(coord_path)
        return None
