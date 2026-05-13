"""Patch embedding API routes."""

import base64
import io
import logging
from collections.abc import Callable
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException
from PIL import Image

from .schemas import EmbedRequest, EmbedResponse

logger = logging.getLogger(__name__)


def create_embedding_router(
    *,
    embedder_provider: Callable[[], Any],
) -> APIRouter:
    router = APIRouter()

    @router.post("/api/embed", response_model=EmbedResponse)
    async def embed_patches(request: EmbedRequest):
        """Generate embeddings for patch images using Path Foundation."""
        embedder = embedder_provider()
        if embedder is None:
            raise HTTPException(status_code=503, detail="Embedder not initialized")

        patches = []
        for i, b64_patch in enumerate(request.patches):
            try:
                if "," in b64_patch:
                    b64_patch = b64_patch.split(",", 1)[1]

                image_data = base64.b64decode(b64_patch)
                image = Image.open(io.BytesIO(image_data))

                if image.mode != "RGB":
                    image = image.convert("RGB")

                if image.size != (224, 224):
                    image = image.resize((224, 224), Image.Resampling.LANCZOS)

                patches.append(np.array(image))
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to decode patch {i}: {str(e)}",
                )

        try:
            embeddings = embedder.embed(patches, show_progress=False)
        except Exception as e:
            logger.error("Embedding generation failed: %s", e)
            raise HTTPException(
                status_code=500,
                detail=f"Embedding generation failed: {str(e)}",
            )

        response = EmbedResponse(
            num_patches=len(patches),
            embedding_dim=embeddings.shape[1] if len(embeddings.shape) > 1 else 384,
        )

        if request.return_embeddings:
            response.embeddings = embeddings.tolist()

        return response

    return router
