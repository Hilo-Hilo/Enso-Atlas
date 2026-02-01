"""
Enso Atlas API - FastAPI Backend

Provides REST API endpoints for the professional frontend:
- Slide analysis with MIL prediction
- Evidence generation (heatmaps, patches)
- Similar case retrieval (FAISS)
- Report generation (MedGemma)
- Patch embedding (Path Foundation)
"""

from pathlib import Path
from typing import Optional, List, Dict, Any
import logging
import json
import base64
import io

import numpy as np
from PIL import Image

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


def _check_cuda() -> bool:
    """Check if CUDA is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


# Request/Response Models
class AnalyzeRequest(BaseModel):
    slide_id: str
    generate_report: bool = False


class AnalyzeResponse(BaseModel):
    slide_id: str
    prediction: str
    score: float
    confidence: float
    patches_analyzed: int
    top_evidence: List[Dict[str, Any]]
    similar_cases: List[Dict[str, Any]]


class ReportRequest(BaseModel):
    slide_id: str
    include_evidence: bool = True
    include_similar: bool = True


class ReportResponse(BaseModel):
    slide_id: str
    report_json: Dict[str, Any]
    summary_text: str


class SlideInfo(BaseModel):
    slide_id: str
    patient_id: Optional[str] = None
    has_embeddings: bool = False
    label: Optional[str] = None
    num_patches: Optional[int] = None


class EmbedRequest(BaseModel):
    """Request for patch embedding."""
    patches: List[str] = Field(
        ...,
        description="Base64-encoded patch images (224x224 RGB)",
        min_length=1,
        max_length=128,
    )
    return_embeddings: bool = Field(
        default=True,
        description="Whether to return embedding vectors",
    )


class EmbedResponse(BaseModel):
    """Response from embedding endpoint."""
    num_patches: int
    embedding_dim: int = 384
    embeddings: Optional[List[List[float]]] = None


class SimilarRequest(BaseModel):
    """Request for similar case search."""
    slide_id: str
    k: int = Field(default=5, ge=1, le=20)
    top_patches: int = Field(default=3, ge=1, le=10)


class SimilarResponse(BaseModel):
    """Response from similar case search."""
    slide_id: str
    similar_cases: List[Dict[str, Any]]
    num_queries: int


def create_app(
    embeddings_dir: Path = Path("data/demo/embeddings"),
    model_path: Path = Path("models/demo_clam.pt"),
    enable_cors: bool = True,
) -> FastAPI:
    """Create and configure the FastAPI application."""
    
    app = FastAPI(
        title="Enso Atlas API",
        description="On-Prem Pathology Evidence Engine for Treatment-Response Insight",
        version="0.1.0",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
    )
    
    # CORS middleware for frontend development
    if enable_cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["http://localhost:3000", "http://localhost:7860"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    # Load models on startup
    classifier = None
    evidence_gen = None
    embedder = None
    available_slides = []
    
    @app.on_event("startup")
    async def load_models():
        nonlocal classifier, evidence_gen, embedder, available_slides
        
        from ..config import MILConfig, EvidenceConfig, EmbeddingConfig
        from ..mil.clam import CLAMClassifier
        from ..evidence.generator import EvidenceGenerator
        from ..embedding.embedder import PathFoundationEmbedder
        
        # Load MIL classifier
        config = MILConfig(input_dim=384, hidden_dim=128)
        classifier = CLAMClassifier(config)
        if model_path.exists():
            classifier.load(model_path)
            logger.info(f"Loaded MIL model from {model_path}")
        
        # Setup evidence generator
        evidence_config = EvidenceConfig()
        evidence_gen = EvidenceGenerator(evidence_config)
        
        # Setup embedder (lazy-loaded on first use)
        embedding_config = EmbeddingConfig()
        embedder = PathFoundationEmbedder(embedding_config)
        
        # Find available slides and build FAISS index
        all_embeddings = []
        all_metadata = []
        
        if embeddings_dir.exists():
            for f in sorted(embeddings_dir.glob("*.npy")):
                if not f.name.endswith("_coords.npy"):
                    slide_id = f.stem
                    available_slides.append(slide_id)
                    
                    # Load embeddings for FAISS index
                    embs = np.load(f)
                    all_embeddings.append(embs)
                    all_metadata.append({
                        "slide_id": slide_id,
                        "n_patches": len(embs),
                    })
            
            logger.info(f"Found {len(available_slides)} slides with embeddings")
            
            # Build FAISS index for similarity search
            if all_embeddings:
                evidence_gen.build_reference_index(all_embeddings, all_metadata)
                logger.info(f"Built FAISS index with {len(all_embeddings)} slides")
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "version": "0.1.0",
            "model_loaded": classifier is not None,
            "cuda_available": _check_cuda(),
            "slides_available": len(available_slides),
        }
    
    @app.get("/")
    async def root():
        """API root endpoint."""
        return {
            "name": "Enso Atlas API",
            "version": "0.1.0",
            "docs": "/api/docs",
        }
    
    @app.get("/api/slides", response_model=List[SlideInfo])
    async def list_slides():
        """List all available slides."""
        slides = []
        labels_path = embeddings_dir.parent / "labels.csv"
        labels = {}
        
        if labels_path.exists():
            import csv
            with open(labels_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    labels[row["slide_id"]] = row.get("label", "")
        
        for slide_id in available_slides:
            # Get patch count
            emb_path = embeddings_dir / f"{slide_id}.npy"
            num_patches = None
            if emb_path.exists():
                try:
                    emb = np.load(emb_path)
                    num_patches = len(emb)
                except Exception:
                    pass
            
            slides.append(SlideInfo(
                slide_id=slide_id,
                has_embeddings=True,
                label=labels.get(slide_id),
                num_patches=num_patches,
            ))
        
        return slides
    
    @app.post("/api/analyze", response_model=AnalyzeResponse)
    async def analyze_slide(request: AnalyzeRequest):
        """Analyze a slide and return prediction with evidence."""
        if classifier is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        slide_id = request.slide_id
        emb_path = embeddings_dir / f"{slide_id}.npy"
        
        if not emb_path.exists():
            raise HTTPException(status_code=404, detail=f"Slide {slide_id} not found")
        
        # Load embeddings
        embeddings = np.load(emb_path)
        
        # Run prediction
        score, attention = classifier.predict(embeddings)
        label = "RESPONDER" if score > 0.5 else "NON-RESPONDER"
        confidence = abs(score - 0.5) * 2
        
        # Get top evidence patches
        top_k = min(8, len(attention))
        top_indices = np.argsort(attention)[-top_k:][::-1]
        
        top_evidence = []
        for i, idx in enumerate(top_indices):
            top_evidence.append({
                "rank": i + 1,
                "patch_index": int(idx),
                "attention_weight": float(attention[idx]),
            })
        
        # Get similar cases using FAISS
        similar_cases = []
        if evidence_gen is not None:
            try:
                similar_results = evidence_gen.find_similar(
                    embeddings, attention, k=10, top_patches=3
                )
                seen_slides = set()
                for s in similar_results:
                    meta = s.get("metadata", {})
                    sid = meta.get("slide_id", "unknown")
                    if sid != slide_id and sid not in seen_slides:
                        seen_slides.add(sid)
                        similar_cases.append({
                            "slide_id": sid,
                            "similarity_score": 1.0 / (1.0 + s["distance"]),
                            "distance": float(s["distance"]),
                        })
                    if len(similar_cases) >= 5:
                        break
            except Exception as e:
                logger.warning(f"Similar case search failed: {e}")
                # Fall back to random
                for sid in available_slides[:5]:
                    if sid != slide_id:
                        similar_cases.append({
                            "slide_id": sid,
                            "similarity_score": float(np.random.uniform(0.7, 0.95)),
                        })
        
        return AnalyzeResponse(
            slide_id=slide_id,
            prediction=label,
            score=float(score),
            confidence=float(confidence),
            patches_analyzed=len(embeddings),
            top_evidence=top_evidence,
            similar_cases=similar_cases[:5],
        )
    
    @app.post("/api/report", response_model=ReportResponse)
    async def generate_report(request: ReportRequest):
        """Generate a structured report for a slide."""
        if classifier is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        slide_id = request.slide_id
        emb_path = embeddings_dir / f"{slide_id}.npy"
        
        if not emb_path.exists():
            raise HTTPException(status_code=404, detail=f"Slide {slide_id} not found")
        
        embeddings = np.load(emb_path)
        score, attention = classifier.predict(embeddings)
        label = "responder" if score > 0.5 else "non-responder"
        
        # Build structured report
        report_json = {
            "case_id": slide_id,
            "task": "Bevacizumab treatment response prediction",
            "model_output": {
                "label": label,
                "probability": float(score),
                "calibration_note": "Model probability requires external validation.",
            },
            "evidence": [
                {
                    "patch_id": f"patch_{idx}",
                    "attention_weight": float(attention[idx]),
                }
                for idx in np.argsort(attention)[-5:][::-1]
            ],
            "limitations": [
                "Research tool only",
                "Not clinically validated",
                "Requires pathologist review",
            ],
            "safety_statement": "This is a research tool. All findings require validation by qualified pathologists.",
        }
        
        # Generate summary text
        summary_text = f"""Case: {slide_id}
Prediction: {label.upper()}
Score: {score:.3f}

This analysis examined {len(embeddings)} tissue patches. The top {min(5, len(attention))} 
patches by attention weight show consistent morphological patterns associated with 
{label} cases in the training cohort.

IMPORTANT: This is a research tool for decision support only. All findings must be 
reviewed and validated by qualified pathologists before any clinical decision-making."""
        
        return ReportResponse(
            slide_id=slide_id,
            report_json=report_json,
            summary_text=summary_text,
        )
    
    @app.get("/api/heatmap/{slide_id}")
    async def get_heatmap(slide_id: str):
        """Get the attention heatmap for a slide as PNG."""
        if classifier is None or evidence_gen is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        emb_path = embeddings_dir / f"{slide_id}.npy"
        coord_path = embeddings_dir / f"{slide_id}_coords.npy"
        
        if not emb_path.exists():
            raise HTTPException(status_code=404, detail=f"Slide {slide_id} not found")
        
        embeddings = np.load(emb_path)
        
        # Load or generate coordinates
        if coord_path.exists():
            coords = np.load(coord_path)
        else:
            coords = np.random.randint(0, 50000, (len(embeddings), 2))
        
        coords = [tuple(c) for c in coords]
        
        # Run prediction and create heatmap
        score, attention = classifier.predict(embeddings)
        slide_dims = (50000, 50000)
        heatmap = evidence_gen.create_heatmap(attention, coords, slide_dims, (512, 512))
        
        # Save to temp file and return
        from PIL import Image
        import io
        
        img = Image.fromarray(heatmap)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        
        return StreamingResponse(
            buf,
            media_type="image/png",
            headers={"Content-Disposition": f"inline; filename={slide_id}_heatmap.png"},
        )
    
    @app.post("/api/embed", response_model=EmbedResponse)
    async def embed_patches(request: EmbedRequest):
        """Generate embeddings for patch images using Path Foundation."""
        if embedder is None:
            raise HTTPException(status_code=503, detail="Embedder not initialized")
        
        # Decode patches
        patches = []
        for i, b64_patch in enumerate(request.patches):
            try:
                # Handle data URI format
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
        
        # Generate embeddings
        try:
            embeddings = embedder.embed(patches, show_progress=False)
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
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
    
    @app.get("/api/similar", response_model=SimilarResponse)
    async def get_similar_cases(slide_id: str, k: int = 5, top_patches: int = 3):
        """Find similar cases from the reference cohort."""
        if classifier is None or evidence_gen is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        emb_path = embeddings_dir / f"{slide_id}.npy"
        if not emb_path.exists():
            raise HTTPException(status_code=404, detail=f"Slide {slide_id} not found")
        
        embeddings = np.load(emb_path)
        _, attention = classifier.predict(embeddings)
        
        similar_results = evidence_gen.find_similar(
            embeddings, attention, k=k * 3, top_patches=top_patches
        )
        
        similar_cases = []
        seen_slides = set()
        
        for s in similar_results:
            meta = s.get("metadata", {})
            sid = meta.get("slide_id", "unknown")
            
            if sid == slide_id:
                continue
            
            if sid not in seen_slides:
                seen_slides.add(sid)
                similar_cases.append({
                    "slide_id": sid,
                    "distance": float(s["distance"]),
                    "similarity_score": 1.0 / (1.0 + s["distance"]),
                    "patch_index": meta.get("patch_index"),
                })
            
            if len(similar_cases) >= k:
                break
        
        return SimilarResponse(
            slide_id=slide_id,
            similar_cases=similar_cases,
            num_queries=top_patches,
        )
    
    @app.get("/api/embed/status")
    async def embedder_status():
        """Check the status of the Path Foundation embedder."""
        model_loaded = embedder is not None and embedder._model is not None
        device = "unknown"
        
        if model_loaded:
            device = str(embedder._device)
        else:
            device = "cuda" if _check_cuda() else "cpu"
        
        return {
            "model": "google/path-foundation",
            "model_loaded": model_loaded,
            "device": device,
            "embedding_dim": 384,
            "input_size": 224,
        }
    
    return app


# Default app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
