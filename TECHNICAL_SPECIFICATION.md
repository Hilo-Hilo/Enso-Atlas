# Enso Atlas: Technical Specification

## On-Premise Pathology Evidence Engine for Treatment-Response Prediction

**Version:** 1.0.0
**Date:** February 2025
**Authors:** Enso Labs
**Classification:** Technical Documentation -- Hackathon Submission / Investor Review

---

# Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Architecture](#2-system-architecture)
3. [Foundation Models](#3-foundation-models)
4. [TransMIL Architecture](#4-transmil-architecture)
5. [Backend API Specification](#5-backend-api-specification)
6. [Agentic Workflow](#6-agentic-workflow)
7. [Database Schema](#7-database-schema)
8. [Frontend Architecture](#8-frontend-architecture)
9. [Config-Driven Project System](#9-config-driven-project-system)
10. [Deployment Guide](#10-deployment-guide)
11. [Data Pipeline](#11-data-pipeline)
12. [Security and Compliance](#12-security-and-compliance)
13. [Performance Benchmarks](#13-performance-benchmarks)
14. [Known Limitations and Future Work](#14-known-limitations-and-future-work)
15. [Appendices](#15-appendices)

---

# 1. Executive Summary

## 1.1 Project Overview

Enso Atlas is an on-premise pathology evidence engine that predicts treatment response from whole-slide histopathology images (WSIs) using Google's HAI-DEF foundation models. The system is designed for deployment within hospital networks, ensuring that Protected Health Information (PHI) never leaves the premises. It combines three Google foundation models -- Path Foundation, MedGemma 4B, and MedSigLIP -- with a Transformer-based Multiple Instance Learning (TransMIL) classifier to deliver interpretable, evidence-grounded predictions.

The platform addresses a critical gap in precision oncology: determining which patients will respond to specific chemotherapy regimens before treatment begins. Currently, treatment response can only be assessed after multiple cycles of therapy, exposing non-responding patients to unnecessary toxicity and delays in receiving effective alternatives. Enso Atlas provides early, morphology-based predictions from standard H&E-stained tissue slides that are already collected during routine clinical care.

### Mission Statement

To build the most transparent, evidence-grounded pathology AI system that gives oncologists, pathologists, and tumor boards the information they need to make better treatment decisions -- while keeping patient data secure and on-premise.

### Key Capabilities

- **Treatment-Response Prediction:** Binary classification of platinum sensitivity in high-grade serous ovarian carcinoma (HGSOC) using TransMIL with attention-based evidence
- **Multi-Model Ensemble:** Five specialized TransMIL models covering platinum sensitivity, chemosensitivity, BRCA status, tumor grade, and survival prediction
- **Evidence Generation:** Attention heatmaps overlaid on WSIs, top-K evidence patches with coordinates, and FAISS-based similar case retrieval
- **Semantic Tissue Search:** MedSigLIP-powered text-to-patch search enabling queries like "tumor infiltrating lymphocytes" or "necrotic tissue"
- **Structured Reporting:** MedGemma 4B generates tumor-board-ready reports with morphology descriptions, clinical significance, limitations, and safety statements
- **Agentic Workflow:** Seven-step AI agent with visible reasoning, SSE streaming, and session memory for follow-up questions
- **PDF Export:** Professional PDF reports suitable for clinical documentation and tumor board presentations

## 1.2 Target Users

| User Role | Primary Use Case | Key Features |
|---|---|---|
| Medical Oncologist | Treatment planning, tumor board discussion | Oncologist summary view, decision support, confidence scores |
| Pathologist | Slide review, morphology verification | WSI viewer with heatmap overlay, annotation tools, pathologist view |
| Tumor Board | Multidisciplinary case discussion | PDF reports, similar case comparison, batch analysis |
| Clinical Researcher | Cohort analysis, biomarker discovery | Batch analysis, semantic search, project system |
| Hospital IT | Deployment, maintenance | Docker Compose, on-premise, PostgreSQL, health checks |

## 1.3 Competition Landscape

Enso Atlas differentiates from existing computational pathology platforms in several key areas:

| Feature | Enso Atlas | Paige.ai | PathAI | Google DeepMind |
|---|---|---|---|---|
| On-Premise | Yes (required) | Cloud + Edge | Cloud | Research only |
| Foundation Models | Path Foundation + MedGemma + MedSigLIP | Proprietary | Proprietary | Internal |
| Explainability | Attention heatmaps + evidence patches + semantic search + structured reports | Limited | Moderate | Research |
| Treatment Response | Yes (platinum sensitivity) | Cancer detection | Drug response (research) | N/A |
| Open Architecture | Config-driven, extensible | Closed | Closed | N/A |
| Cost | Open source / self-hosted | SaaS subscription | SaaS subscription | N/A |

---

# 2. System Architecture

## 2.1 High-Level Architecture

```mermaid
graph TB
    subgraph "Frontend (Next.js 14)"
        UI[React UI - Port 3000]
        OSD[OpenSeadragon WSI Viewer]
        API_CLIENT[API Client - api.ts]
    end

    subgraph "Backend (FastAPI - Port 8003)"
        API[REST API - main.py]
        AGENT[Agent Workflow - workflow.py]
        DB_MOD[Database Module - database.py]
        PROJ[Project Registry - projects.py]
        BATCH[Batch Tasks - batch_tasks.py]
        REPORT[Report Tasks - report_tasks.py]
        PDF[PDF Export - pdf_export.py]
    end

    subgraph "ML Models"
        PF[Path Foundation - 384-dim embeddings]
        MG[MedGemma 4B - Report Generation]
        MS[MedSigLIP - Semantic Search]
        TM[TransMIL - MIL Classification]
        FAISS[FAISS Index - Similarity Search]
    end

    subgraph "Storage"
        PG[(PostgreSQL 16)]
        EMB[/Embeddings .npy/]
        WSI_FILES[/WSI Files .svs/]
        MODELS[/Model Checkpoints/]
    end

    UI --> API_CLIENT
    API_CLIENT -->|HTTP/SSE| API
    OSD -->|DZI Tiles| API
    API --> DB_MOD
    API --> AGENT
    API --> BATCH
    API --> REPORT
    API --> PDF
    AGENT --> MG
    AGENT --> MS
    AGENT --> TM
    AGENT --> FAISS
    API --> PF
    API --> MG
    API --> MS
    API --> TM
    API --> FAISS
    DB_MOD --> PG
    PROJ --> PG
    PF --> EMB
    TM --> EMB
    API --> WSI_FILES
    TM --> MODELS
    MG --> MODELS
    MS --> MODELS
```

## 2.2 Component Overview

### Frontend Layer
- **Framework:** Next.js 14 with React, TypeScript, Tailwind CSS
- **WSI Viewer:** OpenSeadragon for Deep Zoom Image (DZI) tile-based rendering
- **State Management:** React Context (ProjectContext), custom hooks (useAnalysis), localStorage caching
- **API Communication:** Custom fetch wrapper with retry logic, timeout handling, and SSE streaming

### Backend Layer
- **Framework:** FastAPI with Pydantic models for request/response validation
- **Primary Module:** `src/enso_atlas/api/main.py` (5,560 lines) -- houses all core endpoints, model loading, and startup orchestration
- **Database:** asyncpg connection pool to PostgreSQL 16 (raw SQL, no ORM)
- **Background Tasks:** Thread-based batch analysis and report generation with status polling

### ML Models Layer
- **Path Foundation:** TensorFlow SavedModel producing 384-dim patch embeddings (CPU)
- **MedGemma 4B:** Transformers-based causal LM for structured report generation (GPU, bfloat16)
- **MedSigLIP:** SigLIP vision-language model for text-to-patch semantic search (GPU, fp16)
- **TransMIL:** PyTorch Transformer-based MIL classifier with attention extraction (GPU)
- **FAISS:** Facebook AI Similarity Search for slide-level and patch-level retrieval

### Storage Layer
- **PostgreSQL 16:** Slides, patients, metadata, analysis results, embedding tasks, projects
- **File System:** Pre-computed .npy embeddings, .npy coordinate files, .svs whole-slide images
- **Model Storage:** HuggingFace cache + local checkpoints in `models/` directory

## 2.3 Data Flow Diagram

The end-to-end data flow from raw WSI to clinical report follows this pipeline:

```mermaid
flowchart LR
    subgraph "Offline Pipeline"
        WSI[WSI .svs File] --> PATCH[Patch Extraction<br/>224x224 @ Level 0]
        PATCH --> TISSUE[Tissue Masking<br/>Background Removal]
        TISSUE --> PF_EMB[Path Foundation<br/>Embedding]
        PF_EMB --> NPY[.npy Files<br/>N x 384]
    end

    subgraph "Online Analysis"
        NPY --> TRANSMIL[TransMIL<br/>Classification]
        TRANSMIL --> PRED[Prediction<br/>+ Attention Weights]
        PRED --> HEATMAP[Attention<br/>Heatmap]
        PRED --> EVIDENCE[Top-K Evidence<br/>Patches]
        NPY --> FAISS_Q[FAISS<br/>Query]
        FAISS_Q --> SIMILAR[Similar<br/>Cases]
        NPY --> SIGLIP[MedSigLIP<br/>Semantic Search]
        SIGLIP --> TEXT_MATCH[Text-Matched<br/>Patches]
    end

    subgraph "Reporting"
        PRED --> MEDGEMMA[MedGemma 4B<br/>Report Generator]
        EVIDENCE --> MEDGEMMA
        SIMILAR --> MEDGEMMA
        MEDGEMMA --> JSON_REPORT[Structured<br/>JSON Report]
        JSON_REPORT --> PDF_GEN[PDF<br/>Generator]
        PDF_GEN --> PDF_OUT[Tumor Board<br/>PDF]
    end
```

## 2.4 Deployment Topology

The system deploys via Docker Compose with two services:

```
+-----------------------------------+     +------------------+
|  enso-atlas (Application)         |     |  atlas-db        |
|  - FastAPI Backend (port 8003)    |---->|  PostgreSQL 16   |
|  - Path Foundation (CPU)          |     |  (port 5433)     |
|  - MedGemma 4B (GPU, bfloat16)   |     +------------------+
|  - MedSigLIP (GPU, fp16)         |
|  - TransMIL (GPU)                |
|  - FAISS Index (CPU)             |
+-----------------------------------+
        |
        | Port 8003 (API)
        v
+-----------------------------------+
|  Frontend (Next.js)               |
|  - Next.js 14 Dev Server          |
|  - Port 3000                      |
|  - Proxy /api/* -> :8003          |
+-----------------------------------+
```

### Port Mappings

| Service | Internal Port | External Port | Purpose |
|---|---|---|---|
| enso-atlas | 8000 | 8003 | FastAPI REST API |
| enso-atlas | 7860 | 7862 | Gradio UI (legacy) |
| atlas-db | 5432 | 5433 | PostgreSQL |
| frontend | 3000 | 3000 | Next.js development server |

## 2.5 Network Architecture

The Next.js frontend proxies all API requests to the backend via a rewrite rule configured in `frontend/next.config.mjs`:

```javascript
// frontend/next.config.mjs
const nextConfig = {
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: 'http://127.0.0.1:8003/api/:path*',
      },
    ]
  },
};
```

This eliminates CORS issues in production and creates a unified origin for the browser. The backend also configures CORS middleware for direct API access during development:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*",
        "http://localhost:3000",
        "http://localhost:3001",
        "http://100.111.126.23:3000",
        "http://100.111.126.23:8003",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

# 3. Foundation Models

## 3.1 Path Foundation (Google)

### Overview

Path Foundation is Google's histopathology foundation model, based on a DINOv2-style self-supervised learning architecture trained on millions of pathology images. It produces dense 384-dimensional feature vectors from 224x224 pixel tissue patches, capturing morphological patterns that are informative for downstream classification tasks.

### Architecture Specifications

| Parameter | Value |
|---|---|
| Model ID | `google/path-foundation` |
| Architecture | Vision Transformer (DINOv2-based) |
| Input Size | 224 x 224 x 3 (RGB) |
| Output Dimension | 384 |
| Framework | TensorFlow SavedModel |
| Inference Function | `model.signatures["serving_default"]` |
| Preprocessing | Float32, normalize to [0, 1] |

### Implementation

The Path Foundation embedder is implemented in `src/enso_atlas/embedding/embedder_pathfound.py`:

```python
class PathFoundationEmbedder:
    """Feature extractor using Path Foundation model (384-dim embeddings)."""

    EMBEDDING_DIM = 384
    INPUT_SIZE = 224

    def __init__(self, cache_dir: str = "data/embeddings", batch_size: int = 64):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size
        self._model = None
        self._infer = None

    def _load_model(self) -> None:
        if self._model is not None:
            return
        import tensorflow as tf
        from huggingface_hub import from_pretrained_keras

        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

        self._model = from_pretrained_keras("google/path-foundation")
        self._infer = self._model.signatures["serving_default"]

    def _preprocess_batch(self, patches: List[np.ndarray]) -> "tf.Tensor":
        import tensorflow as tf
        batch = np.stack(patches, axis=0)
        batch = batch.astype(np.float32) / 255.0
        return tf.constant(batch)
```

### Patching Strategy

Whole-slide images are divided into non-overlapping 224x224 pixel patches at level 0 (full resolution, typically 40x magnification). The patching process includes tissue masking to exclude background regions:

1. **Grid Generation:** The slide is divided into a regular grid with step size equal to patch size (224 pixels)
2. **Tissue Detection:** Each patch is converted to grayscale; pixels with intensity < 200 are considered tissue. Patches with tissue coverage below 5% are discarded
3. **Batch Embedding:** Tissue patches are batched (default batch size 64) and processed through Path Foundation
4. **Storage:** Embeddings are saved as NumPy arrays (`{slide_id}.npy`) with coordinates saved separately (`{slide_id}_coords.npy`)

For a typical ovarian cancer slide (approximately 80,000 x 40,000 pixels), this produces between 3,000 and 8,000 tissue patches.

### TensorFlow on CPU

Path Foundation uses TensorFlow, which does not support NVIDIA Blackwell GPUs (compute capability sm_121). The system runs Path Foundation on CPU, which is acceptable because:

1. Embeddings are pre-computed offline during the data pipeline phase
2. On-demand embedding is rare and not latency-critical
3. CPU inference takes approximately 2-5 seconds per batch of 64 patches
4. The GPU is reserved for MedGemma and MedSigLIP which benefit more from acceleration

### Performance Benchmarks

| Metric | Value |
|---|---|
| Model Loading Time | ~15 seconds |
| Batch Inference (64 patches, CPU) | ~2.5 seconds |
| Per-Patch Latency (amortized) | ~39 ms |
| Memory Usage (CPU) | ~2 GB |
| Full Slide Embedding (6,000 patches) | ~4 minutes |
| Embedding Dimension | 384 floats = 1.5 KB per patch |

## 3.2 MedGemma 4B

### Overview

MedGemma is Google's medical language model, a 4-billion parameter instruction-tuned variant of Gemma optimized for medical text generation. In Enso Atlas, MedGemma generates structured clinical reports from model predictions and evidence patches, producing tumor-board-ready documentation.

### Architecture Specifications

| Parameter | Value |
|---|---|
| Model ID | `google/medgemma-4b-it` (or local `/app/models/medgemma-4b-it`) |
| Architecture | Gemma-2 Causal LM |
| Parameters | 4 billion |
| Precision | bfloat16 (GPU) or bfloat16 (CPU on modern ARM) |
| VRAM Usage | ~8 GB (bfloat16) |
| Max Input Tokens | 512 (configured) |
| Max Output Tokens | 384 (configured) |
| Temperature | 0.1 |
| Top-p | 0.9 |

### Implementation

The MedGemma reporter is implemented in `src/enso_atlas/reporting/medgemma.py` (908 lines):

```python
@dataclass
class ReportingConfig:
    model: str = "/app/models/medgemma-4b-it"
    max_evidence_patches: int = 4
    max_similar_cases: int = 0
    max_input_tokens: int = 512
    max_output_tokens: int = 384
    max_generation_time_s: float = 300.0
    temperature: float = 0.1
    top_p: float = 0.9
```

### Inference Pipeline

MedGemma inference follows a carefully orchestrated pipeline designed for reliability and performance:

1. **Model Loading:** Lazy-loaded with thread-safe locking (`_load_lock`). The model is loaded in bfloat16 precision on GPU when available, with fallback to CPU.

2. **Warmup:** A test inference is run during server startup to pre-compile CUDA kernels. This is critical because the first inference can take 60-120 seconds due to kernel compilation:

```python
@app.on_event("startup")
async def load_models():
    if reporter is not None:
        await asyncio.wait_for(
            asyncio.to_thread(reporter._warmup_inference),
            timeout=180.0,
        )
```

3. **Prompt Engineering:** The system uses a compact prompt format that forces JSON-first output:

```python
def _build_prompt(self, evidence_patches, score, label, similar_cases,
                  case_id, patient_context):
    categories = []
    for p in evidence_patches[:3]:
        cat = self._classify_patch(p)
        attn = p.get("attention_weight", 0.0)
        categories.append(f"{cat}({attn:.2f})")
    cat_str = ", ".join(categories) if categories else "mixed"

    line1 = f'Ovarian H&E. Prediction: {label} ({score:.2f}). Tissue: {cat_str}.'
    line2 = f'{{"prediction":"{label}","confidence":{score:.2f},"morphology":"'
    prompt = line1 + chr(10) + line2
    return prompt
```

4. **Generation:** Uses `torch.inference_mode()` with configurable max tokens, temperature, and time limits. A custom `StoppingCriteria` class enforces generation time limits:

```python
class _TimeLimitCriteria(StoppingCriteria):
    def __init__(self, start_time: float, max_time_s: float):
        self._start_time = start_time
        self._max_time_s = max_time_s

    def __call__(self, input_ids, scores, **kwargs):
        return (time.monotonic() - self._start_time) >= self._max_time_s
```

5. **Non-Blocking Inference:** All MedGemma inference is wrapped in `asyncio.to_thread()` to prevent blocking the FastAPI event loop:

```python
report = await asyncio.wait_for(
    asyncio.to_thread(
        reporter.generate_report,
        evidence_patches=evidence_patches,
        score=score,
        label=label,
        similar_cases=similar_cases,
        case_id=slide_id,
        patient_context=patient_ctx,
    ),
    timeout=timeout_s,
)
```

6. **JSON Parsing:** The response is parsed with progressive repair strategies for truncated JSON, including regex-based field extraction as a last resort.

### Report Schema

MedGemma reports conform to a structured schema with required safety fields:

```python
REPORT_SCHEMA = {
    "type": "object",
    "required": ["case_id", "task", "model_output", "evidence",
                  "limitations", "safety_statement"],
    "properties": {
        "case_id": {"type": "string"},
        "task": {"type": "string"},
        "model_output": {
            "type": "object",
            "properties": {
                "label": {"type": "string"},
                "probability": {"type": "number"},
                "calibration_note": {"type": "string"}
            }
        },
        "evidence": {"type": "array"},
        "limitations": {"type": "array"},
        "suggested_next_steps": {"type": "array"},
        "safety_statement": {"type": "string"}
    }
}
```

### Safety Validation

Every generated report is validated against prohibited phrases to prevent the system from making direct treatment recommendations:

```python
prohibited_phrases = [
    "start treatment", "stop treatment",
    "recommend starting", "recommend stopping",
    "prescribe", "administer",
]
```

## 3.3 MedSigLIP

### Overview

MedSigLIP is Google's medical vision-language model based on the SigLIP architecture, fine-tuned on medical images including pathology, radiology, dermatology, and ophthalmology. It enables semantic search over tissue patches using natural language queries, a capability unique to Enso Atlas among pathology AI platforms.

### Architecture Specifications

| Parameter | Value |
|---|---|
| Model ID | `google/medsiglip-448` (preferred) or `google/siglip-so400m-patch14-384` (fallback) |
| Architecture | SigLIP (dual encoder: vision + text) |
| Vision Input | 448 x 448 (MedSigLIP) or 384 x 384 (SigLIP) |
| Embedding Dimension | 1152 |
| Precision | fp16 (GPU) |
| VRAM Usage | ~800 MB (fp16) |

### Implementation

The MedSigLIP embedder is implemented in `src/enso_atlas/embedding/medsiglip.py` (426 lines):

```python
class MedSigLIPEmbedder:
    EMBEDDING_DIM = 1152
    INPUT_SIZE = 384

    def __init__(self, config: Optional[MedSigLIPConfig] = None):
        self.config = config or MedSigLIPConfig()
        self._model = None
        self._image_processor = None
        self._tokenizer = None
        self._device = None
```

### Dual Encoder Pipeline

MedSigLIP provides both image and text encoding capabilities:

**Text Embedding (`embed_text`):**
```python
def embed_text(self, text: Union[str, List[str]]) -> np.ndarray:
    inputs = self._tokenizer(text, padding=True, return_tensors="pt")
    inputs = {k: v.to(self._device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = self._model.get_text_features(**inputs)
        embeddings = outputs / outputs.norm(dim=-1, keepdim=True)
    return embeddings.detach().cpu().float().numpy()
```

**Image Embedding (`embed_patches`):**
Processes patches in batches with GPU acceleration. Embeddings are L2-normalized for cosine similarity search.

**Semantic Search (`search`):**
```python
def search(self, query, embeddings, metadata=None, top_k=10):
    query_embedding = self.embed_text(query)
    similarities = np.dot(embeddings, query_embedding.T).squeeze()
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    # Build results with patch_index, similarity_score, metadata
```

### Predefined Pathology Queries

The system includes curated query sets for common pathological features:

```python
PATHOLOGY_QUERIES = {
    "tumor": ["tumor cells", "malignant cells", "cancer cells",
              "neoplastic tissue", "carcinoma"],
    "inflammation": ["lymphocyte infiltration", "inflammatory infiltrate",
                     "tumor infiltrating lymphocytes", "immune cells"],
    "necrosis": ["necrotic tissue", "necrosis", "cell death",
                 "coagulative necrosis"],
    "stroma": ["stromal tissue", "fibrous stroma",
               "desmoplastic stroma", "connective tissue"],
    "mitosis": ["mitotic figures", "cell division", "mitosis"],
    "vessels": ["blood vessels", "vasculature", "angiogenesis"],
}
```

### VRAM Sharing

MedSigLIP shares GPU memory with MedGemma. At fp16, SigLIP requires approximately 800 MB alongside MedGemma's 8 GB, fitting comfortably within typical GPU memory (16-80 GB):

```python
siglip_config = MedSigLIPConfig(
    cache_dir=str(embeddings_dir / "medsiglip_cache"),
    device="auto",  # Shares GPU with MedGemma
)
```

---

# 4. TransMIL Architecture

## 4.1 Transformer-Based Multiple Instance Learning

Multiple Instance Learning (MIL) is the standard paradigm for whole-slide image classification, where a slide (the "bag") contains thousands of patches (the "instances") but only a single slide-level label is available for training. TransMIL extends this paradigm with Transformer self-attention, enabling the model to capture inter-patch relationships and spatial context.

### Why MIL for Pathology

Traditional supervised learning requires per-pixel or per-patch labels, which are prohibitively expensive to obtain in pathology. MIL allows training with only slide-level labels (e.g., "responder" or "non-responder") while still learning which patches are most informative through attention mechanisms. This attention signal provides the interpretability that is critical for clinical adoption.

## 4.2 Architecture Details

The TransMIL architecture is implemented in `models/transmil.py` (236 lines):

```mermaid
graph TD
    INPUT["Patch Embeddings<br/>(N x 384)"] --> FC_IN["Linear Projection<br/>(384 -> 512)"]
    FC_IN --> CLS["Prepend CLS Token<br/>(1 x 512)"]
    CLS --> PPEG["Pyramid Position<br/>Encoding (PPEG)"]
    PPEG --> TRANS1["Transformer Layer 1<br/>(8-head self-attention)"]
    TRANS1 --> TRANS2["Transformer Layer 2<br/>(8-head self-attention)"]
    TRANS2 --> NORM["Layer Norm"]
    NORM --> CLS_OUT["Extract CLS Token"]
    CLS_OUT --> CLASSIFY["Classifier Head<br/>(512 -> 256 -> 1)"]
    CLASSIFY --> SIGMOID["Sigmoid"]
    SIGMOID --> PRED["Prediction<br/>(0-1 probability)"]
    TRANS2 -.->|"Attention Weights"| ATTN["Attention Map<br/>(N weights)"]
```

### Component Breakdown

**Input Projection:**
```python
self.fc_in = nn.Sequential(
    nn.Linear(input_dim, hidden_dim),  # 384 -> 512
    nn.ReLU(),
    nn.Dropout(dropout)
)
```

**CLS Token:** A learnable class token is prepended to the sequence, similar to BERT/ViT:
```python
self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
```

**Pyramid Position Encoding Generator (PPEG):**
PPEG uses multi-scale depthwise convolutions to inject 2D spatial information:
```python
class PPEG(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7 // 2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5 // 2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3 // 2, groups=dim)

    def forward(self, x, H, W):
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat) + cnn_feat + self.proj1(cnn_feat) + self.proj2(cnn_feat)
```

**Transformer Layers:**
Two layers of multi-head self-attention with 8 heads, pre-norm architecture:
```python
class TransLayer(nn.Module):
    def __init__(self, dim=512, num_heads=8, dropout=0.1):
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout,
                                           batch_first=True)
```

**Classifier Head:**
Two-layer MLP with sigmoid output for binary classification:
```python
self.classifier = nn.Sequential(
    nn.Linear(hidden_dim, hidden_dim // 2),  # 512 -> 256
    nn.ReLU(),
    nn.Dropout(dropout),
    nn.Linear(hidden_dim // 2, num_classes),  # 256 -> 1
    nn.Sigmoid()
)
```

### Hyperparameters

| Parameter | Value |
|---|---|
| Input Dimension | 384 (Path Foundation) |
| Hidden Dimension | 512 |
| Number of Classes | 1 (binary with sigmoid) |
| Attention Heads | 8 |
| Transformer Layers | 2 |
| Dropout | 0.1 (inference) / 0.25 (training) |
| Total Parameters | ~3.2M |

## 4.3 Training Pipeline

Training is implemented in `scripts/train_transmil.py`:

### Mixed-Precision Training with Gradient Accumulation

```python
def train_epoch(model, train_data, labels, device, optimizer, scaler,
                accumulation_steps=4):
    model.train()
    for i, sid in enumerate(slide_ids):
        emb = torch.tensor(train_data[sid], dtype=torch.float32).to(device)
        label = torch.tensor([labels[sid]], dtype=torch.float32).to(device)

        with autocast():
            pred = model(emb)
        loss = F.binary_cross_entropy(pred.float().view(-1), label.float())
        loss = loss / accumulation_steps

        scaler.scale(loss).backward()

        if (i + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
```

### Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | AdamW (lr=2e-4, weight_decay=1e-5) |
| Scheduler | CosineAnnealingLR (T_max=50, eta_min=1e-6) |
| Gradient Accumulation | 4 steps |
| Mixed Precision | AMP (autocast + GradScaler) |
| Gradient Clipping | Max norm 1.0 |
| Early Stopping | Patience 15 epochs |
| Train/Val Split | 80/20, stratified |
| Epochs | 100 maximum |

### 5-Fold Cross-Validation

For robust model evaluation, the system supports 5-fold stratified cross-validation with weighted loss for class imbalance. Each fold produces a separate model checkpoint, and the multi-model inference system can ensemble predictions across folds.

## 4.4 Multi-Model Inference

The system supports five specialized TransMIL models, each trained for a different prediction target:

| Model ID | Prediction Target | Category |
|---|---|---|
| `platinum_sensitivity` | Platinum-based chemotherapy response | Ovarian Cancer |
| `chemosensitivity` | General chemosensitivity | Ovarian Cancer |
| `brca_status` | BRCA mutation status | Ovarian Cancer |
| `tumor_grade` | Histological tumor grade | General Pathology |
| `survival_5y` | 5-year overall survival | General Pathology |

Multi-model inference is implemented in `scripts/multi_model_inference.py` and exposed via the `POST /api/analyze-multi` endpoint.

## 4.5 Threshold Optimization

The classification threshold is optimized using the Youden J statistic (J = sensitivity + specificity - 1), which maximizes the sum of sensitivity and specificity on the validation set:

```python
# From threshold_config.json
{
    "method": "youden_j",
    "threshold": 0.9229,
    "sensitivity": 0.846,
    "specificity": 0.912,
    "auc": 0.879
}
```

This threshold (0.9229) is significantly higher than the default 0.5, reflecting the class imbalance in the training data (139 non-responders vs 13 responders) and optimizing for balanced sensitivity/specificity.

## 4.6 Current Performance Metrics

| Metric | Value |
|---|---|
| AUC-ROC | 0.879 |
| Optimized Threshold | 0.9229 |
| Sensitivity | 84.6% |
| Specificity | 91.2% |
| Training Slides | 152 (139:13 class ratio) |
| Embedding Dimension | 384 |
| Patches per Slide | 3,000 - 8,000 |

---

# 5. Backend API Specification

## 5.1 Startup and Initialization

The FastAPI application is created via the `create_app()` factory function in `src/enso_atlas/api/main.py`. The `@app.on_event("startup")` handler orchestrates model loading in a specific sequence:

```
1. Path Foundation Embedder (lazy, CPU)           ~0s (loaded on first use)
2. MIL Classifier (TransMIL checkpoint)           ~2s
3. Multi-Model Inference (5 models)               ~5s
4. Evidence Generator (FAISS index)               ~3s
5. MedGemma Reporter (GPU, bfloat16)              ~30s
6. MedGemma Warmup (CUDA kernel compilation)      ~60-120s
7. MedSigLIP Embedder (GPU, fp16)                 ~10s
8. FAISS Index (slide embeddings)                 ~2s
9. Slide-Mean FAISS Index (cosine similarity)     ~1s
10. Label Loading (CSV parsing)                   ~0.5s
11. PostgreSQL Schema Init + Population           ~5-60s (first run)
12. Project Registry (YAML loading)               ~0.1s
13. Agent Workflow Initialization                  ~0.1s
```

### Timing Benchmarks (DGX Spark, Blackwell GPU)

| Step | Duration | Notes |
|---|---|---|
| TransMIL Loading | 2.1s | 5 model checkpoints |
| MedGemma Loading | 28.4s | 4B params in bfloat16 |
| MedGemma Warmup | 62.3s | First CUDA kernel compilation |
| MedSigLIP Loading | 9.7s | SigLIP-so400m in fp16 |
| FAISS Index Build | 3.2s | 152 slides, ~900K patches |
| PostgreSQL Init | 4.8s | Schema + first-run population |
| **Total Startup** | **~120s** | Dominated by MedGemma warmup |

## 5.2 Core Endpoints

### GET /api/health

Health check endpoint returning server status and capabilities.

| Field | Type | Description |
|---|---|---|
| `status` | string | Always "healthy" if server is running |
| `version` | string | API version ("0.1.0") |
| `model_loaded` | boolean | Whether MIL classifier is loaded |
| `cuda_available` | boolean | Whether CUDA GPU is available |
| `slides_available` | integer | Number of slides with embeddings |
| `db_available` | boolean | Whether PostgreSQL is connected |
| `uptime` | float | Server uptime in seconds |

**Response Example:**
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "model_loaded": true,
  "cuda_available": true,
  "slides_available": 152,
  "db_available": true,
  "uptime": 3842.7
}
```

### GET /api/slides

List all available slides with patient context. Uses PostgreSQL for sub-100ms response times, with flat-file fallback.

**Response Model:** `List[SlideInfo]`

| Field | Type | Description |
|---|---|---|
| `slide_id` | string | Unique slide identifier |
| `patient_id` | string | Associated patient ID |
| `has_embeddings` | boolean | Whether embeddings exist |
| `has_level0_embeddings` | boolean | Full-resolution embeddings available |
| `label` | string | Ground truth label (if known) |
| `num_patches` | integer | Number of tissue patches |
| `patient` | PatientContext | Demographics (age, sex, stage, etc.) |
| `dimensions` | SlideDimensions | Width and height in pixels |
| `mpp` | float | Microns per pixel |
| `magnification` | string | Scan magnification ("40x") |

**Performance:**
- PostgreSQL path: 2-5ms for 152 slides
- Flat-file fallback: 30-60s (reads SVS dimensions via OpenSlide)

### GET /api/slides/{slide_id}/dzi

Returns Deep Zoom Image (DZI) XML descriptor for OpenSeadragon tile-based rendering.

**Response:** XML document
```xml
<?xml version="1.0" encoding="UTF-8"?>
<Image xmlns="http://schemas.microsoft.com/deepzoom/2008"
       Format="jpeg" Overlap="0" TileSize="254">
  <Size Height="40000" Width="80000"/>
</Image>
```

### GET /api/slides/{slide_id}/tiles/{level}/{col}\_{row}.jpeg

Serves individual tiles for the OpenSeadragon viewer. Tiles are extracted from WSI files using OpenSlide at the requested pyramid level.

| Parameter | Type | Description |
|---|---|---|
| `slide_id` | path | Slide identifier |
| `level` | path | Pyramid level (0 = full resolution) |
| `col` | path | Column index |
| `row` | path | Row index |

**Response:** JPEG image bytes

### GET /api/slides/{slide_id}/thumbnail

Returns a thumbnail image of the slide.

**Response:** JPEG image bytes (typically 1024x1024 or smaller)

### POST /api/analyze

Single-slide prediction with evidence generation.

**Request Body:**
```json
{
  "slide_id": "TCGA-04-1331-01A-01-BS1.abc123",
  "generate_report": false
}
```

**Response Model:** `AnalyzeResponse`

| Field | Type | Description |
|---|---|---|
| `slide_id` | string | Slide identifier |
| `prediction` | string | "RESPONDER" or "NON-RESPONDER" |
| `score` | float | Model probability (0-1) |
| `confidence` | float | Confidence score based on threshold distance |
| `patches_analyzed` | integer | Number of patches processed |
| `top_evidence` | array | Top-K evidence patches with coordinates |
| `similar_cases` | array | Similar slides from FAISS retrieval |

**Evidence Patch Structure:**
```json
{
  "rank": 1,
  "patch_index": 4231,
  "attention_weight": 0.0847,
  "coordinates": [52416, 18144],
  "tissue_type": "tumor",
  "tissue_confidence": 0.89
}
```

### POST /api/analyze-multi

Multi-model ensemble prediction running all five specialized models.

**Request Body:**
```json
{
  "slide_id": "TCGA-04-1331-01A-01-BS1.abc123",
  "models": null,
  "return_attention": false,
  "level": 1
}
```

**Response Model:** `MultiModelResponse`

| Field | Type | Description |
|---|---|---|
| `slide_id` | string | Slide identifier |
| `predictions` | object | Map of model_id to ModelPrediction |
| `by_category` | object | Predictions grouped by cancer type |
| `n_patches` | integer | Patches analyzed |
| `processing_time_ms` | float | Total processing time |

### POST /api/embed-slide

On-demand embedding of a slide using Path Foundation. This endpoint triggers background patch extraction and embedding for slides that have not been pre-processed.

### GET /api/similar/{slide\_id}

Find similar slides from the reference cohort using FAISS over L2-normalized mean slide embeddings with cosine similarity (implemented as inner product).

**Query Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `k` | integer | 5 | Number of similar slides |
| `top_patches` | integer | 3 | Legacy parameter (unused) |

**Response:**
```json
{
  "slide_id": "TCGA-04-1331-01A-01-BS1.abc123",
  "similar_cases": [
    {
      "slide_id": "TCGA-13-0883-01A-01-BS1.def456",
      "similarity_score": 0.847,
      "distance": 0.153,
      "label": "responder",
      "n_patches": 5432
    }
  ],
  "num_queries": 1
}
```

### POST /api/semantic-search

Text-to-patch semantic search using MedSigLIP.

**Request Body:**
```json
{
  "slide_id": "TCGA-04-1331-01A-01-BS1.abc123",
  "query": "tumor infiltrating lymphocytes",
  "top_k": 10
}
```

**Response Model:** `SemanticSearchResponse`

### GET /api/heatmap/{slide\_id}

Generate attention heatmap as PNG image.

**Query Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `level` | integer | 2 | Resolution: 0=2048px, 2=512px, 4=128px |
| `smooth` | boolean | true | Apply Gaussian blur |
| `blur` | integer | 31 | Blur kernel size (odd) |

**Response:** PNG image with `X-Slide-Width` and `X-Slide-Height` headers for overlay positioning.

### GET /api/heatmap/{slide\_id}/{model\_id}

Generate attention heatmap for a specific model (e.g., `platinum_sensitivity`, `tumor_grade`).

### POST /api/report

Generate a structured MedGemma report for a slide.

**Request Body:**
```json
{
  "slide_id": "TCGA-04-1331-01A-01-BS1.abc123",
  "include_evidence": true,
  "include_similar": true
}
```

**Response Model:** `ReportResponse`

| Field | Type | Description |
|---|---|---|
| `slide_id` | string | Slide identifier |
| `report_json` | object | Structured report (see Report Schema) |
| `summary_text` | string | Human-readable tumor board summary |

### POST /api/export/pdf

Generate a professional PDF report with attention heatmap and evidence patches.

**Request Body:** `PdfExportRequest` with `slide_id`, `report_data`, `prediction_data`, optional `heatmap_image` and `patient_context`.

**Response:** PDF file as downloadable attachment.

## 5.3 Batch Analysis Endpoints

### POST /api/analyze-batch

Synchronous batch analysis of multiple slides.

**Request:** `BatchAnalyzeRequest` with `slide_ids` (1-100 slides)

**Response:** `BatchAnalyzeResponse` with per-slide results and summary statistics (total, completed, failed, responders, non-responders, uncertain, avg_confidence).

### POST /api/analyze-batch/async

Asynchronous batch analysis with progress tracking.

**Response:** Returns `task_id` immediately. Poll status endpoint for progress.

### GET /api/analyze-batch/status/{task\_id}

Poll batch analysis progress.

| Field | Type | Description |
|---|---|---|
| `status` | string | pending, running, completed, cancelled, failed |
| `progress` | float | 0-100 percentage |
| `current_slide_index` | integer | Current slide being processed |
| `message` | string | Human-readable status |

### POST /api/analyze-batch/cancel/{task\_id}

Cancel a running batch analysis. Already-completed slides are retained.

## 5.4 Project System Endpoints

### GET /api/projects

List all configured projects.

```json
{
  "projects": [
    {
      "id": "ovarian-platinum",
      "name": "Ovarian Cancer - Platinum Sensitivity",
      "cancer_type": "ovarian",
      "prediction_target": "platinum_sensitivity",
      "classes": ["resistant", "sensitive"],
      "positive_class": "sensitive",
      "is_default": true
    }
  ],
  "default_project": "ovarian-platinum",
  "count": 1
}
```

### GET /api/projects/{project\_id}

Get full project configuration including dataset paths, model config, and threshold.

### GET /api/projects/{project\_id}/slides

List slides associated with a specific project.

### GET /api/projects/{project\_id}/status

Check project readiness (slides directory, embeddings, model checkpoint, labels file).

## 5.5 Agent Workflow Endpoints

### POST /api/agent/analyze

Start multi-step agent analysis with SSE streaming.

**Request:**
```json
{
  "slide_id": "TCGA-04-1331-01A-01-BS1.abc123",
  "clinical_context": "58yo female, FIGO stage IIIC, post-debulking",
  "questions": ["What is the predicted treatment response?"]
}
```

**Response:** Server-Sent Events stream (see Section 6 for details).

### POST /api/agent/followup

Ask follow-up questions about an existing analysis.

### GET /api/agent/session/{session\_id}

Get detailed session information including predictions, evidence, report, and reasoning chain.

### GET /api/agent/sessions

List active agent sessions.

### DELETE /api/agent/session/{session\_id}

Delete a session and free memory.

## 5.6 Database Endpoints

### POST /api/db/repopulate

Force re-population of PostgreSQL from flat files.

### GET /api/db/status

Check database connection and population status.

```json
{
  "status": "connected",
  "slides": 152,
  "patients": 98,
  "metadata_entries": 456,
  "slides_with_dimensions": 152
}
```

### Additional Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/history` | Analysis history with filtering and pagination |
| `GET` | `/api/slides/{id}/history` | Per-slide analysis history |
| `GET` | `/api/audit-log` | Audit trail for compliance |
| `POST` | `/api/classify-region` | Tissue type classification at coordinates |
| `POST` | `/api/analyze-uncertainty` | MC Dropout uncertainty quantification |
| `GET` | `/api/embed/status` | Path Foundation embedder status |
| `POST` | `/api/embed` | Embed base64-encoded patch images |
| `GET` | `/api/tags` | List tags (slide organization) |
| `POST` | `/api/tags` | Create tag |
| `GET` | `/api/groups` | List slide groups |
| `POST` | `/api/groups` | Create slide group |
| `POST` | `/api/report/async` | Async report generation |
| `GET` | `/api/report/status/{task_id}` | Report generation status |
| `POST` | `/api/report/pdf` | Generate PDF from report JSON |
| `POST` | `/api/visual-search` | Image-to-image FAISS search |

---

# 6. Agentic Workflow

## 6.1 Overview

The agentic workflow implements a seven-step AI pipeline that performs comprehensive slide analysis with visible reasoning, retrieval-augmented generation, and structured report production. It targets the "Agentic Workflows" special prize track for the MedGemma hackathon.

### Design Principles

1. **Transparency:** Every step produces visible reasoning that the clinician can inspect
2. **Streaming:** Results are streamed via Server-Sent Events (SSE) for real-time progress
3. **Session Memory:** Sessions persist analysis state for follow-up questions
4. **Graceful Degradation:** Each step can be skipped if its dependencies are unavailable

## 6.2 Seven-Step Pipeline

```mermaid
sequenceDiagram
    participant Client
    participant API
    participant Agent
    participant TransMIL
    participant FAISS
    participant MedSigLIP
    participant MedGemma

    Client->>API: POST /api/agent/analyze
    API->>Agent: run_workflow()

    Note over Agent: Step 1: INITIALIZE
    Agent->>Agent: Load embeddings + coordinates
    Agent-->>Client: SSE: initialize complete

    Note over Agent: Step 2: ANALYZE
    Agent->>TransMIL: predict_all(embeddings)
    TransMIL-->>Agent: predictions + attention
    Agent-->>Client: SSE: analyze complete

    Note over Agent: Step 3: RETRIEVE
    Agent->>FAISS: search(mean_embedding, k=10)
    FAISS-->>Agent: similar cases
    Agent-->>Client: SSE: retrieve complete

    Note over Agent: Step 4: SEMANTIC SEARCH
    Agent->>MedSigLIP: search(queries, embeddings)
    MedSigLIP-->>Agent: tissue pattern matches
    Agent-->>Client: SSE: semantic_search complete

    Note over Agent: Step 5: COMPARE
    Agent->>Agent: Analyze similar case patterns
    Agent-->>Client: SSE: compare complete

    Note over Agent: Step 6: REASON
    Agent->>Agent: Synthesize evidence
    Agent-->>Client: SSE: reason complete

    Note over Agent: Step 7: REPORT
    Agent->>MedGemma: generate_report()
    MedGemma-->>Agent: structured report
    Agent-->>Client: SSE: report complete

    Agent-->>Client: SSE: complete
```

## 6.3 Step Details

### Step 1: Initialize
- Load pre-computed Path Foundation embeddings (`{slide_id}.npy`)
- Load patch coordinates (`{slide_id}_coords.npy`)
- Validate slide exists and has embeddings
- Duration: ~50ms

### Step 2: Analyze
- Run multi-model TransMIL inference (5 models)
- Extract attention weights for evidence patch selection
- Select top-8 high-attention patches with coordinates
- Build reasoning chain from predictions
- Duration: ~200ms (GPU)

### Step 3: Retrieve
- Compute L2-normalized mean embedding for the slide
- Search slide-mean FAISS index (IndexFlatIP, cosine similarity)
- Return top-10 similar cases with labels and similarity scores
- Compute responder/non-responder distribution
- Duration: ~10ms

### Step 4: Semantic Search
- Load cached MedSigLIP embeddings for the slide
- Run five key pathology queries: tumor cells, necrosis, lymphocyte infiltration, stroma, mitotic figures
- Return top-5 matching patches per query with similarity scores
- Duration: ~100ms

### Step 5: Compare
- Analyze patterns in similar cases
- Compute similarity-weighted voting (responder vs non-responder)
- Identify most similar responding and non-responding cases
- Duration: ~5ms

### Step 6: Reason
- Synthesize all collected evidence into reasoning chain
- Incorporate model predictions, evidence patches, similar cases, and semantic search results
- Answer user-provided questions using collected context
- Duration: ~10ms

### Step 7: Report
- Generate structured report via MedGemma (when available)
- Fall back to template report if MedGemma is unavailable or times out
- Include evidence, limitations, safety statement, and decision support
- Duration: 10-120s (depending on MedGemma availability)

## 6.4 SSE Streaming Format

Each SSE event contains a JSON payload:

```
data: {
  "step": "analyze",
  "status": "complete",
  "message": "Ran 5 models on 6,234 patches",
  "reasoning": "Model predictions:\n- Platinum Sensitivity: sensitive (score: 0.94, confidence: 89.2%)\n...",
  "data": {
    "predictions": {...},
    "top_evidence": [...]
  },
  "timestamp": "2025-02-06T23:57:42.123Z",
  "duration_ms": 187.3
}
```

## 6.5 Session Management

Sessions are stored in-memory with the `AgentState` dataclass:

```python
@dataclass
class AgentState:
    session_id: str
    created_at: str
    current_step: AgentStep
    context: Optional[AnalysisContext]
    step_results: Dict[AgentStep, StepResult]
    predictions: Dict[str, Dict[str, Any]]
    similar_cases: List[Dict[str, Any]]
    top_evidence: List[Dict[str, Any]]
    semantic_search_results: Dict[str, List[Dict[str, Any]]]
    report: Optional[Dict[str, Any]]
    conversation_history: List[Dict[str, str]]
    reasoning_chain: List[str]
    error: Optional[str]
```

Sessions enable follow-up questions where the agent can reference previous analysis results without re-running the full workflow.

## 6.6 Error Handling

Each step is wrapped in try/except with graceful degradation:

- If multi-model inference is unavailable, Step 2 returns `SKIPPED` status
- If FAISS index is not built, Step 3 returns `SKIPPED` status
- If MedSigLIP embeddings are not cached, Step 4 returns `SKIPPED` status
- If MedGemma fails or times out, Step 7 falls back to template report
- Fatal errors yield an `ERROR` step result and terminate the workflow

---

# 7. Database Schema

## 7.1 PostgreSQL Tables

The database schema is defined in `src/enso_atlas/api/database.py` and consists of seven tables:

### ER Diagram

```mermaid
erDiagram
    patients ||--o{ slides : "has"
    slides ||--o{ slide_metadata : "has"
    slides ||--o{ analysis_results : "has"
    slides ||--o{ embedding_tasks : "has"

    patients {
        text patient_id PK
        integer age
        text sex
        text stage
        text grade
        integer prior_lines
        text histology
        text treatment_response
        text diagnosis
        text vital_status
        timestamptz created_at
        timestamptz updated_at
    }

    slides {
        text slide_id PK
        text patient_id FK
        text filename
        integer width
        integer height
        float mpp
        text magnification
        integer num_patches
        boolean has_embeddings
        boolean has_level0_embeddings
        text label
        text project_id
        timestamptz embedding_date
        text file_path
        bigint file_size_bytes
        timestamptz created_at
        timestamptz updated_at
    }

    slide_metadata {
        bigserial id PK
        text slide_id FK
        text key
        text value
        timestamptz created_at
        timestamptz updated_at
    }

    analysis_results {
        bigserial id PK
        text slide_id FK
        text model_id
        float score
        text label
        float confidence
        float threshold
        text attention_hash
        timestamptz created_at
    }

    embedding_tasks {
        text task_id PK
        text slide_id FK
        integer level
        text status
        float progress
        timestamptz created_at
        timestamptz completed_at
        text error
    }

    projects {
        text id PK
        text name
        text cancer_type
        text prediction_target
        jsonb config_json
        timestamptz created_at
        timestamptz updated_at
    }

    schema_version {
        integer version PK
        timestamptz applied_at
    }
```

## 7.2 Schema DDL

```sql
CREATE TABLE IF NOT EXISTS patients (
    patient_id   TEXT PRIMARY KEY,
    age          INTEGER,
    sex          TEXT,
    stage        TEXT,
    grade        TEXT,
    prior_lines  INTEGER,
    histology    TEXT,
    treatment_response TEXT,
    diagnosis    TEXT,
    vital_status TEXT,
    created_at   TIMESTAMPTZ DEFAULT now(),
    updated_at   TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS slides (
    slide_id              TEXT PRIMARY KEY,
    patient_id            TEXT REFERENCES patients(patient_id) ON DELETE SET NULL,
    filename              TEXT,
    width                 INTEGER DEFAULT 0,
    height                INTEGER DEFAULT 0,
    mpp                   DOUBLE PRECISION,
    magnification         TEXT DEFAULT '40x',
    num_patches           INTEGER,
    has_embeddings        BOOLEAN DEFAULT FALSE,
    has_level0_embeddings BOOLEAN DEFAULT FALSE,
    label                 TEXT,
    embedding_date        TIMESTAMPTZ,
    file_path             TEXT,
    file_size_bytes       BIGINT,
    created_at            TIMESTAMPTZ DEFAULT now(),
    updated_at            TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS slide_metadata (
    id        BIGSERIAL PRIMARY KEY,
    slide_id  TEXT NOT NULL REFERENCES slides(slide_id) ON DELETE CASCADE,
    key       TEXT NOT NULL,
    value     TEXT,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now(),
    UNIQUE(slide_id, key, value)
);

CREATE TABLE IF NOT EXISTS analysis_results (
    id             BIGSERIAL PRIMARY KEY,
    slide_id       TEXT NOT NULL REFERENCES slides(slide_id) ON DELETE CASCADE,
    model_id       TEXT,
    score          DOUBLE PRECISION,
    label          TEXT,
    confidence     DOUBLE PRECISION,
    threshold      DOUBLE PRECISION,
    attention_hash TEXT,
    created_at     TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS embedding_tasks (
    task_id       TEXT PRIMARY KEY,
    slide_id      TEXT REFERENCES slides(slide_id) ON DELETE CASCADE,
    level         INTEGER DEFAULT 0,
    status        TEXT DEFAULT 'pending',
    progress      DOUBLE PRECISION DEFAULT 0,
    created_at    TIMESTAMPTZ DEFAULT now(),
    completed_at  TIMESTAMPTZ,
    error         TEXT
);
```

### Indexes

```sql
CREATE INDEX IF NOT EXISTS idx_slides_patient ON slides(patient_id);
CREATE INDEX IF NOT EXISTS idx_slides_label   ON slides(label);
CREATE INDEX IF NOT EXISTS idx_slides_project ON slides(project_id);
CREATE INDEX IF NOT EXISTS idx_slide_metadata_slide ON slide_metadata(slide_id);
CREATE INDEX IF NOT EXISTS idx_slide_metadata_key   ON slide_metadata(key);
CREATE INDEX IF NOT EXISTS idx_analysis_results_slide ON analysis_results(slide_id);
CREATE INDEX IF NOT EXISTS idx_embedding_tasks_slide ON embedding_tasks(slide_id);
CREATE INDEX IF NOT EXISTS idx_embedding_tasks_status ON embedding_tasks(status);
```

## 7.3 Auto-Population from Flat Files

On first startup, the database is automatically populated from existing flat files:

1. **Parse `labels.csv`:** Extract patient demographics, slide IDs, treatment response labels
2. **Scan `.npy` files:** Count patches, check for level 0 embeddings
3. **Read SVS files via OpenSlide:** Extract dimensions (width, height), microns per pixel (mpp), file size
4. **Import `slide_metadata.json`:** Tags, groups, notes, stars

All inserts use `ON CONFLICT DO UPDATE` for idempotency. The population process runs only on first startup (checked via `is_populated()` which counts existing rows).

## 7.4 Connection Pool

```python
async def get_pool():
    global _pool
    if _pool is None:
        import asyncpg
        for attempt in range(10):
            try:
                _pool = await asyncpg.create_pool(
                    DATABASE_URL,
                    min_size=2,
                    max_size=10,
                    command_timeout=60,
                )
                break
            except Exception as e:
                if attempt < 9:
                    await asyncio.sleep(2)
                else:
                    raise
    return _pool
```

## 7.5 Query Performance

| Query | PostgreSQL | Flat-File Fallback |
|---|---|---|
| List all slides | 2-5ms | 30-60s |
| Get slide by ID | <1ms | N/A |
| Search slides (filtered) | 5-10ms | 100-500ms |
| Count slides | <1ms | N/A |
| Patient lookup | <1ms | Linear scan |

---

# 8. Frontend Architecture

## 8.1 Tech Stack

| Technology | Version | Purpose |
|---|---|---|
| Next.js | 14 | React framework with SSR, routing, API proxying |
| React | 18 | UI component library |
| TypeScript | 5.x | Type-safe frontend development |
| Tailwind CSS | 3.x | Utility-first CSS framework |
| OpenSeadragon | 4.1 | Tile-based whole-slide image viewer |
| Lucide React | - | Icon library |

## 8.2 Component Hierarchy

```
App (layout.tsx)
  +-- ProjectProvider (ProjectContext.tsx)
      +-- HomePage (page.tsx)
          +-- Header
          |   +-- ProjectSwitcher
          |   +-- ViewModeToggle (oncologist/pathologist)
          |   +-- DemoModeToggle
          +-- LeftSidebar
          |   +-- SlideSelector (SlideSelector.tsx)
          |   |   +-- SlideCard
          |   |   +-- SearchInput
          |   |   +-- Filters
          |   +-- QuickStatsPanel
          +-- CenterPanel
          |   +-- WSIViewer (WSIViewer.tsx)
          |   |   +-- OpenSeadragon canvas
          |   |   +-- HeatmapOverlay
          |   |   +-- ScaleBar
          |   |   +-- NavigatorMinimap
          |   +-- MobilePanelTabs
          +-- RightSidebar
          |   +-- PredictionPanel (PredictionPanel.tsx)
          |   +-- MultiModelPredictionPanel
          |   +-- EvidencePanel
          |   +-- SimilarCasesPanel (SimilarCasesPanel.tsx)
          |   +-- SemanticSearchPanel
          |   +-- ReportPanel (ReportPanel.tsx)
          |   +-- BatchAnalysisPanel (BatchAnalysisPanel.tsx)
          |   +-- AIAssistantPanel (AIAssistantPanel.tsx)
          |   +-- CaseNotesPanel
          +-- Footer
          +-- PatchZoomModal
          +-- KeyboardShortcutsModal
          +-- WelcomeModal

Slides Page (/slides/page.tsx)
  +-- SlideManager
      +-- GridView / TableView
      +-- FilterSidebar
      +-- BulkActions
      +-- Pagination
```

## 8.3 State Management

### ProjectContext

The `ProjectContext` provides global project configuration to all components:

```typescript
interface ProjectContextValue {
  projects: Project[];
  currentProject: Project;
  switchProject: (id: string) => void;
  isLoading: boolean;
  error: string | null;
}
```

Projects are fetched from `/api/projects` on mount, with the selected project persisted to `localStorage` under the key `enso-atlas-selected-project`.

### useAnalysis Hook

The `useAnalysis` hook manages analysis state for the currently selected slide:

```typescript
const {
  isAnalyzing,
  analysisResult,
  multiModelResult,
  error,
  analyzeSlide,
  analyzeMultiModel,
} = useAnalysis();
```

### localStorage Caching

Several pieces of state are persisted in `localStorage`:
- Selected project ID (`enso-atlas-selected-project`)
- Welcome modal seen flag (`medgemma-welcome-seen`)
- Case notes per slide (`enso-atlas-case-notes-{slideId}`)
- Analysis history (`enso-atlas-analysis-history`)

## 8.4 API Client

The API client (`frontend/src/lib/api.ts`, 2,625 lines) implements:

### Request Infrastructure

```typescript
const RETRY_CONFIG = {
  maxRetries: 3,
  baseDelayMs: 1000,
  maxDelayMs: 10000,
  retryableStatusCodes: [408, 429, 500, 502, 503, 504],
};

const DEFAULT_TIMEOUT_MS = 30000;
```

### Custom Error Class

```typescript
export class AtlasApiError extends Error {
  code: string;
  details?: Record<string, unknown>;
  statusCode?: number;
  isRetryable: boolean;
  isTimeout: boolean;
  isNetworkError: boolean;

  getUserMessage(): string {
    if (this.isTimeout) return "The request timed out. Please try again.";
    if (this.isNetworkError) return "Unable to connect to the server.";
    if (this.statusCode === 429) return "Too many requests.";
    // ...
  }
}
```

### SSE Streaming Support

For the agentic workflow, the API client implements SSE event parsing:

```typescript
export async function* streamAgentAnalysis(
  slideId: string,
  clinicalContext: string,
  questions: string[]
): AsyncGenerator<AgentStepEvent> {
  const response = await fetch(`${API_BASE_URL}/api/agent/analyze`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ slide_id: slideId, clinical_context: clinicalContext, questions }),
  });

  const reader = response.body!.getReader();
  const decoder = new TextDecoder();

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    const text = decoder.decode(value);
    for (const line of text.split("\n")) {
      if (line.startsWith("data: ")) {
        yield JSON.parse(line.slice(6));
      }
    }
  }
}
```

## 8.5 WSI Viewer

The WSI viewer (`frontend/src/components/viewer/WSIViewer.tsx`, 749 lines) wraps OpenSeadragon with React:

### OpenSeadragon Configuration

```typescript
const viewer = OpenSeadragon({
  id: containerId,
  tileSources: dziUrl,
  showNavigator: true,
  navigatorPosition: "BOTTOM_RIGHT",
  navigatorHeight: "120px",
  navigatorWidth: "160px",
  showZoomControl: false,
  gestureSettingsMouse: {
    clickToZoom: false,
    dblClickToZoom: true,
    scrollToZoom: true,
  },
  minZoomLevel: 0.1,
  maxZoomLevel: 40,
  crossOriginPolicy: "Anonymous",
});
```

### Heatmap Overlay

Attention heatmaps are rendered as a semi-transparent overlay using `addSimpleImage`:

```typescript
viewer.addSimpleImage({
  url: heatmapImageUrl,
  x: 0,
  y: 0,
  width: 1,
  opacity: heatmapOpacity,
  preload: true,
});
```

The heatmap supports multiple models (platinum sensitivity, tumor grade, survival prediction) with a dropdown selector:

```typescript
const HEATMAP_MODELS = [
  { id: "platinum_sensitivity", name: "Platinum Sensitivity" },
  { id: "tumor_grade", name: "Tumor Grade" },
  { id: "survival_5y", name: "5-Year Survival" },
  { id: "survival_3y", name: "3-Year Survival" },
  { id: "survival_1y", name: "1-Year Survival" },
];
```

### Canvas2D Optimization

A patch is applied to suppress Chrome warnings about `willReadFrequently` for OpenSeadragon's pixel data access:

```typescript
if (typeof window !== "undefined") {
  const origGetContext = HTMLCanvasElement.prototype.getContext;
  HTMLCanvasElement.prototype.getContext = function (type, attrs) {
    if (type === "2d") {
      attrs = { ...attrs, willReadFrequently: true };
    }
    return origGetContext.call(this, type, attrs);
  };
}
```

### Scale Bar

A dynamic scale bar displays current magnification and real-world distance based on the slide's microns-per-pixel value.

## 8.6 Slide Manager

The `/slides` page (`frontend/src/app/slides/page.tsx`, 684 lines) provides a comprehensive slide management interface:

### Views
- **Grid View:** Thumbnail cards with quick-action overlays
- **Table View:** Dense data table with sortable columns

### Filtering
- Text search by slide ID
- Label filter (responder/non-responder)
- Embedding status filter
- Patch count range filter
- Tag and group filters
- Date range filter

### Pagination
- Configurable page size (20/50/100)
- Server-side pagination via `/api/slides/search`

### Bulk Actions
- Select multiple slides
- Batch analyze selected slides
- Export slide list as CSV

---

# 9. Config-Driven Project System

## 9.1 YAML Schema

Projects are defined in `config/projects.yaml`:

```yaml
projects:
  ovarian-platinum:
    name: "Ovarian Cancer - Platinum Sensitivity"
    cancer_type: ovarian
    prediction_target: platinum_sensitivity
    classes: ["resistant", "sensitive"]
    positive_class: sensitive
    description: "Predicts platinum-based chemotherapy sensitivity
                  in high-grade serous ovarian carcinoma"
    dataset:
      slides_dir: data/tcga_full/slides
      embeddings_dir: data/embeddings/level0
      labels_file: data/labels.csv
      label_column: platinum_sensitivity
    models:
      embedder: path-foundation
      mil_architecture: transmil
      mil_checkpoint: models/transmil_best.pt
      report_generator: medgemma-4b
      semantic_search: medsiglip
    threshold: 0.9229
    threshold_config: models/threshold_config.json
```

## 9.2 ProjectRegistry Backend

The `ProjectRegistry` class (`src/enso_atlas/api/projects.py`, 207 lines) loads and manages project configurations:

```python
class ProjectRegistry:
    def __init__(self, config_path: str | Path = "config/projects.yaml"):
        self._config_path = Path(config_path)
        self._projects: Dict[str, ProjectConfig] = {}
        self._default_project_id: Optional[str] = None
        self._load()

    def get_project(self, project_id: str) -> Optional[ProjectConfig]:
        return self._projects.get(project_id)

    def list_projects(self) -> Dict[str, ProjectConfig]:
        return dict(self._projects)

    def get_default_project(self) -> Optional[ProjectConfig]:
        if self._default_project_id:
            return self._projects.get(self._default_project_id)
        return None
```

### Data Classes

```python
@dataclass
class DatasetConfig:
    slides_dir: str = "data/slides"
    embeddings_dir: str = "data/embeddings/level0"
    labels_file: str = "data/labels.csv"
    label_column: str = "platinum_sensitivity"

@dataclass
class ModelsConfig:
    embedder: str = "path-foundation"
    mil_architecture: str = "transmil"
    mil_checkpoint: str = "models/transmil_best.pt"
    report_generator: str = "medgemma-4b"
    semantic_search: str = "medsiglip"

@dataclass
class ProjectConfig:
    id: str
    name: str
    cancer_type: str
    prediction_target: str
    classes: List[str]
    positive_class: str
    description: str
    dataset: DatasetConfig
    models: ModelsConfig
    threshold: float
    threshold_config: Optional[str]
```

## 9.3 Frontend ProjectContext

The `ProjectContext` (`frontend/src/contexts/ProjectContext.tsx`, 106 lines) fetches projects from the backend and provides a `switchProject` function. The selected project is persisted to `localStorage`:

```typescript
const DEFAULT_PROJECT: Project = {
  id: "default",
  name: "Ovarian Cancer - Platinum Sensitivity",
  cancer_type: "Ovarian Cancer",
  prediction_target: "platinum_sensitivity",
  classes: ["resistant", "sensitive"],
  positive_class: "sensitive",
  description: "Predicts platinum-based chemotherapy sensitivity...",
  models: {
    embedder: "PathFoundation (DINOv2-L)",
    mil_architecture: "TransMIL",
    report_generator: "MedGemma",
    semantic_search: "MedSigLIP",
  },
};
```

## 9.4 Adding a New Cancer Type

To add support for a new cancer type (e.g., lung adenocarcinoma EGFR mutation):

**Step 1:** Add project to `config/projects.yaml`:
```yaml
projects:
  lung-egfr:
    name: "Lung Adenocarcinoma - EGFR Mutation"
    cancer_type: lung
    prediction_target: egfr_mutation
    classes: ["wildtype", "mutant"]
    positive_class: mutant
    dataset:
      slides_dir: data/lung/slides
      embeddings_dir: data/lung/embeddings/level0
      labels_file: data/lung/labels.csv
      label_column: egfr_status
    models:
      embedder: path-foundation
      mil_architecture: transmil
      mil_checkpoint: models/lung_transmil_best.pt
    threshold: 0.5
```

**Step 2:** Prepare training data in the dataset directories

**Step 3:** Run embedding pipeline:
```bash
python scripts/embed_level0_pipelined.py \
  --slides_dir data/lung/slides \
  --output_dir data/lung/embeddings/level0
```

**Step 4:** Train TransMIL model:
```bash
python scripts/train_transmil.py \
  --embeddings_dir data/lung/embeddings/level0 \
  --labels_file data/lung/labels.json \
  --output_dir outputs/lung_transmil
```

**Step 5:** Restart the server -- the project will be automatically loaded and available in the frontend project switcher.

---

# 10. Deployment Guide

## 10.1 Docker Compose Configuration

The system is deployed via `docker/docker-compose.yaml`:

```yaml
services:
  enso-atlas:
    build:
      context: ..
      dockerfile: docker/Dockerfile
    container_name: enso-atlas
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    environment:
      NVIDIA_VISIBLE_DEVICES: all
      MIL_ARCHITECTURE: transmil
      MIL_THRESHOLD_CONFIG: /app/models/threshold_config.json
      DATABASE_URL: "postgresql://enso:enso_atlas_2024@atlas-db:5432/enso_atlas"
    volumes:
      - /home/hansonwen/med-gemma-hackathon/data:/app/data
      - /home/hansonwen/med-gemma-hackathon/outputs:/app/outputs
      - /home/hansonwen/.cache/huggingface:/root/.cache/huggingface
      - /home/hansonwen/med-gemma-hackathon/models:/app/models
      - /home/hansonwen/med-gemma-hackathon/config:/app/config:ro
      - /home/hansonwen/med-gemma-hackathon/src:/app/src:ro
    ports:
      - "7862:7860"
      - "8003:8000"
    depends_on:
      atlas-db:
        condition: service_healthy

  atlas-db:
    image: postgres:16
    container_name: atlas-db
    environment:
      POSTGRES_DB: enso_atlas
      POSTGRES_USER: enso
      POSTGRES_PASSWORD: enso_atlas_2024
    volumes:
      - atlas-pgdata:/var/lib/postgresql/data
    ports:
      - "5433:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U enso -d enso_atlas"]
      interval: 5s
      timeout: 3s
      retries: 10
      start_period: 10s
```

## 10.2 DGX Spark Specifics

The target deployment platform is the NVIDIA DGX Spark with Blackwell GPU:

| Specification | Value |
|---|---|
| GPU | NVIDIA Blackwell (compute capability sm_121) |
| Architecture | ARM64 (aarch64) |
| VRAM | 128 GB |
| RAM | 128 GB |
| Base Image | `nvcr.io/nvidia/pytorch:24.04-py3` |
| Known Issue | TensorFlow does not support sm_121 |

### ARM64 Considerations

- The Dockerfile uses the NVIDIA PyTorch container as the base image for proper ARM64 + CUDA support
- OpenCV headless version is pinned to 4.8.1.78 to avoid cv2.typing import crashes on aarch64
- NumPy is pinned to 1.26.4 for compatibility with scipy and other packages
- Heatmap generation uses scipy.ndimage instead of OpenCV to avoid cv2 import issues

## 10.3 Volume Mounts

| Container Path | Host Path | Mode | Purpose |
|---|---|---|---|
| `/app/data` | `~/med-gemma-hackathon/data` | rw | Slides, embeddings, labels |
| `/app/outputs` | `~/med-gemma-hackathon/outputs` | rw | Training outputs, model checkpoints |
| `/app/cache` | `atlas-cache` volume | rw | HuggingFace model cache |
| `/root/.cache/huggingface` | `~/.cache/huggingface` | rw | Pre-downloaded models |
| `/app/models` | `~/med-gemma-hackathon/models` | rw | TransMIL checkpoints |
| `/app/config` | `~/med-gemma-hackathon/config` | ro | YAML configuration |
| `/app/src` | `~/med-gemma-hackathon/src` | ro | Source code (dev mode) |

## 10.4 Environment Variables

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | `postgresql://enso:enso_atlas_2024@atlas-db:5432/enso_atlas` | PostgreSQL connection string |
| `MIL_ARCHITECTURE` | `clam` | MIL model type (transmil, clam, abmil) |
| `MIL_THRESHOLD` | (from config) | Override classification threshold |
| `MIL_THRESHOLD_CONFIG` | `models/threshold_config.json` | Threshold configuration file |
| `MIL_MODEL_PATH` | (auto-detected) | Override model checkpoint path |
| `DEFAULT_PROJECT` | (first in YAML) | Default project ID |
| `MEDSIGLIP_MODEL_PATH` | (auto-detected) | MedSigLIP model path |
| `TRANSFORMERS_OFFLINE` | `0` | Set to `1` for air-gapped deployment |
| `HF_HUB_OFFLINE` | `0` | Set to `1` for air-gapped deployment |
| `ENSO_CONFIG` | `/app/config/default.yaml` | Main configuration file |
| `NEXT_PUBLIC_API_URL` | `http://100.111.126.23:8003` | Backend API URL for frontend |

## 10.5 Frontend Build and Deployment

```bash
cd frontend
npm install
npm run build
npm start -- -p 3000
```

The Next.js development server proxies `/api/*` requests to the backend via the rewrite rule in `next.config.mjs`.

## 10.6 Health Check Verification

```bash
# Backend health check
curl http://localhost:8003/api/health

# Expected response:
{
  "status": "healthy",
  "version": "0.1.0",
  "model_loaded": true,
  "cuda_available": true,
  "slides_available": 152,
  "db_available": true,
  "uptime": 3842.7
}

# Database status
curl http://localhost:8003/api/db/status

# Embedder status
curl http://localhost:8003/api/embed/status
```

---

# 11. Data Pipeline

## 11.1 TCGA Data Download

The training data comes from The Cancer Genome Atlas (TCGA) ovarian cancer cohort. The download script (`scripts/download_ovarian_fixed.py`) retrieves whole-slide images and associated clinical metadata:

1. Query TCGA GDC API for ovarian cancer (OV) diagnostic slides
2. Download SVS files (typically 1-3 GB each)
3. Extract clinical metadata (treatment response, survival, demographics)
4. Generate labels.csv with slide-level labels

## 11.2 Whole-Slide Image Processing

Slides are processed using OpenSlide, which provides multi-resolution access to whole-slide images:

```python
import openslide

with openslide.OpenSlide(slide_path) as slide:
    dimensions = slide.dimensions  # (width, height) at level 0
    mpp = float(slide.properties.get(openslide.PROPERTY_NAME_MPP_X))
    # Read a 224x224 patch at position (x, y) from level 0
    patch = slide.read_region((x, y), 0, (224, 224))
```

### Supported Formats

| Format | Extension | Scanner |
|---|---|---|
| Aperio | .svs | Leica/Aperio |
| TIFF | .tiff, .tif | Various |
| Hamamatsu | .ndpi | Hamamatsu |
| MIRAX | .mrxs | 3DHISTECH |
| Ventana | .vms | Ventana |
| Leica | .scn | Leica |

## 11.3 Patch Extraction

The pipelined extraction script (`scripts/embed_level0_pipelined.py`) uses a producer/consumer pattern:

```
+------------------+    Queue    +------------------+
|  CPU Thread      |----------->|  GPU Thread      |
|  - Read patches  |  Batches   |  - Embed patches |
|  - Tissue mask   |  of 512    |  - Save .npy     |
|  - Queue batches |            |                  |
+------------------+            +------------------+
```

**Producer (CPU thread):** Extracts patches from the WSI, applies tissue masking, and enqueues batches of 512 patches.

**Consumer (GPU thread):** Embeds batches using Path Foundation and concatenates results.

### Tissue Masking

```python
def is_tissue(self, patch_array):
    gray = np.mean(patch_array[:, :, :3], axis=2)
    tissue_pixels = np.sum(gray < 200)
    tissue_ratio = tissue_pixels / (self.patch_size * self.patch_size)
    return tissue_ratio > self.tissue_threshold  # 0.05
```

### Output Files

For each slide, two files are produced:
- `{slide_id}.npy` -- Embedding array of shape (N, 384) where N is the number of tissue patches
- `{slide_id}_coords.npy` -- Coordinate array of shape (N, 2) with (x, y) positions at level 0

## 11.4 Label Matching

Labels are stored in CSV format with two supported schemas:

**Format 1 (data/labels.csv):**
```csv
patient_id,slide_file,treatment_response,age,sex,stage,grade,histology
TCGA-04-1331,TCGA-04-1331-01A-01-BS1.svs,non-responder,62,F,IIIC,G3,serous
```

**Format 2 (data/tcga_full/labels.csv):**
```csv
slide_id,patient_id,label,platinum_status
TCGA-04-1331-01A-01-BS1.abc123,TCGA-04-1331,0,resistant
```

The database population code handles both formats and normalizes labels to binary (0/1 or resistant/sensitive).

---

# 12. Security and Compliance

## 12.1 On-Premise Design

Enso Atlas is architecturally designed for on-premise deployment. No data leaves the hospital network:

- **No cloud dependencies at runtime:** All models are downloaded and cached locally before deployment. The `TRANSFORMERS_OFFLINE=1` and `HF_HUB_OFFLINE=1` environment variables enforce air-gapped operation.
- **No telemetry:** `enable_telemetry: false` in the configuration
- **Local database:** PostgreSQL runs as a co-located Docker container
- **Local storage:** All embeddings, slides, and model weights are stored on local volumes

### Data Flow Boundaries

```
+-------------------------------------------------------+
|  Hospital Network (Air-Gapped)                        |
|                                                       |
|  +-------------------+    +----------------------+    |
|  |  PACS / LIS       |--->|  Enso Atlas Server  |    |
|  |  (Slide Export)    |    |  - API (8003)       |    |
|  +-------------------+    |  - DB (5433)         |    |
|                           |  - Frontend (3000)   |    |
|                           +----------------------+    |
|                                    |                  |
|                           +--------v--------+         |
|                           |  Clinician      |         |
|                           |  Workstation    |         |
|                           +-----------------+         |
+-------------------------------------------------------+
           |
           X  No outbound data transfer
           |
```

## 12.2 HIPAA Considerations

While Enso Atlas is a research tool and not a certified medical device, the following HIPAA-aligned controls are implemented:

- **Audit Trail:** All analyses, report generations, and PDF exports are logged with timestamps, user IDs, and slide IDs in an in-memory audit deque (`audit_log`, max 500 entries)
- **No PHI in Logs:** The logging configuration avoids capturing patient identifiable information
- **Access Control:** The system runs within the hospital network; authentication is delegated to the network perimeter
- **Data Encryption:** PostgreSQL connections can be configured with SSL; disk encryption is recommended at the OS level

### Audit Event Types

| Event Type | Trigger |
|---|---|
| `analysis_completed` | Slide analysis completed |
| `batch_analysis_slide` | Individual slide in batch analysis |
| `batch_analysis_completed` | Batch analysis finished |
| `batch_analysis_async_completed` | Async batch analysis finished |
| `batch_analysis_cancelled` | Batch analysis cancelled |
| `uncertainty_analysis_completed` | MC Dropout analysis completed |
| `report_generated` | MedGemma report generated |
| `pdf_exported` | PDF report exported |
| `json_exported` | JSON report exported |

## 12.3 Research-Only Disclaimers

Every report includes mandatory safety statements:

```
"safety_statement": "This is a research decision-support tool, not a diagnostic
device. All findings must be validated by qualified pathologists and clinicians.
Do not use for standalone clinical decision-making."
```

The PDF export includes a prominent header warning:

```
UNCALIBRATED MODEL - RESEARCH USE ONLY
Probabilities are raw model outputs. Not validated for clinical decision-making.
All findings must be verified by qualified pathologists before clinical use.
```

## 12.4 Calibration Notes

The system explicitly marks all predictions as uncalibrated:

```python
"calibration_note": "Model probability, not clinical certainty.
Requires external validation."
```

The threshold (0.9229) was optimized on a specific dataset and may not generalize. Users are warned that:
- The model has not undergone prospective clinical validation
- Predictions are based on morphological patterns only (no molecular data)
- The training cohort (152 slides, 139:13 class ratio) is small and imbalanced
- Performance may vary across institutions and patient populations

---

# 13. Performance Benchmarks

## 13.1 Startup Time Breakdown

| Component | Duration | Notes |
|---|---|---|
| FastAPI App Creation | 0.1s | |
| TransMIL Model Loading | 2.1s | 5 model checkpoints from disk |
| Evidence Generator Setup | 0.1s | |
| Path Foundation Embedder Init | 0.1s | Lazy loaded |
| MedGemma Model Loading | 28.4s | 4B params, bfloat16 |
| MedGemma Warmup | 62.3s | CUDA kernel compilation |
| MedSigLIP Loading | 9.7s | SigLIP-so400m |
| FAISS Index Construction | 3.2s | 152 slides, ~900K patches |
| Slide-Mean FAISS Index | 0.8s | 152 slide means |
| Label Loading | 0.3s | CSV parsing |
| PostgreSQL Init | 4.8s | Schema + population (first run) |
| Project Registry | 0.1s | YAML parsing |
| Agent Workflow Init | 0.1s | |
| **Total** | **~112s** | |

## 13.2 Per-Slide Analysis Latency

| Operation | Latency | Notes |
|---|---|---|
| Load embeddings (.npy) | 5-15ms | Depends on patch count |
| TransMIL inference (single model) | 20-50ms | GPU, 6K patches |
| Multi-model inference (5 models) | 100-250ms | GPU, sequential |
| Attention extraction | Included above | |
| Evidence patch selection (top-8) | <1ms | numpy argsort |
| FAISS similar case search | 2-5ms | Slide-mean index |
| Heatmap generation (512px) | 50-100ms | CPU, scipy blur |
| Semantic search (5 queries) | 100-300ms | MedSigLIP GPU |
| MedGemma report generation | 10-120s | GPU inference |
| **Full analysis (no report)** | **~200ms** | |
| **Full analysis (with report)** | **10-120s** | Dominated by MedGemma |

## 13.3 GPU Memory Usage

| Model | VRAM (approx.) | Precision |
|---|---|---|
| MedGemma 4B | 8 GB | bfloat16 |
| MedSigLIP | 800 MB | fp16 |
| TransMIL (5 models) | 200 MB | float32 |
| FAISS Index | CPU only | float32 |
| **Total GPU** | **~9 GB** | |

## 13.4 Database Query Performance

| Query | Response Time | Method |
|---|---|---|
| List all slides (152) | 2-5ms | PostgreSQL |
| List all slides (152) | 30-60s | Flat-file fallback |
| Get single slide | <1ms | PostgreSQL |
| Search with filters | 5-10ms | PostgreSQL |
| Count slides | <1ms | PostgreSQL |
| Repopulate database | 5-60s | Depends on SVS count |

## 13.5 Concurrent Request Handling

FastAPI runs with uvicorn workers and async request handling:

- **Async endpoints:** Non-blocking I/O for database queries and file operations
- **Thread pool:** MedGemma inference runs in `asyncio.to_thread()` to avoid blocking
- **Background tasks:** Batch analysis and report generation use `BackgroundTasks`
- **Connection pool:** asyncpg pool with 2-10 connections
- **Estimated throughput:** ~50 concurrent analysis requests (without report generation)

---

# 14. Known Limitations and Future Work

## 14.1 Current Limitations

### Class Imbalance

The primary training dataset contains 139 non-responders and 13 responders (10.7:1 ratio). While the Youden J threshold optimization partially addresses this, the model's ability to identify responders is limited by the small positive class.

**Mitigation strategies implemented:**
- Weighted binary cross-entropy loss during training
- Threshold optimization (0.9229 vs default 0.5)
- Confidence scoring based on threshold distance

**Future mitigation:**
- Oversampling of responder embeddings
- Focal loss for hard example mining
- Multi-institutional data collection

### TensorFlow/Blackwell Incompatibility

TensorFlow does not support NVIDIA Blackwell GPUs (compute capability sm_121). Path Foundation runs on CPU, which is acceptable for pre-computed embeddings but limits on-demand embedding performance.

**Workaround:** Embeddings are pre-computed offline. On-demand embedding falls back to CPU inference.

**Future resolution:** Wait for TensorFlow sm_121 support or port Path Foundation to PyTorch.

### Single Cancer Type

Currently, only high-grade serous ovarian carcinoma (HGSOC) platinum sensitivity is trained. The config-driven project system is designed for multi-cancer support but requires training data and model development for each new cancer type.

### No FDA Clearance

Enso Atlas is a research tool and has not undergone regulatory review. It cannot be used for clinical diagnosis or treatment decisions without proper validation and clearance.

### Uncalibrated Predictions

Model probabilities are raw sigmoid outputs and do not represent true clinical probabilities. Platt scaling or isotonic regression could be applied with a dedicated calibration dataset.

### Limited Molecular Integration

The system relies solely on H&E morphology. Integration with IHC, genomic, and molecular data would improve prediction accuracy.

## 14.2 Planned Improvements

### Short-Term (3-6 months)
- Multi-fold ensemble averaging for more robust predictions
- Platt scaling for probability calibration
- Real-time patch embedding pipeline (GPU-accelerated)
- User authentication and role-based access control
- DICOM integration for direct PACS slide import

### Medium-Term (6-12 months)
- Additional cancer types (lung, breast, colorectal)
- Multimodal integration (H&E + IHC)
- Federated learning for multi-institutional training
- Active learning for efficient annotation
- Real-time collaborative annotation tools

### Long-Term (12+ months)
- FDA 510(k) regulatory pathway
- Integration with electronic health records (EHR)
- Prospective clinical validation studies
- Cloud-hybrid deployment option (with encryption)
- Foundation model fine-tuning for institutional adaptation

---

# 15. Appendices

## Appendix A: Full API Endpoint Reference

| Method | Path | Description | Auth |
|---|---|---|---|
| `GET` | `/` | API root | None |
| `GET` | `/health` | Health check | None |
| `GET` | `/api/health` | Health check (aliased) | None |
| `GET` | `/api/slides` | List all slides | None |
| `GET` | `/api/slides/search` | Search slides with filters | None |
| `GET` | `/api/slides/{slide_id}` | Get slide info | None |
| `GET` | `/api/slides/{slide_id}/dzi` | DZI XML for OpenSeadragon | None |
| `GET` | `/api/slides/{slide_id}/tiles/{level}/{col}_{row}.jpeg` | Tile image | None |
| `GET` | `/api/slides/{slide_id}/thumbnail` | Slide thumbnail | None |
| `GET` | `/api/slides/{slide_id}/history` | Slide analysis history | None |
| `POST` | `/api/analyze` | Single-slide analysis | None |
| `POST` | `/api/analyze-multi` | Multi-model analysis | None |
| `POST` | `/api/analyze-uncertainty` | MC Dropout analysis | None |
| `POST` | `/api/analyze-batch` | Synchronous batch analysis | None |
| `POST` | `/api/analyze-batch/async` | Async batch analysis | None |
| `GET` | `/api/analyze-batch/status/{task_id}` | Batch status | None |
| `POST` | `/api/analyze-batch/cancel/{task_id}` | Cancel batch | None |
| `GET` | `/api/analyze-batch/tasks` | List batch tasks | None |
| `POST` | `/api/embed` | Embed base64 patches | None |
| `GET` | `/api/embed/status` | Embedder status | None |
| `POST` | `/api/embed-slide` | On-demand slide embedding | None |
| `GET` | `/api/similar` | Similar case search | None |
| `POST` | `/api/semantic-search` | MedSigLIP text search | None |
| `POST` | `/api/visual-search` | Image-to-image FAISS search | None |
| `GET` | `/api/heatmap/{slide_id}` | Attention heatmap PNG | None |
| `GET` | `/api/heatmap/{slide_id}/{model_id}` | Model-specific heatmap | None |
| `POST` | `/api/classify-region` | Tissue type classification | None |
| `POST` | `/api/report` | Generate MedGemma report | None |
| `POST` | `/api/report/async` | Async report generation | None |
| `GET` | `/api/report/status/{task_id}` | Report generation status | None |
| `GET` | `/api/report/tasks` | List report tasks | None |
| `POST` | `/api/report/pdf` | Generate PDF from report | None |
| `POST` | `/api/export/pdf` | Full PDF export with heatmap | None |
| `GET` | `/api/history` | Analysis history | None |
| `GET` | `/api/audit-log` | Audit trail | None |
| `GET` | `/api/tags` | List tags | None |
| `POST` | `/api/tags` | Create tag | None |
| `GET` | `/api/groups` | List groups | None |
| `POST` | `/api/groups` | Create group | None |
| `GET` | `/api/projects` | List projects | None |
| `GET` | `/api/projects/{id}` | Get project details | None |
| `GET` | `/api/projects/{id}/slides` | Project slides | None |
| `GET` | `/api/projects/{id}/status` | Project readiness | None |
| `POST` | `/api/agent/analyze` | Agent analysis (SSE) | None |
| `POST` | `/api/agent/followup` | Agent follow-up (SSE) | None |
| `GET` | `/api/agent/session/{id}` | Get session | None |
| `GET` | `/api/agent/sessions` | List sessions | None |
| `DELETE` | `/api/agent/session/{id}` | Delete session | None |
| `POST` | `/api/db/repopulate` | Force DB repopulation | None |
| `GET` | `/api/db/status` | Database status | None |

## Appendix B: Environment Variable Reference

| Variable | Required | Default | Description |
|---|---|---|---|
| `DATABASE_URL` | No | `postgresql://enso:enso_atlas_2024@atlas-db:5432/enso_atlas` | PostgreSQL connection |
| `MIL_ARCHITECTURE` | No | `clam` | MIL model type |
| `MIL_THRESHOLD` | No | (from config) | Classification threshold |
| `MIL_THRESHOLD_CONFIG` | No | `models/threshold_config.json` | Threshold config path |
| `MIL_MODEL_PATH` | No | (auto) | Model checkpoint path |
| `DEFAULT_PROJECT` | No | (first YAML) | Default project ID |
| `MEDSIGLIP_MODEL_PATH` | No | (auto) | MedSigLIP model path |
| `TRANSFORMERS_OFFLINE` | No | `0` | Air-gapped mode |
| `HF_HUB_OFFLINE` | No | `0` | Air-gapped mode |
| `HF_HOME` | No | `/app/cache/huggingface` | HuggingFace cache |
| `TRANSFORMERS_CACHE` | No | `/app/cache/huggingface` | Transformers cache |
| `ENSO_CONFIG` | No | `/app/config/default.yaml` | Config file path |
| `NVIDIA_VISIBLE_DEVICES` | No | `all` | GPU selection |
| `NEXT_PUBLIC_API_URL` | No | `http://localhost:8003` | Frontend API URL |
| `EMBED_LOG` | No | `/tmp/embed_pipelined.log` | Embedding log path |

## Appendix C: File Tree

```
med-gemma-hackathon/
+-- config/
|   +-- default.yaml                    # Main configuration
|   +-- projects.yaml                   # Project definitions
+-- docker/
|   +-- Dockerfile                      # Production container
|   +-- docker-compose.yaml             # Service orchestration
+-- frontend/
|   +-- next.config.mjs                 # Next.js config with API proxy
|   +-- tailwind.config.ts              # Tailwind CSS config
|   +-- src/
|       +-- app/
|       |   +-- layout.tsx              # Root layout
|       |   +-- page.tsx                # Main analysis page (1796 lines)
|       |   +-- slides/
|       |       +-- page.tsx            # Slide manager page (684 lines)
|       +-- components/
|       |   +-- layout/
|       |   |   +-- Header.tsx
|       |   |   +-- Footer.tsx
|       |   +-- panels/
|       |   |   +-- PredictionPanel.tsx       (433 lines)
|       |   |   +-- AIAssistantPanel.tsx      (1073 lines)
|       |   |   +-- BatchAnalysisPanel.tsx    (803 lines)
|       |   |   +-- SlideSelector.tsx         (763 lines)
|       |   |   +-- SimilarCasesPanel.tsx     (620 lines)
|       |   |   +-- ReportPanel.tsx           (883 lines)
|       |   +-- viewer/
|       |       +-- WSIViewer.tsx              (749 lines)
|       +-- contexts/
|       |   +-- ProjectContext.tsx              (106 lines)
|       +-- hooks/
|       |   +-- useAnalysis.ts
|       |   +-- useKeyboardShortcuts.ts
|       +-- lib/
|       |   +-- api.ts                         (2625 lines)
|       |   +-- pdfExport.ts
|       |   +-- utils.ts
|       +-- types/
|           +-- index.ts                       (type definitions)
+-- models/
|   +-- transmil.py                            (236 lines)
|   +-- transmil_best.pt                       (TransMIL checkpoint)
|   +-- threshold_config.json                  (Youden J threshold)
|   +-- clam_attention.pt                      (CLAM checkpoint)
+-- scripts/
|   +-- train_transmil.py                      (training)
|   +-- train_transmil_finetune.py             (fine-tuning)
|   +-- evaluate_transmil.py                   (evaluation)
|   +-- embed_level0_pipelined.py              (pipelined embedding)
|   +-- multi_model_inference.py               (multi-model)
|   +-- optimize_threshold.py                  (threshold tuning)
|   +-- download_ovarian_fixed.py              (TCGA download)
+-- src/
|   +-- enso_atlas/
|       +-- api/
|       |   +-- main.py                        (5560 lines)
|       |   +-- database.py                    (778 lines)
|       |   +-- projects.py                    (207 lines)
|       |   +-- project_routes.py              (201 lines)
|       |   +-- batch_tasks.py                 (206 lines)
|       |   +-- report_tasks.py                (164 lines)
|       |   +-- pdf_export.py                  (858 lines)
|       |   +-- slide_metadata.py
|       |   +-- embedding_tasks.py
|       +-- agent/
|       |   +-- workflow.py                    (1227 lines)
|       |   +-- routes.py                      (324 lines)
|       +-- embedding/
|       |   +-- embedder_pathfound.py          (99 lines)
|       |   +-- medsiglip.py                   (426 lines)
|       |   +-- embedder.py
|       +-- evidence/
|       |   +-- generator.py                   (353 lines)
|       +-- mil/
|       |   +-- clam.py
|       +-- reporting/
|       |   +-- medgemma.py                    (908 lines)
|       |   +-- decision_support.py
|       +-- config.py
|       +-- wsi/
|           +-- processor.py
+-- data/
|   +-- labels.csv
|   +-- embeddings/
|   |   +-- level0/                            (N x 384 .npy files)
|   +-- tcga_full/
|   |   +-- slides/                            (.svs files)
|   |   +-- labels.csv
+-- tests/
    +-- test_api_comprehensive.py
```

## Appendix D: Key Design Decisions

### 1. Raw SQL over ORM

The database module uses raw SQL with asyncpg instead of an ORM like SQLAlchemy. This decision was made for:
- **Performance:** asyncpg is the fastest PostgreSQL driver for Python
- **Transparency:** SQL queries are explicit and auditable
- **Simplicity:** No migration framework complexity for a small schema

### 2. In-Memory Audit Trail

The audit trail uses a bounded in-memory deque rather than a database table:
- **Performance:** Zero latency for logging
- **Simplicity:** No additional database writes on every request
- **Trade-off:** Logs are lost on restart (acceptable for research use)

### 3. FAISS IndexFlatIP over IndexIVF

The slide-mean similarity index uses `IndexFlatIP` (exact inner product search) instead of `IndexIVF` (approximate nearest neighbor):
- **Accuracy:** Exact search is preferred for the small dataset (152 slides)
- **Performance:** Search is still <5ms for 152 vectors
- **Scale threshold:** Switch to IVF when dataset exceeds ~10,000 slides

### 4. Template Report Fallback

When MedGemma fails or times out, the system generates a template-based report:
- **Reliability:** Users always receive a report, even without MedGemma
- **Transparency:** Template reports are clearly marked as fallbacks
- **Evidence preservation:** Real attention weights and coordinates are still included

### 5. Heatmap via SciPy instead of OpenCV

Heatmap generation uses scipy.ndimage.gaussian_filter instead of OpenCV:
- **Compatibility:** Avoids OpenCV import crashes on aarch64
- **Reliability:** SciPy is a more stable dependency across platforms
- **Performance:** Comparable speed for the resolution sizes used (128-2048 pixels)

## Appendix E: Error Codes and Troubleshooting

| HTTP Code | Error | Cause | Resolution |
|---|---|---|---|
| 400 | Bad Request | Invalid request body | Check request format against Pydantic model |
| 404 | Slide not found | Slide ID has no embeddings | Run embedding pipeline for the slide |
| 404 | Session not found | Invalid agent session ID | Start a new agent analysis |
| 404 | Task not found | Invalid batch/report task ID | Check task ID spelling |
| 500 | Prediction failed | Model inference error | Check GPU memory, restart server |
| 500 | PDF generation failed | Missing fpdf2/reportlab | Install: `pip install fpdf2` |
| 503 | Model not loaded | Server still starting up | Wait for startup to complete (~120s) |
| 503 | Database not available | PostgreSQL connection failed | Check database container status |
| 503 | Agent workflow not initialized | Agent dependencies unavailable | Check model loading in startup logs |
| 503 | PDF export not available | Missing PDF library | Install reportlab or fpdf2 |

### Common Issues

**Issue:** Server takes 2+ minutes to start
**Cause:** MedGemma warmup compiles CUDA kernels on first run
**Resolution:** This is expected. Subsequent restarts may be faster if CUDA cache is preserved.

**Issue:** Heatmap shows all blue or all red
**Cause:** Attention weights have very low variance
**Resolution:** Check that the TransMIL checkpoint is correctly trained. Try a different model.

**Issue:** MedGemma report times out
**Cause:** CPU inference on Blackwell (TF incompatibility fallback)
**Resolution:** Ensure MedGemma is running on GPU (check startup logs for "Loading MedGemma model on cuda").

**Issue:** Similar cases all show "unknown" label
**Cause:** labels.csv not found or slide IDs don't match
**Resolution:** Verify labels.csv path in config/projects.yaml. Check slide ID format matches between labels CSV and embedding filenames.

**Issue:** Frontend shows "Unable to connect to server"
**Cause:** Backend not running or port mismatch
**Resolution:** Verify backend is running on port 8003. Check NEXT_PUBLIC_API_URL matches.

---

# Document Revision History

| Version | Date | Author | Changes |
|---|---|---|---|
| 1.0.0 | 2025-02-06 | Enso Labs | Initial comprehensive specification |

---

*This document was generated from analysis of the Enso Atlas codebase comprising approximately 25,000 lines of Python backend code and 12,000 lines of TypeScript frontend code.*
