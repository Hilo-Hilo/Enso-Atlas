# Enso Atlas: Technical Specification

## A Modular, Model-Agnostic Pathology Evidence Platform

**Version:** 2.1.0
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
10. [Outlier Detection and Few-Shot Classification](#10-outlier-detection-and-few-shot-classification)
11. [Annotation System](#11-annotation-system)
12. [Heatmap and Overlay System](#12-heatmap-and-overlay-system)
13. [Hospital Deployment Guide](#13-hospital-deployment-guide)
14. [Data Pipeline](#14-data-pipeline)
15. [Security and Compliance](#15-security-and-compliance)
16. [Performance Benchmarks](#16-performance-benchmarks)
17. [Mac Mini Deployment Feasibility](#17-mac-mini-deployment-feasibility)
18. [Known Limitations and Future Work](#18-known-limitations-and-future-work)
19. [Appendices](#19-appendices)

---

# 1. Executive Summary

## 1.1 Project Overview

Enso Atlas is a modular, model-agnostic pathology evidence platform for on-premise deployment in hospital networks. It provides the infrastructure for deploying pathology AI, from slide ingestion and embedding generation through classification, evidence visualization, and structured reporting. Protected Health Information (PHI) does not leave the premises.

The platform is built around four independently swappable layers: a **foundation model layer** (any patch-level embedding model), a **classification layer** (any attention-based MIL architecture), an **evidence layer** (heatmaps, retrieval, semantic search, outlier detection), and a **UI layer** (WSI viewer, annotations, reports). Hospitals bring their own foundation models, train task-specific classification heads, and deploy within existing infrastructure. No code changes are required to add new cancer types, swap models, or adjust feature sets. Everything is driven by a single YAML configuration file.

The current deployment uses Google's HAI-DEF models (Path Foundation, MedGemma 4B, MedSigLIP) with five TransMIL classifiers for ovarian cancer treatment-response prediction. The HAI-DEF models are interchangeable examples. The platform itself is the primary contribution: an on-premise system that makes any pathology AI model deployable, interpretable, and usable in clinical workflows.

The platform addresses a gap in precision oncology: determining which patients will respond to specific chemotherapy regimens before treatment begins. Treatment response can currently only be assessed after multiple cycles of therapy, exposing non-responding patients to unnecessary toxicity and delays in receiving effective alternatives. Enso Atlas provides morphology-based predictions from standard H&E-stained tissue slides already collected during routine clinical care.

### Mission Statement

To build the platform that makes pathology AI deployable: the infrastructure that turns any foundation model and any classification head into an evidence-based decision support tool, running entirely on-premise.

### Design Philosophy: Platform, Not a Demo

Enso Atlas is a modular platform, not a single-purpose tool. Every component can be swapped without code changes:

- **Foundation models are config-driven.** Path Foundation can be replaced with UNI, CONCH, Virchow, DINOv2, or any custom embedding model by editing a YAML file. The system requires only a model name and embedding dimension. All downstream components (classification, retrieval, outlier detection, few-shot learning) adapt automatically to the new embedding space.
- **Classification models are pluggable.** Any attention-based MIL variant (TransMIL, CLAM, ABMIL) can be registered by dropping in a weights file and adding a config entry. After training on local data, the checkpoint appears in the UI with heatmap and evidence support.
- **Cancer types are project-defined.** Ovarian, lung, breast, prostate, colorectal, or any other cancer type can be added by creating a new project block in the configuration. Labels, thresholds, model assignments, class names, and dataset paths are all per-project.
- **Feature toggles are per-project.** Report generation, semantic search, or similar case retrieval can be enabled or disabled independently for each project. Resource-constrained deployments can disable GPU-heavy features while retaining core classification.
- **The UI adapts automatically.** All labels, categories, model lists, and workflow steps derive from the project configuration. No hardcoded cancer types or model IDs exist in the frontend. Switching projects reconfigures the entire interface.
- **Report generators are swappable.** The structured report system works with MedGemma, but any LLM that accepts a prompt and returns JSON can be substituted. A template-based fallback provides reports even without a language model.

### Key Platform Capabilities

- **Plug-and-Play Foundation Models:** Any embedding model (Path Foundation, UNI, CONCH, Virchow, DINOv2, or custom) can be registered via YAML. The platform requires only a model name and output dimension. All downstream layers adapt automatically.
- **Pluggable Classification Heads:** Any attention-based MIL model (TransMIL, CLAM, ABMIL) trained on compatible embeddings can be registered by dropping in the weights file. No code changes. The system handles multi-model ensemble, heatmap generation, and evidence extraction for any registered model.
- **Config-Driven Cancer Types:** Each cancer type is a project definition in YAML. Lung, breast, prostate, or any other cancer can be added by specifying dataset paths, labels, thresholds, and model assignments. The entire UI (labels, categories, model lists, workflows) reconfigures per project.
- **Per-Project Feature Toggles:** Report generation, semantic search, or similar case retrieval can be enabled or disabled independently per project, matching the deployment to available hardware and clinical needs.
- **Evidence Generation:** Per-model attention heatmaps overlaid on WSIs with coverage-based alignment, top-K evidence patches with coordinates, and FAISS-based similar case retrieval
- **Semantic Tissue Search:** Vision-language model text-to-patch search for queries like "tumor infiltrating lymphocytes" or "necrotic tissue" with on-the-fly embedding
- **Outlier Tissue Detector:** Centroid distance analysis on foundation model embeddings identifies morphologically unusual patches, including rare tissue patterns, artifacts, or atypical regions for pathologist review
- **Few-Shot Patch Classifier:** Pathologists define custom tissue classes by selecting example patches on the viewer; a LogisticRegression trained on the active foundation model's embeddings classifies all patches in the slide with per-class heatmap overlay
- **Structured Reporting:** Pluggable LLM report generation (demonstrated with MedGemma 4B) with JSON-first prompting, safety validation, and template-based fallback
- **Agentic Workflow:** Seven-step AI agent with visible reasoning, SSE streaming, and session memory for follow-up questions
- **Pathologist Annotations:** SVG-based drawing tools (circle, rectangle, freehand, point) on the OpenSeadragon viewer with full CRUD persistence to PostgreSQL
- **Batch Analysis:** Async batch processing with model selection, embedding level (L0/L1), force re-embed, real-time progress tracking, and cancellation support
- **PDF Export:** PDF reports for clinical documentation and tumor board presentations
- **On-Premise by Design:** Docker Compose deployment with air-gapped mode, no cloud dependencies at runtime, and full audit trail for HIPAA alignment

### Demo Configuration: Ovarian Cancer with Google HAI-DEF

The current deployment demonstrates the platform with:
- **Foundation model:** Google Path Foundation (384-dim, ViT-S)
- **Classification:** 5 TransMIL models (platinum sensitivity AUC 0.907, tumor grade, survival prediction)
- **Reports:** MedGemma 4B (4 billion parameter medical LLM)
- **Semantic search:** MedSigLIP (medical SigLIP vision-language model)
- **Dataset:** 208 TCGA ovarian cancer slides with clinical labels

This configuration is one example. Replacing the foundation model, training new classification heads for a different cancer type, or swapping the report generator requires zero code changes, only YAML configuration and model weights.

## 1.2 Abstraction Layers

The platform is organized into four abstraction layers, each independently replaceable:

| Layer | Role | Currently Demonstrated With | Swap By |
|---|---|---|---|
| **Foundation Model** | Patch-level embedding (N x D vectors) | Path Foundation (384-dim) | Register new model in YAML, re-embed slides |
| **Classification** | Slide-level prediction from embeddings | TransMIL (5 models) | Train new MIL head, drop in weights, register in YAML |
| **Evidence** | Heatmaps, retrieval, outlier detection, semantic search, few-shot | FAISS + MedSigLIP + LogisticRegression | Foundation model change propagates automatically |
| **Reporting** | Structured clinical reports | MedGemma 4B | Any LLM with JSON output; template fallback always available |

### HAI-DEF Demo Integration

The demo configuration integrates all three Google Health AI Developer Foundations models:

| HAI-DEF Model | Platform Layer | Integration Depth |
|---|---|---|
| **Path Foundation** | Foundation Model | 384-dim embeddings are the universal representation: classification, FAISS retrieval, outlier detection (centroid distance), few-shot classification (LogisticRegression), attention heatmaps, and batch re-embedding |
| **MedGemma 4B** | Reporting | Structured clinical report generation with JSON-first prompting, safety validation, decision support integration, async generation with progress tracking, and template fallback |
| **MedSigLIP** | Evidence | Text-to-patch semantic search with on-the-fly embedding, predefined pathology query sets, image-to-image visual search via FAISS, and GPU-accelerated batch embedding |

Any of these can be replaced with alternative models serving the same role. The platform enforces contracts (embedding dimension, attention output format, JSON report schema) rather than coupling to specific model implementations.

## 1.3 Target Users

| User Role | Primary Use Case | Key Features |
|---|---|---|
| Medical Oncologist | Treatment planning, tumor board discussion | Oncologist summary view, decision support, confidence scores, PDF reports |
| Pathologist | Slide review, morphology verification | WSI viewer with heatmap overlay, annotation tools (SVG on OSD), pathologist view, outlier detector, few-shot classifier |
| Tumor Board | Multidisciplinary case discussion | PDF reports, similar case comparison, batch analysis, multi-model ensemble |
| Clinical Researcher | Cohort analysis, biomarker discovery | Batch analysis, semantic search, project system, slide manager with filtering |
| Hospital IT | Deployment, maintenance | Docker Compose, on-premise, PostgreSQL, health checks, Tailscale Funnel access |

## 1.4 Clinical Validation (Demo Configuration)

The demo configuration predicts platinum-based chemotherapy sensitivity in high-grade serous ovarian carcinoma from standard H&E tissue slides. These results are from a specific foundation model, classification head, and dataset configuration:

| Metric | Value |
|---|---|
| Best single-model AUC (full dataset) | 0.879 |
| Full-dataset AUC (all 208 slides) | 0.907 |
| Optimized threshold (Youden J) | 0.9229 |
| Sensitivity | 84.6% |
| Specificity | 91.2% |
| Training cohort | 208 TCGA ovarian cancer slides |
| Class ratio | 139 non-responders : 13 responders |
| Embedding model | Path Foundation (ViT-S, 384-dim) |
| Classification model | TransMIL (3.2M parameters) |

These results are specific to the demo configuration. The same platform infrastructure (heatmaps, evidence patches, similar cases, outlier detection, structured reports) works identically with any foundation model and classification head. A hospital deploying a breast cancer HER2 predictor or lung cancer subtype classifier would get the same evidence chain, interpretability tools, and reporting pipeline.

## 1.5 Competitive Comparison

| Feature | Enso Atlas | Paige.ai | PathAI | Google DeepMind |
|---|---|---|---|---|
| On-Premise | Yes (required) | Cloud + Edge | Cloud | Research only |
| Model-Agnostic | Yes -- any foundation model, any MIL head via YAML | Proprietary only | Proprietary only | N/A |
| Foundation Models | Any (demo: all 3 HAI-DEF) | Proprietary | Proprietary | Internal |
| Pluggable Classification | Yes -- TransMIL, CLAM, ABMIL, custom | No | No | N/A |
| Multi-Cancer via Config | Yes -- YAML project definitions | Per-product | Per-product | N/A |
| Explainability | Attention heatmaps + evidence patches + semantic search + outlier detection + few-shot classifier + structured reports | Limited | Moderate | Research |
| Interactive Tools | Outlier detector, few-shot classifier, annotation system | None | Limited | N/A |
| Cost | Open source / self-hosted | SaaS subscription | SaaS subscription | N/A |

Paige.ai and PathAI sell single-purpose, closed-source tools for specific cancer types. Enso Atlas is an open platform that allows hospitals to deploy any model for any cancer type on hospital-owned hardware.

---

# 2. System Architecture

## 2.1 Abstraction Layer Architecture

The system is organized into four swappable layers. Each layer communicates through defined contracts (embedding dimension, attention format, JSON report schema) rather than coupling to specific model implementations.

```
+=====================================================================+
|  UI LAYER (independently themed, auto-configures per project)       |
|  Next.js 14 / React 18 / OpenSeadragon / SVG Annotations           |
|  WSI Viewer | Heatmap Overlay | Panels | Dark Mode | PDF Export     |
+=====================================================================+
        |  HTTP/SSE  |  DZI Tiles  |  Annotations CRUD
        v            v             v
+=====================================================================+
|  EVIDENCE LAYER (model-agnostic -- operates on any embedding space) |
|  FastAPI Backend / PostgreSQL / FAISS                               |
|  Attention Heatmaps | Similar Cases | Outlier Detection |           |
|  Few-Shot Classifier | Semantic Search | Agentic Workflow           |
+=====================================================================+
        |  embeddings (N x D)   |  attention weights (N,)
        v                       v
+=====================================================================+
|  CLASSIFICATION LAYER (pluggable -- any attention MIL model)        |
|  Contract: input (N x D) -> prediction (float) + attention (N,)    |
|  [TransMIL]  [CLAM]  [ABMIL]  [Custom MIL]   <- register via YAML |
+=====================================================================+
        |  patch embeddings (N x D)
        v
+=====================================================================+
|  FOUNDATION MODEL LAYER (swappable -- any patch embedding model)    |
|  Contract: image (224x224x3) -> embedding (D-dim vector)            |
|  [Path Foundation]  [UNI]  [CONCH]  [Virchow]  [DINOv2]  [Custom]  |
|                                                  <- register via YAML|
+=====================================================================+
        |  224x224 patches
        v
+=====================================================================+
|  DATA LAYER (format-agnostic slide storage)                         |
|  WSI Files (.svs/.tiff/.ndpi) | Embeddings (.npy) | PostgreSQL 16  |
+=====================================================================+
```

### Detailed Component Diagram

```mermaid
graph TB
    subgraph "UI Layer"
        UI[React UI - Port 3002]
        OSD[OpenSeadragon WSI Viewer]
        GRID[Patch Grid Overlay]
        SVG[SVG Annotation Layer]
        CANVAS[Canvas Heatmap Overlay]
        API_CLIENT[API Client - api.ts]
    end

    subgraph "Evidence Layer (FastAPI - Port 8003)"
        API[REST API - main.py]
        AGENT[Agent Workflow]
        CHAT[Chat Manager]
        DB_MOD[Database Module]
        PROJ[Project Registry - YAML driven]
        BATCH[Batch Tasks]
        BEMBED[Batch Embed]
        REPORT[Report Tasks]
        PDF[PDF Export]
    end

    subgraph "Classification Layer (Pluggable)"
        TM[TransMIL x5]
        CLAM_SLOT["[CLAM slot]"]
        ABMIL_SLOT["[ABMIL slot]"]
        CUSTOM_MIL["[Custom MIL slot]"]
        FAISS[FAISS Index]
        LR[LogisticRegression - Few-Shot]
    end

    subgraph "Foundation Model Layer (Swappable)"
        PF["Path Foundation (demo)"]
        UNI_SLOT["[UNI slot]"]
        CONCH_SLOT["[CONCH slot]"]
        CUSTOM_FM["[Custom model slot]"]
        VLM["Vision-Language Model (Semantic Search)"]
        LLM["Language Model (Reports)"]
    end

    subgraph "Data Layer"
        PG[(PostgreSQL 16<br/>Schema v5)]
        EMB[/Embeddings .npy<br/>N x D vectors/]
        WSI_FILES[/WSI Files .svs/]
        MODELS[/Model Checkpoints/]
        HCACHE[/Heatmap Cache .png/]
        YAML[/projects.yaml<br/>Configuration/]
    end

    UI --> API_CLIENT
    API_CLIENT -->|HTTP/SSE| API
    OSD -->|DZI Tiles| API
    SVG -->|Annotations CRUD| API
    CANVAS -->|Patch Overlay Data| API
    API --> DB_MOD
    API --> AGENT
    API --> CHAT
    API --> BATCH
    API --> BEMBED
    API --> REPORT
    API --> PDF
    AGENT --> VLM
    AGENT --> TM
    AGENT --> FAISS
    API --> PF
    API --> VLM
    API --> LLM
    API --> TM
    API --> FAISS
    API --> LR
    DB_MOD --> PG
    PROJ --> YAML
    PROJ --> PG
    PF --> EMB
    TM --> EMB
    API --> WSI_FILES
    TM --> MODELS
    LLM --> MODELS
    VLM --> MODELS
    API --> HCACHE
```

## 2.2 Component Overview

### Frontend Layer
- **Framework:** Next.js 14 with React 18, TypeScript, Tailwind CSS 3.x
- **WSI Viewer:** OpenSeadragon 4.1 for Deep Zoom Image (DZI) tile-based rendering with SVG annotation overlay, canvas-based patch heatmap overlays, and patch grid overlay
- **Layout:** react-resizable-panels v4 for user-adjustable, collapsible three-panel layout
- **State Management:** React Context (ProjectContext), custom hooks (useAnalysis, useKeyboardShortcuts), localStorage caching
- **API Communication:** Custom fetch wrapper (2,625 lines) with retry logic, timeout handling, SSE streaming, and typed error handling
- **Dark Mode:** Full dark mode support via Tailwind CSS dark variant

### Backend Layer
- **Framework:** FastAPI with Pydantic models for request/response validation
- **Primary Module:** `src/enso_atlas/api/main.py` (6,119 lines), containing all core endpoints, model loading, and startup orchestration
- **Database:** asyncpg connection pool to PostgreSQL 16 (raw SQL, no ORM) with schema v5 including annotations table
- **Background Tasks:** Thread-based batch analysis, report generation, and batch re-embedding with status polling and cancellation

### ML Models Layer
- **Path Foundation:** TensorFlow SavedModel producing 384-dim patch embeddings (CPU-only due to Blackwell TF incompatibility)
- **MedGemma 4B:** Transformers-based causal LM for structured report generation (GPU, bfloat16, ~8 GB VRAM)
- **MedSigLIP:** SigLIP vision-language model for text-to-patch semantic search (GPU, fp16, ~800 MB VRAM)
- **TransMIL:** Five PyTorch Transformer-based MIL classifiers with attention extraction (GPU, ~200 MB total)
- **FAISS:** Facebook AI Similarity Search for slide-level (cosine via IndexFlatIP) and patch-level (L2) retrieval
- **scikit-learn:** LogisticRegression for few-shot patch classification on Path Foundation embeddings

### Storage Layer
- **PostgreSQL 16:** Patients, slides, metadata, analysis results, embedding tasks, projects, junction tables, annotations (schema v5)
- **File System:** Pre-computed .npy embeddings (level 0 and level 1), .npy coordinate files, .svs whole-slide images
- **Model Storage:** HuggingFace cache + local checkpoints in `models/` directory
- **Heatmap Cache:** Disk-based PNG cache (`data/embeddings/heatmap_cache/`) for per-model attention heatmaps

## 2.3 Data Flow Diagram

```mermaid
flowchart LR
    subgraph "Offline Pipeline"
        WSI[WSI .svs File] --> PATCH[Patch Extraction<br/>224x224 @ Level 0/1]
        PATCH --> TISSUE[Tissue Masking<br/>Background Removal]
        TISSUE --> PF_EMB[Path Foundation<br/>Embedding]
        PF_EMB --> NPY[.npy Files<br/>N x 384]
    end

    subgraph "Online Analysis"
        NPY --> TRANSMIL[TransMIL x5<br/>Classification]
        TRANSMIL --> PRED[Prediction<br/>+ Attention Weights]
        PRED --> HEATMAP[Attention Heatmap<br/>Per-Model Cached PNG]
        PRED --> EVIDENCE[Top-K Evidence<br/>Patches]
        NPY --> FAISS_Q[FAISS<br/>Query]
        FAISS_Q --> SIMILAR[Similar<br/>Cases]
        NPY --> SIGLIP[MedSigLIP<br/>Semantic Search]
        SIGLIP --> TEXT_MATCH[Text-Matched<br/>Patches]
        NPY --> OUTLIER[Outlier<br/>Detection]
        OUTLIER --> UNUSUAL[Unusual<br/>Patches]
        NPY --> FEWSHOT[Few-Shot<br/>Classifier]
        FEWSHOT --> TISSUE_MAP[Tissue<br/>Classification]
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

```
+-----------------------------------+     +------------------+
|  enso-atlas (Application)         |     |  atlas-db        |
|  - FastAPI Backend (port 8003)    |---->|  PostgreSQL 16   |
|  - Path Foundation (CPU)          |     |  (port 5433)     |
|  - MedGemma 4B (GPU, bfloat16)   |     +------------------+
|  - MedSigLIP (GPU, fp16)         |
|  - TransMIL x5 (GPU)             |
|  - FAISS Index (CPU)             |
+-----------------------------------+
        |
        | Port 8003 (API)
        v
+-----------------------------------+
|  Frontend (Next.js 14)            |
|  - Production Build               |
|  - Port 3002                      |
|  - Proxy /api/* -> :8003          |
+-----------------------------------+
        |
        | (optional)
        v
+-----------------------------------+
|  Tailscale Funnel                 |
|  - Public HTTPS access            |
|  - Zero-config TLS                |
+-----------------------------------+
```

### Port Mappings

| Service | Internal Port | External Port | Purpose |
|---|---|---|---|
| enso-atlas | 8000 | 8003 | FastAPI REST API |
| atlas-db | 5432 | 5433 | PostgreSQL |
| frontend | 3002 | 3002 | Next.js production server |

## 2.5 Network Architecture

The Next.js frontend proxies all API requests to the backend via a rewrite rule configured in `frontend/next.config.mjs`:

```javascript
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

This eliminates CORS issues in production and creates a unified origin for the browser. The backend also configures CORS middleware for direct API access during development.

For public access without opening firewall ports, Tailscale Funnel provides zero-config HTTPS tunneling to the frontend. This supports demonstrations and remote access while maintaining the on-premise security model.

---

# 3. Foundation Model Layer

The foundation model layer is the bottom of the abstraction stack. It converts 224x224 pixel tissue patches into fixed-dimensional embedding vectors. The platform imposes a single contract on foundation models: **input a patch image, output a D-dimensional vector.** Everything above this layer (classification, retrieval, outlier detection, few-shot learning) operates on the embedding vectors and is agnostic to which model produced them.

Foundation models are registered in `config/projects.yaml` under `foundation_models`. To add a new foundation model, specify its name, embedding dimension, and description. Then re-embed slides with the new model and update project configurations to reference it. No backend code changes are required.

```yaml
foundation_models:
  path_foundation:
    name: "Path Foundation"
    embedding_dim: 384
  uni:
    name: "UNI"
    embedding_dim: 1024
  virchow:
    name: "Virchow"
    embedding_dim: 768
  your_custom_model:
    name: "Institutional Fine-Tuned ViT"
    embedding_dim: 512
```

The demo deployment uses three Google HAI-DEF models. The following subsections document their specifications as reference implementations.

## 3.1 Path Foundation (Google) -- Demo Foundation Model

### Overview

Path Foundation is Google's histopathology foundation model, based on a DINOv2-style self-supervised learning architecture trained on millions of pathology images. It produces 384-dimensional feature vectors from 224x224 pixel tissue patches. In the demo configuration, these embeddings are the universal tissue representation for:

1. TransMIL slide-level classification (5 models)
2. FAISS similar-case retrieval (slide-mean cosine similarity)
3. Outlier tissue detection (centroid distance analysis)
4. Few-shot patch classification (LogisticRegression on embeddings)
5. Attention heatmap generation (per-model, coverage-aligned)
6. Image-to-image visual search (patch-level L2 FAISS)

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

### Patching Strategy

Whole-slide images are divided into non-overlapping 224x224 pixel patches at either level 0 (full resolution, typically 40x) or level 1 (downsampled). The patching process includes tissue masking:

1. **Grid Generation:** Regular grid with step size 224 pixels
2. **Tissue Detection:** Pixels with mean intensity < 245 and standard deviation > 10 are considered tissue (relaxed thresholds for maximum patch coverage)
3. **Batch Embedding:** Tissue patches batched (64 patches per batch) through Path Foundation on CPU
4. **Storage:** Embeddings saved as `{slide_id}.npy` (N x 384) with coordinates saved as `{slide_id}_coords.npy` (N x 2)

For a typical ovarian cancer slide (~80,000 x 40,000 pixels at level 0), this produces between 3,000 and 8,000 tissue patches.

### Performance Benchmarks

| Metric | Value |
|---|---|
| Model Loading Time | ~15 seconds |
| Batch Inference (64 patches, CPU) | ~2.5 seconds |
| Per-Patch Latency (amortized) | ~39 ms |
| Memory Usage (CPU) | ~2 GB |
| Full Slide Embedding (6,000 patches, level 0) | ~4 minutes |
| Embedding Dimension | 384 floats = 1.5 KB per patch |

### TensorFlow on CPU

Path Foundation uses TensorFlow, which does not support NVIDIA Blackwell GPUs (compute capability sm_121). The system runs Path Foundation on CPU. This is acceptable because embeddings are pre-computed offline. On-demand embedding is supported for new slides via background tasks with progress tracking.

## 3.2 MedGemma 4B -- Demo Report Generator

### Overview

The reporting layer accepts any LLM that can produce structured JSON from a clinical prompt. The demo uses MedGemma, Google's medical language model: a 4-billion parameter instruction-tuned variant of Gemma optimized for medical text generation. Alternative LLMs (Llama 3 Med, BioMistral, or proprietary clinical LLMs) can be substituted by implementing the same prompt/response contract. A template-based fallback provides reports even without any LLM.

### Architecture Specifications

| Parameter | Value |
|---|---|
| Model ID | `google/medgemma-4b-it` |
| Architecture | Gemma-2 Causal LM |
| Parameters | 4 billion |
| Precision | bfloat16 (GPU) |
| VRAM Usage | ~8 GB |
| Max Input Tokens | 512 |
| Max Output Tokens | 384 |
| Temperature | 0.1 |
| Top-p | 0.9 |

### Inference Pipeline

1. **Model Loading:** Lazy-loaded with thread-safe locking in bfloat16 on GPU
2. **Warmup:** Test inference at startup pre-compiles CUDA kernels (60-120s)
3. **Prompt Engineering:** Compact JSON-first prompt forcing structured output
4. **Generation:** `torch.inference_mode()` with configurable max tokens and time limits via custom `StoppingCriteria`
5. **Non-Blocking:** All inference wrapped in `asyncio.to_thread()` to avoid blocking the event loop
6. **JSON Parsing:** Progressive repair strategies for truncated JSON with regex fallback
7. **Safety Validation:** Every report checked against prohibited phrases (e.g., "start treatment", "prescribe")

### Report Schema

```json
{
  "case_id": "TCGA-04-1331-01A-01-BS1",
  "task": "Platinum treatment response prediction",
  "model_output": {
    "label": "responder",
    "probability": 0.94,
    "calibration_note": "Uncalibrated research model"
  },
  "evidence": [...],
  "limitations": [...],
  "suggested_next_steps": [...],
  "safety_statement": "Research tool only...",
  "decision_support": {...}
}
```

## 3.3 MedSigLIP -- Demo Vision-Language Model

### Overview

The semantic search layer accepts any vision-language model that can encode both text and images into a shared embedding space. The demo uses MedSigLIP, Google's medical vision-language model based on the SigLIP architecture. Alternative models (BiomedCLIP, PLIP, OpenCLIP fine-tuned on pathology data) can be substituted. Semantic search is an optional feature toggle. Projects can disable it entirely if no vision-language model is available.

### Architecture Specifications

| Parameter | Value |
|---|---|
| Model ID | `google/medsiglip-448` (preferred) or `google/siglip-so400m-patch14-384` (fallback) |
| Architecture | SigLIP (dual encoder: vision + text) |
| Vision Input | 448 x 448 (MedSigLIP) or 384 x 384 (SigLIP) |
| Embedding Dimension | 1152 |
| Precision | fp16 (GPU) |
| VRAM Usage | ~800 MB |

### Capabilities

- **Text-to-Patch Search:** Natural language queries matched against patch embeddings via cosine similarity
- **On-the-Fly Embedding:** When cached MedSigLIP embeddings are unavailable, patches are extracted from the WSI and embedded in real-time
- **Image-to-Image Search:** Visual similarity search via FAISS using Path Foundation embeddings across the entire database
- **Predefined Query Sets:** Curated queries for tumor, inflammation, necrosis, stroma, mitosis, and vessels

### VRAM Sharing

MedSigLIP shares GPU with MedGemma. At fp16, SigLIP requires ~800 MB alongside MedGemma's ~8 GB, fitting within typical GPU memory.

---

# 4. Classification Layer

The classification layer sits above the foundation model layer and consumes embedding vectors. It imposes a single contract: **input N x D embeddings, output a scalar prediction and N attention weights.** The attention weights power the heatmap overlays, evidence patch selection, and interpretability tools. Any MIL architecture that produces per-patch attention scores is compatible.

Classification models are registered in `config/projects.yaml` under `classification_models`. Each entry specifies a model directory (containing the checkpoint), display metadata (name, AUC, labels), and compatible foundation model. Multiple classification models can be assigned to a single project for ensemble analysis.

```yaml
classification_models:
  your_model:
    model_dir: "transmil_your_task"           # Directory under outputs/
    display_name: "Your Prediction Task"
    auc: 0.85                                 # Displayed in UI
    positive_label: "Positive"
    negative_label: "Negative"
    compatible_foundation: "path_foundation"   # Must match embedding_dim
```

The demo deployment uses TransMIL. The following subsections document it as the reference classification architecture.

## 4.1 TransMIL -- Demo Classification Architecture

TransMIL extends the MIL paradigm with Transformer self-attention, allowing the model to capture inter-patch relationships and spatial context. Attention signals provide interpretability for clinical adoption. Other compatible architectures include CLAM (Clustering-constrained Attention MIL), ABMIL (Attention-Based MIL), and DSMIL (Dual-Stream MIL). Any architecture that outputs per-patch attention weights integrates directly.

## 4.2 Architecture Details

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

### Hyperparameters (Demo Configuration)

| Parameter | Value | Configurable |
|---|---|---|
| Input Dimension | 384 (matches Path Foundation) | Auto-adapts to foundation model embedding_dim |
| Hidden Dimension | 512 | Via `--hidden_dim` training arg |
| Number of Classes | 1 (binary with sigmoid) | Per training configuration |
| Attention Heads | 8 | Via `--num_heads` training arg |
| Transformer Layers | 2 | Via `--num_layers` training arg |
| Dropout | 0.1 (inference) / 0.25 (training) | Via `--dropout` training arg |
| Total Parameters | ~3.2M | Varies with hidden_dim/layers |

## 4.3 Multi-Model Inference

The platform supports any number of classification models per project. The demo includes five specialized TransMIL models trained for different prediction targets:

| Model ID | Display Name | Category | AUC | Training Slides |
|---|---|---|---|---|
| `platinum_sensitivity` | Platinum Sensitivity | Ovarian Cancer | 0.907 | 199 |
| `tumor_grade` | Tumor Grade | General Pathology | 0.752 | 918 |
| `survival_5y` | 5-Year Survival | Ovarian Cancer | 0.697 | 965 |
| `survival_3y` | 3-Year Survival | Ovarian Cancer | 0.645 | 1,106 |
| `survival_1y` | 1-Year Survival | Ovarian Cancer | 0.639 | 1,135 |

Multi-model inference runs all models sequentially on the same embeddings in ~100-250ms (GPU). Results are cached to PostgreSQL's `analysis_results` table for instant retrieval on subsequent views (~0.8ms).

## 4.4 Threshold Optimization

The classification threshold is optimized using the Youden J statistic:

```json
{
  "method": "youden_j",
  "threshold": 0.9229,
  "sensitivity": 0.846,
  "specificity": 0.912,
  "auc": 0.879
}
```

The high threshold (0.9229 vs default 0.5) reflects the class imbalance (139:13) and optimizes for balanced sensitivity/specificity.

## 4.5 Training Pipeline

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

---

# 5. Backend API Specification

## 5.1 Startup and Initialization

The FastAPI application is created via `create_app()` in `src/enso_atlas/api/main.py` (6,119 lines). Startup orchestrates model loading:

```
1.  MIL Classifier (TransMIL checkpoint)            ~2s
2.  Multi-Model Inference (5 models)                 ~5s
3.  Evidence Generator (FAISS index)                 ~3s
4.  Path Foundation Embedder Init (lazy, CPU)        ~0s
5.  MedGemma Reporter (GPU, bfloat16)               ~30s
6.  MedGemma Warmup (CUDA kernel compilation)        ~60-120s
7.  MedSigLIP Embedder (GPU, fp16)                   ~10s
8.  Clinical Decision Support Engine                  ~0.1s
9.  FAISS Patch Index Build                          ~3s
10. Slide-Mean FAISS Index (cosine similarity)        ~1s
11. Label Loading (CSV parsing)                       ~0.5s
12. PostgreSQL Schema Init + Population               ~5-60s (first run)
13. Project Registry (YAML loading)                   ~0.1s
14. Agent Workflow Initialization                     ~0.1s
15. Chat Manager Initialization                       ~0.1s
```

### Timing Benchmarks (DGX Spark, Blackwell GPU)

| Step | Duration |
|---|---|
| TransMIL Loading (5 checkpoints) | 2.1s |
| MedGemma Loading (4B params, bfloat16) | 28.4s |
| MedGemma Warmup (CUDA kernels) | 62.3s |
| MedSigLIP Loading (SigLIP-so400m, fp16) | 9.7s |
| FAISS Index Build (208 slides, ~1M patches) | 3.2s |
| PostgreSQL Init (schema + population) | 4.8s |
| **Total Startup** | **~120s** |

## 5.2 Complete API Endpoint Reference

### Health and Status

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/health` | Health check (model status, CUDA, slide count, DB, uptime) |
| `GET` | `/api/db/status` | Database connection and population status |
| `POST` | `/api/db/repopulate` | Force re-population from flat files |
| `GET` | `/api/embed/status` | Path Foundation embedder status |
| `GET` | `/api/semantic-search/status` | MedSigLIP model status |
| `GET` | `/api/search/visual/status` | Visual search FAISS index status |

### Slide Management

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/slides` | List all slides (with ?project_id= filtering) |
| `GET` | `/api/slides/search` | Search slides with filtering and pagination |
| `GET` | `/api/slides/{id}/dzi` | DZI XML for OpenSeadragon |
| `GET` | `/api/slides/{id}/dzi_files/{level}/{tile}` | DZI tile image |
| `GET` | `/api/slides/{id}/thumbnail` | Slide thumbnail (disk-cached) |
| `GET` | `/api/slides/{id}/info` | Detailed slide info (dimensions, levels) |
| `GET` | `/api/slides/{id}/patches/{patch_id}` | Patch image (from WSI or placeholder) |
| `GET` | `/api/slides/{id}/qc` | Slide quality control metrics |
| `GET` | `/api/slides/{id}/cached-results` | All cached analysis results per model |
| `GET` | `/api/slides/{id}/embedding-status` | Embedding and analysis cache status |
| `PATCH` | `/api/slides/{id}` | Rename slide (update display_name) |

### Analysis

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/analyze` | Single-slide prediction with evidence |
| `POST` | `/api/analyze-multi` | Multi-model ensemble (5 models, with caching) |
| `POST` | `/api/analyze-uncertainty` | MC Dropout uncertainty quantification |
| `POST` | `/api/analyze-batch` | Synchronous batch analysis |
| `POST` | `/api/analyze-batch/async` | Async batch with model selection, level, force re-embed |
| `GET` | `/api/analyze-batch/status/{task_id}` | Batch progress |
| `POST` | `/api/analyze-batch/cancel/{task_id}` | Cancel running batch |
| `GET` | `/api/analyze-batch/tasks` | List all batch tasks |

### Embedding

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/embed` | Embed base64-encoded patches |
| `POST` | `/api/embed-slide` | On-demand slide embedding (level 0 or 1) |
| `GET` | `/api/embed-slide/status/{task_id}` | Embedding task progress |
| `GET` | `/api/embed-slide/tasks` | List embedding tasks |
| `POST` | `/api/embed-slides/batch` | Batch re-embedding (all or selected slides) |
| `GET` | `/api/embed-slides/batch/status/{id}` | Batch embed progress |
| `POST` | `/api/embed-slides/batch/cancel/{id}` | Cancel batch embed |
| `GET` | `/api/embed-slides/batch/active` | Currently active batch embed |

### Search and Retrieval

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/similar` | Similar case search (slide-mean cosine FAISS) |
| `POST` | `/api/semantic-search` | MedSigLIP text-to-patch search |
| `POST` | `/api/search/visual` | Image-to-image FAISS search |
| `POST` | `/api/classify-region` | Tissue type classification at coordinates |

### Heatmaps

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/heatmap/{slide_id}` | Default attention heatmap (configurable level, smooth, blur) |
| `GET` | `/api/heatmap/{slide_id}/{model_id}` | Per-model heatmap (disk-cached PNG, coverage-aligned) |

### Outlier Detection and Few-Shot Classification

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/slides/{id}/outlier-detection` | Detect outlier patches (centroid distance) |
| `POST` | `/api/slides/{id}/patch-classify` | Few-shot classification (LogisticRegression) |
| `GET` | `/api/slides/{id}/patch-coords` | Patch coordinates for spatial selection |

### Annotations

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/slides/{id}/annotations` | List all annotations for a slide |
| `POST` | `/api/slides/{id}/annotations` | Create annotation |
| `PUT` | `/api/slides/{id}/annotations/{ann_id}` | Update annotation |
| `DELETE` | `/api/slides/{id}/annotations/{ann_id}` | Delete annotation |
| `GET` | `/api/slides/{id}/annotations/summary` | Annotation summary by label |

### Reports and Export

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/report` | Generate MedGemma report (sync) |
| `POST` | `/api/report/async` | Async report generation |
| `GET` | `/api/report/status/{task_id}` | Report generation status |
| `GET` | `/api/report/tasks` | List report tasks |
| `POST` | `/api/report/pdf` | Generate PDF from report JSON (fpdf2) |
| `POST` | `/api/export/pdf` | Full PDF export with heatmap (reportlab) |

### Projects

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/projects` | List all projects |
| `POST` | `/api/projects` | Create project |
| `GET` | `/api/projects/{id}` | Get project details |
| `PUT` | `/api/projects/{id}` | Update project |
| `DELETE` | `/api/projects/{id}` | Delete project |
| `GET` | `/api/projects/{id}/slides` | List project slides |
| `POST` | `/api/projects/{id}/slides` | Assign slides |
| `DELETE` | `/api/projects/{id}/slides` | Unassign slides |
| `GET` | `/api/projects/{id}/models` | List project models |
| `POST` | `/api/projects/{id}/models` | Assign models |
| `DELETE` | `/api/projects/{id}/models` | Unassign models |
| `POST` | `/api/projects/{id}/upload` | Upload slide file |
| `GET` | `/api/projects/{id}/status` | Project readiness status |
| `GET` | `/api/models` | List available models (with ?project_id= filtering) |

### Agent Workflow

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/agent/analyze` | Start 7-step agent analysis (SSE) |
| `POST` | `/api/agent/followup` | Follow-up questions (SSE) |
| `GET` | `/api/agent/session/{id}` | Get session details |
| `GET` | `/api/agent/sessions` | List active sessions |
| `DELETE` | `/api/agent/session/{id}` | Delete session |

### Chat

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/chat` | RAG-based conversational AI (SSE) |
| `GET` | `/api/chat/session/{id}` | Get chat session history |

### History and Audit

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/history` | Analysis history with filtering |
| `GET` | `/api/slides/{id}/history` | Per-slide history |
| `GET` | `/api/audit-log` | Compliance audit trail |

### Tags and Groups (Stubs)

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/tags` | List tags |
| `POST` | `/api/tags` | Create tag |
| `GET` | `/api/groups` | List groups |
| `POST` | `/api/groups` | Create group |

---

# 6. Agentic Workflow

## 6.1 Overview

The agentic workflow implements a seven-step AI pipeline that performs slide analysis with visible reasoning, retrieval-augmented generation, and structured report production.

### Design Principles

1. **Transparency:** Every step produces visible reasoning that clinicians can inspect
2. **Streaming:** Results are streamed via Server-Sent Events (SSE) for real-time progress
3. **Session Memory:** Sessions persist analysis state for follow-up questions
4. **Graceful Degradation:** Each step can be skipped if dependencies are unavailable

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
    Agent-->>Client: SSE: initialize complete (~50ms)

    Note over Agent: Step 2: ANALYZE
    Agent->>TransMIL: predict_all(embeddings)
    TransMIL-->>Agent: predictions + attention
    Agent-->>Client: SSE: analyze complete (~200ms)

    Note over Agent: Step 3: RETRIEVE
    Agent->>FAISS: search(mean_embedding, k=10)
    FAISS-->>Agent: similar cases
    Agent-->>Client: SSE: retrieve complete (~10ms)

    Note over Agent: Step 4: SEMANTIC SEARCH
    Agent->>MedSigLIP: search(queries, embeddings)
    MedSigLIP-->>Agent: tissue pattern matches
    Agent-->>Client: SSE: semantic_search complete (~100ms)

    Note over Agent: Step 5: COMPARE
    Agent->>Agent: Analyze similar case patterns
    Agent-->>Client: SSE: compare complete (~5ms)

    Note over Agent: Step 6: REASON
    Agent->>Agent: Synthesize evidence
    Agent-->>Client: SSE: reason complete (~10ms)

    Note over Agent: Step 7: REPORT
    Agent->>MedGemma: generate_report()
    MedGemma-->>Agent: structured report
    Agent-->>Client: SSE: report complete (10-120s)
```

## 6.3 SSE Streaming Format

```json
{
  "step": "analyze",
  "status": "complete",
  "message": "Ran 5 models on 6,234 patches",
  "reasoning": "Model predictions:\n- Platinum Sensitivity: sensitive (score: 0.94)...",
  "data": {"predictions": {}, "top_evidence": []},
  "timestamp": "2025-02-06T23:57:42.123Z",
  "duration_ms": 187.3
}
```

## 6.4 Session Management

Sessions are stored in-memory with the `AgentState` dataclass. Follow-up questions can reference previous analysis results without re-running the full workflow.

---

# 7. Database Schema

## 7.1 Schema Version 5

The database schema (defined in `src/enso_atlas/api/database.py`) consists of ten tables across five migration versions:

### ER Diagram

```mermaid
erDiagram
    patients ||--o{ slides : "has"
    slides ||--o{ slide_metadata : "has"
    slides ||--o{ analysis_results : "has"
    slides ||--o{ embedding_tasks : "has"
    slides ||--o{ annotations : "has"
    projects ||--o{ project_slides : "has"
    slides ||--o{ project_slides : "assigned to"
    projects ||--o{ project_models : "has"

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
        text display_name
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

    project_slides {
        text project_id PK_FK
        text slide_id PK_FK
        timestamptz created_at
    }

    project_models {
        text project_id PK_FK
        text model_id PK
        timestamptz created_at
    }

    annotations {
        text id PK
        text slide_id
        text type
        jsonb coordinates
        text label
        text notes
        text color
        text category
        timestamptz created_at
        timestamptz updated_at
    }

    schema_version {
        integer version PK
        timestamptz applied_at
    }
```

## 7.2 Migration History

| Version | Migration | Description |
|---|---|---|
| v1 | Initial schema | patients, slides, slide_metadata, analysis_results, embedding_tasks, projects, schema_version |
| v2 | `_migrate_project_columns` | Added `project_id` column to slides table |
| v3 | `_migrate_project_scoped_tables` | Created project_models and project_slides junction tables, seeded initial data |
| v4 | `_migrate_display_name` | Added `display_name` column to slides table for user-assigned aliases |
| v5 | `_migrate_annotations_table` | Created annotations table with JSONB coordinates, label, notes, color, category |

## 7.3 Connection Pool

```python
_pool = await asyncpg.create_pool(
    DATABASE_URL,
    min_size=2,
    max_size=10,
    command_timeout=60,
)
```

10 retry attempts with 2-second delay for database startup synchronization.

## 7.4 Query Performance

| Query | PostgreSQL | Flat-File Fallback |
|---|---|---|
| List all slides (208) | 2-5ms | 30-60s |
| Get single slide | <1ms | N/A |
| Search with filters | 5-10ms | 100-500ms |
| Cached result lookup | 0.8ms | N/A |
| Annotations for slide | <2ms | N/A |
| Count slides | <1ms | N/A |

---

# 8. Frontend Architecture

## 8.1 Tech Stack

| Technology | Version | Purpose |
|---|---|---|
| Next.js | 14.2 | React framework with SSR, routing, API proxying |
| React | 18 | UI component library |
| TypeScript | 5.x | Type-safe frontend development |
| Tailwind CSS | 3.x | Utility-first CSS with dark mode support |
| OpenSeadragon | 4.1 | Tile-based whole-slide image viewer |
| react-resizable-panels | 4.x | Resizable and collapsible layout panels |
| Lucide React | - | Icon library |

## 8.2 Component Hierarchy

```
App (layout.tsx)
  +-- ProjectProvider (ProjectContext.tsx)
      +-- HomePage (page.tsx, 2142 lines)
          +-- Header
          |   +-- ProjectSwitcher
          |   +-- ViewModeToggle (oncologist/pathologist)
          |   +-- DemoModeToggle
          |   +-- Dark Mode Toggle
          +-- PanelGroup (react-resizable-panels)
              +-- Left Panel (collapsible)
              |   +-- SlideSelector (25,527 bytes)
              |   +-- QuickStatsPanel
              |   +-- AnalysisControls
              +-- Center Panel
              |   +-- WSIViewer (61,161 bytes)
              |   |   +-- OpenSeadragon Canvas
              |   |   +-- SVG Annotation Overlay
              |   |   +-- Canvas Patch Overlay (outlier/classifier heatmaps)
              |   |   +-- Patch Grid Overlay (224px boundaries)
              |   |   +-- Real-Time Scale Bar
              |   |   +-- Navigator Minimap
              |   |   +-- Heatmap Model Selector
              +-- Right Panel (collapsible)
              |   +-- PredictionPanel
              |   +-- MultiModelPredictionPanel
              |   +-- EvidencePanel
              |   +-- SimilarCasesPanel
              |   +-- SemanticSearchPanel
              |   +-- ReportPanel
              |   +-- BatchAnalysisPanel (43,366 bytes)
              |   +-- AIAssistantPanel (38,056 bytes)
              |   +-- OutlierDetectorPanel (11,988 bytes)
              |   +-- PatchClassifierPanel (16,135 bytes)
              |   +-- CaseNotesPanel
              |   +-- UncertaintyPanel
              +-- Footer
              +-- Modals
                  +-- PatchZoomModal
                  +-- KeyboardShortcutsModal
                  +-- SettingsModal
                  +-- SystemStatusModal
                  +-- WelcomeModal

Slides Page (/slides/page.tsx)
  +-- SlideManager
      +-- GridView / TableView (with thumbnails)
      +-- FilterPanel (label, embeddings, patches, date, search)
      +-- BulkActions
      +-- Pagination

Projects Page (/projects/page.tsx)
  +-- ProjectManager
      +-- ProjectList (CRUD cards with readiness status)
      +-- SlideUpload
      +-- ModelAssignment
      +-- SlideAssignment
```

## 8.3 WSI Viewer (61,161 bytes)

The WSI viewer (`frontend/src/components/viewer/WSIViewer.tsx`) is the central component:

### Features

- **OpenSeadragon Integration:** DZI tile-based rendering with navigator minimap, double-click zoom, scroll zoom
- **Attention Heatmap Overlay:** Per-model heatmaps loaded as OSD `addSimpleImage` overlays with configurable opacity and model selection dropdown
- **Canvas-Based Patch Overlay:** Separate HTML canvas for outlier detection and few-shot classifier heatmaps, rendered independently from OSD heatmaps
- **Patch Grid Overlay:** 224px boundary grid drawn on a dedicated canvas with configurable opacity and color (cyan default), toggled via toolbar
- **SVG Annotation Layer:** Real-time drawing of circle, rectangle, freehand, and point annotations using SVG overlay positioned in OSD viewport coordinates
- **Spatial Patch Selection:** Click-to-select mode for few-shot classifier. Clicking on the viewer identifies the nearest patch and adds it to the active class
- **Real-Time Scale Bar:** Imperative DOM updates (bypassing React render) on every OSD animation frame for smooth magnification and scale display
- **Canvas2D Optimization:** Patches HTMLCanvasElement.getContext to add `willReadFrequently: true` for OpenSeadragon pixel data access

### Scale Bar Implementation

The scale bar uses direct DOM manipulation for zero-latency updates:

```typescript
const updateScaleDisplay = useCallback((z: number) => {
    const info = computeScaleBar(z);
    if (scaleTextRef.current) {
        scaleTextRef.current.textContent = `${info.displayValue} ${info.displayUnit}`;
    }
    if (magTextRef.current) {
        magTextRef.current.textContent = `${info.effectiveMag.toFixed(1)}x`;
    }
}, [computeScaleBar]);
```

This avoids React re-renders during continuous zoom/pan operations.

## 8.4 Resizable Panels

All major layout panels use react-resizable-panels v4:

- **Three-panel layout:** Left sidebar (slide selector), center (WSI viewer), right sidebar (analysis results)
- **Collapsible:** Each sidebar can be collapsed via toggle buttons with smooth animation
- **Persistent:** Panel sizes are remembered across sessions
- **Responsive:** Mobile view switches to tabbed panels (Slides / Results)

## 8.5 Dark Mode

Full dark mode support via Tailwind CSS `dark:` variant:

- Toggle in the header persisted to localStorage
- All components styled with dark variants (slate-800, slate-700, etc.)
- OpenSeadragon navigator styled with dark background
- Charts and visualizations adapt to dark theme

## 8.6 State Management

### ProjectContext

Global project configuration providing `currentProject` and `switchProject()`. Selected project persisted to localStorage.

### useAnalysis Hook

Manages analysis state (loading, results, error) with retry support.

### Key State Variables in page.tsx

- `outlierHeatmapData` / `showOutlierHeatmap`: Canvas-based outlier visualization
- `classifyResult` / `showClassifyHeatmap`: Canvas-based classifier visualization
- `patchSelectionMode` / `patchCoordinates`: Spatial selection for few-shot classifier
- `annotations` / `activeAnnotationTool`: Pathologist annotation state
- `multiModelResult` / `selectedModels`: Multi-model analysis
- `resolutionLevel` / `forceReembed`: Batch analysis configuration
- `heatmapModel` / `heatmapLevel`: Per-model heatmap selection

## 8.7 API Client (2,625 lines)

The API client (`frontend/src/lib/api.ts`) implements:

- **Retry Logic:** 3 retries with exponential backoff for 408, 429, 500, 502, 503, 504
- **Typed Errors:** `AtlasApiError` with `getUserMessage()` for display
- **SSE Streaming:** AsyncGenerator for agent and chat endpoints
- **Functions:** `detectOutliers()`, `classifyPatches()`, `getPatchCoords()`, `getAnnotations()`, `saveAnnotation()`, `deleteAnnotation()`, `getSlideCachedResults()`, `embedSlideWithPolling()`, `analyzeSlideMultiModel()`, `visualSearch()`, plus all standard CRUD operations

---

# 9. Config-Driven Project System

The project system is the architectural center of Enso Atlas. It is the mechanism by which the platform achieves model-agnosticism and multi-cancer support without code changes. Every deployment decision (which foundation model to use, which classification heads to run, what labels to display, which features to enable) is encoded in a single YAML file.

A hospital can go from a trained model and a set of slides to a fully deployed clinical decision support tool by editing a configuration file and restarting the server. No Python, no TypeScript, no Docker rebuilds.

## 9.1 YAML Schema

Projects are defined in `config/projects.yaml`:

```yaml
foundation_models:
  path_foundation:
    name: "Path Foundation"
    embedding_dim: 384
    description: "Google Health pathology foundation model (ViT-S, 384-dim)"
  dinov2:
    name: "DINOv2"
    embedding_dim: 768
  uni:
    name: "UNI"
    embedding_dim: 1024
  conch:
    name: "CONCH"
    embedding_dim: 512

classification_models:
  platinum_sensitivity:
    model_dir: "transmil_v2"
    display_name: "Platinum Sensitivity"
    auc: 0.907
    n_slides: 199
    category: "ovarian_cancer"
    positive_label: "Sensitive"
    negative_label: "Resistant"
    compatible_foundation: "path_foundation"
  # ... (5 models total)

projects:
  ovarian-platinum:
    name: "Ovarian Cancer - Platinum Sensitivity"
    cancer_type: Ovarian Cancer
    prediction_target: platinum_sensitivity
    classes: ["resistant", "sensitive"]
    positive_class: sensitive
    foundation_model: path_foundation
    classification_models:
      - platinum_sensitivity
      - tumor_grade
      - survival_5y
      - survival_3y
      - survival_1y
    features:
      medgemma_reports: true
      medsiglip_search: true
    threshold: 0.9229
```

## 9.2 ProjectRegistry Backend

The `ProjectRegistry` class loads and manages project configurations with PostgreSQL synchronization on startup. Junction tables (`project_slides`, `project_models`) provide many-to-many relationships.

## 9.3 Adding a New Cancer Type

No code changes required. The entire process is configuration-driven:

1. **Define the project** in `config/projects.yaml` (cancer type, classes, labels, thresholds)
2. **Add slides** to the configured `slides_dir`
3. **Generate embeddings:** `python scripts/embed_level0_pipelined.py --slides_dir ... --output_dir ...`
4. **Train a classification model:** `python scripts/train_transmil.py --embeddings_dir ... --labels_file ...`
5. **Register the model** under `classification_models` in `projects.yaml`
6. **Restart the server.** The new project auto-loads in the frontend project switcher with all labels, models, and features configured

The frontend renders entirely from project metadata: class names, positive/negative labels, model lists, feature availability, and threshold values all derive from the YAML. No frontend code is aware of specific cancer types or model names.

## 9.4 Switching Between Projects

The frontend `ProjectContext` reads available projects from the API and renders a project switcher in the header. When the user switches projects:

1. The slide list re-filters to show only project-assigned slides
2. The model picker updates to show only project-assigned classification models
3. Labels, class names, and threshold values update to match the project config
4. Feature panels show/hide based on project feature toggles
5. The selected project persists to localStorage across sessions

A single Enso Atlas deployment can serve multiple clinical teams working on different cancer types simultaneously.

---

# 10. Outlier Detection and Few-Shot Classification

## 10.1 Outlier Tissue Detector

The outlier detector identifies morphologically unusual tissue patches by measuring each patch's distance from the embedding centroid. It operates on whatever foundation model embeddings are configured for the active project. No model-specific code is required.

### Algorithm

```
1. Load foundation model embeddings (N x D) for the slide
2. Compute centroid = mean(embeddings, axis=0)
3. Compute distances = ||embedding_i - centroid||_2 for all patches
4. Compute mean_dist, std_dist
5. Flag patches where distance > mean_dist + threshold * std_dist
6. Normalize all distances to [0, 1] for heatmap rendering
7. Return outlier patches sorted by distance (descending)
```

### API

```
POST /api/slides/{slide_id}/outlier-detection?threshold=2.0
```

**Response:** `OutlierDetectionResponse` with:
- `outlier_patches`: List of outlier patches with (x, y, distance, z_score)
- `heatmap_data`: All patches with normalized scores for canvas overlay
- Statistics: `mean_distance`, `std_distance`, `outlier_count`, `total_patches`

### Frontend Integration

The `OutlierDetectorPanel` component:
- Configurable z-score threshold (default 2.0)
- Toggle heatmap overlay on/off (renders on dedicated canvas, not OSD)
- Click outlier patches to navigate WSI viewer to that location
- Shows top outliers with z-scores and coordinates

## 10.2 Few-Shot Patch Classifier

The few-shot classifier enables pathologists to define custom tissue classes by selecting example patches, then classifies all patches in the slide. It operates on the active project's foundation model embeddings, making it compatible with any registered foundation model.

### Algorithm

```
1. User defines 2+ classes with example patch indices
   (via text input "1,2,3,10-20" or spatial click selection on viewer)
2. Load foundation model embeddings for example patches
3. Train LogisticRegression(max_iter=1000) on example embeddings
4. Predict all N patches in the slide
5. Compute per-class probabilities
6. Leave-one-out cross-validation for accuracy estimate
7. Return per-patch predictions with confidence and class heatmap data
```

### API

```
POST /api/slides/{slide_id}/patch-classify
{
  "classes": {
    "tumor": [10, 42, 103, 255],
    "stroma": [5, 88, 190, 301]
  }
}
```

**Response:** `PatchClassifyResponse` with:
- `predictions`: Per-patch (patch_idx, x, y, predicted_class, confidence, probabilities)
- `class_counts`: Distribution across classes
- `accuracy_estimate`: Leave-one-out accuracy on training examples
- `heatmap_data`: Per-patch (x, y, class_idx, confidence) for canvas overlay

### Spatial Selection Mode

The frontend supports click-to-select on the WSI viewer:

1. User creates class names in PatchClassifierPanel
2. Clicks "Select on Map" for a class, entering spatial selection mode
3. Clicks on the WSI viewer. The nearest patch (by coordinate) is added to the class
4. Patch coordinates are loaded via `GET /api/slides/{id}/patch-coords`
5. Selected patches highlighted with class-specific colors on the viewer

### Frontend Integration

The `PatchClassifierPanel` component:
- Add/remove classes with auto-assigned colors (8 predefined colors)
- Input patch indices via text or spatial selection
- Parse ranges (e.g., "10-20")
- Run classification and display class distribution
- Toggle class heatmap overlay (canvas-based, separate from OSD heatmap)

---

# 11. Annotation System

## 11.1 Overview

The annotation system provides markup tools for WSI review, persisted to PostgreSQL for multi-session access.

## 11.2 Annotation Types

| Type | Drawing Method | Coordinates |
|---|---|---|
| Circle | Click + drag | {x, y, width, height} |
| Rectangle | Click + drag | {x, y, width, height} |
| Freehand | Click + draw path | {x, y, width, height, points: [{x, y}]} |
| Point | Single click | {x, y, width: 0, height: 0} |

## 11.3 SVG Rendering

Annotations are rendered as an SVG overlay positioned in OpenSeadragon viewport coordinates:

- Drawing preview updates in real-time as the user drags
- Completed annotations are stored with image-space coordinates
- SVG elements are styled with the annotation's color property
- Selected annotations are highlighted with a distinct border
- Annotations remain correctly positioned during zoom/pan

## 11.4 PostgreSQL Persistence

```sql
CREATE TABLE annotations (
    id TEXT PRIMARY KEY,           -- "ann_{uuid12}"
    slide_id TEXT NOT NULL,
    type TEXT NOT NULL,            -- circle, rectangle, freehand, point
    coordinates JSONB NOT NULL,    -- {x, y, width, height, points?}
    label TEXT,
    notes TEXT,
    color TEXT DEFAULT '#3b82f6',
    category TEXT,                 -- mitotic, tumor, stroma, etc.
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);
```

## 11.5 API

Full CRUD operations:
- `GET /api/slides/{id}/annotations` -- List all annotations
- `POST /api/slides/{id}/annotations` -- Create (auto-generates ID)
- `PUT /api/slides/{id}/annotations/{ann_id}` -- Update label/notes/color/category
- `DELETE /api/slides/{id}/annotations/{ann_id}` -- Delete
- `GET /api/slides/{id}/annotations/summary` -- Count by label

All annotation operations are logged to the audit trail.

---

# 12. Heatmap and Overlay System

## 12.1 Attention Heatmaps (OSD Layer)

Attention heatmaps are generated from TransMIL attention weights and rendered as OpenSeadragon overlays:

### Per-Model Heatmaps

Each of the 5 TransMIL models produces its own attention heatmap:
- Endpoint: `GET /api/heatmap/{slide_id}/{model_id}`
- Format: PNG with RGBA (transparent background)
- Resolution: 1 pixel per 224px patch (grid_w x grid_h)
- Alignment: Coverage-based -- `ceil(slide_dim / 224) * 224` ensures the heatmap covers exactly the patch grid area

### Coverage-Based Alignment

The heatmap dimensions are computed to match the coverage area of 224px patches:

```python
grid_w = int(np.ceil(slide_w / patch_size))
grid_h = int(np.ceil(slide_h / patch_size))
```

Response headers include both slide dimensions and coverage dimensions:
```
X-Slide-Width: 80000
X-Slide-Height: 40000
X-Coverage-Width: 80136    # grid_w * 224
X-Coverage-Height: 40096   # grid_h * 224
```

The frontend uses coverage dimensions for positioning the heatmap overlay to ensure pixel-perfect alignment with patch boundaries.

### Disk-Based Caching

Heatmaps are cached to `data/embeddings/heatmap_cache/{slide_id}_{model_id}.png` after first generation. Subsequent requests serve the cached file directly via `FileResponse`.

### Configurable Rendering

The default (non-model-specific) heatmap endpoint supports:
- `level`: 0=2048px, 1=1024px, 2=512px (default), 3=256px, 4=128px
- `smooth`: Gaussian blur (default true)
- `blur`: Kernel size (default 31, odd number)

## 12.2 Canvas-Based Patch Overlays

Outlier detection and few-shot classification produce per-patch data that is rendered on a separate HTML canvas overlaid on the viewer, independent of the OSD heatmap system:

### Outlier Heatmap
- Data: Array of {x, y, score} where score is normalized distance [0, 1]
- Only outlier patches are highlighted (filtered by z-score threshold)
- Color: Warm palette (yellow to red) based on score

### Classifier Heatmap
- Data: Array of {x, y, class_idx, confidence}
- Each class assigned a distinct color from 8 predefined colors
- Opacity modulated by confidence

### PatchOverlay Type
```typescript
interface PatchOverlay {
    data: Array<{ x: number; y: number; score?: number; class_idx?: number; confidence?: number }>;
    type: "outlier" | "classifier";
    classColors?: string[];
}
```

## 12.3 Patch Grid Overlay

A dedicated canvas renders 224px patch boundaries:
- Toggle via toolbar button (Grid icon)
- Configurable opacity (0-1 range via slider)
- Configurable color (default cyan #00ffff)
- Lines drawn at every 224px interval aligned to the patch extraction grid
- Helps pathologists understand which regions map to which patches

---

# 13. Hospital Deployment Guide

This section provides step-by-step instructions for deploying Enso Atlas in a hospital environment. It covers hardware selection, software installation, configuration, slide ingestion, model customization, network security, and database management. A hospital IT team can follow this guide end-to-end to bring Enso Atlas into production on local infrastructure.

---

## 13.1 Hardware Requirements

### Minimum: Mac mini or Desktop Workstation (CPU-only MedGemma, slower reports)

| Component | Specification |
|---|---|
| CPU | Apple M2 / Intel i7-12th gen or newer |
| RAM | 16 GB (unified or DDR5) |
| Storage | 256 GB SSD (slides stored externally or on NAS) |
| GPU | None required (Apple MPS or integrated) |
| OS | macOS 14+ / Ubuntu 22.04+ / Windows WSL2 |

With 16 GB, the system runs Path Foundation (CPU, ~2 GB), TransMIL (5 models, ~200 MB), MedSigLIP (~3 GB), and PostgreSQL (~200 MB). MedGemma 4B (~8 GB bfloat16) will fit but leaves minimal headroom; report generation will be slow (~60-120s). For comfortable operation with MedGemma, 32 GB is recommended.

### Recommended: GPU Workstation

| Component | Specification |
|---|---|
| CPU | AMD Ryzen 9 / Intel i9 / Apple M4 Pro |
| RAM | 32-64 GB |
| Storage | 1 TB NVMe SSD |
| GPU | NVIDIA RTX 4090 (24 GB VRAM) or A6000 (48 GB) |
| OS | Ubuntu 22.04 with NVIDIA Container Toolkit |

With 24 GB VRAM: MedGemma 4B (~8 GB) + MedSigLIP (~800 MB) + TransMIL (~200 MB) = ~9 GB, with headroom remaining. Report generation takes ~20s on GPU.

### Production: DGX Spark or GPU Server

| Component | Specification |
|---|---|
| CPU | ARM64 or x86_64 |
| RAM | 64-128 GB |
| Storage | 2+ TB NVMe (for slide storage) |
| GPU | NVIDIA A100/H100/Blackwell (40-128 GB VRAM) |
| Network | 10 GbE for PACS integration |

### Storage Sizing

| Data Type | Per-Slide Size | 200 Slides | 1,000 Slides |
|---|---|---|---|
| WSI files (.svs) | 1-3 GB | 200-600 GB | 1-3 TB |
| Level 0 embeddings (.npy) | 5-15 MB | 1-3 GB | 5-15 GB |
| Level 1 embeddings (.npy) | 0.1-1 MB | 20-200 MB | 100 MB-1 GB |
| Heatmap cache (.png) | 10-50 KB each | 10-50 MB | 50-250 MB |
| PostgreSQL | ~1 KB per slide | ~1 MB | ~5 MB |
| HuggingFace model cache | N/A | ~15 GB | ~15 GB |

---

## 13.2 Step-by-Step Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/Hilo-Hilo/med-gemma-hackathon.git
cd med-gemma-hackathon
```

### Step 2: Download Foundation Models

Models must be downloaded once (requires internet access and a HuggingFace account with access to gated models). After download, the system runs fully offline.

```bash
# Install huggingface-hub CLI
pip install huggingface-hub

# Log in (required for gated models)
huggingface-cli login

# Download Path Foundation (~500 MB, TensorFlow SavedModel)
huggingface-cli download google/path-foundation --local-dir ~/.cache/huggingface/hub/models--google--path-foundation

# Download MedGemma 4B (~8 GB, PyTorch)
huggingface-cli download google/medgemma-4b-it

# Download MedSigLIP (~3 GB, PyTorch)
huggingface-cli download google/medsiglip-448
# Fallback if MedSigLIP-448 unavailable:
# huggingface-cli download google/siglip-so400m-patch14-384
```

### Step 3a: Docker Deployment (Recommended for GPU Servers)

```bash
# Build the backend container
docker compose -f docker/docker-compose.yaml build

# Start backend + database
docker compose -f docker/docker-compose.yaml up -d

# Monitor startup logs (wait for "Application startup complete", ~2-3 minutes)
docker logs -f enso-atlas

# Verify backend is healthy
curl http://localhost:8003/api/health
```

### Step 3b: Native Deployment (Mac mini / No Docker)

```bash
# Create Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install Python dependencies
pip install -e .

# Install and start PostgreSQL
# macOS:
brew install postgresql@16
brew services start postgresql@16
createdb enso_atlas

# Set environment variables
export DATABASE_URL="postgresql://$(whoami)@localhost:5432/enso_atlas"
export MIL_ARCHITECTURE=transmil
export MIL_THRESHOLD_CONFIG=models/threshold_config.json

# Start the backend
python -m uvicorn enso_atlas.api.main:app --host 0.0.0.0 --port 8003
```

### Step 4: Build and Start the Frontend

```bash
cd frontend
npm install
npm run build
npx next start -p 3002
```

The frontend is now available at `http://localhost:3002`.

### Step 5: Verify the Installation

```bash
# Backend health
curl http://localhost:8003/api/health
# Expected: {"status":"healthy","model_loaded":true,"slides_available":208,"db_available":true}

# Database status
curl http://localhost:8003/api/db/status
# Expected: {"status":"connected","slides":208,"patients":98}

# Available models
curl http://localhost:8003/api/models
# Expected: 5 TransMIL models listed

# Open browser
open http://localhost:3002
```

### Step 6: Enable Air-Gapped Mode (Production)

After confirming the system works, lock it down so it never attempts to download models:

```bash
# In docker-compose.yaml or shell environment:
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
```

Restart the backend. It will now refuse any network model downloads and use only local cache.

---

## 13.3 YAML Configuration Guide

All project configuration lives in `config/projects.yaml`. This file defines foundation models, classification models, and projects. Editing this file customizes Enso Atlas for a specific institution.

### 13.3.1 Foundation Model Registry

The `foundation_models` section registers embedding models. Each model produces a fixed-dimensional embedding vector. Classification models must use a compatible foundation model (matching `embedding_dim`).

```yaml
foundation_models:
  # Google Path Foundation (included)
  path_foundation:
    name: "Path Foundation"
    embedding_dim: 384
    description: "Google Health pathology foundation model (ViT-S, 384-dim)"

  # Additional foundation model example
  dinov2:
    name: "DINOv2"
    embedding_dim: 768
    description: "Meta DINOv2 self-supervised vision transformer (768-dim)"

  uni:
    name: "UNI"
    embedding_dim: 1024
    description: "Mass General pathology foundation model (1024-dim)"

  conch:
    name: "CONCH"
    embedding_dim: 512
    description: "Contrastive learning pathology model (512-dim)"

  # Hospital-specific fine-tuned model
  institutional_vit:
    name: "Institutional ViT"
    embedding_dim: 512
    description: "In-house ViT fine-tuned on institutional H&E slides"
```

**Constraint:** The `embedding_dim` must exactly match the output dimension of the model. All classification models referencing this foundation model must have `compatible_foundation` set to the same key.

### 13.3.2 Classification Model Registry

The `classification_models` section registers TransMIL (or other MIL) models. Each entry points to a directory under `outputs/` containing the trained checkpoint.

```yaml
classification_models:
  # Included model (platinum sensitivity)
  platinum_sensitivity:
    model_dir: "transmil_v2"          # Directory under outputs/
    display_name: "Platinum Sensitivity"
    description: "Predicts response to platinum-based chemotherapy"
    auc: 0.907                        # Validation AUC (displayed in UI)
    n_slides: 199                     # Training set size (displayed in UI)
    category: "ovarian_cancer"        # Groups models in the UI
    positive_label: "Sensitive"       # Label when score > threshold
    negative_label: "Resistant"       # Label when score <= threshold
    compatible_foundation: "path_foundation"  # Must match foundation model

  # Example: adding a custom model
  her2_status:
    model_dir: "transmil_her2"        # Directory to create
    display_name: "HER2 Status"
    description: "Predicts HER2 amplification from H&E"
    auc: 0.82
    n_slides: 500
    category: "breast_cancer"
    positive_label: "HER2+"
    negative_label: "HER2-"
    compatible_foundation: "path_foundation"
```

**Directory structure expected for each model:**

```
outputs/
  transmil_her2/                  # Matches model_dir
    best_model.pt                 # PyTorch checkpoint (TransMIL state_dict)
    config.json                   # Optional: training hyperparameters
    threshold_config.json         # Optional: optimized threshold
    training_curves.png           # Optional: training visualization
```

The checkpoint file must contain either a bare `state_dict` or a dict with a `model_state_dict` key. The TransMIL architecture (input_dim, hidden_dim=512, num_heads=8, num_layers=2) must match the training configuration.

### 13.3.3 Project Configuration

Projects tie together a foundation model, one or more classification models, a dataset, and feature toggles.

```yaml
projects:
  # Existing ovarian cancer project
  ovarian-platinum:
    name: "Ovarian Cancer - Platinum Sensitivity"
    cancer_type: Ovarian Cancer
    prediction_target: platinum_sensitivity
    classes: ["resistant", "sensitive"]
    positive_class: sensitive
    description: "Predicts platinum-based chemotherapy sensitivity in HGSOC"
    foundation_model: path_foundation
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
    classification_models:
      - platinum_sensitivity
      - tumor_grade
      - survival_5y
      - survival_3y
      - survival_1y
    features:
      medgemma_reports: true         # Enable MedGemma report generation
      medsiglip_search: true         # Enable MedSigLIP semantic search
      semantic_search: true          # Enable text-to-patch search UI
    threshold: 0.9229
    threshold_config: models/threshold_config.json

  # Example: adding a new cancer type
  breast-her2:
    name: "Breast Cancer - HER2 Prediction"
    cancer_type: Breast Cancer
    prediction_target: her2_status
    classes: ["negative", "positive"]
    positive_class: positive
    description: "Predicts HER2 amplification status from H&E slides"
    foundation_model: path_foundation
    dataset:
      slides_dir: data/breast/slides
      embeddings_dir: data/breast/embeddings/level0
      labels_file: data/breast/labels.csv
      label_column: her2_status
    models:
      embedder: path-foundation
      mil_architecture: transmil
      mil_checkpoint: outputs/transmil_her2/best_model.pt
      report_generator: medgemma-4b
      semantic_search: medsiglip
    classification_models:
      - her2_status
    features:
      medgemma_reports: true
      medsiglip_search: true
      semantic_search: true
    threshold: 0.5
```

### 13.3.4 Feature Toggles

| Feature | Effect When `false` |
|---|---|
| `medgemma_reports` | Disables MedGemma report generation; system uses template reports only |
| `medsiglip_search` | Disables MedSigLIP semantic search; saves ~800 MB VRAM |
| `semantic_search` | Hides semantic search panel in the frontend UI |

For resource-constrained deployments (e.g., 16 GB Mac mini), disabling `medsiglip_search` saves memory. Template reports can substitute for MedGemma if report generation is too slow without a GPU.

---

## 13.4 Plugin Model System: Adding Custom Models

### 13.4.1 Training a New TransMIL Model

**Prerequisites:**
- Pre-computed embeddings (`.npy` files, one per slide)
- Labels file (CSV with slide_id and binary label)
- Python environment with PyTorch, scikit-learn, numpy

**Step 1: Prepare the labels file.**

Format must include `slide_id` (matching the `.npy` filenames) and a binary label column:

```csv
slide_id,label
SLIDE-001,1
SLIDE-002,0
SLIDE-003,1
```

Or the expanded clinical format:

```csv
patient_id,slide_file,treatment_response,age,sex,stage,grade,histology
PATIENT-001,SLIDE-001.svs,responder,58,F,IIIA,High,Serous
PATIENT-002,SLIDE-002.svs,non-responder,67,F,IV,High,Serous
```

**Step 2: Generate embeddings (if not already done).**

```bash
python scripts/embed_level0_pipelined.py \
  --slides_dir data/breast/slides \
  --output_dir data/breast/embeddings/level0 \
  --patch_size 224 \
  --batch_size 512
```

This produces `{slide_id}.npy` (N x 384) and `{slide_id}_coords.npy` (N x 2) for each slide.

**Step 3: Train the TransMIL model.**

```bash
python scripts/train_transmil.py \
  --embeddings_dir data/breast/embeddings/level0 \
  --labels_file data/breast/labels.csv \
  --output_dir outputs/transmil_her2 \
  --epochs 100 \
  --lr 2e-4 \
  --hidden_dim 512 \
  --num_heads 8 \
  --num_layers 2 \
  --dropout 0.1 \
  --patience 15 \
  --accumulation_steps 4
```

Alternatively, the fine-tuning script supports 5-fold cross-validation:

```bash
python scripts/train_transmil_finetune.py \
  --embeddings_dir data/breast/embeddings/level0 \
  --labels_file data/breast/labels.csv \
  --output_dir outputs/transmil_her2 \
  --n_folds 5 \
  --epochs 100
```

**Training output:**
```
outputs/transmil_her2/
  best_model.pt                # Best checkpoint (by validation AUC)
  config.json                  # Hyperparameters used
  training_curves_fold1.png    # Loss/AUC curves per fold
  summary_metrics.png          # Cross-validation summary
  results.json                 # Full metrics (AUC, sensitivity, specificity)
```

**Step 4: Optimize the classification threshold.**

```bash
python scripts/optimize_threshold.py \
  --embeddings_dir data/breast/embeddings/level0 \
  --labels_file data/breast/labels.csv \
  --model_path outputs/transmil_her2/best_model.pt \
  --output outputs/transmil_her2/threshold_config.json
```

This produces a `threshold_config.json` with the Youden J optimal threshold:
```json
{
  "method": "youden_j",
  "threshold": 0.62,
  "sensitivity": 0.88,
  "specificity": 0.79,
  "auc": 0.82
}
```

### 13.4.2 Registering a New Model

**Step 1:** Add the classification model to `config/projects.yaml` under `classification_models` (see Section 13.3.2).

**Step 2:** Add or update a project that references the new model (see Section 13.3.3).

**Step 3:** Restart the backend. The `MultiModelInference` loader automatically discovers models under `outputs/` matching the `model_dir` field.

**Step 4:** Verify via API:

```bash
curl http://localhost:8003/api/models | python3 -m json.tool
# Should list the new model with its display_name and AUC
```

### 13.4.3 Foundation Model Compatibility

The TransMIL input dimension must match the foundation model's embedding dimension. The default configuration uses Path Foundation (384-dim). To use a different foundation model:

1. Register it in `foundation_models` with the correct `embedding_dim`
2. Set `compatible_foundation` in the classification model entry
3. Modify the `--hidden_dim` in training if needed (the TransMIL input projection adapts automatically to `input_dim`)
4. Re-embed all slides with the new foundation model before training

**Supported foundation models and their dimensions:**

| Model | embedding_dim | Framework | Notes |
|---|---|---|---|
| Path Foundation | 384 | TensorFlow | Default, runs on CPU |
| DINOv2-Large | 768 | PyTorch | Requires GPU for embedding |
| UNI | 1024 | PyTorch | Mass General model |
| CONCH | 512 | PyTorch | Contrastive learning model |
| Custom ViT | varies | varies | Must output fixed-dim vectors |

Changing foundation models requires re-embedding all slides and retraining all classification models. The system does not support mixing embeddings from different foundation models within a single project.

---

## 13.5 Slide Ingestion

### 13.5.1 Supported Formats

| Format | Extension | Scanner |
|---|---|---|
| Aperio | .svs | Leica/Aperio |
| TIFF | .tiff, .tif | Various |
| Hamamatsu | .ndpi | Hamamatsu |
| MIRAX | .mrxs | 3DHISTECH |
| Ventana | .vms | Ventana |
| Leica | .scn | Leica |

### 13.5.2 Adding Slides

**Option A: Copy files to the slides directory.**

```bash
# Copy slides to the project's slides directory
cp /path/to/new_slides/*.svs data/tcga_full/slides/

# Trigger database repopulation to pick up new slides
curl -X POST http://localhost:8003/api/db/repopulate
```

**Option B: Upload via the API.**

```bash
curl -X POST http://localhost:8003/api/projects/ovarian-platinum/upload \
  -F "file=@/path/to/SLIDE-001.svs"
```

**Option C: Upload via the frontend.**

Navigate to the Projects page, select the target project, and use the drag-and-drop slide upload interface.

### 13.5.3 Generating Embeddings for New Slides

After adding slides, generate embeddings:

**Option A: Via the API (per-slide, with progress tracking).**

```bash
# Start background embedding (level 0 = full resolution)
curl -X POST http://localhost:8003/api/embed-slide \
  -H "Content-Type: application/json" \
  -d '{"slide_id": "SLIDE-001", "level": 0}'

# Response:
# {"status":"started","task_id":"abc123","estimated_time_minutes":15}

# Poll for progress
curl http://localhost:8003/api/embed-slide/status/abc123
# {"status":"running","progress":55,"message":"Generating embeddings: 3000/6000 patches"}
```

**Option B: Batch embedding via the API.**

```bash
curl -X POST http://localhost:8003/api/embed-slides/batch \
  -H "Content-Type: application/json" \
  -d '{"level": 0, "force": false, "concurrency": 1}'

# Embeds all slides that don't yet have embeddings
```

**Option C: Batch embedding via command line (fastest).**

```bash
python scripts/embed_level0_pipelined.py \
  --slides_dir data/tcga_full/slides \
  --output_dir data/embeddings/level0 \
  --batch_size 512
```

### 13.5.4 Adding Labels

Create or update `data/labels.csv` (or the project-specific labels file):

```csv
patient_id,slide_file,treatment_response,age,sex,stage,grade,histology
NEW-001,SLIDE-001.svs,responder,55,F,IIB,High,Serous
NEW-002,SLIDE-002.svs,non-responder,63,F,IIIC,High,Serous
```

Then repopulate the database:

```bash
curl -X POST http://localhost:8003/api/db/repopulate
```

---

## 13.6 Docker Compose Configuration

### 13.6.1 Production docker-compose.yaml

Adapt the volume mounts to the local directory structure:

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
      DATABASE_URL: "postgresql://enso:${POSTGRES_PASSWORD}@atlas-db:5432/enso_atlas"
      TRANSFORMERS_OFFLINE: "1"      # Air-gapped: no model downloads
      HF_HUB_OFFLINE: "1"
    volumes:
      # EDIT THESE PATHS for the local installation:
      - /data/enso-atlas/data:/app/data                 # Slides + embeddings
      - /data/enso-atlas/outputs:/app/outputs           # Trained model checkpoints
      - /data/enso-atlas/huggingface:/root/.cache/huggingface  # Pre-downloaded models
      - /data/enso-atlas/models:/app/models             # TransMIL default checkpoints
      - /data/enso-atlas/config:/app/config             # projects.yaml
    ports:
      - "8003:8000"                   # API
    restart: unless-stopped
    depends_on:
      atlas-db:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 180s             # MedGemma warmup takes ~2-3 min

  atlas-db:
    image: postgres:16
    container_name: atlas-db
    environment:
      POSTGRES_DB: enso_atlas
      POSTGRES_USER: enso
      POSTGRES_PASSWORD: "${POSTGRES_PASSWORD:-change_me_in_production}"
    volumes:
      - atlas-pgdata:/var/lib/postgresql/data
    ports:
      - "5433:5432"
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U enso -d enso_atlas"]
      interval: 5s
      timeout: 3s
      retries: 10
      start_period: 10s

volumes:
  atlas-pgdata:
    driver: local
```

### 13.6.2 Environment Variables Reference

| Variable | Required | Default | Description |
|---|---|---|---|
| `DATABASE_URL` | Yes | `postgresql://enso:...@atlas-db:5432/enso_atlas` | PostgreSQL connection string |
| `POSTGRES_PASSWORD` | Yes | `enso_atlas_2024` | Change in production |
| `MIL_ARCHITECTURE` | No | `transmil` | MIL model type |
| `MIL_THRESHOLD_CONFIG` | No | `models/threshold_config.json` | Threshold config path |
| `MIL_MODEL_PATH` | No | (auto-detected) | Override default model checkpoint |
| `TRANSFORMERS_OFFLINE` | No | `0` | Set to `1` for air-gapped deployment |
| `HF_HUB_OFFLINE` | No | `0` | Set to `1` for air-gapped deployment |
| `NVIDIA_VISIBLE_DEVICES` | No | `all` | GPU device selection |
| `NEXT_PUBLIC_API_URL` | No | `http://localhost:8003` | Frontend API URL |
| `EMBEDDINGS_DIR` | No | `data/embeddings` | Override embeddings directory |

### 13.6.3 Volume Mounts

| Container Path | Purpose | Mode |
|---|---|---|
| `/app/data` | Slides (.svs), embeddings (.npy), labels (.csv), heatmap cache | rw |
| `/app/outputs` | Trained TransMIL model checkpoints | rw |
| `/root/.cache/huggingface` | Pre-downloaded HuggingFace models (Path Foundation, MedGemma, MedSigLIP) | ro |
| `/app/models` | Default TransMIL checkpoint and threshold config | ro |
| `/app/config` | `projects.yaml` and `default.yaml` | ro |

### 13.6.4 Without GPU (Mac mini / CPU-only)

Remove the `deploy.resources` GPU section from docker-compose.yaml. PyTorch automatically falls back to CPU for TransMIL. MedGemma is significantly slower on CPU (~60-120s per report instead of ~20s). For Mac mini, native deployment (Step 3b) is preferable over Docker for access to Apple MPS acceleration.

---

## 13.7 Network and Security

### 13.7.1 Network Architecture

Enso Atlas requires no inbound internet access at runtime. All models are pre-downloaded.

```
+---------------------------------------------------------------+
|  Hospital Network                                             |
|                                                               |
|  +-------------------+      +----------------------------+   |
|  |  PACS / LIS       |----->|  Enso Atlas Server         |   |
|  |  (Slide Export)    |      |  Backend:  localhost:8003  |   |
|  +-------------------+      |  Frontend: localhost:3002  |   |
|                              |  Database: localhost:5433  |   |
|  +-------------------+      +----------------------------+   |
|  |  Clinician        |<---->|           ^                    |
|  |  Workstation      |      |           |                    |
|  +-------------------+      +-----------+                    |
|                                                               |
+---------------------------------------------------------------+
            |
            X  No outbound data transfer
            |
```

### 13.7.2 Firewall Configuration

**Internal access only (default):** No ports need to be exposed to the internet.

| Port | Service | Access |
|---|---|---|
| 3002 | Frontend (Next.js) | Hospital LAN only |
| 8003 | Backend API | Hospital LAN only |
| 5433 | PostgreSQL | localhost only (or hospital LAN) |

**Restrict PostgreSQL** to localhost if the database runs on the same machine:

```yaml
# In docker-compose.yaml, change:
ports:
  - "127.0.0.1:5433:5432"  # Bind to localhost only
```

### 13.7.3 Tailscale Funnel (Optional Remote Access)

For secure remote access without opening firewall ports or configuring VPNs:

```bash
# Install Tailscale on the server
curl -fsSL https://tailscale.com/install.sh | sh
tailscale up

# Expose the frontend publicly via HTTPS
tailscale funnel 3002
# Now accessible at https://your-server.ts.net/
```

Tailscale Funnel provides:
- Automatic TLS certificates (no Let's Encrypt configuration)
- Zero-config NAT traversal (works behind hospital firewalls)
- Access control via Tailscale ACLs (restrict to specific users)
- End-to-end encryption

The backend API (port 8003) is NOT exposed via Funnel, only the frontend. API access is mediated through the Next.js proxy.

### 13.7.4 HIPAA Compliance Notes

Enso Atlas is designed for HIPAA alignment, though formal certification is the hospital's responsibility:

| HIPAA Requirement | Enso Atlas Implementation |
|---|---|
| Access controls | Network-perimeter authentication; no built-in auth (delegate to hospital SSO/VPN) |
| Audit trail | All analyses, reports, annotations, and exports logged with timestamps |
| Data encryption at rest | Delegate to OS-level disk encryption (BitLocker, FileVault, LUKS) |
| Data encryption in transit | HTTPS via Tailscale Funnel or reverse proxy; PostgreSQL SSL configurable |
| Minimum necessary | Project-scoped slide/model access; no unnecessary data exposure |
| Backup and recovery | PostgreSQL backup procedures (see Section 13.8) |
| No cloud transmission | `TRANSFORMERS_OFFLINE=1` enforced; no telemetry; no outbound connections |

**Recommendation:** Run Enso Atlas on an encrypted volume. Place it behind the institution's existing authentication proxy (e.g., OAuth2 proxy, Shibboleth). The audit log (`GET /api/audit-log`) provides a compliance-ready event stream.

---

## 13.8 Database Management

### 13.8.1 PostgreSQL Setup

The Docker Compose configuration includes a PostgreSQL 16 container that starts automatically. For native deployment:

```bash
# macOS
brew install postgresql@16
brew services start postgresql@16
createuser enso
createdb -O enso enso_atlas

# Ubuntu
sudo apt install postgresql-16
sudo -u postgres createuser enso
sudo -u postgres createdb -O enso enso_atlas
sudo -u postgres psql -c "ALTER USER enso PASSWORD 'your_secure_password';"
```

### 13.8.2 Schema Initialization

The schema is created automatically on first startup. The backend runs migrations through v5:

| Version | Tables Created/Modified |
|---|---|
| v1 | patients, slides, slide_metadata, analysis_results, embedding_tasks, projects, schema_version |
| v2 | Added `project_id` column to slides |
| v3 | project_models and project_slides junction tables |
| v4 | Added `display_name` column to slides |
| v5 | annotations table |

Migrations are idempotent and safe to re-run.

### 13.8.3 Backup and Restore

**Backup (run regularly via cron):**

```bash
# Docker deployment
docker exec atlas-db pg_dump -U enso enso_atlas > enso_atlas_backup_$(date +%Y%m%d).sql

# Native deployment
pg_dump -U enso enso_atlas > enso_atlas_backup_$(date +%Y%m%d).sql
```

**Restore from backup:**

```bash
# Docker deployment
docker exec -i atlas-db psql -U enso enso_atlas < enso_atlas_backup_20250210.sql

# Native deployment
psql -U enso enso_atlas < enso_atlas_backup_20250210.sql
```

**Automated daily backup (cron):**

```bash
# Add to crontab -e
0 2 * * * docker exec atlas-db pg_dump -U enso enso_atlas | gzip > /backups/enso_atlas_$(date +\%Y\%m\%d).sql.gz
# Keep last 30 days
0 3 * * * find /backups -name "enso_atlas_*.sql.gz" -mtime +30 -delete
```

### 13.8.4 Database Repopulation

If slides, labels, or embeddings are modified outside the API:

```bash
curl -X POST http://localhost:8003/api/db/repopulate
# Re-reads all CSV, .npy, and .svs files and updates the database
```

This is safe to run at any time -- it uses `ON CONFLICT DO UPDATE` for idempotency.

### 13.8.5 Changing the Database Password

**Step 1:** Update the PostgreSQL password:
```bash
docker exec -it atlas-db psql -U enso -c "ALTER USER enso PASSWORD 'new_secure_password';"
```

**Step 2:** Update `DATABASE_URL` in docker-compose.yaml:
```yaml
DATABASE_URL: "postgresql://enso:new_secure_password@atlas-db:5432/enso_atlas"
```

**Step 3:** Restart the backend:
```bash
docker compose -f docker/docker-compose.yaml restart enso-atlas
```

---

## 13.9 Post-Deployment Checklist

Checklist for verifying a complete hospital deployment:

```
[ ] Hardware meets minimum requirements (16 GB RAM, 256 GB SSD)
[ ] Repository cloned and dependencies installed
[ ] HuggingFace models downloaded (Path Foundation, MedGemma, MedSigLIP)
[ ] PostgreSQL running and accessible
[ ] Backend starts and passes health check (curl /api/health)
[ ] Frontend builds and loads in browser (http://localhost:3002)
[ ] At least one project configured in config/projects.yaml
[ ] Slide files copied to the configured slides_dir
[ ] Embeddings generated for all slides (check /api/slides for has_embeddings=true)
[ ] Labels file populated with clinical data
[ ] Database populated (curl /api/db/status shows correct slide count)
[ ] Multi-model analysis works (test with one slide in the UI)
[ ] Heatmap overlay renders correctly on the WSI viewer
[ ] PDF report generation works (try "Generate Report" button)
[ ] Air-gapped mode enabled (TRANSFORMERS_OFFLINE=1)
[ ] Database backup scheduled (cron job)
[ ] Firewall rules restrict access to hospital LAN
[ ] PostgreSQL password changed from default
[ ] Disk encryption enabled on the host
```

---

## 13.10 Upgrading

To upgrade Enso Atlas to a new version:

```bash
# Pull latest code
cd med-gemma-hackathon
git pull origin main

# Rebuild backend container
docker compose -f docker/docker-compose.yaml build enso-atlas
docker compose -f docker/docker-compose.yaml up -d enso-atlas

# Rebuild frontend
cd frontend
npm install
npm run build
npx next start -p 3002
```

Database migrations run automatically on startup. Data, models, and configuration are preserved (they live in mounted volumes, not inside the container).

For native deployments:

```bash
git pull origin main
pip install -e .  # Update Python dependencies
cd frontend && npm install && npm run build  # Update frontend
# Restart the backend process
```

## 13.6 Hospital Deployment Walkthrough

This section provides a step-by-step guide for hospital IT teams to deploy Enso Atlas on local infrastructure with local data and models.

### Step 1: Hardware Selection

| Configuration | RAM | GPU | Use Case |
|---|---|---|---|
| Mac mini M2/M4 (minimum) | 16 GB | Integrated (MPS) | Small cohorts (<500 slides), CPU embedding |
| Mac mini M4 Pro/Max | 32-64 GB | Integrated (MPS) | Medium cohorts, faster MedGemma inference |
| Linux workstation + A40/L40 | 32+ GB + 48 GB VRAM | NVIDIA CUDA | Full performance, GPU embedding |
| NVIDIA DGX Spark | 128 GB unified | Blackwell (limited TF support) | Reference deployment |

### Step 2: Install and Start

```bash
git clone https://github.com/Hilo-Hilo/med-gemma-hackathon.git
cd med-gemma-hackathon/docker
docker compose up -d
```

Wait approximately 3-4 minutes for model loading and warmup. Verify with:

```bash
curl http://localhost:8003/api/health
```

### Step 3: Add a New Cancer Type

Edit `config/projects.yaml` to add a new project. Example for lung adenocarcinoma:

```yaml
foundation_models:
  path_foundation:
    name: "Path Foundation"
    embedding_dim: 384
    description: "Google Health pathology foundation model"
  # Additional foundation model:
  uni_v2:
    name: "UNI v2"
    embedding_dim: 1024
    description: "Mass General Brigham universal pathology encoder"

classification_models:
  # Custom trained model:
  egfr_mutation:
    model_dir: "egfr_transmil_v1"
    display_name: "EGFR Mutation Prediction"
    auc: 0.83
    n_slides: 450
    category: "lung_cancer"
    positive_label: "EGFR Mutant"
    negative_label: "Wild Type"
    compatible_foundation: "path_foundation"
    embedding_dim: 384

projects:
  # Keep existing demo project:
  ovarian-platinum:
    # ... (unchanged)

  # New project:
  lung-egfr:
    name: "Lung Adenocarcinoma - EGFR Mutation"
    cancer_type: "Lung Cancer"
    prediction_target: egfr_mutation
    classes: ["wild_type", "mutant"]
    positive_class: mutant
    foundation_model: path_foundation
    classification_models:
      - egfr_mutation
    features:
      medgemma_reports: true
      medsiglip_search: false    # Disable if not needed
      semantic_search: false
    threshold: 0.5               # Adjust after validation
```

### Step 4: Train a Classification Model

```bash
# 1. Place WSI files in a slides directory
mkdir -p /data/lung_slides

# 2. Extract patches and generate embeddings
python scripts/embed_level0_pipelined.py \
  --slides-dir /data/lung_slides \
  --output-dir /data/lung_embeddings \
  --batch-size 512 --threads 32

# 3. Prepare labels CSV (slide_id, label columns)
# Format: slide_id,egfr_status
# Example: TCGA-XX-XXXX.svs,mutant

# 4. Train TransMIL
python scripts/train_transmil.py \
  --embeddings-dir /data/lung_embeddings \
  --labels /data/lung_labels.csv \
  --output-dir models/egfr_transmil_v1 \
  --target-column egfr_status \
  --positive-class mutant

# 5. Copy weights to the models directory
cp models/egfr_transmil_v1/best_model.pt models/egfr_transmil_v1/

# 6. Restart the server
cd docker && docker compose restart enso-atlas
```

### Step 5: Swap Foundation Models

To use a different foundation model (e.g., UNI instead of Path Foundation), follow these steps:

1. Register the model in `foundation_models` section of `projects.yaml` with the correct `embedding_dim`
2. Implement the embedding interface in `src/enso_atlas/embeddings/` (a function that takes image patches and returns vectors)
3. Update the project `foundation_model` field to reference the new model
4. Re-embed slides using the new model (existing embeddings are model-specific)
5. Re-train classification heads on the new embeddings (embedding dimensions must match)

The classification models specify `compatible_foundation` and `embedding_dim` to prevent mismatched embeddings from being used.

### Step 6: Configure Feature Toggles

Each project independently controls which features are available:

```yaml
features:
  medgemma_reports: true    # AI-generated clinical reports (requires MedGemma)
  medsiglip_search: true    # Semantic text-to-patch search (requires MedSigLIP)
  semantic_search: true     # Similar case retrieval (requires FAISS index)
```

Setting any feature to `false` disables it for that project. This is useful when:
- MedGemma or MedSigLIP are not installed (e.g., CPU-only deployment)
- Certain features are not relevant for the cancer type
- A simplified interface is needed for specific user groups

### Step 7: Network and Security

- **Air-gapped deployment:** The system runs entirely on-premise. No external API calls are made during inference.
- **Tailscale (optional):** For secure remote access without opening firewall ports: `tailscale funnel 3002`
- **Reverse proxy:** Place behind nginx or Caddy for TLS termination and authentication integration with the hospital SSO (SAML, LDAP)
- **Database backups:** PostgreSQL data is stored in the Docker volume. Schedule `pg_dump` via cron for backup.

```bash
# Backup database
docker exec atlas-db pg_dump -U postgres enso_atlas > backup_$(date +%Y%m%d).sql

# Restore
cat backup_20260210.sql | docker exec -i atlas-db psql -U postgres enso_atlas
```

---

# 14. Data Pipeline

## 14.1 TCGA Data

Training data: 208 TCGA ovarian cancer whole-slide images with platinum sensitivity labels, survival data, and demographics.

## 14.2 Patch Extraction

The pipelined extraction script (`scripts/embed_level0_pipelined.py`) uses producer/consumer:

```
CPU Thread                  GPU Thread
(extract patches,    --->   (Path Foundation embed,
 tissue mask,                save .npy files)
 batch to queue)
```

### Tissue Masking

```python
# Relaxed thresholds for maximum tissue coverage
if mean_val < 245 and std_val > 10:
    patches.append(patch_array)
```

### Output Files

- `{slide_id}.npy` -- Embedding array (N x 384)
- `{slide_id}_coords.npy` -- Coordinate array (N x 2) with (x, y) at level 0

## 14.3 Multi-Level Embeddings

| Level | Resolution | Typical Patches | Use Case |
|---|---|---|---|
| Level 0 | Full resolution (40x) | 3,000-8,000 | Production analysis |
| Level 1 | Downsampled | 100-500 | Quick preview, development |

The batch analysis panel allows users to select embedding level and force re-embedding.

---

# 15. Security and Compliance

## 15.1 On-Premise Design

No data leaves the hospital network:

- **No cloud dependencies at runtime:** `TRANSFORMERS_OFFLINE=1` and `HF_HUB_OFFLINE=1` enforce air-gapped operation
- **No telemetry:** All analytics disabled
- **Local database:** PostgreSQL co-located Docker container
- **Local storage:** All embeddings, slides, and model weights on local volumes

```
+-------------------------------------------------------+
|  Hospital Network (Air-Gapped)                        |
|                                                       |
|  +-------------------+    +----------------------+    |
|  |  PACS / LIS       |--->|  Enso Atlas Server  |    |
|  |  (Slide Export)    |    |  - API (8003)       |    |
|  +-------------------+    |  - DB (5433)         |    |
|                           |  - Frontend (3002)   |    |
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

## 15.2 HIPAA Considerations

- **Audit Trail:** All analyses, reports, annotations, and exports logged with timestamps and user IDs
- **No PHI in Logs:** Logging configured to avoid patient-identifiable information
- **Access Control:** Runs within hospital network; authentication delegated to network perimeter
- **Data Encryption:** PostgreSQL SSL configurable; disk encryption recommended at OS level

### Audit Event Types

| Event Type | Trigger |
|---|---|
| `analysis_completed` | Slide analysis |
| `multi_model_analysis` | Multi-model ensemble |
| `batch_analysis_slide` | Batch slide analysis |
| `batch_analysis_async_completed` | Async batch completed |
| `uncertainty_analysis_completed` | MC Dropout analysis |
| `report_generated` | MedGemma report |
| `pdf_exported` | PDF export |
| `annotation_created` | Annotation created |
| `annotation_deleted` | Annotation deleted |
| `visual_search` | Image-to-image search |
| `patch_classification` | Few-shot classification |
| `outlier_detection` | Outlier tissue detection |

## 15.3 Research-Only Disclaimers

Every report includes mandatory safety statements. The PDF export includes a header warning: "UNCALIBRATED MODEL - RESEARCH USE ONLY". Prohibited phrase validation prevents the system from making direct treatment recommendations.

---

# 16. Performance Benchmarks

## 16.1 Startup Time

| Component | Duration |
|---|---|
| TransMIL Loading (5 models) | 2.1s |
| MedGemma Loading + Warmup | ~90s |
| MedSigLIP Loading | 9.7s |
| FAISS Index Build (208 slides) | 3.2s |
| PostgreSQL Init | 4.8s |
| **Total** | **~120s** |

## 16.2 Per-Slide Analysis

| Operation | Latency |
|---|---|
| Load embeddings (.npy) | 5-15ms |
| TransMIL single model | 20-50ms |
| Multi-model (5 models) | 100-250ms |
| Evidence patch selection (top-8) | <1ms |
| FAISS similar case search | 2-5ms |
| Heatmap generation (first, per-model) | 50-100ms |
| Heatmap serving (cached) | <5ms |
| Semantic search (5 queries) | 100-300ms |
| Outlier detection | 10-50ms |
| Few-shot classification | 50-200ms |
| MedGemma report | ~20s (GPU) |
| **Full analysis (no report)** | **~200ms** |
| **Full analysis (with report)** | **10-120s** |

## 16.3 GPU Memory Usage

| Model | VRAM | Precision |
|---|---|---|
| MedGemma 4B | ~8 GB | bfloat16 |
| MedSigLIP | ~800 MB | fp16 |
| TransMIL (5 models) | ~200 MB | float32 |
| FAISS Index | CPU only | float32 |
| **Total GPU** | **~9 GB** | |

## 16.4 Database Performance

| Query | Response Time |
|---|---|
| List 208 slides (PostgreSQL) | 2-5ms |
| List 208 slides (flat-file fallback) | 30-60s |
| Cached result lookup | 0.8ms |
| Annotation CRUD | <2ms |

## 16.5 TransMIL Model Performance

| Metric | Value |
|---|---|
| Best single-model AUC (optimized threshold) | 0.879 |
| Full-dataset AUC (platinum sensitivity) | 0.907 |
| Optimized threshold (Youden J) | 0.9229 |
| Sensitivity | 84.6% |
| Specificity | 91.2% |
| Cached result lookup | 0.8ms |

---

# 17. Mac Mini Deployment Feasibility

## 17.1 Overview

Enso Atlas supports on-premise deployment on consumer hardware. A modern Mac mini (M4 Pro/Max with 16-32 GB unified memory) can run the complete system:

## 17.2 Model Memory Requirements

| Model | Memory Required | Notes |
|---|---|---|
| Path Foundation | ~2 GB | TensorFlow on CPU -- works natively on Apple Silicon |
| TransMIL (5 models) | ~200 MB | PyTorch MPS or CPU, 3.2M params each |
| MedGemma 4B | ~8 GB (bfloat16) | Fits in 16 GB unified memory via MPS |
| MedSigLIP | ~3 GB | SigLIP-so400m full model |
| FAISS Index | ~50 MB | 208 slides, CPU-only |
| PostgreSQL | ~200 MB | Standard footprint |
| **Total** | **~13.5 GB** | Fits in 16 GB Mac mini with headroom |

## 17.3 Performance Expectations

| Operation | DGX Spark (Blackwell) | Mac mini (M4 Pro) |
|---|---|---|
| Multi-model inference (5 models) | 100-250ms | 500ms-2s (MPS) |
| MedGemma report generation | ~20s | 30-60s (MPS) |
| Path Foundation embedding (64 patches) | 2.5s (CPU) | 3-5s (CPU) |
| Heatmap generation | 50-100ms | 100-200ms |
| PostgreSQL queries | 2-5ms | 2-5ms |

## 17.4 Deployment Configuration

For Mac mini deployment:
- Use `device: "mps"` for PyTorch models (MedGemma, TransMIL)
- Path Foundation remains on CPU (TensorFlow)
- PostgreSQL via Homebrew or Docker
- Frontend via Node.js (native ARM64)
- No Docker GPU passthrough needed -- run directly

## 17.5 Comparison

- **Cost:** Mac mini M4 Pro (~$1,600) vs DGX Spark (~$3,999+)
- **Noise:** Silent operation, suitable for clinical environments
- **Power:** ~40W vs ~500W
- **Size:** Desktop form factor
- **Maintenance:** macOS with automatic updates
- **Data locality:** Data does not leave the device

---

# 18. Known Limitations and Future Work

## 18.1 Current Limitations

### Class Imbalance
139 non-responders vs 13 responders (10.7:1 ratio). Mitigated with weighted loss and Youden J threshold (0.9229).

### Path Foundation CPU-Only
TensorFlow does not support Blackwell GPUs. Embeddings are pre-computed offline.

### Single Cancer Type in Demo
The demo covers HGSOC platinum sensitivity only. The config-driven project system supports additional cancer types pending training data.

### No FDA Clearance
Research tool only. Predictions are uncalibrated raw sigmoid outputs.

### Semantic Search Cold Start
First MedSigLIP embedding run for a slide takes ~12-14 minutes for ~6,680 patches. Cached after first run.

## 18.2 Planned Improvements

### Short-Term (3-6 months)
- Multi-fold ensemble averaging
- Platt scaling for probability calibration
- GPU-accelerated real-time embedding pipeline
- User authentication and RBAC
- DICOM integration for PACS slide import

### Medium-Term (6-12 months)
- Additional cancer types (lung, breast, colorectal)
- Multimodal integration (H&E + IHC)
- Federated learning
- Active learning for efficient annotation
- Real-time collaborative annotation

### Long-Term (12+ months)
- FDA 510(k) regulatory pathway
- EHR integration
- Prospective clinical validation
- Foundation model fine-tuning

---

# 19. Appendices

## Appendix A: Environment Variables

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | `postgresql://enso:enso_atlas_2024@atlas-db:5432/enso_atlas` | PostgreSQL connection |
| `MIL_ARCHITECTURE` | `transmil` | MIL model type |
| `MIL_THRESHOLD` | (from config) | Classification threshold |
| `MIL_THRESHOLD_CONFIG` | `models/threshold_config.json` | Threshold config |
| `MIL_MODEL_PATH` | (auto) | Model checkpoint path |
| `TRANSFORMERS_OFFLINE` | `0` | Air-gapped mode |
| `HF_HUB_OFFLINE` | `0` | Air-gapped mode |
| `HF_HOME` | `/app/cache/huggingface` | HuggingFace cache |
| `ENSO_CONFIG` | `/app/config/default.yaml` | Config file |
| `NVIDIA_VISIBLE_DEVICES` | `all` | GPU selection |
| `NEXT_PUBLIC_API_URL` | `http://localhost:8003` | Frontend API URL |

## Appendix B: File Tree

```
med-gemma-hackathon/
+-- config/
|   +-- default.yaml
|   +-- projects.yaml                    # Project + model definitions
+-- docker/
|   +-- Dockerfile
|   +-- docker-compose.yaml
+-- frontend/
|   +-- next.config.mjs
|   +-- tailwind.config.ts
|   +-- src/
|       +-- app/
|       |   +-- layout.tsx
|       |   +-- page.tsx                 # Main page (2142 lines)
|       |   +-- slides/page.tsx          # Slide manager
|       +-- components/
|       |   +-- layout/
|       |   |   +-- Header.tsx           # Project switcher, view mode, dark mode
|       |   |   +-- Footer.tsx
|       |   |   +-- UserDropdown.tsx
|       |   +-- panels/
|       |   |   +-- AIAssistantPanel.tsx        (38,056 bytes)
|       |   |   +-- AnalysisControls.tsx
|       |   |   +-- BatchAnalysisPanel.tsx      (43,366 bytes)
|       |   |   +-- CaseNotesPanel.tsx
|       |   |   +-- EvidencePanel.tsx
|       |   |   +-- ModelPicker.tsx
|       |   |   +-- MultiModelPredictionPanel.tsx
|       |   |   +-- OncologistSummaryView.tsx
|       |   |   +-- OutlierDetectorPanel.tsx    (11,988 bytes)
|       |   |   +-- PatchClassifierPanel.tsx    (16,135 bytes)
|       |   |   +-- PathologistView.tsx         (30,114 bytes)
|       |   |   +-- PredictionPanel.tsx
|       |   |   +-- QuickStatsPanel.tsx
|       |   |   +-- ReportPanel.tsx
|       |   |   +-- SemanticSearchPanel.tsx
|       |   |   +-- SimilarCasesPanel.tsx
|       |   |   +-- SlideSelector.tsx           (25,527 bytes)
|       |   |   +-- UncertaintyPanel.tsx
|       |   +-- viewer/
|       |   |   +-- WSIViewer.tsx               (61,161 bytes)
|       |   +-- slides/
|       |   |   +-- FilterPanel.tsx
|       |   |   +-- SlideGrid.tsx
|       |   |   +-- SlideTable.tsx
|       |   |   +-- BulkActions.tsx
|       |   |   +-- SlideModals.tsx
|       |   +-- modals/
|       |   |   +-- PatchZoomModal.tsx
|       |   |   +-- KeyboardShortcutsModal.tsx
|       |   |   +-- SettingsModal.tsx
|       |   |   +-- SystemStatusModal.tsx
|       |   +-- demo/
|       |   |   +-- DemoMode.tsx
|       |   +-- ui/
|       |       +-- Badge, Button, Card, Logo, NetworkStatus,
|       |           PredictionGauge, ProgressStepper, Skeleton,
|       |           Slider, Spinner, Toast, Toggle
|       +-- contexts/
|       |   +-- ProjectContext.tsx
|       +-- hooks/
|       |   +-- useAnalysis.ts
|       |   +-- useKeyboardShortcuts.ts
|       +-- lib/
|       |   +-- api.ts                         (2,625 lines)
|       |   +-- pdfExport.ts
|       |   +-- utils.ts
|       +-- types/
|           +-- index.ts                       (561 lines)
+-- models/
|   +-- transmil.py
|   +-- transmil_best.pt
|   +-- threshold_config.json
+-- scripts/
|   +-- train_transmil.py
|   +-- embed_level0_pipelined.py
|   +-- multi_model_inference.py
|   +-- optimize_threshold.py
|   +-- download_ovarian_fixed.py
+-- src/enso_atlas/
|   +-- api/
|   |   +-- main.py                        (6,119 lines)
|   |   +-- database.py                    (1,234 lines)
|   |   +-- projects.py
|   |   +-- project_routes.py
|   |   +-- batch_tasks.py
|   |   +-- batch_embed_tasks.py
|   |   +-- report_tasks.py
|   |   +-- embedding_tasks.py
|   |   +-- pdf_export.py
|   |   +-- slide_metadata.py
|   +-- agent/
|   |   +-- workflow.py                    (1,227 lines)
|   |   +-- routes.py
|   +-- llm/
|   |   +-- chat.py                        (RAG-based chat)
|   +-- embedding/
|   |   +-- embedder_pathfound.py
|   |   +-- medsiglip.py
|   |   +-- embedder.py
|   +-- evidence/
|   |   +-- generator.py
|   +-- mil/
|   |   +-- clam.py
|   +-- reporting/
|   |   +-- medgemma.py                    (908 lines)
|   |   +-- decision_support.py
|   +-- config.py
|   +-- wsi/
|       +-- processor.py
+-- data/
|   +-- labels.csv
|   +-- embeddings/
|   |   +-- level0/                        (208 x .npy files)
|   |   +-- heatmap_cache/                 (cached PNG heatmaps)
|   |   +-- medsiglip_cache/               (cached MedSigLIP embeddings)
|   |   +-- thumbnail_cache/               (cached slide thumbnails)
|   +-- tcga_full/
|       +-- slides/                        (208 .svs files)
|       +-- labels.csv
+-- outputs/                               (TransMIL training outputs)
+-- tests/
+-- docs/
    +-- screenshots/
    +-- reproduce.md
```

## Appendix C: Key Design Decisions

### 1. Raw SQL over ORM
asyncpg with raw SQL for maximum performance and query transparency. No migration framework for small schema.

### 2. Coverage-Based Heatmap Alignment
Heatmap dimensions use `ceil(dim/224)*224` to ensure pixel-perfect alignment with the 224px patch grid. Coverage dimensions are returned as response headers.

### 3. Disk-Based Heatmap Cache
Per-model heatmaps cached to PNG files. First generation takes 50-100ms; subsequent requests serve cached files in <5ms.

### 4. Canvas-Based Overlays vs OSD Overlays
Outlier and classifier heatmaps use a separate HTML canvas rather than OSD's `addSimpleImage`. This allows independent toggle/opacity control and avoids conflicts with the attention heatmap overlay.

### 5. Imperative Scale Bar Updates
Direct DOM manipulation for the scale bar bypasses React's reconciliation on every animation frame, preventing jank during continuous zoom/pan.

### 6. Template Report Fallback
When MedGemma fails or times out, a template-based report preserving real attention weights and coordinates ensures users always receive actionable output.

### 7. FAISS IndexFlatIP for Similar Cases
Exact inner product search on L2-normalized mean embeddings (cosine similarity). Acceptable for 208 slides; switch to IndexIVF at ~10,000+ slides.

## Appendix D: Error Codes

| HTTP Code | Error | Resolution |
|---|---|---|
| 400 | Bad Request / LEVEL0_EMBEDDINGS_REQUIRED | Check request format or generate level 0 embeddings |
| 404 | Slide/Session/Task not found | Verify ID exists |
| 500 | Prediction/PDF/Embedding failed | Check GPU memory, restart server |
| 503 | Model not loaded / DB unavailable | Wait for startup (~120s) or check container status |

---

# Document Revision History

| Version | Date | Author | Changes |
|---|---|---|---|
| 1.0.0 | 2025-02-06 | Enso Labs | Initial specification |
| 2.0.0 | 2025-02-10 | Enso Labs | Complete revamp: added outlier detector, few-shot classifier, patch grid overlay, canvas-based overlays, annotation system (SVG + PostgreSQL), batch embedding, heatmap caching, coverage-based alignment, real-time scale bar, resizable panels, dark mode, Mac mini feasibility, Tailscale Funnel, schema v5, 80+ API endpoints, updated architecture diagrams, all current features |
| 2.1.0 | 2025-02-10 | Enso Labs | Reframed as modular platform architecture: added abstraction layer diagrams (foundation/classification/evidence/UI), emphasized plug-and-play model system, reframed HAI-DEF models as demo configuration rather than core value, added comprehensive Hospital Deployment Guide (13.1-13.10), strengthened config-driven project system documentation |

---

*Enso Atlas consists of approximately 30,000 lines of Python backend code and 15,000 lines of TypeScript frontend code.*
