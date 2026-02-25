# Enso Atlas: Technical Specification

## A Modular, Model-Agnostic Pathology Evidence Platform

**Version:** 2.3.0
**Date:** February 24, 2026
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

In this repository, the reference deployment uses Google's HAI-DEF models (Path Foundation, MedGemma 4B, MedSigLIP) across two demonstration projects: ovarian platinum-response prediction and lung adenocarcinoma stage classification. Those projects are included examples, and the model choices are interchangeable. The platform itself is the primary contribution: an on-premise system that makes any pathology AI model deployable, interpretable, and usable in clinical workflows.

The platform addresses a gap in precision oncology: determining which patients will respond to specific chemotherapy regimens before treatment begins. Treatment response can currently only be assessed after multiple cycles of therapy, exposing non-responding patients to unnecessary toxicity and delays in receiving effective alternatives. Enso Atlas provides morphology-based predictions from standard H&E-stained tissue slides already collected during routine clinical care.

### Mission Statement

To build the platform that makes pathology AI deployable: the infrastructure that turns any foundation model and any classification head into an evidence-based decision support tool, running entirely on-premise.

### Design Philosophy: Platform, Not a Demo

Enso Atlas is a modular platform, not a single-purpose tool. Every component can be swapped without code changes:

- **Foundation model metadata is config-defined.** This repo currently executes Path Foundation-specific embedding code paths. Swapping to a different backbone requires code integration plus config updates.
- **Classification models are pluggable within implemented architectures.** Current implementation supports TransMIL and CLAM; additional MIL variants (for example ABMIL) require implementation before registration.
- **Cancer types are project-defined.** The ovarian and lung projects in this repo are demonstration project blocks; breast, prostate, colorectal, or any other cancer type can be added by creating a new project block in the configuration. Labels, thresholds, model assignments, class names, and dataset paths are all per-project.
- **Feature toggles are per-project in config.** Report generation, semantic search, and similar case retrieval toggles are defined per project, but backend/frontend enforcement is partial and should be treated as in-progress.
- **The UI is mostly project-aware.** Labels, categories, and model lists are primarily derived from project configuration, but some demo/test frontend files still contain hardcoded example model IDs.
- **Report generators are swappable.** The structured report system works with MedGemma, but any LLM that accepts a prompt and returns JSON can be substituted. A template-based fallback provides reports even without a language model.

### Key Platform Capabilities

- **Foundation Models (Current State):** Foundation model metadata is defined in YAML. Path Foundation is the integrated runtime backbone in this repo; adding a new embedding backbone requires integration work plus config updates.
- **Pluggable Classification Heads:** Implemented MIL architectures (TransMIL, CLAM) can be registered by adding weights + config. Additional architectures (for example ABMIL) require implementation before use.
- **Config-Driven Cancer Types:** Each cancer type is a project definition in YAML. Lung, breast, prostate, or any other cancer can be added by specifying dataset paths, labels, thresholds, and model assignments. The entire UI (labels, categories, model lists, workflows) reconfigures per project.
- **Per-Project Feature Toggles:** Feature toggles are declared per project. Enforcement across backend/frontend panels is partial and should be validated per deployment.
- **Evidence Generation:** Per-model attention heatmaps overlaid on WSIs with coverage-based alignment (regenerated each analysis run with checkpoint-aware invalidation), top-K evidence patches with coordinates, and FAISS-based similar case retrieval
- **Semantic Tissue Search:** Vision-language model text-to-patch search for queries like "tumor infiltrating lymphocytes" or "necrotic tissue" with on-the-fly embedding
- **Outlier Tissue Detector:** Centroid distance analysis on foundation model embeddings identifies morphologically unusual patches, including rare tissue patterns, artifacts, or atypical regions for pathologist review
- **Few-Shot Patch Classifier:** Pathologists define custom tissue classes by selecting example patches on the viewer; a LogisticRegression trained on the active foundation model's embeddings classifies all patches in the slide with per-class heatmap overlay
- **Structured Reporting:** Pluggable LLM report generation (demonstrated with MedGemma 4B) with JSON-first prompting, safety validation, and template-based fallback
- **Agentic Workflow:** Seven-step AI agent with visible reasoning, SSE streaming, and session memory for follow-up questions
- **Pathologist Annotations:** SVG-based drawing tools (circle, rectangle, freehand, point) on the OpenSeadragon viewer with full CRUD persistence to PostgreSQL
- **Batch Analysis:** Async batch processing with model selection, project scoping, force re-embed, real-time progress tracking, and cancellation support
- **PDF Export:** PDF reports for clinical documentation and tumor board presentations
- **On-Premise by Design:** Docker Compose deployment with optional air-gapped mode, local-first data handling, and audit logging for HIPAA-aligned operations

### Demo Configuration: Multi-Project Deployment with Google HAI-DEF

The current deployment demonstrates the platform with:
- **Foundation model:** Google Path Foundation (384-dim, ViT-S)
- **Classification:** 6 TransMIL models across two projects (`ovarian-platinum` + `lung-stage`)
- **Reports:** MedGemma 4B (4 billion parameter medical LLM)
- **Semantic search:** MedSigLIP (enabled per project via feature flags)
- **Datasets:** Project-scoped TCGA cohorts under `data/projects/{project-id}/...`

This configuration is one example. Training new classification heads for different cancer types and swapping report generators is largely config-driven; replacing the foundation embedding backbone currently requires integration work plus config/model updates.

## 1.2 Abstraction Layers

The platform is organized into four abstraction layers, each independently replaceable:

| Layer | Role | Currently Demonstrated With | Swap By |
|---|---|---|---|
| **Foundation Model** | Patch-level embedding (N x D vectors) | Path Foundation (384-dim) | Register new model in YAML, re-embed slides |
| **Classification** | Slide-level prediction from embeddings | TransMIL (6 project-scoped models) | Train new MIL head, drop in weights, register in YAML |
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

## 1.4 Demo Metrics Snapshot (Research-Only; Not Clinical Validation)

The active demo deployment includes two project-specific prediction tasks on TCGA-derived data using the same core platform.

| Project | Primary Endpoint | Metrics/Threshold Source | Snapshot Values | Caveat |
|---|---|---|---|---|
| `ovarian-platinum` | Platinum chemotherapy response | `config/projects.yaml` model metadata and project threshold | AUC `0.907`, threshold `0.9229` | Training/cohort counts vary across repo notes and historical exports; treat these as demo metadata, not externally validated clinical performance. |
| `lung-stage` | Early (I/II) vs advanced (III/IV) stage | `config/projects.yaml` model metadata and project threshold | AUC `0.648`, threshold `0.5` | Value reflects current config snapshot only; no multi-site or prospective clinical validation in this repo. |

Both projects use Path Foundation embeddings (384-dim) and TransMIL classification heads, with project-scoped labels, models, and feature flags resolved from `config/projects.yaml`.

These numbers are for research/demo communication only. They should be interpreted as configuration-linked performance snapshots, not as clinically validated operating characteristics.

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
- **WSI Viewer:** OpenSeadragon 5.0.1 for Deep Zoom Image (DZI) tile-based rendering with SVG annotation overlay, canvas-based patch heatmap overlays, and patch grid overlay
- **Layout:** react-resizable-panels v4 for user-adjustable, collapsible three-panel layout
- **State Management:** React Context (ProjectContext), custom hooks (useAnalysis, useKeyboardShortcuts), localStorage caching
- **API Communication:** Custom fetch wrapper (3,054 lines) with retry logic, timeout handling, polling-based progress tracking, and typed error handling
- **Dark Mode:** Theme toggling is implemented via a root `.dark` class and CSS variables; Tailwind `dark:` utility usage is limited

### Backend Layer
- **Framework:** FastAPI with Pydantic models for request/response validation
- **Primary Module:** `src/enso_atlas/api/main.py` (7,072 lines), containing all core endpoints, model loading, and startup orchestration
- **Database:** asyncpg connection pool to PostgreSQL 16 (raw SQL, no ORM) with schema v5 including annotations table
- **Background Tasks:** Thread-based batch analysis, report generation, and batch re-embedding with status polling and cancellation

### ML Models Layer
- **Path Foundation:** 384-dim patch embeddings served via mixed runtimes: Transformers/PyTorch for `/api/embed` (device auto-selection) and TensorFlow SavedModel for slide re-embedding flows
- **MedGemma 4B:** Transformers-based causal LM for structured report generation (GPU, bfloat16, ~8 GB VRAM)
- **MedSigLIP:** SigLIP vision-language model for text-to-patch semantic search (GPU, fp16, ~800 MB VRAM)
- **TransMIL:** Six project-scoped PyTorch Transformer-based MIL classifiers with attention extraction (GPU, ~240 MB total)
- **FAISS:** Facebook AI Similarity Search for slide-level (cosine via IndexFlatIP) and patch-level (L2) retrieval
- **scikit-learn:** LogisticRegression for few-shot patch classification on Path Foundation embeddings

### Storage Layer
- **PostgreSQL 16:** Patients, slides, metadata, analysis results, embedding tasks, projects, junction tables, annotations (schema v5)
- **File System:** Project-scoped pre-computed .npy embeddings (level 0 dense), .npy coordinate files, and .svs whole-slide images
- **Model Storage:** HuggingFace cache + local checkpoints in `models/` directory
- **Heatmap Cache:** Disk-based PNG cache (`data/embeddings/heatmap_cache/`) for per-model attention heatmaps

## 2.3 Data Flow Diagram

```mermaid
flowchart LR
    subgraph Offline_Pipeline ["Offline Pipeline"]
        WSI["WSI (.svs file)"] --> PATCH["Patch extraction (224x224, level 0 dense)"]
        PATCH --> PF_EMB["Path Foundation embedding (dense level-0 grid)"]
        PF_EMB --> NPY[".npy embeddings (N x 384)"]
    end

    subgraph Online_Analysis ["Online Analysis"]
        NPY --> TRANSMIL["TransMIL models (project scoped)"]
        TRANSMIL --> PRED["Predictions + attention weights"]
        PRED --> HEATMAP["Attention heatmap (regenerated per analysis run)"]
        PRED --> EVIDENCE["Top-K evidence patches"]
        NPY --> FAISS_Q["FAISS retrieval query"]
        FAISS_Q --> SIMILAR["Similar cases"]
        NPY --> SIGLIP["MedSigLIP semantic search"]
        SIGLIP --> TEXT_MATCH["Text-matched patches"]
        NPY --> OUTLIER["Outlier detection"]
        OUTLIER --> UNUSUAL["Unusual patches"]
        NPY --> FEWSHOT["Few-shot classifier"]
        FEWSHOT --> TISSUE_MAP["Tissue classification"]
    end

    subgraph Reporting ["Reporting"]
        PRED --> MEDGEMMA["MedGemma 4B report generator"]
        EVIDENCE --> MEDGEMMA
        SIMILAR --> MEDGEMMA
        MEDGEMMA --> JSON_REPORT["Structured JSON report"]
        JSON_REPORT --> PDF_GEN["PDF generator"]
        PDF_GEN --> PDF_OUT["Tumor board PDF"]
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
|  - TransMIL x6 (GPU)             |
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
        destination: `${process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000'}/api/:path*',
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

1. TransMIL slide-level classification (project-scoped model set)
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
| Framework | Mixed runtime: Transformers AutoModel (for `/api/embed`) and TensorFlow SavedModel (for `/api/embed-slide` re-embedding) |
| Inference Function | `model.signatures["serving_default"]` |
| Preprocessing | Float32, normalize to [0, 1] |

### Patching Strategy

Whole-slide images are divided into non-overlapping 224x224 pixel patches at level 0 (full resolution, typically 40x) using a dense grid policy. The patching process includes tissue masking for offline pipelines:

1. **Grid Generation:** Regular grid with step size 224 pixels
2. **Patch Selection:** `/api/embed-slide` uses a dense level-0 grid with no tissue filtering (all 224x224 tiles kept). Optional offline scripts use stricter patch-level filters (mean intensity 30-230, std >= 15, non-white ratio threshold).
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
| Model ID | `/app/models/medgemma-4b-it` (default local path); `google/medgemma-4b-it` when configured |
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
3. **Prompt Engineering:** JSON-first prompt with explicit section-level requirements (overview, key findings, patch significance, next steps, optional decision-support block)
4. **Generation:** `torch.inference_mode()` with configurable max tokens and time limits via `max_time` or custom `StoppingCriteria`
5. **Non-Blocking:** Inference runs in worker threads: `asyncio.to_thread()` in async paths and dedicated `threading.Thread` + timeout in report-task execution
6. **Structured Parsing:** Multi-stage parser handles code-fenced JSON, truncated JSON repair, field alias normalization, list coercion, and regex extraction fallback when parsing fails
7. **Safety Constraints:** Validation enforces required fields and rejects prohibited treatment-directive phrases (e.g., "start treatment", "prescribe")
8. **Fallback Handling:** If generation/validation still fails after retries, the system emits a deterministic fallback report with safety language and manual-review guidance

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
| Model ID | Auto-resolved local path (`MEDSIGLIP_MODEL_PATH`, `/app/models/medsiglip`, etc.); fallback `google/siglip-so400m-patch14-384` |
| Architecture | SigLIP (dual encoder: vision + text) |
| Vision Input | 448 x 448 (MedSigLIP) or 384 x 384 (SigLIP) |
| Embedding Dimension | 1152 |
| Precision | fp16 (GPU) |
| VRAM Usage | ~800 MB |

### Capabilities

- **Text-to-Patch Search:** Natural language queries matched against patch embeddings via cosine similarity
- **On-the-Fly Embedding + Fallback:** If cached embeddings are missing, the service attempts real-time MedSigLIP embedding from WSI patches; if unavailable or failed, it falls back to query-aware tissue-type ranking (`model_used: tissue-type-fallback`)
- **Image-to-Image Search:** Visual similarity search via FAISS using Path Foundation embeddings across the entire database
- **Predefined Query Sets:** Curated queries for tumor, inflammation, necrosis, stroma, mitosis, and vessels

### VRAM Sharing

MedSigLIP shares GPU with MedGemma. At fp16, SigLIP requires ~800 MB alongside MedGemma's ~8 GB, fitting within typical GPU memory.

---

# 4. Classification Layer

The classification layer converts a bag of patch embeddings (`N x D`) into:

- a slide-level score,
- a thresholded label,
- a confidence value,
- and (optionally) per-patch attention weights.

In this repository there are two runtime paths:

1. **Single-model MIL runtime** (`create_classifier`): supports both **CLAM** and **TransMIL**.
2. **Multi-model runtime** (`POST /api/analyze-multi`): runs a bank of project-registered **TransMIL** models from `scripts/multi_model_inference.py`.

Model metadata is declared in `config/projects.yaml` under `classification_models`.

```yaml
classification_models:
  your_model:
    model_dir: "transmil_your_task"            # Directory under outputs/
    display_name: "Your Prediction Task"
    auc: 0.85
    positive_label: "Positive"
    negative_label: "Negative"
    compatible_foundation: "path_foundation"
```

## 4.1 Runtime Selection and Threshold Resolution

On API startup (`create_app` -> `load_models` in `src/enso_atlas/api/main.py`):

- `MIL_ARCHITECTURE` selects the single-model MIL backend (`clam` default, `transmil` optional).
- `MIL_MODEL_PATH` overrides checkpoint path.
- If `MIL_ARCHITECTURE=transmil` and `MIL_MODEL_PATH` is unset, runtime prefers `models/transmil_best.pt` when present; otherwise it falls back to the default model path.

Single-model threshold precedence is implemented in `CLAMClassifier._resolve_threshold` and reused by `TransMILClassifier`:

1. `MIL_THRESHOLD` (explicit numeric override)
2. `MIL_THRESHOLD_CONFIG` -> `recommended_threshold` from JSON
3. default `0.5`

`create_classifier(config)` returns:

- `TransMILClassifier` when `architecture == "transmil"`
- `CLAMClassifier` otherwise

## 4.2 Single-Model MIL Behavior (CLAM/TransMIL)

Both classifiers expose the same public interface (`load`, `predict`, `classify`, `predict_batch`, `predict_with_uncertainty`) so endpoint code does not branch on architecture.

Operational details from current code:

- **CLAM checkpoint compatibility:** legacy checkpoints are detected by key pattern (`encoder.*`) and loaded with `LegacyCLAMModel`.
- **TransMIL config hydration:** constructor args can be overridden by checkpoint `config` payload.
- **TransMIL OOM handling:** `predict()` retries on the same device with uniform subsampling (`N -> N/2 -> ...`) down to `MIL_MIN_PATCHES_FOR_ATTENTION` (default `128`, hard floor `16`).
- **Attention shape stability:** when subsampling is used, attention is projected back to full bag length with zero-fill for unsampled indices, then renormalized.

For the main single-slide analysis endpoint, label assignment is threshold-based and confidence uses an exponential margin transform:

```text
margin = abs(score - threshold)
confidence = min(1.0 - 0.5 * (2.0 ** (-20.0 * margin)), 0.99)
```

This differs from the linear threshold-distance confidence used in multi-model scoring.

## 4.3 Multi-Model Inference (`/api/analyze-multi`)

The multi-model path is implemented by `MultiModelInference` in `scripts/multi_model_inference.py` and orchestrated in `src/enso_atlas/api/main.py`.

Current runtime rules:

- endpoint is **level-0 only** (`level must be 0`)
- requests are blocked while level-0 batch embedding is active (GPU contention guard)
- requested model IDs are validated against project scope (DB assignment first, YAML fallback)
- embeddings are resolved from project-scoped directories when `project_id` is provided

Threshold resolution for each model (`_resolve_decision_threshold`) is:

1. `MULTIMODEL_THRESHOLD_<MODEL_ID>` env override
2. YAML `decision_threshold` (or legacy `threshold`)
3. per-model default (`DEFAULT_DECISION_THRESHOLDS`, typically `0.5`)

Resolved thresholds are clamped to `(0.01, 0.99)`.

Score-to-label conversion is centralized in `score_to_prediction`:

```text
if score >= threshold:
    confidence = (score - threshold) / (1 - threshold)
else:
    confidence = (threshold - score) / threshold
confidence = clamp(confidence, 0.0, 0.99)
```

Model-specific post-processing currently includes temperature scaling for survival heads before thresholding:

- `survival_5y`: `2.5`
- `survival_3y`: `2.0`
- `survival_1y`: `2.5`

## 4.4 Cached vs Fresh Consistency Guarantees

`/api/analyze-multi` intentionally keeps cached and fresh responses aligned on threshold semantics.

### Cached path (`analysis_results` reuse)

When cache exists and `force=false`:

- stored raw `score` is reused,
- `label` and `confidence` are **recomputed** with `score_to_prediction(...)` using the **current** model threshold from config,
- response includes `decision_threshold` and capped confidence.

### Fresh path

Fresh inference also routes through `score_to_prediction(...)` inside `predict_single`, then API normalization preserves/patches `decision_threshold` and caps confidence.

### Persistence

Fresh outputs are written back with:

- `score`
- `label`
- `confidence`
- `threshold=pred.decision_threshold`

This keeps DB cache records threshold-aware and allows threshold updates to take effect without manual cache invalidation.

## 4.5 Training Path in This Repository

Primary training flow is `scripts/train_transmil_finetune.py`.

| Component | Current behavior |
|---|---|
| Split strategy (default) | Patient-grouped CV with `StratifiedGroupKFold` (default 5 folds) |
| Single-split mode | `--single_split` with grouped split (`StratifiedGroupKFold` when feasible, `GroupShuffleSplit` fallback) |
| Optimizer | AdamW |
| Scheduler | `CosineAnnealingWarmRestarts(T_0=10, T_mult=2, eta_min=1e-6)` |
| Loss | `FocalLoss(alpha=0.25, gamma=2.0)` with negative-class weighting (`n_pos / n_neg`) |
| Class balancing | Per-epoch balanced stream by oversampling each class to majority count |
| Minority augmentation | Optional feature-space augmentation (noise, dropout, same-class mixup) |
| Patch caps | `--max_train_patches`, `--max_eval_patches` |
| Precision | fp32 training path (no AMP/GradScaler in this script) |
| Gradient clipping | `clip_grad_norm_` with max norm `1.0` |
| Early stop | Validation-AUC patience |
| Runtime guard | Optional `--max_minutes` per fold |
| Metrics | ROC-AUC, PR-AUC, accuracy, sensitivity, specificity |
| Calibration artifact | pooled `calibration_curve.png` |

A CLAM in-repo training path (`CLAMClassifier.fit`) is still present, but the shipped multi-model family and current endpoint path are TransMIL-based.

---

# 5. Backend API Specification

## 5.1 Startup and Initialization

The FastAPI app is built in `create_app()` in `src/enso_atlas/api/main.py` (**7,401 lines** at this audit). The startup work is centralized in `@app.on_event("startup")` and executes in a fixed, dependency-aware order.

### Startup sequence (code-audited)

| Order | Component | What happens |
|---|---|---|
| 1 | Embedding/slide directory resolution | Validates configured directories; falls back to `data/demo/...` if primary embeddings are empty. |
| 2 | MIL classifier | Loads CLAM/TransMIL base classifier from env-driven config (`MIL_ARCHITECTURE`, optional threshold config). |
| 3 | Multi-model inference | Initializes project model inference service from `outputs/*` checkpoints (when dependencies exist). |
| 4 | Evidence generator | Initializes FAISS-backed evidence engine. |
| 5 | Path Foundation embedder | Instantiates embedder object (model is lazy-loaded on first embedding call). |
| 6 | MedGemma reporter | Instantiates reporter, then performs warmup inference to compile CUDA kernels. |
| 7 | Clinical decision support | Initializes decision support helper service. |
| 8 | MedSigLIP embedder | Initializes and eagerly loads MedSigLIP for semantic search. |
| 9 | In-memory slide inventory + FAISS indices | Loads available slide embeddings and builds patch-level and slide-mean FAISS indices. |
| 10 | Label loading | Loads labels from CSV candidates and maps short/full slide IDs. |
| 11 | PostgreSQL initialization | Creates schema; populates from flat files on first run. |
| 12 | Project registry | Loads `config/projects.yaml`, injects registry, and syncs project metadata to DB if available. |
| 13 | Agent workflow | Initializes workflow after models/indices are ready. |

**Outside startup hook:** route wiring occurs in app construction; `ChatManager` initialization happens after router registration logic (still during app creation, not inside the startup event body).

## 5.2 Complete API Endpoint Reference (Audited)

This section is audited against:
- `src/enso_atlas/api/main.py`
- `src/enso_atlas/api/project_routes.py`
- `src/enso_atlas/api/slide_metadata.py`
- `src/enso_atlas/agent/routes.py`

### Core service and compatibility routes

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Canonical health probe (status, CUDA, slide counts, DB, uptime). |
| `GET` | `/api/health` | Health alias used by frontend compatibility code. |
| `GET` | `/` | API root metadata (`name`, `version`, docs path). |
| `GET` | `/docs` | Redirects to `/api/docs`. |
| `GET` | `/redoc` | Redirects to `/api/redoc`. |
| `GET` | `/api/tags` | Legacy stub endpoint (returns empty tag list). |
| `POST` | `/api/tags` | Legacy stub tag-create endpoint. |
| `GET` | `/api/groups` | Legacy stub endpoint (returns empty groups list). |
| `POST` | `/api/groups` | Legacy stub group-create endpoint. |

### Database and model status

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/db/status` | Database availability and population state. |
| `POST` | `/api/db/repopulate` | Force DB repopulation from flat files. |
| `GET` | `/api/embed/status` | Path Foundation embedder runtime status. |
| `GET` | `/api/semantic-search/status` | MedSigLIP semantic-search status. |
| `GET` | `/api/search/visual/status` | Visual search FAISS index status. |

### Slide management and imagery

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/slides` | List slides (`project_id` optionally scopes to project assignments). |
| `GET` | `/api/slides/search` | Search/filter/paginate slides. |
| `GET/HEAD` | `/api/slides/{slide_id}/dzi` | Deep Zoom descriptor for OpenSeadragon. |
| `GET` | `/api/slides/{slide_id}/dzi_files/{level}/{tile_spec}` | Deep Zoom tile image. |
| `GET` | `/api/slides/{slide_id}/thumbnail` | Slide thumbnail (disk-cached). |
| `GET` | `/api/slides/{slide_id}/info` | Detailed slide/WSI metadata. |
| `GET` | `/api/slides/{slide_id}/patches/{patch_id}` | Patch preview; supports `project_id` and explicit `x`,`y`,`patch_size` for coordinate-accurate extraction. |
| `GET` | `/api/slides/{slide_id}/qc` | Slide QC metrics. |
| `GET` | `/api/slides/{slide_id}/embedding-status` | Embedding status and related task state. |
| `GET` | `/api/slides/{slide_id}/cached-results` | Cached analysis results for a slide. |
| `PATCH` | `/api/slides/{slide_id}` | Rename slide (`display_name`). |

### Analysis and batch inference

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/analyze` | Single-slide prediction + evidence retrieval. |
| `POST` | `/api/analyze-multi` | Multi-model inference with project-aware model authorization and per-model `decision_threshold` in response. |
| `POST` | `/api/analyze-uncertainty` | MC Dropout uncertainty analysis. |
| `POST` | `/api/analyze-batch` | Synchronous multi-slide analysis. |
| `POST` | `/api/analyze-batch/async` | Async batch analysis with task tracking. |
| `GET` | `/api/analyze-batch/status/{task_id}` | Batch task progress/details. |
| `POST` | `/api/analyze-batch/cancel/{task_id}` | Cancel running batch task. |
| `GET` | `/api/analyze-batch/tasks` | List batch tasks. |

### Heatmaps

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/heatmap/{slide_id}` | Default MIL attention heatmap. Requires level-0 embeddings + coords for truthful localization. |
| `GET` | `/api/heatmap/{slide_id}/{model_id}` | Model-specific attention heatmap; supports cache bypass via `refresh=true` or `analysis_run_id`, checkpoint-aware cache invalidation, and response `Cache-Control: no-store`. |

### Embedding and embedding tasks

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/embed` | Embed base64 patch images. |
| `POST` | `/api/embed-slide` | On-demand slide embedding (level fixed to 0 in current policy). |
| `GET` | `/api/embed-slide/status/{task_id}` | Per-slide embedding task status. |
| `GET` | `/api/embed-slide/tasks` | List embedding tasks. |
| `POST` | `/api/embed-slides/batch` | Start batch embedding job. |
| `GET` | `/api/embed-slides/batch/status/{batch_task_id}` | Batch embedding progress/details. |
| `POST` | `/api/embed-slides/batch/cancel/{batch_task_id}` | Cancel batch embedding job. |
| `GET` | `/api/embed-slides/batch/active` | Active batch embedding job (if any). |

### Search and retrieval

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/similar` | Slide-mean FAISS similar-case search (`project_id` filters candidate cohort). |
| `POST` | `/api/semantic-search` | Text-to-patch semantic search (MedSigLIP or tissue-type fallback). Returns patch-level coordinates/size metadata for viewer alignment. |
| `POST` | `/api/search/visual` | Image-to-image patch similarity search via FAISS. |
| `POST` | `/api/classify-region` | Tissue-type classification for given coordinates. |

### Outlier detection and few-shot patch classification

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/slides/{slide_id}/outlier-detection` | Distance-based outlier patch detection. |
| `POST` | `/api/slides/{slide_id}/patch-classify` | Few-shot patch classification (logistic regression on embeddings). |
| `GET` | `/api/slides/{slide_id}/patch-coords` | Patch coordinate retrieval for spatial workflows. |

### Annotations

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/slides/{slide_id}/annotations` | List annotations for a slide. |
| `POST` | `/api/slides/{slide_id}/annotations` | Create annotation. |
| `PUT` | `/api/slides/{slide_id}/annotations/{annotation_id}` | Update annotation. |
| `DELETE` | `/api/slides/{slide_id}/annotations/{annotation_id}` | Delete annotation. |
| `GET` | `/api/slides/{slide_id}/annotations/summary` | Annotation summary by label/type. |

### Reports and export

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/report` | Synchronous MedGemma report generation. |
| `POST` | `/api/report/async` | Async report generation task. |
| `GET` | `/api/report/status/{task_id}` | Report task status. |
| `GET` | `/api/report/tasks` | List report tasks. |
| `POST` | `/api/report/pdf` | Render PDF from report JSON (`fpdf2`). |
| `POST` | `/api/export/pdf` | Full export PDF with optional heatmap/evidence (`reportlab`). |

### Projects and model scoping

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/projects` | List projects. |
| `POST` | `/api/projects` | Create project. |
| `GET` | `/api/projects/{project_id}` | Get project details. |
| `PUT` | `/api/projects/{project_id}` | Update project config. |
| `DELETE` | `/api/projects/{project_id}` | Remove project entry from `config/projects.yaml` registry; does **not** delete slide/embedding files already on disk. |
| `GET` | `/api/projects/{project_id}/slides` | List slides assigned to project. |
| `POST` | `/api/projects/{project_id}/slides` | Assign slides to project. |
| `DELETE` | `/api/projects/{project_id}/slides` | Unassign slides from project. |
| `GET` | `/api/projects/{project_id}/models` | List project model assignments. |
| `POST` | `/api/projects/{project_id}/models` | Assign models to project. |
| `DELETE` | `/api/projects/{project_id}/models` | Unassign models from project. |
| `GET` | `/api/projects/{project_id}/available-models` | List models compatible with this project configuration. |
| `POST` | `/api/projects/{project_id}/upload` | Upload slide into project-scoped dataset path. |
| `GET` | `/api/projects/{project_id}/status` | Project readiness/status summary. |
| `GET` | `/api/models` | List available models (optionally scoped by `project_id`). |

### Agent workflow and chat

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/agent/analyze` | Start streaming agent workflow analysis (SSE). |
| `POST` | `/api/agent/followup` | Follow-up question in existing workflow session (SSE). |
| `GET` | `/api/agent/session/{session_id}` | Retrieve one agent session. |
| `GET` | `/api/agent/sessions` | List active agent sessions. |
| `DELETE` | `/api/agent/session/{session_id}` | Delete agent session. |
| `POST` | `/api/chat` | Streaming RAG chat endpoint (SSE). |
| `GET` | `/api/chat/session/{session_id}` | Retrieve chat session history/context. |

### History, audit, and slide metadata

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/history` | Analysis history with filters. |
| `GET` | `/api/slides/{slide_id}/history` | Slide-scoped history view. |
| `GET` | `/api/audit-log` | Audit/event log feed. |
| `GET` | `/api/metadata/tags` | List metadata tags with usage counts. |
| `GET` | `/api/metadata/groups` | List metadata groups. |
| `POST` | `/api/metadata/groups` | Create metadata group. |
| `GET` | `/api/metadata/groups/{group_id}` | Get metadata group details. |
| `PATCH` | `/api/metadata/groups/{group_id}` | Update metadata group. |
| `DELETE` | `/api/metadata/groups/{group_id}` | Delete metadata group. |
| `POST` | `/api/metadata/groups/{group_id}/slides` | Add slides to metadata group. |
| `DELETE` | `/api/metadata/groups/{group_id}/slides` | Remove slides from metadata group. |
| `GET` | `/api/metadata/slides/{slide_id}` | Get metadata for one slide. |
| `POST` | `/api/metadata/slides/{slide_id}/tags` | Add tags to a slide. |
| `DELETE` | `/api/metadata/slides/{slide_id}/tags` | Remove tags from a slide. |
| `POST` | `/api/metadata/slides/{slide_id}/star` | Toggle starred state for a slide. |
| `PATCH` | `/api/metadata/slides/{slide_id}` | Update notes/custom metadata for a slide. |
| `GET` | `/api/metadata/search` | Search slides by metadata predicates. |
| `POST` | `/api/metadata/bulk/metadata` | Fetch metadata for multiple slides in one request. |

## 5.3 Scope and Behavioral Contracts

### Project scope resolution and isolation

When `project_id` is present, backend behavior is intentionally scoped instead of global:

1. **Model visibility** is resolved via `_resolve_project_model_ids()` and enforced through `resolve_project_model_scope()` + `require_model_allowed_for_scope()`.
2. **Embedding lookup** is routed through `_resolve_project_embeddings_dir()` and `_resolve_embedding_path()` so analysis/search reads project-scoped arrays.
3. **Similar-case retrieval** (`/api/similar`) filters candidate slides to the project assignment set.
4. **Batch/report jobs** carry project context so async execution preserves project-specific labels and display semantics.

### `POST /api/analyze-multi`: threshold and cache behavior

- Response payload includes `decision_threshold` for each model (`ModelPrediction.decision_threshold`).
- Cached scores are **re-evaluated against current YAML threshold config** before returning results, so threshold updates take effect without requiring cache invalidation.
- Freshly computed results are persisted with threshold context in DB cache.

### Heatmap refresh and cache-control behavior

For `GET /api/heatmap/{slide_id}/{model_id}`:

- Regeneration is forced when any of the following is true:
  - `refresh=true`
  - non-empty `analysis_run_id`
  - model checkpoint signature changes (mtime/size-based)
- Disk cache key includes project scope, slide, model, render mode, and checkpoint signature.
- Responses are deliberately non-cacheable client-side (`Cache-Control: no-store, max-age=0`, plus `Pragma: no-cache`, `Expires: 0`) even when served from server-side disk cache.

### Semantic patch field contract

`POST /api/semantic-search` returns patch-level fields designed to support deterministic patch preview and viewer navigation:

- `patch_index`
- `similarity_score`
- `coordinates` (normalized to level-0 when SigLIP cache coordinates originate from lower pyramid levels)
- `patch_size` (level-0 pixel span for extraction)
- `attention_weight` (when available)

The patch preview endpoint (`GET /api/slides/{slide_id}/patches/{patch_id}`) accepts explicit `x`, `y`, and `patch_size`, which allows the frontend to request the exact semantic patch crop that corresponds to these fields.

---

# 6. Agentic Workflow

## 6.1 Current Implementation Status

The agent workflow is implemented in `src/enso_atlas/agent/workflow.py` and exposed through `src/enso_atlas/agent/routes.py`.

At startup, the API creates a single `AgentWorkflow` instance and injects shared dependencies (embeddings directory, multi-model inference, MedSigLIP embedder, MedGemma reporter, and FAISS mean-slide index metadata).

Two practical details matter for operations:

1. Agent routes are registered only when the agent module imports successfully (`AGENT_AVAILABLE`).
2. Session state is in-memory only (`workflow._sessions`), so it is cleared on process restart.

## 6.2 API Surface

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/agent/analyze` | Run full seven-step workflow as SSE stream |
| `POST` | `/api/agent/followup` | Ask a follow-up question against an existing session |
| `GET` | `/api/agent/session/{session_id}` | Fetch full session state, outputs, and reasoning chain |
| `GET` | `/api/agent/sessions` | List active in-memory sessions |
| `DELETE` | `/api/agent/session/{session_id}` | Delete a session from memory |

All streaming responses use `text/event-stream` and emit `data: <json>` events.

## 6.3 Seven-Step Execution Path (Backend)

`run_workflow()` executes steps in this fixed order:

1. `initialize`
2. `analyze`
3. `retrieve`
4. `semantic_search`
5. `compare`
6. `reason`
7. `report`
8. final `complete` event

### Step behavior summary

| Step | What it does now | Skip/Error behavior |
|---|---|---|
| `initialize` | Loads `<slide_id>.npy` embeddings and optional `<slide_id>_coords.npy` | If embeddings file is missing, emits `error` and stops workflow |
| `analyze` | Calls `multi_model_inference.predict_all(..., return_attention=True)` and extracts top attention patches | If inference module is unavailable, emits `skipped` |
| `retrieve` | Computes normalized mean embedding for slide and queries FAISS slide-mean index for up to 10 non-self matches | If index is unavailable, emits `skipped` |
| `semantic_search` | Loads cached MedSigLIP patch embeddings from `embeddings_dir/medsiglip_cache/{slide_id}_siglip.npy` and runs 5 fixed pathology text queries | If MedSigLIP model or cache is missing, emits `skipped` |
| `compare` | Summarizes responder/non-responder distribution in retrieved neighbors and computes similarity-weighted vote | If no similar cases were retrieved, emits `skipped` |
| `reason` | Builds a textual evidence summary from predictions, top evidence, similar cases, semantic hits, and optional user questions | Emits `complete` unless unexpected exception |
| `report` | Uses MedGemma reporter when available (run in `asyncio.to_thread`); otherwise builds structured fallback report from collected workflow artifacts | Emits `error` on failure, otherwise `complete` |

## 6.4 SSE Event Contract

Each step is streamed as a JSON object with this shape:

```json
{
  "step": "analyze",
  "status": "complete",
  "message": "Ran 3 models on 6,234 patches",
  "reasoning": "Model predictions: ...",
  "data": {
    "predictions": {},
    "top_evidence": []
  },
  "timestamp": "2026-02-24T22:32:10.123Z",
  "duration_ms": 187.3
}
```

Notes tied to the current implementation:

- The backend emits one event per completed/skipped/error step (it does not currently emit intermediate `running` step events).
- The final `complete` event includes `data.session_id` and total workflow duration.
- Nginx buffering is explicitly disabled for `/api/agent/analyze` via `X-Accel-Buffering: no`.

## 6.5 Session and Follow-up Semantics

Session objects are `AgentState` dataclasses keyed by `session_id`.

A follow-up request (`/api/agent/followup`) appends user/assistant messages to `conversation_history` and returns a streamed reasoning response generated from stored workflow artifacts, not by rerunning the seven-step pipeline.

`GET /api/agent/session/{session_id}` returns:

- core identifiers and status
- predictions, similar cases, top evidence, semantic search results
- generated report
- reasoning chain
- conversation history
- summed per-step duration

## 6.6 Frontend Behavior (Current)

The `AIAssistantPanel` component implements SSE parsing for `/api/agent/analyze` and `/api/agent/followup`, updates step cards incrementally, and hydrates report state when a `report` step arrives.

In the current main page (`frontend/src/app/page.tsx`), that panel is intentionally not rendered (commented as removed for demo readiness). The backend workflow and endpoints are still implemented and callable.

---

# 7. Database Schema

## 7.1 Current Schema (v5)

The PostgreSQL schema is defined in `src/enso_atlas/api/database.py`.

Current state:
- 10 tables total
- schema migrations `v1` through `v5`
- project membership is now modeled with junction tables (`project_slides`, `project_models`)
- `slides.project_id` is still present as a legacy compatibility column

### ER Diagram

```mermaid
erDiagram
    patients ||--o{ slides : has
    slides ||--o{ slide_metadata : has
    slides ||--o{ analysis_results : has
    slides ||--o{ embedding_tasks : has
    slides ||--o{ annotations : has
    projects ||--o{ project_slides : contains
    slides ||--o{ project_slides : assigned_to
    projects ||--o{ project_models : allows

    patients {
        string patient_id PK
        int age
        string sex
        string stage
        string grade
        int prior_lines
        string histology
        string treatment_response
        string diagnosis
        string vital_status
        datetime created_at
        datetime updated_at
    }

    slides {
        string slide_id PK
        string patient_id FK
        string filename
        int width
        int height
        float mpp
        string magnification
        int num_patches
        boolean has_embeddings
        boolean has_level0_embeddings
        string label
        datetime embedding_date
        string file_path
        int file_size_bytes
        string project_id_legacy
        string display_name
        datetime created_at
        datetime updated_at
    }

    slide_metadata {
        int id PK
        string slide_id FK
        string meta_key
        string value
        datetime created_at
        datetime updated_at
    }

    analysis_results {
        int id PK
        string slide_id FK
        string model_id
        float score
        string label
        float confidence
        float threshold
        string attention_hash
        datetime created_at
    }

    embedding_tasks {
        string task_id PK
        string slide_id FK
        int level
        string status
        float progress
        datetime created_at
        datetime completed_at
        string error
    }

    projects {
        string id PK
        string name
        string cancer_type
        string prediction_target
        json config_json
        datetime created_at
        datetime updated_at
    }

    project_slides {
        string project_id PK,FK
        string slide_id PK,FK
        datetime created_at
    }

    project_models {
        string project_id PK,FK
        string model_id PK
        datetime created_at
    }

    annotations {
        string id PK
        string slide_id
        string type
        json coordinates
        string label
        string notes
        string color
        string category
        datetime created_at
        datetime updated_at
    }

    schema_version {
        int version PK
        datetime applied_at
    }
```

Notes:
- Mermaid uses simplified display types (`string`, `int`, `datetime`, `json`) to keep GitHub rendering stable; actual PostgreSQL types are defined in `src/enso_atlas/api/database.py`.
- `slide_metadata.meta_key` in the diagram corresponds to the real DB column `slide_metadata.key`.
- `slides.project_id_legacy` in the diagram corresponds to the real DB column `slides.project_id`.
- `annotations.slide_id` is indexed but currently not declared as a DB foreign key.
- `projects.name` is `NOT NULL` in DDL.
- `embedding_tasks` exists in schema, but current embedding job progress is tracked in-memory by `EmbeddingTaskManager` (`src/enso_atlas/api/embedding_tasks.py`).

## 7.2 Migration History

| Version | Migration | What changed |
|---|---|---|
| v1 | Initial schema | Created core tables (`patients`, `slides`, `slide_metadata`, `analysis_results`, `embedding_tasks`, `projects`, `project_models`, `project_slides`, `schema_version`) and indexes. |
| v2 | `_migrate_project_columns` | Added legacy `slides.project_id` column (default `'ovarian-platinum'`) and index `idx_slides_project`. |
| v3 | `_migrate_project_scoped_tables` | Ensured `project_slides`/`project_models` tables + indexes exist, then seeded `ovarian-platinum` assignments when that project exists. |
| v4 | `_migrate_display_name` | Added `slides.display_name` (user-facing alias). |
| v5 | `_migrate_annotations_table` | Added `annotations` table + `idx_annotations_slide`. |

## 7.3 Connection Pool and Startup Behavior

```python
_pool = await asyncpg.create_pool(
    DATABASE_URL,
    min_size=2,
    max_size=10,
    command_timeout=60,
)
```

Startup behavior:
- up to 10 connection attempts
- 2-second delay between retries
- migrations run during `init_schema()` in order: v2 -> v3 -> v4 -> v5

## 7.4 Query Behavior (Current Backend)

| Area | Current behavior |
|---|---|
| Global slide listing (`GET /api/slides`) | Uses `db.get_all_slides()` (`slides` LEFT JOIN `patients`) when DB is available. |
| Project-scoped slide listing (`GET /api/slides?project_id=...`) | Intentionally uses flat-file/project-directory scan for fresh patch counts; project filtering uses authoritative IDs from DB/labels with guardrails against cross-project leakage. |
| Project slide membership | Resolution order in backend: `project_slides` (authoritative) -> legacy `slides.project_id` -> project labels file -> embeddings-dir fallback (optional in some paths). |
| Project model scope | Resolution order: `project_models` assignments -> `projects.yaml` `classification_models` fallback. |
| Cached model outputs | `analysis_results` stores all runs; reads return latest per model using `DISTINCT ON (model_id)` + `ORDER BY created_at DESC`. |
| Slide metadata | Stored as key/value rows in `slide_metadata`; uniqueness is enforced on `(slide_id, key, value)`. |
| Annotations | CRUD is PostgreSQL-backed via `annotations`; list endpoint returns rows ordered by `created_at`. |

## 7.5 Practical Performance Notes

The DB path replaces expensive directory scans for common global queries. In normal operation:
- global slide list and single-slide lookups are low-latency SQL calls
- cached result fetch is a cheap indexed lookup on `analysis_results`
- annotation reads are index-assisted by `idx_annotations_slide`

Project-scoped list views are the deliberate exception: they prioritize live filesystem state over DB-cached patch counts.

---
# 8. Frontend Architecture

## 8.1 Tech Stack

| Technology | Version (repo) | Purpose |
|---|---|---|
| Next.js | 14.2.22 | App Router shell and page routing (`/`, `/slides`, `/projects`) |
| React | 18.2.0 | Component and state model for all UI flows |
| TypeScript | 5.7.x | Typed contracts for panels, API responses, and viewer state |
| Tailwind CSS | 3.4.17 | Utility-first styling and design tokens from `globals.css` |
| OpenSeadragon | 5.0.1 | Whole-slide tile rendering and viewport controls |
| react-resizable-panels | 4.6.2 | Desktop three-column split layout with collapsible sidebars |
| Lucide React | 0.469.0 | Icons across layout, panels, and modals |
| react-joyride | 2.9.3 | Demo/tutorial walkthrough UI |
| jsPDF | 4.0.0 | Client-side PDF export pipeline |

Notes:
- The viewer currently uses OpenSeadragon runtime from npm, but still references the 4.1 CDN image sprite assets (`prefixUrl`) for control icons.
- Most pages are client components (`"use client"`) with backend access via `fetch` calls.

## 8.2 Component Hierarchy

```text
RootLayout (frontend/src/app/layout.tsx)
  +-- ToastProvider
  +-- ProjectProvider
  +-- DisclaimerBanner
  +-- ErrorBoundary
      +-- App routes

Home route (/frontend/src/app/page.tsx)
  +-- HomePageWrapper (Suspense)
      +-- HomePage (~2485 lines)
          +-- Header
          |   +-- ProjectSwitcher
          |   +-- Nav links (/slides, /projects)
          |   +-- View-mode toggle (oncologist/pathologist/batch)
          |   +-- Demo toggle
          |   +-- Utility menu (System Status, Shortcuts, Docs, Settings)
          |   +-- SettingsModal + SystemStatusModal (mounted in Header)
          +-- Main
          |   +-- Batch mode: BatchAnalysisPanel (full width)
          |   +-- Non-batch mode:
          |       +-- Mobile tabs (Slides / Results)
          |       +-- Desktop PanelGroup (react-resizable-panels)
          |           +-- Left sidebar
          |           |   +-- SlideSelector
          |           |   +-- AnalysisControls
          |           |   +-- Inline error card
          |           |   +-- CaseNotesPanel
          |           +-- Center panel
          |           |   +-- WSIViewer (dynamic import; SSR disabled)
          |           |   +-- OncologistSummaryView (summary mode)
          |           +-- Right sidebar
          |               +-- PathologistView (pathologist mode)
          |               +-- or oncologist stack:
          |                   +-- PredictionPanel
          |                   +-- MultiModelPredictionPanel (conditional)
          |                   +-- EvidencePanel
          |                   +-- SemanticSearchPanel
          |                   +-- SimilarCasesPanel
          |                   +-- OutlierDetectorPanel
          |                   +-- ReportPanel
          +-- Footer
          +-- Page-level modals
              +-- PatchZoomModal
              +-- KeyboardShortcutsModal
              +-- DemoMode
              +-- WelcomeModal

Slides route (/frontend/src/app/slides/page.tsx)
  +-- Header
  +-- Toolbar (refresh, filters toggle, grid/table switch)
  +-- FilterPanel (collapsible side column)
  +-- SlideGrid or SlideTable
  +-- BulkActions bar
  +-- Pagination
  +-- Modals: CreateTag, CreateGroup, ConfirmDelete, AddToGroup
  +-- Footer

Projects route (/frontend/src/app/projects/page.tsx)
  +-- Header
  +-- Project cards with readiness status
  +-- ProjectFormModal (create/edit)
  +-- UploadModal
  +-- DeleteConfirmModal
  +-- Footer
```

Components present in `frontend/src/components/panels` but currently not mounted in the home layout:
- `AIAssistantPanel`
- `PatchClassifierPanel`
- `UncertaintyPanel`

## 8.3 WSI Viewer (74,056 bytes, 1,940 lines)

`frontend/src/components/viewer/WSIViewer.tsx` remains the most complex frontend component.

### Current behavior

- **OpenSeadragon initialization:** tile source loading, navigator minimap, zoom/pan controls, and fullscreen support.
- **Heatmap overlay via `addSimpleImage`:** attention maps are aligned in world coordinates using backend metadata (coverage bounds and patch size), not CSS overlays.
- **Heatmap controls:** model selection, resolution level, alpha-power tuning, smoothing toggle, opacity slider, and heatmap-only mode.
- **Patch overlays on canvas:** outlier scores and few-shot classifier outputs render on a dedicated canvas layer independent of OSD tile rendering.
- **Patch grid overlay:** grid is drawn with requestAnimationFrame and synchronized to heatmap geometry; defaults to 224px when metadata is missing.
- **Annotation layer:** SVG-based drawing/selection for point, circle, rectangle, and freehand annotations.
- **Spatial patch selection hooks:** nearest-patch click selection is implemented in viewer callbacks for few-shot workflows.
- **Realtime scale bar:** magnification + distance labels are updated through direct DOM refs on animation frames to avoid React render churn.
- **Canvas2D patching:** context creation is patched to include `willReadFrequently: true` to avoid browser warnings and improve pixel-read paths.

## 8.4 Resizable Layout Behavior

Desktop layout uses `react-resizable-panels` with three panes (left/center/right):

- Left sidebar: default 22%, min 10%, max 35%, collapsible.
- Center viewer: default 50%, min 30%.
- Right sidebar: default 28%, min 5%, max 45%, collapsible.

Important implementation detail: the current `PanelGroup` uses an `id` but no `autoSaveId`, so pane sizes are **not persisted across browser sessions** by default.

On mobile, this is replaced with a tabbed Slides/Results pattern instead of draggable panel resizing.

## 8.5 Theme and Dark Mode Status

Theme settings live in `SettingsModal` and use localStorage key `atlas-theme`.

- `light`, `dark`, and `system` options are implemented.
- A `.dark` CSS variable theme exists in `globals.css`.
- Tailwind `dark:` utility usage is currently absent, so dark mode coverage is partial.
- Saved theme preference is stored reliably, but startup-wide theme bootstrapping is still lightweight (theme behavior is mostly driven from modal logic).

## 8.6 State Management and UI Flow

### Global context

`ProjectContext` provides:
- `projects`
- `currentProject`
- `switchProject(id)`
- loading/error state

Project selection priority is: URL `?project=` -> localStorage (`enso-atlas-selected-project`) -> first returned project.

### Home page flow

`frontend/src/app/page.tsx` keeps most interaction state local and coordinates several flows:

- **Project change reset:** clears selected slide, cached analysis/report state, selected models, and repopulates project-scoped model options.
- **Deep-link bootstrapping:** `?slide=<id>` autoselects a slide; `?analyze=true` triggers analysis once the slide is loaded.
- **Analysis orchestration:** combines single-model and multi-model runs, embedding generation (including level-0 polling), cache reuse, and report generation.
- **Viewer state bridging:** WSI viewer controls are exposed through refs for keyboard shortcuts and panel actions.
- **Annotation lifecycle:** fetch, create, delete, and selection state are synced with backend annotation APIs.

### Hooks in active use

- `useAnalysis`: analysis/report loading state, progress simulation, retry flows.
- `useKeyboardShortcuts`: global shortcuts with input-field guardrails.

## 8.7 API Client (3,079 lines)

`frontend/src/lib/api.ts` is the typed transport layer used by most frontend surfaces.

### Core client behavior

- Central `fetchApi<T>()` wrapper with timeout + retries.
- Retry policy: 3 retries with exponential backoff + jitter for 408/429/5xx/network/timeout classes.
- `AtlasApiError` carries status, retryability, network/timeout flags, and user-friendly message helpers.

### Functional coverage

- Slides: listing, search/filtering, metadata updates, stars, tags/groups bulk operations.
- Analysis: single-model, multi-model, semantic search, visual search, outlier detection, patch classification, patch coordinates.
- Reports: sync and async report generation with polling progress.
- Embeddings and batch: async task start/status/cancel/poll helpers.
- Project scoping: available models per project and assign/unassign operations.
- Connectivity: health checks and connection-state listeners.

### Known split responsibilities

- SSE parsing for chat/agent streams lives in `AIAssistantPanel.tsx` (`/api/agent/analyze`, `/api/agent/followup`, `/api/chat`), not in `frontend/src/lib/api.ts`.
- `frontend/src/app/projects/page.tsx` still uses local page-level fetch helpers for project CRUD/upload instead of `frontend/src/lib/api.ts` wrappers.

---

# 9. Config-Driven Project System

Enso Atlas project behavior is defined in `config/projects.yaml`. The backend loads this file at startup through `ProjectRegistry` and uses it to resolve project metadata, dataset paths, and model scoping.

## 9.1 Current YAML Structure (`config/projects.yaml`)

The current file has three top-level sections:

- `foundation_models`: global embedding model catalog
- `classification_models`: global prediction model catalog
- `projects`: per-project configuration

Current config snapshot:

- **Foundation models (4):** `path_foundation`, `dinov2`, `uni`, `conch`
- **Classification models (6):** `lung_stage`, `platinum_sensitivity`, `tumor_grade`, `survival_5y`, `survival_3y`, `survival_1y`
- **Projects (2):** `ovarian-platinum`, `lung-stage`

Each project entry includes:

- Core metadata (`name`, `cancer_type`, `prediction_target`, `classes`, `positive_class`)
- Dataset paths (`dataset.slides_dir`, `dataset.embeddings_dir`, `dataset.labels_file`, `dataset.label_column`)
- Model/tool metadata (`models.*`, `features.*`)
- Model scoping (`foundation_model`, `classification_models`)
- Decision settings (`threshold`, optional `threshold_config`)

The expected path convention is project-scoped:

`data/projects/<project_id>/{slides,embeddings,labels.*}`

## 9.2 Backend Loading and Resolution

`src/enso_atlas/api/projects.py` implements `ProjectRegistry`.

At startup it:

1. Loads `foundation_models` into `FoundationModelConfig`
2. Loads `classification_models` into `ClassificationModelConfig`
3. Loads `projects` into `ProjectConfig`
4. Sets default project from `DEFAULT_PROJECT` env var (if valid), else first YAML project

Then `src/enso_atlas/api/main.py` wires the registry into the API and, when PostgreSQL is available, syncs project rows via `db.populate_projects_from_registry(...)`.

Important implementation details from current code:

- `get_project_classification_models(project_id)` filters models by both project membership and `compatible_foundation`.
- Project-scoped `/api/models?project_id=...` resolves allowlists in this order:
  1) DB `project_models` assignments
  2) YAML `projects.<id>.classification_models` fallback
- `ProjectRegistry.save()` currently writes only the `projects` section. If API create/update/delete flows call `save()`, top-level `foundation_models` and `classification_models` are not preserved in the output YAML.
- YAML may include extra classification-model keys (for example `decision_threshold`), but `ClassificationModelConfig` currently loads only the typed fields defined in code.

## 9.3 Adding a New Project / Cancer Task

Configuration-first workflow:

1. Add or reuse a foundation model in `foundation_models`
2. Add classification model definitions in `classification_models`
3. Add a project under `projects` with project-scoped dataset paths
4. Put slide files in that project's `slides_dir`
5. Generate embeddings into that project's `embeddings_dir`
6. Run layout validation:
   - `python scripts/validate_project_modularity.py --check-embedding-layout`
7. Train/export prediction checkpoints and ensure model IDs match config
8. Restart backend so `ProjectRegistry` reloads

Practical note: for full-fidelity config management, edit `config/projects.yaml` directly until `ProjectRegistry.save()` preserves top-level model catalogs.

## 9.4 Runtime Project Switching (Frontend)

Frontend project state lives in `frontend/src/contexts/ProjectContext.tsx`.

Selection priority is:

1. URL query param `?project=<id>`
2. `localStorage` key `enso-atlas-selected-project`
3. First project returned by `/api/projects`

When project changes, frontend calls project-scoped APIs:

- Slide list: `/api/slides?project_id=<id>`
- Model list: `/api/models?project_id=<id>` (with fallback to `/api/projects/{id}/models` and then project-derived safe defaults)

Current behavior notes:

- The selected project persists in `localStorage`.
- If project APIs are unavailable, frontend falls back to `DEFAULT_PROJECT` metadata.
- Project-aware model and label wiring is in place, but there are still fallback heuristics in UI components (for example model/label derivation in `ModelPicker`).
- YAML `features` toggles are defined and loaded, but panel-level rendering in `frontend/src/app/page.tsx` is not fully feature-flag-driven yet.

---

# 10. Outlier Detection and Few-Shot Classification

## 10.1 Outlier Tissue Detector

### Backend implementation (current)

The backend endpoint is:

```http
POST /api/slides/{slide_id}/outlier-detection?threshold=2.0
```

It loads:
- embeddings: `{embeddings_dir}/{slide_id}.npy`
- coordinates: `{embeddings_dir}/{slide_id}_coords.npy`

`embeddings_dir` comes from app startup config (`EMBEDDINGS_DIR` or `data/embeddings`; if unset and `data/embeddings/level0` exists, level-0 is auto-selected).

The endpoint is slide-based only (`slide_id` path param). It does not accept `project_id`.

Algorithm in code:

1. Compute centroid of all patch embeddings (`mean` over axis 0)
2. Compute Euclidean distance from each patch embedding to centroid
3. Compute `mean_dist` and `std_dist`
4. Flag outliers where `distance > mean_dist + threshold * std_dist` (strict `>`)
5. Build `outlier_patches` with `(patch_idx, x, y, distance, z_score)`
6. Sort outliers by raw distance descending
7. Min-max normalize all patch distances to `[0, 1]` for `heatmap_data`

Validation/error behavior:
- Returns `404` if embedding or coordinate file is missing
- Returns `400` if embedding array is empty
- If `std_dist == 0`, `z_score` is set to `0.0`

Response fields (`OutlierDetectionResponse`):
- `outlier_patches`
- `total_patches`, `outlier_count`
- `mean_distance`, `std_distance`, `threshold`
- `heatmap_data` for all patches

### Frontend behavior (current)

`OutlierDetectorPanel` is active in `page.tsx`.

- Threshold slider: `1.0` to `4.0` SD (default `2.0`)
- Runs backend detection on demand
- Shows counts, distance stats, and top outliers
- Clicking an outlier recenters the viewer at `(x, y)` with a `224x224` patch box

Heatmap behavior detail:
- Backend returns normalized scores for all patches
- Panel currently filters to only detected outlier coordinates before rendering overlay
- Overlay is drawn on the viewer canvas layer (not the model-attention OpenSeadragon heatmap)

## 10.2 Few-Shot Patch Classifier

### Backend implementation (current)

The backend endpoint is:

```http
POST /api/slides/{slide_id}/patch-classify
```

Request shape:

```json
{
  "classes": {
    "tumor": [10, 42, 103],
    "stroma": [5, 88, 190]
  }
}
```

Validation rules in code:
- At least 2 classes required
- Each class must contain at least 1 patch index
- Every patch index must be in `[0, n_patches - 1]`

Data loading is the same pattern as outlier detection:
- `{embeddings_dir}/{slide_id}.npy`
- `{embeddings_dir}/{slide_id}_coords.npy`

This endpoint is also slide-based only and does not accept `project_id`.

Classification flow:

1. Collect training examples from user-provided indices
2. Sort class names alphabetically (`class_names = sorted(request.classes.keys())`)
3. Train `LogisticRegression(max_iter=1000, random_state=42)` on selected embeddings
4. Predict class and probabilities for all patches in the slide
5. Build per-patch output (`patch_idx`, `x`, `y`, `predicted_class`, `confidence`, `probabilities`)
6. Build `heatmap_data` with `(x, y, class_idx, confidence)`

`class_idx` is based on the sorted `class_names` order.

Accuracy estimate:
- Uses leave-one-out retraining over the selected examples
- Computed only when total examples `> number_of_classes`
- Returns `accuracy_estimate` as `null` when not computed

Response fields (`PatchClassifyResponse`):
- `classes`, `total_patches`
- `predictions` (all patches)
- `class_counts`
- `accuracy_estimate`
- `heatmap_data`

### Frontend status (current)

Backend/API wiring exists, but the user-facing panel is not currently mounted in the main page.

What is implemented in frontend code:
- `classifyPatches()` API client
- `PatchClassifierPanel` component with:
  - class add/remove (2-8 classes)
  - index parsing with ranges (for example `10-20`)
  - optional spatial selection mode support
  - class distribution and heatmap toggle after inference
- `WSIViewer` logic for click-to-nearest-patch selection using loaded patch coordinates

Current limitation:
- `PatchClassifierPanel` is imported but not rendered in `frontend/src/app/page.tsx` (marked as disabled), so few-shot classification is not exposed in the primary UI without additional wiring.


---

# 11. Annotation System

## 11.1 Overview

The annotation system supports pathologist review workflows on the WSI viewer. Annotations are stored in PostgreSQL and loaded per slide, so they persist across sessions when the backend is available.

In the current frontend, create/list/delete are server-backed. If create fails, the UI falls back to a local (non-persisted) annotation so review can continue.

## 11.2 Annotation Types in Practice

The API accepts these types:
- `circle`
- `rectangle`
- `freehand`
- `point`
- `marker`
- `note`
- `measurement`

Current drawing behavior in `WSIViewer` is:
- Interactive drawing: `circle`, `rectangle`, `freehand`, `point`
- `freehand` stores a `points` array plus a computed bounding box
- `note`, `marker`, and `measurement` are supported as stored types and panel-generated metadata entries

## 11.3 Coordinate and Rendering Model

Annotations are stored in slide image coordinates (pixel space), not viewport pixels.

Frontend coordinate flow:
1. Mouse event in screen pixels
2. Screen -> OpenSeadragon viewport coordinates
3. Viewport -> image coordinates (rounded)

Rendering flow:
1. Stored image coordinates are transformed back to screen space on render
2. An SVG overlay is drawn on top of the viewer
3. Selected annotations use stronger visual emphasis (higher stroke width/opacity)

This keeps annotations spatially aligned with the slide as users zoom and pan.

## 11.4 PostgreSQL Persistence

```sql
CREATE TABLE annotations (
    id TEXT PRIMARY KEY,           -- "ann_{uuid12}"
    slide_id TEXT NOT NULL,
    type TEXT NOT NULL,
    coordinates JSONB NOT NULL,    -- {x, y, width, height, points?}
    label TEXT,
    notes TEXT,
    color TEXT DEFAULT '#3b82f6',
    category TEXT,                 -- mitotic, tumor, stroma, etc.
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);
```

Index:
- `idx_annotations_slide` on `(slide_id)`

## 11.5 API and Audit Behavior

Endpoints:
- `GET /api/slides/{id}/annotations` -- list annotations for a slide
- `POST /api/slides/{id}/annotations` -- create annotation (`id` auto-generated as `ann_{12hex}`)
- `PUT /api/slides/{id}/annotations/{ann_id}` -- update `label`, `notes`, `color`, `category`
- `DELETE /api/slides/{id}/annotations/{ann_id}` -- delete annotation by ID
- `GET /api/slides/{id}/annotations/summary` -- aggregate counts (prefers `label`, falls back to `category`)

Request/response notes:
- On create, `text` is accepted and mapped to `notes` if `notes` is not provided
- `GET`/`POST`/`PUT` responses include `text` derived from `notes` or `label`
- Current frontend API client consumes create/list/delete; update and summary are available but not wired into the primary UI flow

Audit logging:
- Logged: annotation create/delete
- Not logged: list/update/summary

---

# 12. Heatmap and Overlay System

## 12.1 Attention Heatmaps (OSD Layer)

The viewer supports two heatmap paths:

- **Model-specific endpoint (primary):** `GET /api/heatmap/{slide_id}/{model_id}`
- **Legacy endpoint (compatibility):** `GET /api/heatmap/{slide_id}`

The model-specific path is the one used for project-scoped multi-model analysis.

### Model-Specific Heatmaps (`/api/heatmap/{slide_id}/{model_id}`)

Key behavior in the current implementation:

- Works with project-scoped models (not a fixed model count).
- Requires level-0 embeddings and `*_coords.npy` coordinates.
  - If level-0 embeddings are missing: `400 LEVEL0_EMBEDDINGS_REQUIRED`
  - If coordinates are missing: `409 COORDS_REQUIRED_FOR_HEATMAP`
- Renders PNG RGBA with transparent background.
- Uses patch-resolution output: **1 heatmap pixel = 1 patch cell**.
- Uses `patch_size=224` for grid coverage math.
- Supports:
  - `alpha_power` (default `0.7`)
  - `smooth` (default `false`; blurred/interpolated visualization)
  - `refresh` (default `false`)
  - `analysis_run_id` (optional nonce)

### Refresh and Regeneration Controls

Regeneration is forced when either of the following is present:

- `refresh=true`
- `analysis_run_id=<non-empty value>`

Frontend behavior is aligned with this contract:

- The UI increments an `analysisRunId` whenever users run, retry, or re-run analysis.
- That nonce is appended to model heatmap URLs as `analysis_run_id`.
- The proxy also sets `refresh=true` whenever `analysis_run_id` is provided.

Result: model heatmaps are recomputed for each analysis run even when embeddings are reused from cache.

### Coverage-Based Alignment

Heatmap sizing uses deterministic coverage math:

```python
grid_w = ceil(slide_width / 224)
grid_h = ceil(slide_height / 224)
coverage_w = grid_w * 224
coverage_h = grid_h * 224
```

Model heatmap responses include:

- `X-Slide-Width`, `X-Slide-Height`
- `X-Coverage-Width`, `X-Coverage-Height`

The viewer uses these coverage headers to compute overlay world-space width/height, so the heatmap and patch grid stay aligned to extraction boundaries.

### Checkpoint-Aware Disk Cache

Model heatmaps are cached under:

- `<embedding_dir>/heatmap_cache/`

Cache key fields include:

- scope (`project_id` or `global`)
- `slide_id`
- `model_id`
- rendering mode (`truthful` vs `smooth`)
- checkpoint signature (`mtime_ns + file size`)
- version suffix (`v4`)

Example shape:

- `{scope}_{slide}_{model}_{truthful|smooth}_{checkpoint_sig}_v4.png`

Additional cache rules:

- Disk cache is read/written only for default `alpha_power` (~`0.7`).
- Checkpoint signature changes force refresh and clear in-memory model entries so inference reloads the updated checkpoint.

### No-Store Response Policy

Model heatmap responses (cached or regenerated) are returned with no-store semantics (`Cache-Control: no-store, max-age=0`, plus `Pragma: no-cache`, `Expires: 0`).

The Next.js heatmap proxy also uses no-store fetch/response behavior for model heatmaps to avoid stale browser or intermediary reuse.

### Legacy Heatmap Endpoint (`/api/heatmap/{slide_id}`)

The legacy single-model endpoint remains available and supports:

- `level`: `0..4` mapped to thumbnail target sizes (`2048, 1024, 512, 256, 128`)
- `smooth` (default `true`)
- `blur` kernel size (default `31`)

This path is compatibility-oriented; the project-scoped model heatmap route above is the primary overlay path.

## 12.2 Canvas-Based Patch Overlays

Outlier and few-shot classifier overlays are rendered on a separate HTML canvas (independent of the OSD heatmap image layer).

### Outlier Overlay

Backend outlier detection returns normalized scores for all patches. In the UI, only coordinates flagged as outliers are rendered.

Overlay payload (rendered subset):

- `{ x, y, score }`
- score range: `[0, 1]`
- color ramp: warm yellow → red

### Few-Shot Classifier Overlay

Few-shot classification trains a logistic regression model from user-provided labeled patch indices, then predicts all patches.

Overlay payload:

- `{ x, y, class_idx, confidence }`
- class colors are selected from a fixed 8-color palette
- alpha is scaled by confidence

### PatchOverlay Type

```typescript
interface PatchOverlay {
  type: "outlier" | "classifier";
  data: Array<{ x: number; y: number; score?: number; classIdx?: number; confidence?: number }>;
  classes?: string[];
}
```

## 12.3 Patch Grid Overlay

A dedicated canvas draws the patch grid in viewer coordinates.

Controls in the viewer:

- toggle on/off
- opacity slider
- color picker (default cyan `#00ffff`)

Alignment behavior:

- Grid spacing uses heatmap metadata-derived patch width/height when available.
- Fallback spacing is `224px`.
- Drawing is clipped to slide bounds to prevent lines from extending outside the image area.

## 12.4 Coordinate-Space Alignment Notes (Including Semantic Paths)

Coordinate consistency matters because multiple systems share patch coordinates:

- attention heatmap generation
- patch overlay canvases
- semantic search previews and patch extraction
- click-to-navigate / click-to-select interactions

Current rule set:

- Canonical coordinate space is level-0 pixel space.
- Heatmaps require explicit coordinate files (`*_coords.npy`) and do not synthesize fallback grids.
- Legacy SigLIP semantic caches can contain lower-level coordinates; backend normalization upscales them to level-0 when slide-coverage checks indicate misalignment.

This keeps semantic patch previews and viewer overlays operating in the same coordinate frame.

---

# 13. Hospital Deployment Guide

This chapter is an operations runbook for hospital IT and MLOps teams deploying Enso Atlas on local infrastructure. The commands and paths below are aligned with the current repository structure (`docker/docker-compose.yaml`, `config/projects.yaml`, and `scripts/*`) and the DGX-style topology used in this codebase (backend on `8003`, frontend on `3002`, PostgreSQL on `5433`).

---

## 13.1 Hardware Requirements

Choose the profile that matches your workload and turnaround requirements.

### Minimum: CPU-first pilot (small cohorts, slower report generation)

| Component | Specification |
|---|---|
| CPU | Apple M2/M4, Intel i7 12th gen+, or AMD Ryzen 7+ |
| RAM | 16 GB (32 GB strongly preferred) |
| Storage | 512 GB SSD minimum |
| GPU | Optional |
| OS | macOS 14+, Ubuntu 22.04+, or Windows via WSL2 |

This profile supports validation, demos, and low-throughput use. MedGemma report generation can take ~60-120s on CPU.

### Recommended: GPU workstation (department deployment)

| Component | Specification |
|---|---|
| CPU | Ryzen 9 / Intel i9 |
| RAM | 32-64 GB |
| Storage | 1+ TB NVMe SSD |
| GPU | RTX 4090 (24 GB) or RTX/A-series 48 GB |
| OS | Ubuntu 22.04 + NVIDIA Container Toolkit |

This profile is suitable for routine clinical workflow testing and faster report generation.

### Production: DGX-class server

| Component | Specification |
|---|---|
| CPU | x86_64 or ARM64 |
| RAM | 64-128 GB |
| Storage | 2+ TB NVMe |
| GPU | A100/H100/Blackwell class |
| Network | 10 GbE recommended |

Note on DGX Spark / Blackwell: TensorFlow support for Path Foundation GPU paths may be limited. In practice, this deployment keeps Path Foundation embedding workflows CPU-safe and precomputes embeddings offline, while inference/reporting workloads use GPU acceleration.

### Storage Planning

| Data Type | Per Slide | 200 Slides | 1,000 Slides |
|---|---|---|---|
| WSI (`.svs`, `.ndpi`, etc.) | 1-3 GB | 200-600 GB | 1-3 TB |
| Embeddings (`.npy`) | 5-15 MB | 1-3 GB | 5-15 GB |
| Coords (`*_coords.npy`) | 1-3 MB | 0.2-0.6 GB | 1-3 GB |
| Heatmap cache/images | 10-50 KB each | 10-50 MB | 50-250 MB |
| PostgreSQL metadata | ~1-5 KB/slide | <5 MB | <25 MB |
| Model artifacts/cache | N/A | 15-30 GB | 15-30 GB |

---

## 13.2 Step-by-Step Installation

### Step 1: Clone the repository

```bash
git clone https://github.com/Hilo-Hilo/enso-atlas.git
cd enso-atlas
```

### Step 2: Download model assets before going offline

The runtime expects local model availability for air-gapped operation.

```bash
python3 -m pip install --upgrade "huggingface_hub[cli]"
huggingface-cli login

# Shared Hugging Face cache (mount this into /app/cache/huggingface for Docker)
export HF_HOME="$PWD/.hf-cache"
mkdir -p "$HF_HOME"

# Path Foundation (TensorFlow SavedModel; required in HF cache)
huggingface-cli download google/path-foundation --cache-dir "$HF_HOME"

# MedGemma report model (kept in repo-local models/ for deterministic loading)
huggingface-cli download google/medgemma-4b-it --local-dir models/medgemma-4b-it --cache-dir "$HF_HOME"

# MedSigLIP (preferred), with SigLIP fallback
huggingface-cli download google/medsiglip-448 --local-dir models/medsiglip --cache-dir "$HF_HOME" \
  || huggingface-cli download google/siglip-so400m-patch14-384 --local-dir models/medsiglip --cache-dir "$HF_HOME"
```

### Step 3A: Docker deployment (recommended for DGX/GPU hosts)

Important: the repository `docker/docker-compose.yaml` currently contains developer-specific absolute mounts (for example `/home/hansonwen/med-gemma-hackathon/...`). Do not deploy those paths unchanged.

Use an override file for hospital paths:

```bash
cp docker/docker-compose.yaml docker/docker-compose.hospital.yaml
# Edit docker/docker-compose.hospital.yaml:
# - volume mounts
# - DATABASE_URL / POSTGRES_PASSWORD
# - optional port bindings
```

Then start services:

```bash
docker compose -f docker/docker-compose.yaml -f docker/docker-compose.hospital.yaml up -d --build
curl http://localhost:8003/api/health
```

### Step 3B: Native deployment (Mac mini / no Docker)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

# PostgreSQL (example: macOS)
brew install postgresql@16
brew services start postgresql@16
createuser enso
createdb -O enso enso_atlas

# Backend env
export DATABASE_URL="postgresql://enso@localhost:5432/enso_atlas"
export MIL_ARCHITECTURE=transmil
export MIL_THRESHOLD_CONFIG=models/threshold_config.json

# Start backend on 8000 (native)
python -m uvicorn enso_atlas.api.main:app --host 0.0.0.0 --port 8000
```

### Step 4: Build and run frontend

```bash
cd frontend
npm install

# Docker backend default:
export NEXT_PUBLIC_API_URL=http://127.0.0.1:8003
# If using native backend on port 8000, use:
# export NEXT_PUBLIC_API_URL=http://127.0.0.1:8000
# For public tunnel/same-origin proxy deployments, leave empty:
# unset NEXT_PUBLIC_API_URL

npm run build
npx next start -p 3002
```

Frontend URL: `http://localhost:3002`

### Step 5: Verify service health

```bash
# Backend health
curl http://localhost:8003/api/health

# Database status
curl http://localhost:8003/api/db/status

# Project-scoped model visibility
curl "http://localhost:8003/api/models?project_id=ovarian-platinum"

# Project-scoped slides
curl "http://localhost:8003/api/slides?project_id=ovarian-platinum"
```

### Step 6: Enable strict offline mode

After validating model availability and startup behavior, enforce offline operation:

```bash
# Environment (compose or shell)
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
```

Then restart backend services.

---

## 13.3 YAML Configuration Guide (`config/projects.yaml`)

`config/projects.yaml` is the deployment control plane. It defines:

1. `foundation_models` (embedding backbones)
2. `classification_models` (TransMIL heads and metadata)
3. `projects` (dataset paths, model assignments, feature flags)

### 13.3.1 Foundation model registry

Each foundation model must declare the output embedding dimension:

```yaml
foundation_models:
  path_foundation:
    name: "Path Foundation"
    embedding_dim: 384
    description: "Google Health pathology foundation model (ViT-S, 384-dim)"

  uni:
    name: "UNI"
    embedding_dim: 1024
    description: "Mass General pathology foundation model"
```

### 13.3.2 Classification model registry

`model_dir` must map to a folder under `outputs/` containing `best_model.pt`.

```yaml
classification_models:
  lung_stage:
    model_dir: "luad_transmil_stage"
    display_name: "Tumor Stage"
    auc: 0.648
    positive_label: "Advanced (III/IV)"
    negative_label: "Early (I/II)"
    compatible_foundation: "path_foundation"

  tumor_grade:
    model_dir: "transmil_grade"
    display_name: "Tumor Grade"
    auc: 0.752
    decision_threshold: 0.9935
    compatible_foundation: "path_foundation"
```

Expected model artifact layout:

```text
outputs/
  luad_transmil_stage/
    best_model.pt
    config.json
    threshold_config.json
```

### 13.3.3 Project definitions

Projects scope dataset paths, visible models, and enabled features.

```yaml
projects:
  lung-stage:
    name: "Lung Adenocarcinoma - Stage Classification"
    foundation_model: path_foundation
    dataset:
      slides_dir: data/projects/lung-stage/slides
      embeddings_dir: data/projects/lung-stage/embeddings
      labels_file: data/projects/lung-stage/labels.json
      label_column: tumor_stage
    classification_models:
      - lung_stage
    features:
      medgemma_reports: true
      medsiglip_search: false
      semantic_search: false
    threshold: 0.5
```

### 13.3.4 Feature toggles

| Feature | If disabled (`false`) |
|---|---|
| `medgemma_reports` | Report generation endpoints return fallback behavior |
| `medsiglip_search` | Text-to-patch retrieval disabled |
| `semantic_search` | Similar-case semantic panel hidden/disabled |

---

## 13.4 Plugin Model System: Adding Custom Models

### 13.4.1 Train a new TransMIL model

1) Prepare labels as numeric binary targets for training scripts (`label` must be `0/1`):

```csv
slide_id,label,patient_id
SLIDE-001,1,PAT-001
SLIDE-002,0,PAT-002
```

2) Generate embeddings:

```bash
python scripts/embed_level0_pipelined.py \
  --slide-dir data/projects/breast-her2/slides \
  --output-dir data/projects/breast-her2/embeddings \
  --patch-size 224 \
  --batch-size 512
```

3) Train:

```bash
python scripts/train_transmil_finetune.py \
  --embeddings_dir data/projects/breast-her2/embeddings \
  --labels_file data/projects/breast-her2/labels.csv \
  --output_dir outputs/transmil_her2 \
  --n_folds 5 \
  --epochs 100 \
  --augment_minority \
  --max_train_patches 1024 \
  --max_eval_patches 1024
```

4) Optimize threshold:

```bash
python scripts/optimize_threshold.py \
  --checkpoint outputs/transmil_her2/best_model.pt \
  --embeddings_dir data/projects/breast-her2/embeddings \
  --labels_file data/projects/breast-her2/labels.csv \
  --output_dir outputs/transmil_her2
```

### 13.4.2 Register and activate model

1. Add the model under `classification_models`
2. Add it to the target project's `classification_models` list
3. Restart backend
4. Verify:

```bash
curl "http://localhost:8003/api/models?project_id=<project_id>" | python3 -m json.tool
```

### 13.4.3 Compatibility rules

- `compatible_foundation` in each classifier must match the project `foundation_model`
- Embedding dimensions must match training-time configuration
- Switching foundation models requires full re-embedding + retraining

---

## 13.5 Slide Ingestion

### 13.5.1 Supported WSI formats

| Format | Extension |
|---|---|
| Aperio | `.svs` |
| TIFF | `.tiff`, `.tif` |
| Hamamatsu | `.ndpi` |
| MIRAX | `.mrxs` |
| Ventana | `.vms` |
| Leica | `.scn` |

### 13.5.2 Add slides

Option A: copy directly into project directory

```bash
cp /path/to/new_slides/*.svs data/projects/ovarian-platinum/slides/
curl -X POST http://localhost:8003/api/db/repopulate
```

Option B: upload via API

```bash
curl -X POST http://localhost:8003/api/projects/lung-stage/upload \
  -F "file=@/path/to/LUAD-001.svs"
```

Option C: upload from frontend `/projects` page.

### 13.5.3 Generate embeddings for new slides

Per-slide async embedding:

```bash
curl -X POST http://localhost:8003/api/embed-slide \
  -H "Content-Type: application/json" \
  -d '{"slide_id":"LUAD-001","level":0,"force":false}'

curl http://localhost:8003/api/embed-slide/status/<task_id>
```

Batch embedding through API:

```bash
curl -X POST http://localhost:8003/api/embed-slides/batch \
  -H "Content-Type: application/json" \
  -d '{"level":0,"force":true,"concurrency":1}'
```

Batch embedding through script (useful on DGX overnight):

```bash
python scripts/batch_reembed_level0.py --api-url http://127.0.0.1:8003
# Tailnet example:
# python scripts/batch_reembed_level0.py --api-url http://100.x.y.z:8003
```

### 13.5.4 Update labels and refresh DB

After modifying `labels.csv`/`labels.json` in a project dataset, repopulate:

```bash
curl -X POST http://localhost:8003/api/db/repopulate
```

---

## 13.6 Docker Compose Configuration

### 13.6.1 Production hardening for current compose file

Before production use, update these repo defaults:

- Replace developer-specific host mounts (`/home/hansonwen/med-gemma-hackathon/...`)
- Remove hardcoded DB credentials from committed compose config
- Bind PostgreSQL to localhost (`127.0.0.1:5433:5432`) unless remote DB access is required
- Keep source-code bind mounts only for development (`/app/src:ro`)
- Keep `8003:8000`; expose `7862` only if Gradio/UI experiments are needed

Recommended override pattern:

```bash
docker compose -f docker/docker-compose.yaml -f docker/docker-compose.hospital.yaml up -d
```

### 13.6.2 Reference environment variables

| Variable | Required | Description |
|---|---|---|
| `DATABASE_URL` | Yes | PostgreSQL DSN used by backend |
| `MIL_ARCHITECTURE` | No | `transmil` or `clam` |
| `MIL_MODEL_PATH` | No | Explicit checkpoint path override |
| `MIL_THRESHOLD_CONFIG` | No | Threshold config JSON path |
| `ENSO_CONFIG` | No | Path to `config/default.yaml` |
| `EMBEDDINGS_DIR` | No | Override default embeddings root |
| `HF_HOME` | No | Hugging Face cache root (useful for offline Docker mounts) |
| `TRANSFORMERS_OFFLINE` | No | Set `1` for air-gapped mode |
| `HF_HUB_OFFLINE` | No | Set `1` for air-gapped mode |
| `NEXT_PUBLIC_API_URL` | Frontend | Backend base URL for frontend/proxy |

### 13.6.3 CPU-only deployment notes

If GPU is unavailable:

- Remove `deploy.resources.reservations.devices` from compose
- Keep `concurrency=1` for long embedding runs
- Expect slower MedGemma report generation
- Consider disabling `medgemma_reports` or `medsiglip_search` per project

---

## 13.7 Network and Security

### 13.7.1 Recommended DGX network topology

```text
Hospital LAN / Tailnet

Clinician Browser ---> Frontend (Next.js, :3002)
                           |
                           | /api proxy
                           v
                     Backend API (FastAPI, :8003)
                           |
                           v
                    PostgreSQL (atlas-db, :5433)
```

Operational rule: keep backend and DB private to LAN/Tailnet. Expose frontend only when external access is required.

### 13.7.2 Firewall baseline

| Port | Service | Recommendation |
|---|---|---|
| 3002 | Frontend | Allow hospital LAN; optional public tunnel |
| 8003 | Backend API | LAN/Tailnet only |
| 5433 | PostgreSQL | Localhost only in most deployments |

### 13.7.3 Public/demo tunnel options

Tailscale Funnel:

```bash
tailscale up
tailscale funnel 3002
```

Cloudflare Tunnel:

```bash
cloudflared tunnel --url http://127.0.0.1:3002
```

For public tunnel deployments:

- expose only frontend (`3002`)
- keep backend private
- avoid publishing private Tailnet backend URLs in browser-exposed `NEXT_PUBLIC_API_URL`

### 13.7.4 HIPAA implementation notes

Enso Atlas supports HIPAA-aligned deployments, but institutional compliance remains the operator's responsibility.

| Control Domain | Recommended Implementation |
|---|---|
| Access control | Place frontend behind hospital SSO/VPN/reverse proxy |
| Auditability | Retain `/api/audit-log` and infrastructure logs |
| Encryption at rest | Use FileVault/LUKS/BitLocker + encrypted backups |
| Encryption in transit | TLS via reverse proxy, tunnel, or internal PKI |
| Data locality | Air-gapped mode + no cloud inference dependencies |

---

## 13.8 Database Management

### 13.8.1 PostgreSQL setup

Docker compose includes `atlas-db` (PostgreSQL 16). For native installs:

```bash
# Ubuntu example
sudo apt install postgresql-16
sudo -u postgres createuser enso
sudo -u postgres createdb -O enso enso_atlas
sudo -u postgres psql -c "ALTER USER enso PASSWORD 'change_me';"
```

### 13.8.2 Schema and migrations

Schema bootstrap/migrations run automatically at startup (currently through v5). Repeated startup is idempotent.

### 13.8.3 Backup and restore

Backup:

```bash
# Docker
docker exec atlas-db pg_dump -U enso enso_atlas > enso_atlas_$(date +%Y%m%d).sql

# Native
pg_dump -U enso enso_atlas > enso_atlas_$(date +%Y%m%d).sql
```

Restore:

```bash
# Docker
docker exec -i atlas-db psql -U enso enso_atlas < enso_atlas_20260224.sql

# Native
psql -U enso enso_atlas < enso_atlas_20260224.sql
```

### 13.8.4 Repopulate metadata after file changes

```bash
curl -X POST http://localhost:8003/api/db/repopulate
```

Use after adding/removing slides, embeddings, or labels outside API workflows.

### 13.8.5 Rotate DB credentials

1. Rotate PostgreSQL user password
2. Update `DATABASE_URL` in runtime environment
3. Restart backend services
4. Verify with `GET /api/db/status`

---

## 13.9 Post-Deployment Checklist

Use this checklist before clinical pilot rollout:

```text
[ ] Hardware sized for expected throughput and cohort growth
[ ] docker-compose paths and credentials replaced from repo defaults
[ ] Path Foundation, MedGemma, and MedSigLIP assets available locally
[ ] Backend health endpoint returns healthy
[ ] Frontend reachable on hospital LAN at :3002
[ ] /api/models and /api/slides return project-scoped results
[ ] New slide upload + embedding + analysis tested end-to-end
[ ] Database backup/restore tested with a real backup artifact
[ ] Offline mode enabled (TRANSFORMERS_OFFLINE=1, HF_HUB_OFFLINE=1)
[ ] Firewall rules restrict backend/database exposure
[ ] Audit logging and authentication controls reviewed with IT/security
```

---

## 13.10 Upgrading

### Docker deployment

```bash
cd enso-atlas
git pull origin main

docker compose -f docker/docker-compose.yaml -f docker/docker-compose.hospital.yaml up -d --build

cd frontend
npm install
npm run build
npx next start -p 3002
```

### Native deployment

```bash
cd enso-atlas
git pull origin main
source .venv/bin/activate
pip install -e .

cd frontend
npm install
npm run build
npx next start -p 3002
```

After either upgrade path, run:

```bash
curl http://localhost:8003/api/health
curl http://localhost:8003/api/db/status
```

If model metadata changed, re-check:

```bash
curl "http://localhost:8003/api/models?project_id=ovarian-platinum"
```

# 14. Data Pipeline

## 14.1 Project-Scoped Datasets

Datasets are defined per project in `config/projects.yaml`. Current runtime projects are:

- `ovarian-platinum`: TCGA ovarian cohort for platinum sensitivity and related ovarian endpoints
- `lung-stage`: TCGA LUAD cohort for stage classification (early vs advanced)

Each project is expected to stay within `data/projects/<project_id>/...`. The repository includes modularity checks (`scripts/validate_project_modularity.py`) to catch path drift and cross-project leakage.

## 14.2 Canonical Data Layout

```text
data/
  projects/
    <project_id>/
      slides/
      embeddings/
        *.npy
        *_coords.npy
        level0/
          *.npy
          *_coords.npy
      labels.csv | labels.json
```

Notes:

- `dataset.embeddings_dir` in `projects.yaml` may point to `.../embeddings` or `.../embeddings/level0`.
- Current guardrails expect root `embeddings/` and `embeddings/level0/` to stay synchronized for slide IDs and coordinate pairs.
- Non-project/legacy paths (for example `data/tcga_full`) still exist as compatibility fallbacks in some code paths.

## 14.3 Embedding Generation and Runtime Routing

Primary dense embedding workflow:

- `scripts/embed_level0_pipelined.py` extracts level-0 patches and writes:
  - `{slide_id}.npy` (embedding matrix, typically N x 384)
  - `{slide_id}_coords.npy` (level-0 patch coordinates, N x 2)

Operational caveat:

- Some historical ingestion scripts can leave embeddings without coords. Use `scripts/recover_missing_coords.py` when needed before heatmap generation.

Runtime behavior in the API:

- `/api/analyze-multi` and both heatmap endpoints use `_resolve_embedding_path(...)` with strict level-0 lookup.
- Heatmap endpoints require `*_coords.npy`; missing coords return an explicit error instead of synthetic fallback grids.
- Several legacy endpoints (`/api/analyze`, `/api/analyze-batch`, `/api/report`, async report variants) still read `{project_embeddings_dir}/{slide_id}.npy` directly.

## 14.4 Dataset Preparation and Pool Rebuilding

Pool-generation scripts are CLI-driven and reusable across projects:

- `scripts/prepare_ov_endpoint_api_pool.py`: builds OV endpoint pools (`survival_1y/3y/5y`, `tumor_grade`) from GDC + bucket manifests.
- `scripts/prepare_lung_stage_api_pool.py`: builds LUAD stage pools from GDC + bucket manifests.
- `scripts/prepare_platinum_api_expanded_pool.py` and `scripts/prepare_platinum_bucket_pool.py`: build ovarian platinum pools with strict/expanded labeling variants.
- `scripts/rebuild_multimodel_pools_barcode_balanced.py`: rebuilds multi-endpoint pools using barcode matching, balancing, and reproducible sampling.

Output location is configurable via `--out_dir` (commonly `data/tcga_full/trainval_pool/`). Typical artifacts are:

- `<prefix>_meta.csv`
- `<prefix>_labels.csv`
- `<prefix>_file_ids.txt`
- `<prefix>_h5_uris.txt`
- balanced variants for scripts that emit them

Training/evaluation scripts consume these files through explicit CLI arguments (`--labels_file`, `--embeddings_dir`, etc.), so pool location is not hardcoded.

---

# 15. Security and Compliance

## 15.1 On-Premise Deployment Posture

Enso Atlas is designed to run on local infrastructure, but true air-gapped operation depends on deployment policy as well as application settings.

- **Local runtime:** `docker/docker-compose.yaml` deploys the API service and PostgreSQL on the same host/network.
- **Local data paths:** Slides, embeddings, model weights, and cache directories are mounted as local volumes/bind mounts.
- **Offline model loading (when enabled):** Set `TRANSFORMERS_OFFLINE=1` and `HF_HUB_OFFLINE=1` after pre-caching model artifacts. The embedding loader then uses local-only Hugging Face resolution.
- **Telemetry surface:** The default config sets `enable_telemetry: false`; no external telemetry backend is wired in the current codebase.
- **Network boundary requirement:** The application itself does not enforce egress firewall rules. If you require zero outbound traffic, enforce it at the network layer.

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
           X  Outbound paths should be blocked by policy
           |
```

## 15.2 HIPAA-Oriented Controls (Current Implementation)

The codebase includes several controls that support HIPAA-aligned deployments, but some controls are deployment responsibilities rather than built-in guarantees.

- **Audit logging exists, but is in-memory by default:** The API writes audit events to an in-process deque (`maxlen=500`) and serves them via `GET /api/audit-log`.
- **Event content:** Audit records include timestamp, event type, `slide_id`, `user_id`, and structured details. Treat identifiers as potentially sensitive based on local naming practices.
- **Access control is externalized:** There is no built-in app-level authentication/authorization layer in the API; production deployments should place Enso Atlas behind institutional identity/access controls.
- **Encryption is configurable at deployment time:** TLS and PostgreSQL SSL are not enforced by default in Compose; enable them through your reverse proxy/DB connection settings and infrastructure hardening.

### Audit Event Types (Implemented)

| Event Type | Trigger |
|---|---|
| `analysis_completed` | Single-slide analysis completion |
| `multi_model_analysis` | Multi-model analysis run |
| `batch_analysis_slide` | Per-slide event during batch analysis |
| `batch_analysis_completed` | Synchronous batch completion |
| `batch_analysis_async_completed` | Asynchronous batch completion |
| `batch_analysis_cancelled` | Batch cancellation request |
| `uncertainty_analysis_completed` | MC Dropout uncertainty analysis |
| `pdf_exported` | PDF export |
| `annotation_created` | Annotation creation |
| `annotation_deleted` | Annotation deletion |
| `visual_search` | Visual similarity search |
| `patch_classification` | Few-shot patch classification |
| `outlier_detection` | Outlier tissue detection |

## 15.3 Research-Use Safeguards

Current report-generation paths include explicit research-use protections:

- Generated MedGemma report payloads include a required `safety_statement` field.
- PDF export adds a prominent "UNCALIBRATED MODEL - RESEARCH USE ONLY" warning banner.
- Report validation rejects a small set of treatment-directive phrases (for example, "start treatment" and "prescribe").

These safeguards reduce obvious misuse, but they are not a substitute for clinical validation, governance review, or regulatory clearance.

---

# 16. Performance Benchmarks

This section keeps only numbers that are directly supported by the current repository (startup timing table, code-level limits/targets, and project config values). Where full benchmark methodology is not packaged in-repo, values are presented as operational estimates from the audited environment, not universal guarantees.

## 16.1 Startup Time (Operational Estimate)

These startup timings come from the audited dev/deployment environment and should be treated as approximate planning numbers.

| Component | Duration |
|---|---|
| TransMIL loading | 2.1s |
| MedGemma loading | 28.4s |
| MedGemma warmup | 62.3s |
| MedSigLIP loading | 9.7s |
| FAISS index build | 3.2s |
| PostgreSQL init (schema + population) | 4.8s |
| **Model/service startup subtotal** | **~120s** |
| **End-to-end Docker startup (README reference)** | **~3.5 minutes** |

## 16.2 Request Latency (Verified Constraints)

| Path / Operation | Verified Behavior |
|---|---|
| `GET /api/slides` (PostgreSQL path) | Code path is designed for sub-100ms response time |
| `GET /api/slides` (flat-file fallback) | Documented fallback can take 30-60s when DB is unavailable |
| MedGemma report generation | Runtime enforces timeout bounds (`max_generation_time_s` defaults to 300s; async wrapper allows `max_time + 60s`) |

## 16.3 GPU Memory Budget (Sizing Estimates)

| Model | VRAM | Precision |
|---|---|---|
| MedGemma 4B | ~8 GB | bfloat16 |
| MedSigLIP | ~800 MB | fp16 |
| TransMIL model set | ~240 MB | float32 |
| FAISS index | CPU only | float32 |
| **Estimated total GPU footprint** | **~9 GB** | |

## 16.4 Database Performance Notes

The strongest verified performance delta in the current codebase is the slide listing path:

- PostgreSQL-backed slide listing is intended to stay under 100ms for normal operation.
- If PostgreSQL is unavailable, the system falls back to flat-file scanning, which is documented at 30-60s.

These two numbers are retained because they correspond to explicit behavior documented in the backend implementation.

## 16.5 Project Model Metrics (Config Snapshot, Research-Only)

| Project / Model | Metric | Value | Provenance / Caveat |
|---|---|---|---|
| Ovarian / `platinum_sensitivity` | AUC | 0.907 | From `config/projects.yaml` metadata; not an externally audited benchmark in this repo snapshot. |
| Ovarian / `platinum_sensitivity` | Threshold (project config) | 0.9229 | Project-level operating threshold in config; historical threshold scripts/configs may differ by training snapshot. |
| Lung / `lung_stage` | AUC | 0.648 | From `config/projects.yaml` metadata; no multi-site validation evidence included here. |

---

# 17. Mac Mini Deployment Feasibility

## 17.1 Overview

Enso Atlas supports on-premise deployment on consumer Apple hardware. The 2024 Mac mini ships in two configurations: M4 (16 GB unified memory, configurable to 32 GB) and M4 Pro (24 GB unified memory, configurable to 64 GB). There is no M4 Max Mac mini; that chip is limited to MacBook Pro and Mac Studio. The M4 Pro with at least 24 GB is the recommended minimum for running the full inference stack, since the combined model footprint leaves little room on a 16 GB machine once the OS and application overhead are accounted for.

## 17.2 Model Memory Requirements

| Model | Memory Required | Notes |
|---|---|---|
| Path Foundation | ~2 GB | TensorFlow on CPU; runs natively on Apple Silicon via tensorflow-macos |
| TransMIL (6 project-scoped models) | ~240 MB | PyTorch MPS or CPU, ~3.2M params each |
| MedGemma 4B | ~8 GB (bfloat16) | 4B params x 2 bytes; requires MPS backend on Apple Silicon |
| MedSigLIP | ~3 GB | SigLIP-so400m full model |
| FAISS Index | ~50 MB | 208 slides, CPU-only |
| PostgreSQL | ~200 MB | Standard working-set footprint |
| **Total** | **~13.5 GB** | Fits on M4 Pro (24 GB) with headroom for OS and services |

On the base M4 (16 GB), the models technically fit, but macOS itself consumes 4-6 GB at idle. Sustained inference with all models loaded simultaneously will trigger memory pressure and swap. For reliable operation, use the M4 Pro or configure the M4 with 24 GB or more.

## 17.3 Performance Expectations (Illustrative Estimates)

The following ranges are directional estimates from internal testing and implementation constraints; they are not a formal benchmark suite.

| Operation | DGX Spark (Blackwell) | Mac mini (M4 Pro) |
|---|---|---|
| Multi-model inference (project-scoped model set) | 100-250ms | 500ms-2s (MPS) |
| MedGemma report generation | ~20s | 30-60s (MPS) |
| Path Foundation embedding (64 patches) | 2.5s (CPU) | 3-5s (CPU) |
| Heatmap generation | 50-100ms | 100-200ms |
| PostgreSQL queries | 2-5ms | 2-5ms |

The MPS backend on M4 Pro provides roughly 2-5x slower throughput than a Blackwell GPU for these workloads. For single-user clinical review, this latency is acceptable. Batch processing large slide cohorts is better suited to server hardware.

## 17.4 Deployment Configuration

- Set `device: "mps"` for PyTorch models (MedGemma, TransMIL). The MPS backend is stable on PyTorch 2.1+ with macOS Sonoma or later.
- Path Foundation remains on CPU (TensorFlow). The tensorflow-metal plugin can offload to GPU, but Path Foundation's TensorFlow graph runs faster on Apple Silicon CPU cores for this workload.
- PostgreSQL via Homebrew (`brew install postgresql@16`) or Docker. No GPU passthrough is involved.
- Frontend via Node.js (native ARM64 builds available).
- All services run as native processes; Docker is optional and only needed for PostgreSQL isolation.

## 17.5 Comparison

| Attribute | Mac mini M4 Pro | DGX Spark |
|---|---|---|
| Cost | ~$1,600 | ~$3,999+ |
| Peak power draw | 155W (Apple spec) | ~500W |
| Noise at idle | 5 dBA (near-silent) | Active cooling |
| Form factor | 12.7 cm x 12.7 cm x 5.0 cm | Desktop workstation |
| Maintenance | macOS with automatic updates | Ubuntu/NVIDIA stack |
| Data locality | All data stays on-device | All data stays on-device |

The Mac mini draws significantly less power than the DGX Spark, though Apple's rated maximum of 155W is well above the ~40W sometimes cited for idle operation. Under sustained ML inference, expect 50-100W depending on MPS and CPU load.

---

# 18. Known Limitations and Future Work

## 18.1 Current Limitations

### Class Imbalance
One documented ovarian training subset in project notes reports 139 non-responders vs 13 responders (10.7:1 ratio). Because cohort manifests differ across snapshots/exports, treat this as an example imbalance profile rather than a canonical count for every run. Training uses focal loss with inverse-frequency class weighting, which helps but does not eliminate the problem. The threshold optimization script exists (`scripts/optimize_threshold.py`) but threshold artifacts and configured project thresholds have diverged across snapshots; these should be re-derived on a pinned validation split and versioned with provenance.

### Path Foundation Is CPU-Only
Path Foundation uses a TensorFlow SavedModel. TensorFlow has no GPU support on Apple Silicon and limited support on newer NVIDIA architectures. In practice, all Path Foundation embeddings are pre-computed offline and stored as .npy files. Real-time embedding at inference is not feasible with this model.

### No Authentication or Access Control
There is no user login, no RBAC, no API key enforcement. Anyone with network access to the backend can run predictions, view slides, and generate reports. For single-user local deployments this is acceptable; for any shared or networked deployment it is not.

### TCGA-Only, Single-Site Data
Every model is trained exclusively on TCGA data. There is no multi-site validation. Performance on slides from other institutions, scanners, or staining protocols is unknown and likely worse.

### Variable Model Quality
The six classification models span a wide AUC range. Platinum sensitivity sits at 0.907, but the survival models (1-year, 3-year, 5-year) range from 0.639 to 0.697 -- barely above chance for a binary classifier. The lung stage model is at 0.648. The UI shows these AUCs, but users may not appreciate what a 0.64 AUC means in practice.

### No FDA Clearance
This is a research tool. Predictions are raw sigmoid outputs with no probability calibration. Nothing here is validated for clinical decision-making.

### Semantic Search Cold Start
The first MedSigLIP embedding run for a slide takes roughly 12-14 minutes for around 6,680 patches. Results are cached after the first run, so subsequent searches are fast, but the initial wait is real and there is no progress indicator in the UI beyond a polling spinner.

### No DICOM or PACS Integration
Slides must be manually placed as files in the project data directory. There is no DICOM import, no PACS connectivity, and no way to pull slides from a clinical image management system.

## 18.2 Planned Improvements

### Near-Term
- Run threshold optimization on all models and commit real Youden J / F1 thresholds to replace the placeholder config
- Platt scaling or isotonic regression for probability calibration (the "uncalibrated" disclaimers are honest but the outputs should still be better)
- Basic API key or token authentication -- even a shared secret would be better than nothing
- UI progress feedback for long-running embedding and analysis tasks

### Medium-Term
- Additional cancer programs: BRCA setup script exists (`scripts/setup_brca_project.py`) but is not yet in production. Colorectal and prostate are candidates after that
- Multi-site validation with non-TCGA cohorts
- GPU-accelerated embedding pipeline for DINOv2/UNI (these are PyTorch and already support CUDA/MPS, but the integration into the real-time API path is not wired up)
- Multimodal integration (H&E + IHC on the same case)
- DICOM/PACS slide import

### Long-Term
- FDA 510(k) regulatory pathway (requires prospective clinical validation, which has not started)
- EHR integration (FHIR)
- Federated learning across institutions
- Foundation model fine-tuning on institutional data

---

# 19. Appendices

## Appendix A: Environment Variables (Current)

| Variable | Default / Example | Used By | Notes |
|---|---|---|---|
| `DATABASE_URL` | `postgresql://enso:enso_atlas_2024@atlas-db:5432/enso_atlas` | Backend DB pool (`src/enso_atlas/api/database.py`) | Compose value matches code fallback. |
| `ENSO_CONFIG` | `config/default.yaml` (local) / `/app/config/default.yaml` (container) | UI launcher + backend config loading | Compose pins container path; code supports local fallback. |
| `MIL_ARCHITECTURE` | `clam` (code default), commonly `transmil` in Compose | MIL classifier initialization | Controls architecture selection at startup. |
| `MIL_THRESHOLD` | *(unset by default)* | MIL classifier initialization | Optional explicit threshold override; otherwise model/config threshold is used. |
| `MIL_THRESHOLD_CONFIG` | Auto-resolved to `threshold_config.json` near model checkpoint | MIL classifier initialization | Compose typically sets `/app/models/threshold_config.json`. |
| `MIL_MODEL_PATH` | *(auto-resolved)* | MIL classifier initialization | Optional explicit checkpoint path override. |
| `MIL_MIN_PATCHES_FOR_ATTENTION` | `128` | MIL inference (`src/enso_atlas/mil/clam.py`) | Lower bound used when subsampling very large bags. |
| `TRANSFORMERS_OFFLINE` | `0` | Embedding + reporting model loaders | Set to `1` for fully offline HuggingFace behavior. |
| `HF_HUB_OFFLINE` | `0` | Embedding + reporting model loaders | Same operational intent as `TRANSFORMERS_OFFLINE`. |
| `HF_HOME` | `~/.cache/huggingface` (local fallback) / `/app/cache/huggingface` (Compose) | Embedding + reporting model cache | Primary HuggingFace cache location. |
| `TRANSFORMERS_CACHE` | *(unset unless provided)* | Embedding loader fallback path | Used if `HF_HOME` is not set. |
| `HF_TOKEN` / `HUGGINGFACE_TOKEN` / `HUGGINGFACE_HUB_TOKEN` | *(optional)* | MedGemma model loading | Required only for gated/private HuggingFace assets. |
| `NVIDIA_VISIBLE_DEVICES` | `all` (Compose) | Container GPU visibility | Relevant in GPU container deployments. |
| `NEXT_PUBLIC_API_URL` | Local dev commonly `http://127.0.0.1:8000`; Docker backend commonly `http://127.0.0.1:8003` | Next.js frontend | `frontend/.env.example` defaults to `8000`; set explicitly to `8003` when targeting Docker backend. |

## Appendix B: Repository Map (Audited)

```text
<repo-root>/
+-- .github/
+-- config/
|   +-- default.yaml
|   +-- projects.yaml
+-- data/
|   +-- brca/
|   +-- luad/
|   +-- tcga_full/
|   +-- labels.csv
|   +-- slide_metadata.json
+-- docker/
|   +-- Dockerfile
|   +-- docker-compose.yaml
+-- frontend/
|   +-- src/app/
|   |   +-- page.tsx
|   |   +-- slides/page.tsx
|   |   +-- projects/page.tsx
|   +-- src/components/
|   |   +-- layout/
|   |   +-- panels/
|   |   +-- viewer/
|   |   +-- slides/
|   |   +-- modals/
|   |   +-- demo/
|   |   +-- ui/
|   +-- src/contexts/ProjectContext.tsx
|   +-- src/hooks/
|   +-- src/lib/
|   +-- src/types/
+-- models/
|   +-- transmil.py
|   +-- threshold_config.json
|   +-- medgemma-4b-it/
|   +-- path-foundation/
+-- scripts/
|   +-- download_models.py
|   +-- embed_level0_pipelined.py
|   +-- batch_reembed_level0.py
|   +-- multi_model_inference.py
|   +-- train_transmil*.py
|   +-- evaluate_transmil.py
|   +-- setup_luad_project.py
|   +-- setup_brca_project.py
|   +-- validate_project_modularity.py
|   +-- validate_project_scoping.py
|   +-- start.sh
+-- src/enso_atlas/
|   +-- api/
|   +-- agent/
|   +-- embedding/
|   +-- evidence/
|   +-- llm/
|   +-- mil/
|   +-- reporting/
|   +-- ui/
|   +-- wsi/
|   +-- cli.py
|   +-- config.py
|   +-- core.py
+-- tests/
+-- docs.md
+-- README.md
+-- pyproject.toml
+-- requirements.txt
```

Runtime convention from `config/projects.yaml` remains project-scoped (`data/projects/<project-id>/{slides,embeddings,labels.*}`), even though this repository snapshot also includes dataset-specific directories under `data/` for training and setup workflows.

## Appendix C: Key Design Decisions

### 1. Configuration-Driven Project Scoping
`config/projects.yaml` is the source of truth for project IDs, dataset paths, model compatibility, and feature flags. API behavior is intentionally scoped by `project_id` to prevent cross-project leakage.

### 2. Raw SQL over ORM
The backend uses `asyncpg` with explicit SQL for predictable performance and clear operational behavior. This keeps schema and query paths transparent during debugging.

### 3. Coverage-Aligned Heatmap Geometry
Heatmap dimensions are aligned to the patch grid (`224x224`) to keep overlays pixel-consistent with model attention outputs.

### 4. Disk-Backed Caching for Expensive Artifacts
Heatmaps, embeddings, and derived assets are cached on disk to make repeat requests fast and to reduce repeated compute.

### 5. Canvas Overlay Layering in the Frontend
Overlay rendering is separated from base slide rendering so visibility, opacity, and interaction controls remain independent and predictable.

### 6. Resilient Report Generation Path
When LLM-based generation is unavailable or times out, the system can still return structured, evidence-grounded outputs instead of failing hard.

### 7. FAISS for Retrieval Workloads
Similar-case retrieval uses FAISS indexes over normalized embedding representations to keep nearest-neighbor lookup efficient as slide volume grows.

## Appendix D: Error Codes

| HTTP Code | Typical Error Class | Resolution Guidance |
|---|---|---|
| 400 | Invalid request shape or missing prerequisites (for example, level-0 embeddings required) | Validate request payload and ensure prerequisite embeddings/resources exist. |
| 404 | Slide/session/task/resource not found | Verify IDs and project scope, then retry. |
| 409 | Resource conflict (state mismatch or duplicate operation) | Re-fetch latest state before retrying the operation. |
| 500 | Internal processing failure (prediction/report/embedding pipeline) | Inspect backend logs, model artifacts, and runtime resources (memory/storage). |
| 503 | Dependency unavailable (model/database/service warming up) | Wait for startup completion or recover dependent services, then retry. |

---

# Document Revision History

| Version | Date | Author | Changes |
|---|---|---|---|
| 1.0.0 | 2025-02-06 | Enso Labs | Initial specification |
| 2.0.0 | 2025-02-10 | Enso Labs | Complete revamp: added outlier detector, few-shot classifier, patch grid overlay, canvas-based overlays, annotation system (SVG + PostgreSQL), batch embedding, heatmap caching, coverage-based alignment, real-time scale bar, resizable panels, dark mode, Mac mini feasibility, Tailscale Funnel, schema v5, 80+ API endpoints, updated architecture diagrams, all current features |
| 2.1.0 | 2025-02-10 | Enso Labs | Reframed as modular platform architecture: added abstraction layer diagrams (foundation/classification/evidence/UI), emphasized plug-and-play model system, reframed HAI-DEF models as demo configuration rather than core value, added comprehensive Hospital Deployment Guide (13.1-13.10), strengthened config-driven project system documentation |
| 2.2.0 | 2026-02-20 | Enso Labs | Updated for multi-project architecture (ovarian + lung): project-scoped API documentation (`/api/analyze-multi` canonical path), `project_id` scope enforcement, level-0 dense embedding routing, project isolation guarantees across backend/frontend/task queues, and project-based data layout (`data/projects/{project-id}/...`). |

---

*Enso Atlas consists of approximately 30,000 lines of Python backend code and 15,000 lines of TypeScript frontend code.*
