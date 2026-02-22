# Enso Atlas: On-Premise Pathology Evidence Engine for Interpretable Multi-Project Clinical Prediction

## MedGemma Impact Challenge Submission

**Author:** Hanson Wen, UC Berkeley
**Repository:** https://github.com/Hilo-Hilo/enso-atlas

---

## 1. Problem Statement and Motivation

Ovarian cancer remains one of the deadliest gynecologic malignancies. Platinum-based chemotherapy is the standard first-line treatment following cytoreductive surgery, but approximately 30% of patients do not respond. Predicting platinum sensitivity from routine histopathology would enable personalized therapy selection, spare non-responders from ineffective treatment toxicity, and optimize clinical resources.

Current limitations in computational pathology hinder adoption: many AI tools require cloud infrastructure (raising PHI concerns), provide black-box predictions without interpretable evidence, and lack integration with clinical workflows. Pathologists need tools that explain why a prediction was made while keeping all patient data on-premise.

Enso Atlas addresses these gaps as an on-premise pathology evidence engine integrating all three Google HAI-DEF foundation models (Path Foundation, MedGemma, MedSigLIP) in a unified workflow. Recent project-system extensions also demonstrate platform extensibility beyond ovarian cancer, including lung adenocarcinoma stage classification.

---

## 2. System Architecture and HAI-DEF Model Integration

### 2.1 Pipeline Overview

Enso Atlas processes whole-slide images through a four-stage pipeline:

1. **WSI Ingestion**: OpenSlide reads SVS/NDPI/TIFF formats; Otsu thresholding detects tissue regions
2. **Feature Extraction**: Path Foundation (ViT-S) extracts 384-dimensional embeddings from 224x224 patches at level 0 (full resolution), yielding 6,000-30,000 patches per slide
3. **Slide-Level Classification**: TransMIL aggregates patch embeddings into slide-level predictions with per-patch attention weights
4. **Evidence Generation**: Attention heatmaps, top-K evidence patches, FAISS similar case retrieval, MedSigLIP semantic search, and MedGemma report generation

### 2.2 HAI-DEF Model Usage

**Path Foundation** is the feature backbone. Each patch is embedded into a 384-dimensional vector optimized for histopathology morphology. Level-0 dense embeddings are the default analysis pathway and are cached for reuse across project-scoped models.

**MedGemma 1.5 4B** generates structured clinical reports from visual evidence. The model receives evidence patches, prediction scores, and attention weights, producing morphology-focused summaries with safety disclaimers.

**MedSigLIP** enables semantic text-to-patch retrieval for natural-language tissue queries and clinician-guided validation.

### 2.3 Project-Scoped TransMIL Classification

The current system supports two projects (`ovarian-platinum`, `lung-stage`) and six classification models:

| Project | Model | AUC-ROC | Slides |
|---------|-------|---------|--------|
| ovarian-platinum | Platinum Sensitivity | 0.907 | 199 |
| ovarian-platinum | Tumor Grade | 0.752 | 918 |
| ovarian-platinum | 5-Year Survival | 0.697 | 965 |
| ovarian-platinum | 3-Year Survival | 0.645 | 1,106 |
| ovarian-platinum | 1-Year Survival | 0.639 | 1,135 |
| lung-stage | Tumor Stage (Early I/II vs Advanced III/IV) | 0.648 | 130 |

Best single-model AUC remains 0.907 (ovarian platinum sensitivity). The LUAD stage model demonstrates config-driven extension to a second cancer type without core service rewrites.

### 2.4 Agentic AI Assistant

A 7-step agentic workflow orchestrates full analysis:

1. Initialize case context from project configuration
2. Run project-scoped TransMIL predictions
3. Retrieve similar cases via FAISS
4. Perform semantic tissue search (feature-flagged by project)
5. Compare against project-specific cohort statistics
6. Generate a reasoning chain from accumulated evidence
7. Produce a MedGemma report with patch-level citations

---

## 3. Implementation, Results, and Impact

### 3.1 Architecture

- **Backend**: FastAPI + PostgreSQL (asyncpg) with Docker deployment on port 8003
- **Frontend**: Next.js 14.2 + TypeScript + Tailwind CSS on port 3002
- **Database**: PostgreSQL tables for patients, slides, analysis results, embedding tasks, projects, and project-model/project-slide junctions
- **Config-driven project system**: `projects.yaml` defines project datasets, model sets, and feature flags
- **Project isolation**: API routing, model resolution, slide retrieval, heatmap generation, and report generation are enforced per project ID end-to-end

### 3.2 Clinical Interface

The frontend provides a 3-panel resizable layout (Case Selection, WSI Viewer, Analysis Results) with three view modes:

- **Oncologist View**: Summary dashboard with prediction scores and confidence bands
- **Pathologist View**: Annotation tools, grading controls, and morphology review
- **Batch View**: Multi-slide parallel processing with export support

Key features include TransMIL attention heatmaps, evidence patches, FAISS similar-case retrieval, MedSigLIP semantic search, project management CRUD, project-scoped model selection, and PostgreSQL result caching.

### 3.3 Dataset Coverage

**TCGA-OV (ovarian-platinum):** 208 whole-slide images with platinum labels plus larger subsets for grade/survival tasks.

**TCGA-LUAD (lung-stage):** 130 slides with stage labels used for early (I/II) vs advanced (III/IV) classification.

Both projects use level-0 Path Foundation embeddings with project-scoped storage and retrieval.

### 3.4 Performance

| Metric | Value |
|--------|-------|
| Path Foundation embedding (CPU) | ~2-3 min/slide |
| TransMIL inference | <1 second |
| FAISS similarity search | <100 ms |
| MedGemma report generation (GPU) | ~20 seconds |
| Cached analysis retrieval (PostgreSQL) | ~0.8 ms |

### 3.5 Deployment

Hardware: NVIDIA DGX Spark (ARM64, GB10 GPU, 128GB unified memory). Deployment is fully local after setup with Docker Compose and a production Next.js frontend build.

### 3.6 Limitations and Future Work

- Path Foundation currently runs on CPU in this environment
- Additional external cohorts are needed for broader validation
- Cross-validation variance remains high for low-prevalence endpoints
- Future directions: additional cancer projects beyond current ovarian + lung support, EHR/LIMS integration, and prospective validation

---

## References

1. Google Health AI. Path Foundation Model Documentation.
2. Google Health AI. MedGemma 1.5 Model Card.
3. Shao Z, et al. TransMIL: Transformer-based Correlated Multiple Instance Learning for Whole Slide Image Classification. NeurIPS 2021.
4. Cancer Genome Atlas Research Network. Integrated genomic analyses of ovarian carcinoma. Nature 2011.
5. TCGA-LUAD Program Documentation, Genomic Data Commons.

---

*This work is a research prototype for decision support. It is not intended for autonomous clinical decision-making and has not been validated as a medical device. Outputs must be interpreted by qualified healthcare professionals in full clinical context.*
