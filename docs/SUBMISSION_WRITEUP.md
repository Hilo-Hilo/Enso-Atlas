# Enso Atlas: On-Premise Pathology Evidence Engine for Ovarian Cancer Treatment Response Prediction

## MedGemma Impact Challenge Submission

**Author:** Hanson Wen, UC Berkeley
**Repository:** https://github.com/Hilo-Hilo/med-gemma-hackathon

---

## 1. Problem Statement and Motivation

Ovarian cancer remains one of the deadliest gynecologic malignancies. Platinum-based chemotherapy is the standard first-line treatment following cytoreductive surgery, but approximately 30% of patients do not respond. Predicting platinum sensitivity from routine histopathology would enable personalized therapy selection, spare non-responders from ineffective treatment toxicity, and optimize clinical resources.

Current limitations in computational pathology hinder adoption: most AI tools require cloud infrastructure (raising PHI concerns), provide black-box predictions without interpretable evidence, and lack integration with clinical workflows. Pathologists need tools that explain *why* a prediction was made -- not just the prediction itself -- while keeping all patient data on-premise.

Enso Atlas addresses these gaps as an **on-premise pathology evidence engine** that integrates all three Google HAI-DEF foundation models (Path Foundation, MedGemma, MedSigLIP) into a unified, evidence-first clinical decision support system.

---

## 2. System Architecture and HAI-DEF Model Integration

### 2.1 Pipeline Overview

Enso Atlas processes whole-slide images through a four-stage pipeline:

1. **WSI Ingestion**: OpenSlide reads SVS/NDPI/TIFF formats; Otsu thresholding detects tissue regions
2. **Feature Extraction**: Path Foundation (ViT-S) extracts 384-dimensional embeddings from 224x224 patches at level 0 (full resolution), yielding 6,000--30,000 patches per slide
3. **Slide-Level Classification**: TransMIL (Transformer-based Multiple Instance Learning) with pyramid position encoding aggregates patch embeddings into slide-level predictions with per-patch attention weights
4. **Evidence Generation**: Attention heatmaps, top-K evidence patches, FAISS similar case retrieval, MedSigLIP semantic search, and MedGemma clinical report generation

### 2.2 HAI-DEF Model Usage

**Path Foundation** serves as the feature backbone. Each patch is embedded into a 384-dimensional vector optimized for histopathology morphology. Embeddings are cached as FP16 arrays (~15 MB per slide), enabling rapid reprocessing for multiple classification tasks. Path Foundation currently runs on CPU due to TensorFlow/Blackwell GPU incompatibility on our ARM64 deployment target.

**MedGemma 1.5 4B** generates structured clinical reports from visual evidence. The model receives evidence patches, prediction scores, and attention weights, producing JSON-structured morphology descriptions with safety disclaimers. Reports are constrained to describe visible morphological features and explicitly avoid treatment recommendations. Generation takes approximately 20 seconds per report on GPU.

**MedSigLIP** enables semantic text-to-patch retrieval. Pathologists type natural language queries (e.g., "tumor infiltrating lymphocytes," "necrosis," "mitotic figures") and instantly retrieve matching tissue regions with similarity scores and coordinates. This human-centered feature allows clinicians to validate AI predictions against their domain expertise.

### 2.3 TransMIL Classification

We train five specialized TransMIL models on TCGA ovarian cancer data, each targeting a different clinical endpoint:

| Model | AUC-ROC | Slides |
|-------|---------|--------|
| Platinum Sensitivity | 0.907 | 199 |
| Tumor Grade | 0.752 | 918 |
| 5-Year Survival | 0.697 | 965 |
| 3-Year Survival | 0.645 | 1,106 |
| 1-Year Survival | 0.639 | 1,135 |

Best single-model AUC: 0.879 on the full dataset. The platinum sensitivity model achieves an optimal threshold of 0.917 (Youden's J statistic) with 83.5% sensitivity and 84.6% specificity. 5-fold stratified cross-validation yields a mean AUC of 0.707 +/- 0.117, with high variance attributable to the small negative class (only 2--3 negatives per fold in 199 slides).

Training configuration: AdamW optimizer, lr=2e-4, weight decay=0.01, focal loss with class weighting (91.4% positive / 8.6% negative), 100 epochs with early stopping (patience 15), pyramid position encoding (512-dim, 8 heads, 2 layers).

### 2.4 Agentic AI Assistant

A 7-step agentic workflow orchestrates the full analysis pipeline:

1. Initialize case context from project configuration
2. Run TransMIL prediction across all applicable models
3. Retrieve similar cases via FAISS index (208-slide reference cohort)
4. Perform semantic tissue search with MedSigLIP
5. Compare against reference cohort statistics
6. Generate reasoning chain from accumulated evidence
7. Produce MedGemma clinical report with citations to evidence patches

This agentic approach provides comprehensive, multi-model analysis in a single interaction -- a key differentiator from single-prediction tools.

---

## 3. Implementation, Results, and Impact

### 3.1 Architecture

- **Backend**: FastAPI + PostgreSQL (asyncpg) with Docker deployment on port 8003
- **Frontend**: Next.js 14.2 + TypeScript + Tailwind CSS on port 3002
- **Database**: PostgreSQL with tables for patients, slides, analysis results, embedding tasks, projects, and project-model/project-slide junction tables
- **Deployment**: Docker Compose on NVIDIA DGX Spark (ARM64, GB10 GPU, 128GB unified memory)
- **Config-driven project system**: projects.yaml defines available models, slides, and parameters; DB junction tables enable many-to-many project-scoped associations

### 3.2 Clinical Interface

The frontend provides a 3-panel resizable layout (Case Selection, WSI Viewer, Analysis Results) with three view modes:

- **Oncologist View**: Summary dashboard with prediction scores, confidence bands, and actionable recommendations
- **Pathologist View**: Full annotation tools (circle, rectangle, freehand, measure), mitotic counter, grading interface, and detailed morphology analysis
- **Batch View**: Multi-slide parallel processing with CSV export for cohort-level analysis

Key features include TransMIL attention heatmaps (jet colormap overlay on WSI), evidence patches with normalized attention weights, FAISS similar case retrieval with outcome display, MedSigLIP semantic search, Project Management UI with CRUD operations, Slide Manager with thumbnails and filtering, dark mode, PDF/JSON report export, and PostgreSQL result caching (0.8ms for cached results).

### 3.3 Dataset

We use the TCGA Ovarian Cancer cohort: 208 whole-slide images with platinum sensitivity labels derived from clinical follow-up data. Level 0 (full resolution) embeddings capture cellular-level morphological detail across 6,000--30,000 patches per slide.

Note: We originally planned to use the Ovarian Bevacizumab Response dataset from PathDB/TCIA (288 WSIs from 78 patients), but the PathDB download server returned 0-byte files for 217 of 286 slides, blocking that approach. The TCGA dataset provides a larger cohort with well-characterized clinical annotations.

### 3.4 Performance

| Metric | Value |
|--------|-------|
| Patch embedding (Path Foundation, CPU) | 2--3 min/slide |
| TransMIL inference | < 1 second |
| FAISS similarity search | < 100 ms |
| MedGemma report generation (GPU) | ~20 seconds |
| Cached analysis retrieval (PostgreSQL) | 0.8 ms |
| Backend startup (model loading) | ~3.5 min |

### 3.5 Deployment

Hardware: NVIDIA DGX Spark (ARM64, GB10 GPU, 128GB unified memory). Deployment: `docker compose -f docker/docker-compose.yaml up -d` for the backend; `cd frontend && npm run build && npx next start -p 3002` for the frontend. The system operates fully offline after initial setup -- no outbound network connections required, no PHI leaves the hospital network.

### 3.6 Limitations and Future Work

- Path Foundation runs on CPU only (TensorFlow/Blackwell incompatibility); a PyTorch port would enable GPU acceleration
- Training cohort limited to TCGA (potential single-institution bias); multi-site validation needed
- High cross-validation variance due to small negative class; additional labeled data would improve stability
- Not validated as a medical device; research use only
- Future directions: stain normalization for cross-site robustness, multi-cancer type support, EHR/LIMS integration, prospective clinical validation

---

## References

1. Google Health AI. "Path Foundation Model." https://developers.google.com/health-ai-developer-foundations/path-foundation
2. Google Health AI. "MedGemma 1.5 Model Card." https://developers.google.com/health-ai-developer-foundations/medgemma
3. Shao Z, et al. "TransMIL: Transformer based Correlated Multiple Instance Learning for Whole Slide Image Classification." NeurIPS 2021.
4. Cancer Genome Atlas Research Network. "Integrated genomic analyses of ovarian carcinoma." Nature 2011.
5. Johnson J, et al. "Billion-scale similarity search with GPUs." IEEE Transactions on Big Data 2019.

---

*This work is a research prototype for decision support. It is not intended for autonomous clinical decision-making and has not been validated as a medical device. All predictions should be interpreted by qualified healthcare professionals in the context of complete clinical information.*
