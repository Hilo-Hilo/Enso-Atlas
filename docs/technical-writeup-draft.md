# Enso Atlas: On-Premise Pathology Evidence Engine for Treatment Response Prediction

## Abstract

Enso Atlas is an on-premise pathology decision support system that predicts treatment response in ovarian cancer patients using whole-slide histopathology images. Built on all three Google HAI-DEF foundation models (Path Foundation, MedGemma 1.5 4B, MedSigLIP), the system provides interpretable evidence through attention heatmaps, similar case retrieval, semantic tissue search, and structured clinical reports. Deployed via Docker on NVIDIA DGX Spark (ARM64, 128GB unified memory), Enso Atlas processes slides entirely on local hardware while maintaining patient data privacy.

## 1. Problem and Motivation

### Clinical Need

Ovarian cancer treatment decisions rely heavily on histopathological assessment of tumor morphology. Predicting response to platinum-based chemotherapy remains challenging, with approximately 30% of patients not responding to initial treatment. The ability to predict platinum sensitivity before initiating therapy would enable personalized treatment selection and spare non-responders from ineffective toxicity.

### Gap

Existing computational pathology tools often require cloud infrastructure, raising PHI concerns. They also provide predictions without interpretable evidence, limiting clinical adoption. Pathologists need tools that explain *why* a prediction was made, not just the prediction itself.

### Our Approach

Enso Atlas addresses both gaps:
1. **On-premise deployment** via Docker Compose on consumer/workstation GPUs
2. **Evidence-first design** providing attention heatmaps, similar case retrieval, semantic search, and structured clinical reports alongside every prediction
3. **All three HAI-DEF models** integrated into a unified pipeline

## 2. System Architecture

### Data Pipeline

1. **Whole-slide image ingestion** via OpenSlide (supports .svs, .ndpi, .tiff)
2. **Tissue detection and patching** at level 0 (full resolution, 224x224 patches)
3. **Feature extraction** using Path Foundation (384-dimensional embeddings per patch)
4. **Slide-level prediction** via TransMIL (Transformer-based Multiple Instance Learning)
5. **Evidence generation** through attention weight analysis, FAISS similarity search, MedSigLIP semantic search, and MedGemma report generation

### HAI-DEF Model Integration

- **Path Foundation**: Frozen feature extractor for H&E histopathology patches. ViT-S architecture producing 384-dim embeddings. Runs on CPU due to TensorFlow/Blackwell GPU incompatibility on ARM64.
- **MedGemma 1.5 4B**: Local inference for structured clinical report generation. Receives prediction scores, evidence patches, and attention weights; outputs JSON-structured morphology descriptions with safety disclaimers. ~20s/report on GPU.
- **MedSigLIP**: Semantic search over tissue regions using natural language queries (e.g., "tumor infiltrating lymphocytes"). Enables pathologist-guided exploration of slide content.

### Agentic Workflow

The AI Assistant orchestrates a 7-step analysis pipeline:
1. Initialize case context from project configuration
2. Run TransMIL prediction across all applicable models
3. Retrieve similar cases via FAISS index
4. Perform semantic tissue search (MedSigLIP)
5. Compare against reference cohort statistics
6. Generate reasoning chain from accumulated evidence
7. Produce MedGemma clinical report with evidence citations

### Application Architecture

- **Backend**: FastAPI + PostgreSQL (asyncpg), Docker deployment, port 8003
- **Frontend**: Next.js 14.2 + TypeScript + Tailwind CSS, port 3002
- **Database**: PostgreSQL with tables for patients, slides, slide_metadata, analysis_results, embedding_tasks, projects, project_models, project_slides
- **Config**: projects.yaml drives project definitions; DB junction tables enable many-to-many project-scoped associations

Three viewing modes: Oncologist (summary dashboard), Pathologist (annotation tools, grading), and Batch (multi-case parallel processing).

## 3. Experiments and Results

### Dataset

TCGA Ovarian Cancer cohort: 208 whole-slide images with platinum sensitivity labels (binary: responder/non-responder). 5-fold stratified cross-validation.

Note: Originally planned to use the Ovarian Bevacizumab Response dataset from PathDB/TCIA (288 WSIs, 78 patients), but the PathDB server returned 0-byte files for 217/286 slides, blocking that approach.

### TransMIL Training

- Input: Path Foundation embeddings (384-dim, level 0 patches, ~6,000--30,000 per slide)
- Architecture: TransMIL with pyramid position encoding (512-dim, 8 heads, 2 layers)
- Training: AdamW optimizer, lr=2e-4, weight decay=0.01, 100 epochs, early stopping (patience=15)
- Loss: Focal loss with class weighting for severe imbalance (91.4% positive, 8.6% negative)

### Results: 5 Specialized Models

| Model | AUC-ROC | Slides |
|-------|---------|--------|
| Platinum Sensitivity | 0.907 | 199 |
| Tumor Grade | 0.752 | 918 |
| 5-Year Survival | 0.697 | 965 |
| 3-Year Survival | 0.645 | 1,106 |
| 1-Year Survival | 0.639 | 1,135 |

### Cross-Validation (Platinum Sensitivity)

| Fold | AUC-ROC | Best Epoch |
|------|---------|------------|
| 1 | 0.810 | 11 |
| 2 | 0.667 | 1 |
| 3 | 0.661 | 2 |
| 4 | 0.536 | 8 |
| 5 | 0.864 | 4 |

- **Mean AUC: 0.707 +/- 0.117** (5-fold CV)
- **Best model AUC: 0.879** (full dataset evaluation)
- Optimal threshold via Youden's J: 0.917 (sensitivity=83.5%, specificity=84.6%)

High variance across folds is attributable to the small negative class (only 2--3 negatives per fold).

### Performance Benchmarks

| Operation | Time |
|-----------|------|
| Patch embedding (Path Foundation, CPU) | 2--3 min/slide |
| TransMIL inference | < 1 second |
| FAISS similarity search | < 100 ms |
| MedGemma report generation (GPU) | ~20 seconds |
| Cached result retrieval (PostgreSQL) | 0.8 ms |
| Backend startup (model loading) | ~3.5 min |

## 4. Discussion

### Strengths

- Fully on-premise: zero PHI exposure
- Evidence-first: attention maps + retrieval + semantic search + reports
- All three HAI-DEF models integrated end-to-end
- 7-step agentic workflow for comprehensive analysis
- Config-driven project system with multi-model ensemble
- PostgreSQL result caching for sub-millisecond repeated queries
- Batch processing for clinical throughput

### Limitations

- Path Foundation runs on CPU only (TensorFlow/Blackwell incompatibility)
- Training cohort limited to TCGA (potential single-institution bias)
- High CV variance due to small negative class
- No prospective clinical validation
- Not validated as a medical device; research use only

### Future Work

- PyTorch Path Foundation port for GPU acceleration
- Multi-cancer type support
- Integration with LIMS/EHR systems
- Stain normalization (Macenko) for cross-site robustness
- Prospective validation study

## 5. Reproducibility

- Source code: https://github.com/Hilo-Hilo/med-gemma-hackathon
- Deployment: `docker compose -f docker/docker-compose.yaml up -d`
- Frontend: `cd frontend && npm run build && npx next start -p 3002`
- Hardware tested: NVIDIA DGX Spark (ARM64, GB10 GPU, 128GB unified memory)
- All models available via HuggingFace (Path Foundation, MedGemma 1.5 4B, MedSigLIP)

## Safety Statement

Enso Atlas is a research decision-support tool. It is not a medical device and has not been validated for clinical use. All predictions and reports must be reviewed by qualified pathologists before informing treatment decisions.
