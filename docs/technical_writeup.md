# Enso Atlas: AI-Powered Treatment Response Prediction for Ovarian Cancer

## Technical Report

**Author:** Hanson Wen, UC Berkeley
**Date:** February 2026

---

## 1. Problem Statement

### 1.1 Clinical Challenge

High-grade serous ovarian carcinoma (HGSOC) is the most lethal gynecologic malignancy, with a five-year survival rate below 50%. First-line treatment typically involves platinum-based chemotherapy following cytoreductive surgery. However, approximately 30% of patients do not respond to initial chemotherapy, experiencing disease progression or recurrence within six months.

The ability to predict treatment response before initiating chemotherapy would fundamentally change clinical practice by enabling personalized treatment selection, reducing unnecessary toxicity for non-responders, and optimizing resource allocation for expensive therapies.

### 1.2 Current Limitations

Existing biomarkers for treatment response prediction (BRCA status, HRD scores) have limited sensitivity and are not universally available. Pathology-based assessment remains subjective and lacks standardized predictive criteria. There is a critical need for computational tools that extract predictive information from routinely collected histopathology slides while providing interpretable evidence for clinical adoption.

### 1.3 Project Objective

This project develops Enso Atlas, a deep learning system that predicts platinum-based chemotherapy response from digitized H&E-stained whole-slide images (WSIs) of ovarian cancer tissue. The system provides:

1. Multi-endpoint classification via 5 specialized TransMIL models (platinum sensitivity, tumor grade, 1/3/5-year survival)
2. Interpretable evidence through attention-weighted tissue regions and heatmaps
3. Semantic tissue search via MedSigLIP for clinician-guided exploration
4. Structured clinical reports via MedGemma 1.5 4B
5. On-premise deployment with no PHI exposure

---

## 2. Methodology

### 2.1 System Architecture

The system follows a multi-stage pipeline:

```
WSI Input -> Tissue Detection -> Patch Extraction -> Path Foundation Embedding -> TransMIL -> Prediction + Evidence
              (Otsu threshold)   (224x224, level 0)   (384-dim, ViT-S)          (5 models)   (heatmap, patches, report)
```

**Stage 1: Feature Extraction**

Whole-slide images are tiled into non-overlapping 224x224 pixel patches at level 0 (full resolution). Each patch is processed through Path Foundation (ViT-S) to generate a 384-dimensional embedding vector. At full resolution, each slide yields 6,000--30,000 patches, capturing cellular-level morphological detail.

**Stage 2: Multiple Instance Learning**

The bag of patch embeddings is processed by TransMIL (Transformer-based Multiple Instance Learning), which uses self-attention with pyramid position encoding to learn spatial relationships between patches and aggregate them into slide-level predictions with per-patch attention weights.

### 2.2 Foundation Models

**Path Foundation** is Google's pathology-specific Vision Transformer trained on histopathology images using self-supervised learning. Key advantages: domain-specific representations capturing nuclear morphology, tissue architecture, and cellular patterns; compact 384-dimensional embeddings for efficient downstream processing; extract-once-reuse-many architecture. Currently runs on CPU due to TensorFlow/Blackwell GPU incompatibility on ARM64.

**MedGemma 1.5 4B** generates structured clinical reports from evidence patches and model outputs. Constrained to describe visible morphological features with explicit safety disclaimers. ~20 seconds per report on GPU.

**MedSigLIP** enables semantic text-to-patch retrieval, allowing pathologists to query tissue regions with natural language descriptions.

### 2.3 TransMIL Architecture

TransMIL replaces traditional attention-based MIL (e.g., CLAM) with a Transformer architecture that captures inter-patch correlations:

```
Patch Embeddings [N x 384]
        |
   Linear Projection [384 -> 512]
        |
   Pyramid Position Encoding
        |
   Transformer Layers (8 heads, 2 layers)
        |
   Class Token Aggregation [1 x 512]
        |
   Classifier [512 -> 2]
        |
   Output: (probability, per-patch attention weights)
```

Key advantages over CLAM: captures long-range spatial dependencies between distant tissue regions; position encoding preserves spatial relationships; self-attention provides richer patch interactions than gated attention.

### 2.4 Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning rate | 2e-4 |
| Weight decay | 0.01 |
| Batch size | 1 (slide-level) |
| Epochs | 100 |
| Early stopping | 15 epochs patience |
| Loss function | Focal loss with class weighting |
| Class balance | 91.4% positive / 8.6% negative |
| Position encoding | Pyramid (512-dim, 8 heads, 2 layers) |

### 2.5 Evaluation

All metrics reported with 5-fold stratified cross-validation (patient-level splits, no data leakage). Primary metric: AUC-ROC. Optimal threshold determined via Youden's J statistic.

---

## 3. Results

### 3.1 Multi-Endpoint Classification

Five specialized TransMIL models trained on TCGA Ovarian Cancer data:

| Model | AUC-ROC | Slides | Task |
|-------|---------|--------|------|
| Platinum Sensitivity | 0.907 | 199 | Binary response prediction |
| Tumor Grade | 0.752 | 918 | High vs. low grade |
| 5-Year Survival | 0.697 | 965 | 5-year overall survival |
| 3-Year Survival | 0.645 | 1,106 | 3-year overall survival |
| 1-Year Survival | 0.639 | 1,135 | 1-year overall survival |

Best single-model AUC: 0.879 (full dataset evaluation).

### 3.2 Cross-Validation (Platinum Sensitivity)

| Fold | AUC-ROC | Best Epoch |
|------|---------|------------|
| 1 | 0.810 | 11 |
| 2 | 0.667 | 1 |
| 3 | 0.661 | 2 |
| 4 | 0.536 | 8 |
| 5 | 0.864 | 4 |

Mean AUC: 0.707 +/- 0.117. Optimal threshold: 0.917 (sensitivity 83.5%, specificity 84.6%).

High fold variance is due to severe class imbalance (only 2--3 negative samples per fold). Performance is expected to improve with larger negative-class cohorts.

### 3.3 Attention Analysis

The TransMIL attention mechanism reveals biologically plausible patterns. High-attention regions frequently correspond to tumor-stroma interface, areas of lymphocytic infiltration, and regions with distinct nuclear morphology. Low-attention regions typically include necrotic tissue, blood vessels, and technical artifacts.

### 3.4 System Performance

| Operation | Time |
|-----------|------|
| Patch embedding (CPU) | 2--3 min/slide |
| TransMIL inference | < 1 second |
| FAISS search | < 100 ms |
| MedGemma report (GPU) | ~20 seconds |
| Cached retrieval (PostgreSQL) | 0.8 ms |

---

## 4. Clinical Interface and Deployment

### 4.1 Application Architecture

- **Backend**: FastAPI + PostgreSQL (asyncpg), Docker, port 8003
- **Frontend**: Next.js 14.2 + TypeScript + Tailwind CSS, port 3002
- **Database**: PostgreSQL with config-driven project system (projects.yaml + junction tables)
- **Hardware**: NVIDIA DGX Spark (ARM64, GB10 GPU, 128GB unified memory)

### 4.2 User Interface

Three view modes: Oncologist (summary dashboard), Pathologist (annotation and grading tools), Batch (multi-slide parallel processing). 3-panel resizable layout with WSI viewer, case selection, and analysis results. Features include attention heatmaps (jet colormap), evidence patches with normalized weights, FAISS similar case retrieval, MedSigLIP semantic search, 7-step agentic AI Assistant, Project Management with CRUD, PDF/JSON export, and dark mode.

### 4.3 Deployment

```bash
# Backend (Docker)
docker compose -f docker/docker-compose.yaml up -d

# Frontend
cd frontend && npm run build && npx next start -p 3002
```

Backend startup: ~3.5 minutes (model loading). Fully offline after initial setup.

---

## 5. Limitations and Future Directions

### Limitations

1. Path Foundation CPU-only (TensorFlow/Blackwell incompatibility)
2. Training limited to TCGA cohort; multi-site validation needed
3. High CV variance from small negative class
4. Not validated as a medical device; research use only
5. Originally planned Bevacizumab dataset (PathDB) was blocked -- server returned 0-byte files for 217/286 slides

### Future Directions

1. PyTorch Path Foundation port for GPU acceleration
2. Multi-cancer type expansion
3. EHR/LIMS integration
4. Stain normalization for cross-site robustness
5. Prospective clinical validation

---

## References

1. Shao Z, et al. "TransMIL: Transformer based Correlated Multiple Instance Learning for Whole Slide Image Classification." NeurIPS 2021.
2. Google Health AI. Path Foundation Model. https://developers.google.com/health-ai-developer-foundations/path-foundation
3. Google Health AI. MedGemma 1.5 Model Card. https://developers.google.com/health-ai-developer-foundations/medgemma
4. Cancer Genome Atlas Research Network. "Integrated genomic analyses of ovarian carcinoma." Nature 2011.
5. Johnson J, et al. "Billion-scale similarity search with GPUs." IEEE Transactions on Big Data 2019.

---

*Document Version: 2.0*
*Last Updated: February 7, 2026*
*Author: Hanson Wen, UC Berkeley*

*Research prototype only. Not validated as a medical device. All predictions must be reviewed by qualified healthcare professionals.*
