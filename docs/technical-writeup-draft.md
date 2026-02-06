# Enso Atlas: On-Premise Pathology Evidence Engine for Treatment Response Prediction

## Abstract
Enso Atlas is an on-premise pathology decision support system that predicts treatment response in ovarian cancer patients using whole-slide histopathology images. Built on Google's HAI-DEF foundation models (Path Foundation, MedGemma, MedSigLIP), the system provides interpretable evidence through attention heatmaps, similar case retrieval, semantic tissue search, and structured clinical reports. Designed for deployment in hospital pathology labs without requiring cloud connectivity, Enso Atlas processes slides entirely on local hardware while maintaining patient data privacy.

## 1. Problem and Motivation

### Clinical Need
Ovarian cancer treatment decisions rely heavily on histopathological assessment of tumor morphology. Predicting response to therapies like platinum-based chemotherapy and bevacizumab remains challenging, with pathologist interpretation varying significantly across institutions.

### Gap
Existing computational pathology tools often require cloud infrastructure, raising PHI concerns. They also provide predictions without interpretable evidence, limiting clinical adoption. Pathologists need tools that explain *why* a prediction was made, not just the prediction itself.

### Our Approach
Enso Atlas addresses both gaps:
1. **On-premise deployment** via Docker on consumer/workstation GPUs (tested on NVIDIA DGX Spark)
2. **Evidence-first design** providing attention heatmaps, similar case retrieval, and structured clinical reports alongside every prediction

## 2. System Architecture

### Data Pipeline
1. **Whole-slide image ingestion** via OpenSlide (supports .svs, .ndpi, .tiff)
2. **Tissue detection and patching** at level 0 (full resolution, 224x224 patches)
3. **Feature extraction** using Path Foundation (384-dimensional embeddings per patch)
4. **Slide-level prediction** via TransMIL (Transformer-based Multiple Instance Learning)
5. **Evidence generation** through attention weight analysis and FAISS similarity search

### HAI-DEF Model Integration
- **Path Foundation**: Frozen feature extractor for H&E histopathology patches. ConvNext-based architecture producing 384-dim embeddings. Runs on CPU (TensorFlow, Blackwell GPU not yet supported).
- **MedGemma 1.5 4B**: Local inference for structured clinical report generation. Receives prediction scores, tissue categories, and attention weights; outputs JSON-structured morphology descriptions and clinical recommendations.
- **MedSigLIP**: Semantic search over tissue regions using natural language queries (e.g., "tumor infiltrating lymphocytes"). Enables pathologist-guided exploration.

### Agentic Workflow
The AI Assistant orchestrates a multi-step analysis pipeline:
1. Initialize case context
2. Run TransMIL prediction
3. Retrieve similar cases via FAISS
4. Perform semantic tissue search (MedSigLIP)
5. Compare against reference cohort
6. Generate reasoning chain
7. Produce MedGemma clinical report

### Frontend
React/Next.js application with OpenSeadragon WSI viewer, attention heatmap overlay, and clinical report panels. Three viewing modes: Oncologist (summary), Pathologist (detailed), and Batch (multi-case processing).

## 3. Experiments and Results

### Dataset
TCGA Ovarian Cancer cohort: 208 whole-slide images with platinum sensitivity labels (binary: responder/non-responder). 5-fold cross-validation with stratified splits.

### TransMIL Training
- Input: Path Foundation embeddings (384-dim, level 0 patches)
- Architecture: TransMIL with pyramid position encoding (512-dim, 8 heads, 2 layers)
- Training: AdamW optimizer, lr=2e-4, weight decay=0.01, 100 epochs, early stopping (patience=15)
- Focal loss with class weighting for severe imbalance (91.4% positive, 8.6% negative)
- 5-fold stratified cross-validation

| Fold | AUC-ROC | Best Epoch |
|------|---------|------------|
| 1    | 0.810   | 11         |
| 2    | 0.667   | 1          |
| 3    | 0.661   | 2          |
| 4    | 0.536   | 8          |
| 5    | 0.864   | 4          |

- **Mean AUC: 0.707 +/- 0.117** (5-fold CV)
- **Best model AUC: 0.879** (full dataset evaluation)
- Optimal threshold via Youden's J: 0.917 (sensitivity=83.5%, specificity=84.6%)

Note: High variance across folds due to small negative class (only 2-3 negatives per fold). Performance expected to improve with more negative samples from additional datasets.

### Embedding Quality
- Level 0 (full resolution): ~6,000-30,000 patches per slide
- Embedding time: ~2-3 min/slide on CPU (Path Foundation)
- FAISS index: 208 slides, cosine similarity

### Report Generation
- MedGemma inference: ~120s on CPU (bfloat16)
- Evidence patches enriched with morphology descriptions
- JSON-structured output with safety disclaimers

## 4. Discussion

### Strengths
- Fully on-premise: zero PHI exposure
- Evidence-first: attention maps + retrieval + reports
- Three HAI-DEF models integrated end-to-end
- Agentic workflow for comprehensive analysis
- Batch processing for clinical throughput

### Limitations
- Path Foundation runs on CPU only (TensorFlow/Blackwell incompatibility)
- MedGemma report generation is slow (~2 min/case on CPU)
- Training cohort limited to TCGA (single institution bias)
- No prospective clinical validation

### Future Work
- PyTorch Path Foundation port for GPU acceleration
- Multi-cancer type support
- Integration with LIMS/EHR systems
- Prospective validation study

## 5. Reproducibility
- Source code: [GitHub link]
- Docker Compose deployment
- Hardware tested: NVIDIA DGX Spark (ARM64, 128GB RAM)
- All models available via HuggingFace

## Safety Statement
Enso Atlas is a research decision-support tool. It is not a medical device and has not been validated for clinical use. All predictions and reports must be reviewed by qualified pathologists before informing treatment decisions.
