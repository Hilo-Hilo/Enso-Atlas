# Kaggle Submission: MedGemma Impact Challenge

---

## 1. Title

**Enso Atlas: On-Premise Pathology Evidence Engine for Ovarian Cancer Treatment Response Prediction**

---

## 2. Summary

Platinum-based chemotherapy is the standard first-line treatment for ovarian cancer, but ~30% of patients do not respond. Enso Atlas is an on-premise pathology evidence engine that predicts platinum sensitivity from whole-slide histopathology images using all three Google HAI-DEF foundation models. Path Foundation extracts 384-dimensional patch embeddings, TransMIL classifies slides with interpretable attention weights (AUC 0.907 for platinum sensitivity), MedSigLIP enables semantic text-to-patch search for clinician-guided exploration, and MedGemma generates structured clinical reports grounded in visual evidence. Every prediction is accompanied by attention heatmaps, evidence patches, similar case retrieval, and auditable reports. The local-first architecture ensures no patient data leaves hospital networks, running entirely on a single GPU workstation via Docker.

---

## 3. Video Demo

**Link:** [placeholder -- recording pending upload]

Recording plan and required challenge coverage are documented in `VIDEO_SCRIPT.md`.
Target export filename: `enso-atlas-3min-demo.mp4` (1080p, <= 3:00).

---

## 4. GitHub Repository

**https://github.com/Hilo-Hilo/med-gemma-hackathon**

Repository includes:
- Complete source code (FastAPI backend + Next.js frontend)
- Docker Compose deployment configuration
- 5 trained TransMIL models with evaluation metrics
- Documentation, reproduction guide, and benchmark results

---

## 5. Technical Writeup

See: [SUBMISSION_WRITEUP.md](./SUBMISSION_WRITEUP.md)

Key sections:
- Problem statement and clinical motivation
- System architecture with all 3 HAI-DEF models
- TransMIL classification results (AUC 0.907, 5-fold CV 0.707)
- 7-step agentic AI workflow
- Deployment on NVIDIA DGX Spark

---

## 6. HAI-DEF Models Used

### Path Foundation
- **Role:** Histopathology feature extraction
- **Usage:** ViT-S model extracts 384-dimensional embeddings from 224x224 H&E patches at level 0 (full resolution)
- **Details:** Runs on CPU (TensorFlow/Blackwell incompatibility); embeddings cached as FP16 for downstream tasks
- **Impact:** Provides domain-optimized representations that enable TransMIL to achieve AUC 0.907

### MedGemma 1.5 4B
- **Role:** Clinical report generation
- **Usage:** Generates structured tumor board summaries from evidence patches and model outputs (~20s/report on GPU)
- **Details:** Constrained to describe morphological observations only; avoids prescriptive language; JSON-structured output with safety disclaimers

### MedSigLIP
- **Role:** Semantic text-to-patch search
- **Usage:** Pathologists query tissue regions using natural language (e.g., "tumor infiltrating lymphocytes") to find matching patches
- **Details:** Dual encoder architecture enables real-time similarity search across thousands of patches per slide

---

## 7. Team

**Hanson Wen** (solo) -- UC Berkeley

---

## 8. Acknowledgments

- **Google HAI-DEF** for Path Foundation, MedGemma, MedSigLIP, and the Impact Challenge
- **TCGA (The Cancer Genome Atlas)** for ovarian cancer whole-slide image data and clinical annotations
- **OpenSlide** for whole-slide image processing
- **FAISS** (Meta AI) for similarity search infrastructure
- **NVIDIA** for DGX Spark hardware

---

## Short Pitch

Enso Atlas: on-premise pathology AI that shows its work. Predicts ovarian cancer platinum sensitivity (AUC 0.907) with interpretable evidence -- attention heatmaps, semantic search, and MedGemma reports. All 3 HAI-DEF models, fully local, no PHI leaves the hospital.
