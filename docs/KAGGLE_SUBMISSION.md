# Kaggle Submission: MedGemma Impact Challenge

---

## 1. Title

**Enso Atlas: On-Premise Pathology Evidence Engine for Multi-Cancer Prediction (Ovarian Platinum Response + Lung Stage Classification)**

---

## 2. Summary

Platinum-based chemotherapy is the standard first-line treatment for ovarian cancer, but a substantial subset of patients does not respond. Enso Atlas is an on-premise pathology evidence engine that was initially developed for ovarian platinum sensitivity prediction and then extended to lung adenocarcinoma stage classification through a config-driven project system. The platform now supports two isolated projects (`ovarian-platinum`, `lung-stage`) with six total classification models.

Path Foundation extracts 384-dimensional embeddings from level-0 (full-resolution) tissue patches and serves as the shared feature backbone. TransMIL performs project-scoped slide-level classification with interpretable attention weights (best AUC 0.907 for ovarian platinum sensitivity; AUC 0.648 for LUAD early vs advanced stage). MedSigLIP provides semantic text-to-patch search where enabled, and MedGemma generates structured clinical reports grounded in visual evidence. Every prediction is accompanied by attention heatmaps, evidence patches, similar-case retrieval, and auditable report output.

The architecture is local-first and project-isolated end-to-end: datasets, embeddings, model access, and API routing are all scoped by project ID so no cross-project leakage occurs.

---

## 3. Video Demo

**Link:** [placeholder -- recording pending upload]

Recording plan and required challenge coverage are documented in `VIDEO_SCRIPT.md`.
Target export filename: `enso-atlas-3min-demo.mp4` (1080p, <= 3:00).

---

## 4. GitHub Repository

**https://github.com/Hilo-Hilo/enso-atlas**

Repository includes:
- Complete source code (FastAPI backend + Next.js frontend)
- Docker Compose deployment configuration
- 6 trained TransMIL classification models across 2 projects
- Project-scoped API/model routing with regression tests
- Documentation, reproduction guide, and benchmark results

---

## 5. Technical Writeup

See: [SUBMISSION_WRITEUP.md](./SUBMISSION_WRITEUP.md)

Key sections:
- Problem statement and clinical motivation
- System architecture with all 3 HAI-DEF models
- Multi-project, project-isolated backend/frontend architecture
- Ovarian and LUAD classification results
- 7-step agentic AI workflow
- Deployment on NVIDIA DGX Spark

---

## 6. HAI-DEF Models Used

### Path Foundation
- **Role:** Histopathology feature extraction
- **Usage:** ViT-S model extracts 384-dimensional embeddings from 224x224 H&E patches at level 0 (full resolution)
- **Details:** Level-0 dense embeddings are the default inference pathway; embeddings are reused across project-scoped models
- **Impact:** Enables a shared morphology representation across ovarian and lung projects

### MedGemma 1.5 4B
- **Role:** Clinical report generation
- **Usage:** Generates structured summaries from evidence patches and model outputs (~20s/report on GPU)
- **Details:** Constrained to morphology-focused descriptions with safety disclaimers

### MedSigLIP
- **Role:** Semantic text-to-patch search
- **Usage:** Natural language retrieval of histologic patterns for clinician-guided exploration
- **Details:** Integrated with project feature flags and scoped endpoints

---

## 7. Team

**Hanson Wen** (solo) -- UC Berkeley

---

## 8. Acknowledgments

- **Google HAI-DEF** for Path Foundation, MedGemma, MedSigLIP, and the Impact Challenge
- **TCGA (The Cancer Genome Atlas)** for TCGA-OV and TCGA-LUAD whole-slide datasets and clinical annotations
- **OpenSlide** for whole-slide image processing
- **FAISS** (Meta AI) for similarity search infrastructure
- **NVIDIA** for DGX Spark hardware

---

## Short Pitch

Enso Atlas is an on-premise pathology AI platform that shows its work. It supports ovarian platinum-response prediction (AUC 0.907) and config-driven extension to lung adenocarcinoma stage classification (AUC 0.648), with project-isolated routing, interpretable heatmaps, semantic evidence search, and MedGemma-generated reports. All inference runs locally; no PHI leaves hospital infrastructure.
