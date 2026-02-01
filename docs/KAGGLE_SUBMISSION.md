# Kaggle Submission: MedGemma Impact Challenge

---

## 1. Title

**Enso Atlas: Evidence-Based Pathology AI for Ovarian Cancer Treatment Response**

---

## 2. Summary

Ovarian cancer treatment response prediction remains a critical clinical challenge, with no reliable histopathology biomarkers for bevacizumab therapy selection. Enso Atlas addresses this gap with an evidence-based pathology AI system that combines foundation model embeddings with MedGemma-generated clinical reports. Unlike black-box approaches, every prediction is grounded in visual evidence: attention heatmaps, similar case retrieval, and structured pathology summaries. The local-first architecture ensures no patient data leaves hospital networks, meeting stringent healthcare privacy requirements while delivering interpretable, clinical-grade decision support.

---

## 3. Video Demo

**Link:** [placeholder - recording pending]

See `VIDEO_SCRIPT.md` for the planned demonstration walkthrough.

---

## 4. GitHub Repository

**https://github.com/Hilo-Hilo/med-gemma-hackathon**

Repository includes:
- Complete source code with reproducible pipeline
- Documentation and usage instructions
- Sample outputs and benchmark results
- Edge case testing framework

---

## 5. Technical Writeup

See: [SUBMISSION_WRITEUP.md](./SUBMISSION_WRITEUP.md)

**Key sections:**
- Problem statement and clinical context
- System architecture (Path Foundation, CLAM, FAISS, MedGemma)
- Implementation details and performance benchmarks
- Edge case handling and failure mode analysis
- Deployment considerations for clinical environments

---

## 6. HAI-DEF Models Used

### MedGemma 4B
- **Role:** Clinical report generation
- **Usage:** Generates structured tumor board summaries from visual evidence
- **Key feature:** Constrained to describe morphological observations only; avoids prescriptive language

### Path Foundation (Primary) vs DINOv2 (Fallback)
- **Path Foundation:** Google's histopathology-optimized ViT-S model (384-dim embeddings)
  - Provides domain-specific representations for H&E tissue patches
  - Currently gated; requires access approval
- **DINOv2:** Self-supervised vision transformer fallback
  - Used when Path Foundation access is unavailable
  - Maintains pipeline functionality with general-purpose embeddings

The modular architecture supports swapping embedding models without pipeline modifications.

---

## 7. Team

**Hanson Wen** (solo)

---

## 8. Acknowledgments

- **Google HAI-DEF** for MedGemma and the Impact Challenge
- **TCGA (The Cancer Genome Atlas)** for ovarian cancer whole-slide image data
- **OpenSlide** for whole-slide image processing
- **CLAM** (Mahmood Lab) for attention-based multiple instance learning
- **FAISS** (Meta AI) for similarity search infrastructure

---

## Short Pitch (Tweet-length)

> Enso Atlas: AI pathology that shows its work. Predicts ovarian cancer treatment response with interpretable evidence and MedGemma-generated reports. Local-first, no PHI leaves the hospital.

---

## Submission Checklist

- [x] Title and summary
- [x] GitHub repository link
- [x] Technical writeup
- [x] HAI-DEF model documentation
- [x] Team information
- [x] Acknowledgments
- [ ] Video demo (pending recording)
