# Enso Atlas: An On-Premise Pathology Evidence Engine for Treatment-Response Prediction in Ovarian Cancer

**MedGemma Impact Challenge — Technical Writeup**

---

## 1. Problem Statement

Ovarian cancer is the deadliest gynecologic malignancy, with approximately 70% of patients diagnosed at advanced stages. Platinum-based chemotherapy remains the standard first-line treatment, yet roughly 30% of patients receive platinum regimens that will ultimately prove ineffective [1]. This translates to months of toxic, futile treatment — time that could be spent on alternative therapies or clinical trials.

Current molecular biomarkers for platinum sensitivity, including BRCA mutation status and homologous recombination deficiency (HRD) scores, leave substantial gaps. Many BRCA-wild-type patients respond well to platinum, while some HRD-positive patients do not. These tests are also expensive, time-consuming, and unavailable in many clinical settings worldwide.

Meanwhile, hematoxylin and eosin (H&E) stained histopathology slides are collected for virtually every cancer patient during diagnostic biopsy. These slides encode rich morphological information — stromal composition, immune infiltration patterns, nuclear atypia, and architectural features — that correlates with treatment response but remains largely unquantified in clinical practice.

Existing computational pathology tools suffer from three critical limitations: (1) they require uploading patient data to cloud services, creating regulatory and privacy barriers; (2) they produce opaque predictions without supporting evidence; and (3) they are designed as single-purpose research tools rather than integrated clinical decision-support systems.

**Enso Atlas** addresses these gaps by providing an on-premise, evidence-rich pathology platform that predicts platinum sensitivity from routine histopathology slides while surfacing the morphological evidence behind each prediction.

## 2. Approach and Architecture

Enso Atlas is designed around a core principle: clinicians do not need another black-box score — they need an evidence engine that integrates into tumor board workflows. The platform combines three Google Health AI Developer Foundations (HAI-DEF) models into a unified analysis pipeline, with each model serving a distinct role.

### 2.1 System Overview

The platform follows a modular, three-stage architecture:

**Stage 1 — Embedding and Feature Extraction.** Whole-slide images (WSIs) are tessellated into 224x224-pixel patches at level 0 magnification. Each patch is embedded using Google Path Foundation [2], producing a 384-dimensional feature vector. A typical ovarian cancer slide yields approximately 6,934 tissue patches. Outlier detection via z-score analysis on the Path Foundation feature space flags morphologically unusual regions without requiring any labeled training data.

**Stage 2 — Classification and Attention.** Patch-level embeddings are aggregated into slide-level predictions using TransMIL [3], a Transformer-based multiple instance learning architecture. Five independent classification heads operate in parallel: platinum sensitivity, tumor grade, and survival prediction at 1-year, 3-year, and 5-year horizons. The Transformer attention mechanism produces patch-level importance scores, enabling spatial attribution of each prediction back to specific tissue regions.

**Stage 3 — Evidence Synthesis and Reporting.** High-attention patches are extracted as an evidence gallery. FAISS-indexed Path Foundation embeddings enable similar-case retrieval across the slide corpus. MedSigLIP [4] provides natural-language semantic search over tissue patches (e.g., querying "tumor infiltrating lymphocytes" or "stromal desmoplasia"). Finally, MedGemma 1.5 4B [5] generates structured tumor board reports by synthesizing prediction outputs, attention patterns, and morphological observations into natural-language summaries suitable for clinical discussion.

> **Figure 1 (Architecture Diagram).** The system diagram depicts the three-stage pipeline. On the left, a WSI is tessellated and passed through Path Foundation to produce patch embeddings. The center shows TransMIL classification with five parallel heads and attention weight extraction. The right panel illustrates the evidence layer: attention heatmaps overlaid on the WSI via OpenSeadragon, a ranked patch gallery, FAISS-based similar case retrieval, MedSigLIP semantic search, and MedGemma report generation. All components run within a Docker Compose deployment on local hardware.

### 2.2 Integration of HAI-DEF Models

Enso Atlas uses all three HAI-DEF foundation models in complementary roles:

- **Path Foundation** serves as the universal feature backbone. Every downstream task — classification, retrieval, outlier detection, and attention mapping — operates on Path Foundation embeddings. This unified representation ensures consistency across the platform and eliminates the need for task-specific feature extractors.

- **MedGemma 1.5 4B** runs locally (tested on NVIDIA DGX Spark) to generate structured clinical reports. Given a slide's prediction outputs, top-attention patches, and case metadata, MedGemma produces a tumor board summary covering morphological findings, predicted treatment response, confidence assessment, and suggested next steps. Local inference ensures no patient data leaves the hospital network.

- **MedSigLIP** enables free-text semantic search over the tissue patch corpus. Pathologists can query for specific morphological patterns using natural language, bypassing the need for pre-defined annotation categories. This supports exploratory analysis and quality assurance workflows that structured classifiers alone cannot address.

### 2.3 Implementation Details

The backend is implemented in Python using FastAPI, with the frontend built in Next.js and React. WSI visualization uses OpenSeadragon with DZI tile serving for smooth pan-and-zoom navigation. The entire system is containerized via Docker Compose, with a YAML-based project configuration system that allows new cancer types and classification tasks to be added without code changes. The platform has been tested on hardware ranging from a Mac mini with 16 GB RAM (CPU inference) to an NVIDIA DGX Spark (GPU-accelerated inference with MedGemma).

## 3. Results

### 3.1 Classification Performance

Models were trained and evaluated on The Cancer Genome Atlas Ovarian Cancer (TCGA-OV) dataset [6] using 5-fold cross-validation:

| Task | Slides | AUC | Architecture |
|------|--------|-----|--------------|
| Platinum Sensitivity | 152 | **0.752 (mean) / 0.857 (best fold)** | TransMIL |
| Tumor Grade | 208 | 0.750 | TransMIL |

The platinum sensitivity model achieves a mean cross-validated AUC of 0.752 across 5 folds, with the best fold reaching 0.857. At the Youden-optimal threshold (0.923), the model achieves 73% sensitivity and 77% specificity. Performance is constrained by severe class imbalance in TCGA-OV (91% platinum-sensitive), which limits the model's ability to learn negative-class features. All 208 TCGA-OV slides were fully embedded at level 0 magnification, averaging 6,934 patches per slide, with 152 slides having confirmed platinum response labels available for training.

### 3.2 System Performance

End-to-end inference — from raw WSI to completed analysis with all five predictions, attention heatmap, evidence gallery, and similar case retrieval — completes in under 60 seconds per slide on CPU hardware. MedGemma report generation adds approximately 10-20 seconds on GPU. These latencies are compatible with real-time tumor board workflows.

### 3.3 Clinical Workflow Integration

The user interface presents a three-panel layout: WSI viewer with attention heatmap overlay (left), prediction dashboard with confidence indicators (center), and evidence panel with patch gallery, similar cases, semantic search, and generated report (right). An adjustable sensitivity slider on the attention heatmap allows pathologists to explore regions of varying predictive importance. Clicking any evidence patch navigates directly to that region in the WSI viewer, maintaining spatial context. Reports can be exported as PDF documents for inclusion in clinical records.

## 4. Impact and Deployment Considerations

Enso Atlas is designed for deployment within hospital networks where patient data governance is paramount. The on-premise architecture ensures that no protected health information (PHI) traverses external networks — a prerequisite for adoption in most healthcare systems. The modular design means that as HAI-DEF models improve or new foundation models emerge, they can be swapped in via configuration without retraining downstream classifiers.

The platform's evidence-first design philosophy directly addresses a key barrier to clinical AI adoption: trust. Rather than presenting an opaque prediction score, Enso Atlas surfaces the tissue regions driving each prediction, retrieves morphologically similar historical cases with known outcomes, and generates natural-language explanations. This gives pathologists and oncologists the information needed to contextualize AI predictions within their clinical judgment.

The project system — where new cancer types and classification tasks are defined via YAML configuration files — provides a path toward a general-purpose pathology evidence platform. The same architecture that predicts platinum sensitivity in ovarian cancer can be extended to other tumor types and treatment-response questions by adding new training data and configuration entries.

## 5. Limitations and Future Work

We acknowledge several important limitations. First, all training and evaluation data comes from TCGA-OV; the models have not been validated on external cohorts or prospective clinical data. Clinical deployment would require multi-site validation studies with appropriate regulatory review. Second, model outputs are raw logits rather than calibrated probabilities — presenting these as confidence scores without proper calibration could mislead clinical users. Implementing Platt scaling or isotonic regression calibration is planned. Third, the current demonstration covers a single cancer type, though the architecture is cancer-type-agnostic by design. Fourth, the training set of 199 slides is small by modern standards; performance on larger, more diverse datasets remains to be established. Finally, MedSigLIP semantic search can experience timeouts on slides with very large patch counts, requiring optimization for production-scale deployment.

Future work includes: external validation on non-TCGA ovarian cancer cohorts, expansion to additional tumor types (beginning with breast and lung), probability calibration for all classification heads, integration of genomic and clinical covariates alongside morphological features, and a prospective feasibility study in a clinical tumor board setting.

## 6. Acknowledgments

We thank the Google Health AI Developer Foundations (HAI-DEF) team for developing and releasing Path Foundation, MedGemma, and MedSigLIP as open foundation models for biomedical AI research. We thank the Kaggle team for organizing the MedGemma Impact Challenge and fostering innovation in clinical AI applications. TCGA-OV data was generated by the TCGA Research Network (https://www.cancer.gov/tcga).

## References

[1] Bowtell, D.D. et al. "Rethinking ovarian cancer II: reducing mortality from high-grade serous ovarian cancer." *Nature Reviews Cancer* 15, 668-679 (2015).

[2] Vorontsov, E. et al. "Path Foundation: A foundation model for computational pathology." Google Health AI, 2024.

[3] Shao, Z. et al. "TransMIL: Transformer based Correlated Multiple Instance Learning for Whole Slide Image Classification." *NeurIPS* 2021.

[4] Google Health AI. "MedSigLIP: Medical vision-language foundation model." 2024.

[5] Google Health AI. "MedGemma: A family of medical AI models." 2024.

[6] Cancer Genome Atlas Research Network. "Integrated genomic analyses of ovarian carcinoma." *Nature* 474, 609-615 (2011).
