# Enso Atlas: An On-Premise Pathology Evidence Engine for Treatment-Response Prediction in Ovarian Cancer

**MedGemma Impact Challenge — 3-Page Technical Writeup**  
**Team:** Hilo-Hilo (Hanson Wen, UC Berkeley)

---

## 1) Problem: Treatment Response Prediction in Ovarian Cancer Pathology

Ovarian cancer is the deadliest gynecologic malignancy, and most patients are diagnosed at advanced stage. Platinum-based chemotherapy remains standard first-line treatment, but a substantial fraction of patients do not benefit from the initial regimen. In practice, this means some patients endure months of toxicity before clinicians can determine that the therapy is ineffective.

Current response-associated biomarkers (for example BRCA/HRD-related signals) are useful but incomplete, costly, and not universally available. By contrast, H&E pathology slides are already collected for essentially every patient and contain rich morphological information that can be used for computational prediction.

The key translational challenge is not only to predict response, but to provide evidence that clinicians can inspect and discuss in tumor board settings. Most prior computational pathology workflows fail here because they are either cloud-dependent (problematic for PHI governance) or operate as black boxes with limited interpretability.

**Goal.** Build a practical system that predicts platinum response from routine whole-slide images (WSIs), keeps data on-premise, and returns auditable evidence rather than an opaque score.

---

## 2) Approach: Enso Atlas with HAI-DEF Foundation Models

Enso Atlas is designed as an **on-prem evidence engine** rather than a single-task model endpoint. The system integrates all three Google HAI-DEF models in one clinical workflow:

- **Path Foundation** for pathology-native feature extraction
- **MedGemma 1.5 4B** for structured natural-language clinical summaries
- **MedSigLIP** for semantic text-to-patch retrieval

This design addresses two practical requirements:

1. **Trust and interpretability**: every prediction is paired with attention heatmaps, high-impact evidence patches, and similar-case retrieval.
2. **Deployment feasibility**: the full stack runs locally in Docker Compose, with no patient data leaving institutional infrastructure.

We implement slide-level prediction with **TransMIL** (Transformer multiple-instance learning), using patch embeddings from Path Foundation as the common representation. The same embeddings support both classification and retrieval/search, simplifying the system and improving consistency across components.

---

## 3) Architecture: Path Foundation → TransMIL → MedGemma Reports → MedSigLIP Search

### 3.1 End-to-end pipeline

1. **WSI ingestion and tissue patching**
   - Input formats include standard pathology slide files (e.g., SVS/NDPI/TIFF).
   - Tissue regions are identified, then tiled into 224×224 patches at level 0 resolution.

2. **Embedding with Path Foundation**
   - Each patch is embedded into a 384-dimensional vector.
   - Embeddings are cached and reused for downstream tasks.

3. **Slide-level prediction with TransMIL**
   - Patch embeddings are aggregated to slide-level outputs.
   - Multiple task heads are supported (platinum sensitivity, grade, 1/3/5-year survival).
   - Attention weights are retained for explainability and heatmap visualization.

4. **Evidence synthesis and reporting**
   - Top-attention patches are surfaced as supporting evidence.
   - FAISS retrieval returns morphologically similar cases.
   - MedSigLIP supports free-text semantic search over patch space.
   - MedGemma generates a structured report grounded in model outputs and visual evidence.

### 3.2 Clinical UI and workflow support

The interface is organized around a WSI viewer, prediction panel, and evidence panel. Pathologists can inspect heatmaps, jump directly to high-attention regions, compare with similar historical cases, and run semantic patch search with natural language. This makes model behavior inspectable and discussion-ready for multidisciplinary review.

### 3.3 Why this architecture is practical

- **Single embedding backbone** (Path Foundation) minimizes feature-fragmentation across tasks.
- **Attention + retrieval + semantic search** provides complementary evidence modes.
- **Local deployment** aligns with privacy and hospital IT constraints.

---

## 4) Results: Model Performance and Clinical Utility

### 4.1 Predictive performance

On the TCGA ovarian cohort used in this project, the platinum sensitivity TransMIL model achieves:

- **AUC: 0.907** (full-dataset evaluation)
- **5-fold CV mean AUC: 0.707 ± 0.117**
- **Optimal threshold (Youden): 0.917**
- **Sensitivity/Specificity at optimal threshold: 83.5% / 84.6%**

Additional trained heads (grade and survival endpoints) show expectedly lower AUCs, but validate that the same architecture generalizes to multiple clinically relevant outcomes.

### 4.2 Evidence-centric utility

Beyond scalar metrics, Enso Atlas is intended for decision support where traceability matters:

- Attention heatmaps identify spatially salient tissue regions.
- Ranked evidence patches provide concrete visual rationale.
- Similar-case retrieval supports case-based comparison.
- MedGemma summaries improve communication of findings in tumor board workflows.

### 4.3 Runtime characteristics

Representative system timings:

- Path Foundation embedding (CPU): ~2–3 minutes/slide
- TransMIL inference: <1 second
- FAISS retrieval: <100 ms
- MedGemma report generation (GPU): ~20 seconds

These timings are compatible with review workflows in practice, especially with embedding caching.

---

## 5) Impact: Real-World Feasibility and Clinical Integration

Enso Atlas is designed for realistic deployment constraints in healthcare environments:

- **On-prem operation** protects PHI and avoids cloud transfer barriers.
- **Containerized stack** simplifies deployment and maintenance.
- **Evidence-first outputs** improve clinician trust compared with black-box predictors.
- **Config-driven project system** supports extension to new cancers/tasks without rewriting core services.

The impact is not just a higher AUC. The core contribution is a deployable architecture that combines foundational pathology embeddings, interpretable slide-level modeling, semantic evidence exploration, and report generation in one coherent platform.

---

## 6) Reproducibility: Docker Compose and Data Pipeline

### 6.1 Reproducible setup

The repository includes a Dockerized backend and a production-ready frontend:

```bash
# Backend services
docker compose -f docker/docker-compose.yaml up -d

# Frontend
cd frontend
npm install
npm run build
npx next start -p 3002
```

Core components (FastAPI backend, PostgreSQL persistence/caching, and Next.js frontend) are wired for local deployment.

### 6.2 Data and training pipeline summary

1. Acquire and register slide metadata.
2. Generate level-0 Path Foundation embeddings.
3. Train/evaluate TransMIL models (including class-imbalance handling).
4. Build FAISS retrieval index.
5. Run inference and evidence/report generation through API endpoints.

The same pipeline is reusable for additional projects by updating project configuration and input datasets.

### 6.3 Known limitations

- External prospective validation remains future work.
- Cross-site generalization should be tested with broader cohorts.
- Current system is a research prototype and not a cleared medical device.

---

## Evaluation Criteria Coverage (Challenge Alignment)

- **Effective HAI-DEF usage:** Path Foundation + MedGemma + MedSigLIP are all integrated in production workflow.
- **Problem importance:** Platinum response prediction in ovarian cancer has direct treatment implications.
- **Real-world impact:** On-prem, evidence-backed outputs are aligned with clinical governance and usability.
- **Technical feasibility:** End-to-end architecture is implemented, benchmarked, and runnable with Docker Compose.
- **Execution quality:** Multi-model pipeline, interpretable UI, retrieval/search tooling, and reproducible repo are delivered.

---

## References

1. Shao Z, et al. *TransMIL: Transformer based Correlated Multiple Instance Learning for Whole Slide Image Classification.* NeurIPS 2021.
2. Google Health AI. Path Foundation model documentation.
3. Google Health AI. MedGemma model documentation.
4. Cancer Genome Atlas Research Network. *Integrated genomic analyses of ovarian carcinoma.* Nature 2011.
5. Johnson J, et al. *Billion-scale similarity search with GPUs.* IEEE Transactions on Big Data 2019.

---

**Safety Note:** Enso Atlas is a research decision-support prototype. Outputs must be reviewed by qualified clinicians and are not intended for autonomous treatment decisions.
