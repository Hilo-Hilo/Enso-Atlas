# Option A PRD + Technical Design Document

## Implementation Status Legend
- [DONE] - Fully implemented and working
- [WIP] - Work in progress
- [TODO] - Not yet started

## Enso Atlas (Prototype): On-Prem Pathology Evidence Engine for Treatment-Response Insight

> This document describes a **winning** MedGemma Impact Challenge project in the "Option A" family: an **offline, local-first, on-prem WSI evidence engine** that predicts a clinically meaningful outcome (or proxy) from histopathology and produces **interactive, auditable evidence** (heatmaps + similar patch retrieval + structured report).
>
> It is designed to be **foundation-model-agnostic** so you can later swap in Enso's own histopathology slide foundation model without rewriting the product.

---

## Table of contents

1. Executive summary
2. Why this can win (judging rubric mapping)
3. Product vision & value proposition
4. Clinical users, real-life use cases, and "when would a doctor use it?"
5. PRD: goals, non-goals, scope, success metrics
6. System overview and architecture
7. Technical pipeline (end-to-end)
8. Model strategy (encoders + slide head + explainability)
9. MedGemma role (report generation + safety + grounding)
10. Compute estimates (DGX Spark target)
11. System requirements and limitations
12. Deployment plan (offline / hospital on-prem)
13. Security, privacy, and compliance posture (practical, not legal advice)
14. Evaluation plan (ML + UX + safety)
15. Roadmap and "swap-in Enso FM later" plan
16. Appendix: alternative open(-weight) models and libraries

---

## 1) Executive summary

**Enso Atlas** is an **on-prem pathology evidence engine** that takes a whole-slide image (WSI) and produces:

* A **patient/slide-level score** for a target outcome (starting with the public bevacizumab-response dataset; later extend to other therapies/biomarkers). ([Nature][1])
* A **heatmap** that highlights regions most responsible for that score (evidence patches).
* A **similarity search** view: "show me the most similar patches/cases" inside the local reference cohort. (This is critical: evidence is not just attention-clinicians can inspect precedent.) ([Google for Developers][2])
* A **structured, cautious tumor-board-style summary** generated locally by **MedGemma**, grounded only in the evidence patches and a small set of facts you provide. ([Google for Developers][3])

The core design principle is **local-first**: the hospital runs everything on a single workstation-class box (DGX Spark) and **no PHI needs to leave the hospital network** by default. DGX Spark is explicitly marketed around **128GB unified system memory**, local inference/fine-tuning, and "desktop-scale" development. ([NVIDIA Docs][4])

This aligns with Enso's startup direction (histopath foundation models → treatment response) while leveraging open(-weight) models now: **Path Foundation** for patch embeddings + **MedGemma** for multimodal reporting (and optionally **MedSigLIP** for text-to-patch retrieval). Path Foundation produces **384-dim embeddings from 224×224 H&E patches** and is designed to reduce compute for downstream classifiers. ([Google for Developers][5])

---

## 2) Why this can win (rubric mapping)

The MedGemma Impact Challenge is judged more like a product hackathon: video, writeup, reproducible code; emphasis on realistic healthcare deployment rather than isolated benchmarks. ([EdTech Innovation Hub][6])

This project maps cleanly:

### Execution & communication

* A visually compelling demo: upload WSI → heatmap → click evidence patches → retrieve similar cases → generate structured summary.

### Effective use of HAI-DEF models

* Path Foundation is not cosmetic; it is the **feature backbone**. ([Google for Developers][5])
* MedGemma is not a chatbot add-on; it is the **report compiler** that turns evidence into a tumor-board packet. ([Google for Developers][3])
* Optional: MedSigLIP enables "semantic evidence search" (text query → patch retrieval), a highly "human-centered" feature. ([Google for Developers][7])

### Product feasibility

* Compute is made feasible by the embedding-first pipeline: embed patches once, then train small heads. ([Google for Developers][5])
* DGX Spark specs support a credible "single-box" story (20-core Arm CPU, 128GB unified memory, substantial bandwidth). ([NVIDIA Docs][4])

### Impact potential + domain importance

* Treatment response prediction is a high-impact domain (avoid ineffective therapy, reduce toxicity, better trial matching).
* There is a public dataset specifically for ovarian bevacizumab response with WSIs + clinical info, giving you a defensible prototype. ([Nature][1])

---

## 3) Product vision & value proposition

### Vision statement

Build a **pathology-first decision-support assistant** that produces *inspectable evidence* from histology images to support oncologists and pathologists during tumor board and therapy planning-**without cloud compute** and without requiring the startup to host PHI.

### The real value proposition (what's actually different)

Most "AI in pathology" demos fail in the same ways:

* They give a number but no evidence.
* They require cloud infrastructure (hard in hospitals).
* They don't integrate into tumor board preparation.

**Enso Atlas differentiates** by making three things first-class:

1. **Evidence map that is human-readable**

   * Heatmap on the slide
   * Top evidence patches
   * Similar patch/case retrieval
     This turns the AI output from "a magic probability" into something clinicians can interrogate.

2. **Local-first deployment**

   * Runs on a single on-prem box (DGX Spark class)
   * No PHI transfer required by default
     This aligns with "privacy-first, offline-capable healthcare AI" positioning. ([EdTech Innovation Hub][6])

3. **Tumor-board packet automation (MedGemma)**

   * Converts evidence into structured summaries, limitations, suggested confirmatory tests, and an exportable "case note draft." ([Google for Developers][3])

### How this aligns with Enso's startup trajectory

* Today: use open(-weight) encoders (Path Foundation) to ship the "evidence engine" and workflow. ([Google for Developers][5])
* Tomorrow: swap the encoder with **Enso's slide foundation model** to improve performance/robustness, while keeping the entire product architecture (tiler, retrieval, UI, reporting) unchanged.

The product is the workflow + evidence interface; the foundation model is an interchangeable component.

---

## 4) Clinical users, real-life use cases, and "when would a doctor use it?"

### Primary persona 1: Pathologist (sign-out + tumor board prep)

**Situation:** A pathologist is reviewing a surgical specimen slide (or biopsy). They already use a digital viewer (Aperio, HALO, QuPath, etc.). They are preparing a tumor board slide that summarizes morphology and key biomarkers.

**How they use Enso Atlas**

* They open the WSI and run Enso Atlas on that slide (or select a region-of-interest).
* Atlas outputs:

  * Heatmap of model-attended regions
  * Top evidence patches (clickable)
  * Similar patch retrieval from reference cohort (shows "what it resembles")
  * A draft summary with limitations (for tumor board packet)

**Why this is valuable**

* It reduces time spent hunting for the "most informative" regions.
* It provides a structured way to justify and communicate findings to the oncology team.
* It supports research discussions: "These patches look like X pattern seen in responders/non-responders in this local cohort."

**What it is not**

* Not a final diagnosis.
* Not a replacement for pathology interpretation.
* Not a treatment recommendation engine.

### Primary persona 2: Medical oncologist (tumor board + therapy planning)

**Situation:** The oncologist is deciding between therapy options (or considering eligibility for an anti-VEGF regimen / clinical trial). They have pathology, radiology, labs, and genomics-often fragmented.

**How they use Enso Atlas**

* During tumor board: the pathologist shares the Atlas evidence view.
* The oncologist sees:

  * A response-likelihood score (or a therapy-relevant phenotype score)
  * Evidence patches driving that score
  * Similar cases in the local reference set (if available)
  * A structured summary (MedGemma) focusing on what evidence supports the output, limitations, and next-step tests

**Why this is valuable**

* It helps the oncologist ask better questions: "Is this score driven by necrosis? stromal patterns? immune infiltration?"
* It supports "right test, right patient" decisions (e.g., whether to pursue additional IHC/NGS or trial screening).

### Secondary persona: Tumor board coordinator / clinical research coordinator

**Situation:** Tumor boards are time constrained; coordinators need standardized packet materials and missing-data detection.

**How they use Atlas**

* Atlas outputs a consistent tumor board draft (JSON + human-readable).
* Coordinator can export the summary into the tumor board deck template and mark missing metadata.

### Real-life "doctor would use it when…"

This tool is most defensible in settings where:

* The institution already has **digital pathology** workflows.
* There is a **clinical trial program** or therapy selection decisions that would benefit from additional evidence.
* The hospital has strong data privacy constraints and prefers **on-prem compute**.

It is especially compelling for:

* **Tumor board**: the tool creates shared, inspectable artifacts for multidisciplinary discussion.
* **Retrospective cohort review**: identifying morphological correlates of response and hypothesis generation.
* **Triage**: highlight "regions worth looking at" (not "this is the answer").

---

## 5) PRD

### 5.1 Problem statement

Clinicians routinely rely on H&E histology for diagnosis and biomarker workflows, but:

* slide review is time-consuming,
* important morphological signals may be subtle or diffuse,
* therapy response labels are hard to operationalize,
* cloud AI deployment is often blocked by privacy/security requirements.

### 5.2 Product goals (v1)

1. **Make therapy-response insight inspectable**

   * Model score + heatmap + evidence patches + similar-case retrieval.

2. **Make it deployable on a single on-prem machine**

   * Offline mode, limited compute, simple installation (Docker).

3. **Make it usable in tumor board workflows**

   * Generate a structured tumor board summary and exports.

4. **Make it modular and pivotable**

   * Swap target tasks without rebuilding the whole system.

### 5.3 Non-goals (v1)

* Not an autonomous diagnostic device.
* Not prescribing treatment.
* Not integrating directly into the EHR in v1 (export-only).
* Not training a new foundation model (Enso FM comes later).

### 5.4 Target "winning demo" outcomes

* A working demo that runs end-to-end on a sample WSI.
* A short evaluation on a public dataset with patient-level splits.
* A UI with evidence exploration + report export.
* A clear writeup that emphasizes safety/limitations and real-world fit.

### 5.5 Success metrics

**Model/ML**

* AUC/PR-AUC on the chosen public target (patient-level cross-validation).
* Calibration curve or simple reliability metric.
* Evidence sanity checks (pathologist review of top patches: "do these look plausible?")

**Product**

* Time-to-result per slide under a defined patch budget.
* One-command local run.
* "Tumor board packet generated" success on sample cases.

**Human-centered**

* User testing with at least 1-2 clinicians (even informal) demonstrating that evidence view changes/helps discussion.

---

## 6) System overview and architecture

### 6.1 High-level architecture diagram (text)

```
[WSI File]                                          [DONE]
   |
[WSI Reader + Tissue Mask + Patch Sampler]          [DONE] src/enso_atlas/wsi/processor.py
   | patches (224x224) + coords
[Patch Embedding Service]                           [DONE] src/enso_atlas/embedding/embedder.py
   - Path Foundation embeddings (primary)           [DONE]
   - (Optional) MedSigLIP embeddings                [TODO]
   |
[Local Storage Cache]                               [DONE] data/demo/embeddings/
   |
[Slide Head Service]                                [DONE] src/enso_atlas/mil/clam.py
   - CLAM Attention MIL head                        [DONE]
   - Outputs: score + attention weights             [DONE]
   |
[Evidence Engine]                                   [DONE] src/enso_atlas/evidence/generator.py
   - Heatmap compositor                             [DONE]
   - Top-K patch selector                           [DONE]
   - FAISS similarity search                        [DONE]
   |
[MedGemma Report Generator]                         [DONE] src/enso_atlas/reporting/medgemma.py
   - Inputs: evidence patches + score + constraints [DONE]
   - Output: structured JSON + human summary        [DONE]
   |
[UI + Exports]
   - Gradio demo interface                          [DONE] src/enso_atlas/ui/demo_app.py
   - Next.js professional frontend                  [DONE] frontend/
   - OpenSeadragon slide viewer                     [DONE]
   - Evidence patch gallery                         [DONE]
   - Similar cases/patches                          [DONE]
   - Export PDF/JSON                                [WIP]
```

### 6.2 Key design choices

* **Embedding-first**: compute patch embeddings once, reuse for many tasks. Path Foundation is explicitly designed to reduce compute and enable multiple downstream classifiers from embeddings. ([Google for Developers][5])
* **Evidence-first**: heatmaps + retrieval are product features, not research afterthoughts.
* **Local-first**: no internet requirement at inference; hospital controls data boundary.

---

## 7) Technical pipeline (end-to-end)

This section is the "exact technical pipeline" you asked for.

### Implementation Status Summary
| Component | Status | Location |
|-----------|--------|----------|
| WSI Processing | [DONE] | `src/enso_atlas/wsi/processor.py` |
| Tissue Detection | [DONE] | Uses Otsu thresholding |
| Patch Sampling | [DONE] | Grid-based, configurable budget |
| Path Foundation Embedding | [DONE] | `src/enso_atlas/embedding/embedder.py` |
| CLAM MIL Head | [DONE] | `src/enso_atlas/mil/clam.py` |
| Heatmap Generation | [DONE] | `src/enso_atlas/evidence/generator.py` |
| FAISS Similarity Search | [DONE] | `src/enso_atlas/evidence/generator.py` |
| MedGemma Reporting | [DONE] | `src/enso_atlas/reporting/medgemma.py` |
| FastAPI Backend | [DONE] | `src/enso_atlas/api/main.py` |
| Gradio Demo UI | [DONE] | `src/enso_atlas/ui/demo_app.py` |
| Next.js Frontend | [DONE] | `frontend/` |

### 7.1 Input formats and ingestion

**Supported inputs (v1):**

* WSI formats supported by OpenSlide/cucim ecosystem: SVS, NDPI, MRXS, TIFF variants (depending on build).
* Optional: clinical metadata JSON (non-required).

**Ingestion steps:**

1. Read WSI metadata (mpp, magnification levels).
2. Generate low-res thumbnail at a chosen level.
3. Compute tissue mask (fast segmentation).
4. Determine patch grid at target magnification.

**Implementation notes**

* Use `OpenSlide` (broad format support) OR `cuCIM` (GPU-accelerated I/O where supported).
* Always keep coordinate transforms so patches map back to slide.

### 7.2 Tissue detection and patch sampling

**Why sampling matters:** WSI can contain tens of thousands of valid tissue patches. You need bounded compute.

**Two-phase sampling (recommended for "winning")**

* Phase 1: coarse sampling (e.g., 1,000-2,000 patches) to produce an initial score + rough heatmap.
* Phase 2: adaptive refinement: sample more patches in high-uncertainty / high-attention regions (another 2,000-8,000 patches).

**Patch size:** 224×224 for Path Foundation input. ([Google for Developers][5])
**Magnification:** configurable; start with 20× equivalent if metadata allows; otherwise use a pixel-size-based approach.

### 7.3 Patch embedding

**Primary embedding model: Path Foundation**

* Produces **384-dimensional embeddings** from **224×224 H&E patches**.
* Uses **ViT-S architecture**.
* Intended for histopathology tasks; reduces compute for downstream models. ([Google for Developers][5])

**Optional embedding model: MedSigLIP**

* Dual encoder: medical image + text → shared embedding space.
* Trained on de-identified medical image-text pairs including histopathology.
* Supports 448×448 images and up to 64 text tokens (useful for semantic patch search). ([Google for Developers][7])

**Caching strategy (critical for speed)**

* Store embeddings as FP16 in a local cache keyed by:

  * slide hash
  * patch coordinates
  * model version
* Store thumbnails of evidence patches (compressed JPEG/PNG).

### 7.4 Slide-level prediction head (trainable on small compute)

Use an **attention-based MIL head** (ABMIL-style) on top of patch embeddings.

**Inputs:** `{(e_i, coord_i)}` for i in sampled patches.
**Outputs:**

* `p` = probability of label (e.g., responder vs non-responder)
* `a_i` = attention weight per patch

**Why MIL is the right default**

* Works with slide-level labels (no pixel annotations needed).
* Produces attention weights that translate into evidence patches (explainability).

### 7.5 Evidence generation

Evidence outputs are what clinicians actually use.

**Evidence artifacts**

1. **Heatmap overlay** on slide thumbnail

   * Map attention weights to patch grid
   * Smooth/interpolate for readability
2. **Top-K evidence patch gallery**

   * show patch thumbnails + coordinates
3. **Similarity search** (FAISS)

   * For each evidence patch, retrieve:

     * nearest patches within same slide (pattern repetition)
     * nearest patches across cohort library (precedent)

Path Foundation explicitly supports similar-image search use cases. ([Google for Developers][2])

### 7.6 MedGemma report generation (local)

MedGemma 1.5 supports:

* Whole-slide histopathology use via multiple patches as input.
* EHR/document understanding tasks.
  This makes it a fit for "evidence patches → structured tumor board summary." ([Google for Developers][3])

**MedGemma inputs (v1)**

* Top 6-12 evidence patches (images)
* Model score + a confidence descriptor
* A short "task card" prompt that defines:

  * intended use (research / decision support)
  * limitations
  * schema constraints

**Outputs**

* Strict JSON adhering to a schema (enforced by validator).
* Human-readable summary for tumor board packet.

### 7.7 Export and integration

**Exports (v1)**

* PDF/HTML report (score, heatmap snapshot, top patches, limitations)
* JSON evidence bundle (patch coords, embeddings references, report)
* Optional: CSV of patch coordinates and weights

**Integration (v2+)**

* Plugin mode for QuPath / slide viewer integration.
* Simple folder-watcher mode: drop a WSI into a directory → Atlas processes and outputs a report bundle.

---

## 8) Model strategy: pick targets that are defensible and winnable

You said you "only want a winning project," not a locked-in one. This is how to remain pivotable without losing months.

### 8.1 Target selection strategy (what to predict)

Option A can be framed as either:

**Track 1 (high impact, higher risk): direct treatment-response prediction**

* Demo dataset: ovarian bevacizumab response WSIs (public). ([Nature][1])
* Pros: strongest "impact story"
* Cons: label noise, small n, risk of overfitting.

**Track 2 (lower risk, still clinically aligned): therapy-relevant phenotype / biomarker proxy**
Examples:

* TIL-rich vs TIL-poor morphology proxy
* Necrosis/stroma dominance patterns
* Tumor purity / cellularity proxy
* Slide QA artifact detection (as a gating tool before response prediction)

Pros: easier to validate, less overclaiming, still meaningful for therapy discussion.

**Winning approach:** Build the product for Track 1 but keep a "task head plug-in" so you can pivot to Track 2 if Track 1 performance is shaky.

### 8.2 Evidence-based precedent: foundation models can support response prediction + explainability

There is published work benchmarking histopathology foundation models for bevacizumab response prediction from WSIs, explicitly noting that high-attention regions can aid explainability and may serve as imaging biomarkers. ([PubMed][8])

This is useful in your writeup: you're not claiming a miracle; you're building a tool consistent with existing research and pushing it into a deployable workflow.

### 8.3 Avoid the most common WSI ML pitfalls (PRD-level requirements)

* **Patient-level split** (not slide-level) to avoid leakage (multiple slides per patient).
* **Slide stain/scanner variability**: include stain normalization or robust augmentation.
* **Patch sampling bias**: ensure tissue coverage; don't only sample tumor-dense regions unless that's the intended behavior.

---

## 9) MedGemma role: make it safe, grounded, and "doctor-usable"

### 9.1 Why MedGemma here is not fluff

MedGemma 1.5 explicitly supports multiple domains including WSI multi-patch interpretation and medical document/EHR understanding. ([Google for Developers][3])

In Atlas, MedGemma is the **clinical communication layer**:

* Converts evidence artifacts into a consistent summary.
* Enforces limitations and "not for standalone decision-making."
* Produces a structured output that can be audited.

### 9.2 Grounding strategy (to reduce hallucinations)

MedGemma can hallucinate if you ask it to "explain why treatment will work." Don't.

Instead, constrain the task:

**You want MedGemma to:**

* Describe what it sees in the evidence patches (morphology descriptors).
* Summarize what the model output is and what evidence was used.
* Provide limitations and suggested confirmatory steps (general, non-prescriptive).
* Avoid any direct clinical recommendations ("start drug X").

**Mechanisms**

* Provide only:

  * evidence patch images,
  * the model output,
  * the allowed vocabulary / schema.
* Add a "must cite evidence patch IDs" rule inside the schema.
* Validate JSON output; if invalid, rerun with stricter prompt.

### 9.3 Example structured report schema (v1)

*(This is a design artifact; tune to your dataset/task.)*

```json
{
  "case_id": "string",
  "task": "string",
  "model_output": {
    "label": "string",
    "probability": 0.0,
    "calibration_note": "string"
  },
  "evidence": [
    {
      "patch_id": "string",
      "coords_level0": [0, 0],
      "morphology_description": "string",
      "why_this_patch_matters": "string"
    }
  ],
  "similar_examples": [
    { "example_id": "string", "label": "string", "distance": 0.0 }
  ],
  "limitations": ["string"],
  "suggested_next_steps": ["string"],
  "safety_statement": "string"
}
```

---

## 10) Compute estimates for DGX Spark deployment (practical planning)

### 10.1 DGX Spark baseline assumptions

DGX Spark hardware overview includes:

* NVIDIA Grace Blackwell architecture (integrated GPU+CPU),
* 20-core Arm CPU,
* **128GB unified system memory**,
* memory bandwidth reported in docs,
* "support for AI models up to 200B parameters" (marketing + user guide). ([NVIDIA Docs][4])

Your project (Path Foundation + MIL head + MedGemma 4B) is far below "200B parameter" scale, but the unified memory is still valuable for:

* caching embeddings,
* holding WSI tiles,
* running multiple services in one box.

### 10.2 Pipeline compute cost drivers

In practice, your runtime is dominated by:

1. Patch extraction / I/O
2. Patch embedding forward passes (Path Foundation)
3. MedGemma generation time (optional, can be asynchronous in UI)

The MIL head and FAISS retrieval are cheap.

### 10.3 Estimating patch counts

Typical tissue area varies widely, but for planning:

* **Sampling budget (recommended):** 3,000-12,000 patches/slide total per run (coarse + refine).
* "Full slide" can exceed 30,000 patches at 224×224 depending on tissue area.

### 10.4 Embedding throughput estimate

You should measure this early, but a planning model:

* Let `N` = number of patches embedded
* Let `R` = embedding rate (patches/sec)
* Then `T_embed ≈ N / R`

**Conservative planning ranges (you should benchmark):**

* `R` might range ~200-1500 patches/sec depending on:

  * batching,
  * image decode/I/O,
  * precision,
  * GPU utilization.

**Example**

* `N = 8,000 patches`, `R = 800 patches/sec` → `T_embed ≈ 10 sec`

### 10.5 Embedding cache storage/memory

Path Foundation embedding is 384 floats. ([Google for Developers][5])

* FP32: 384 × 4 bytes = 1,536 bytes/patch
* FP16: 384 × 2 bytes = 768 bytes/patch

For 20,000 patches:

* FP16 embeddings ≈ 15.4 MB/slide
* FP32 embeddings ≈ 30.7 MB/slide

This is **tiny** relative to system memory, which is why "embed once, reuse forever" is such a strong product strategy.

### 10.6 FAISS retrieval compute

FAISS approximate search with 10k-1M vectors is typically sub-second per query on CPU, often milliseconds.
For your UI:

* 10 evidence patches × top-20 neighbors: effectively instantaneous.

### 10.7 MedGemma inference compute

MedGemma 1.5 is used for multi-patch interpretation and medical summarization tasks. ([Google for Developers][3])

In v1, treat it as optional:

* Run MedGemma only when the user clicks "Generate tumor board summary"
* Keep patch count small (6-12 evidence patches)
* Keep output short (300-600 tokens)

This makes interactive latency acceptable even on a single box.

### 10.8 End-to-end latency targets (v1)

**Goal:** "useful in tumor board prep," not real-time.

* Slide ingest + tissue mask: 1-5 sec
* Embedding (8k patches): 5-30 sec (depends; measure)
* MIL + heatmap + retrieval prep: <1 sec
* Report generation (MedGemma): 2-20 sec depending on runtime config

**Total:** ~10-60 seconds per slide for "evidence-ready," plus optional report.

---

## 11) System requirements and limitations

### 11.1 Functional requirements

**Core workflow**

* Load WSI and render thumbnail
* Run tissue detection and patch sampling
* Compute embeddings (Path Foundation)
* Predict target label with MIL head
* Display heatmap overlay
* Display top evidence patches with coordinates
* Similarity search within cohort (FAISS)
* Generate structured report (MedGemma) and export

**Configurability**

* Patch sampling budget
* Magnification level
* Task head selection (response vs phenotype vs QC)
* Offline mode enforcement (no outbound calls)

### 11.2 Non-functional requirements

**Privacy / data locality**

* Default configuration: no telemetry, no remote logging.
* Logs must avoid PHI by default (hash identifiers).

**Reproducibility**

* Deterministic seeds for sampling (configurable).
* Model versioning in every output report.

**Reliability**

* If MedGemma fails, the evidence engine still works.
* If retrieval index missing, show evidence patches only.

### 11.3 Known limitations (you should acknowledge in the writeup)

These acknowledgements increase judge trust.

1. **Domain shift**

* Scanner differences, stain differences, tissue prep variation.
* Mitigation: stain augmentation, normalization, site-specific calibration.

2. **Label quality for response prediction**
   The ovarian bevacizumab dataset provides a binary effective/invalid label based on defined criteria and clinical info, but response labels are inherently noisy and cohort-specific. ([Nature][1])

3. **Not clinically validated**
   Path Foundation model card emphasizes that downstream tasks require end-user validation and notes limitations of validation scope. ([Google for Developers][5])

4. **Explainability is not causality**
   Attention heatmaps are evidence cues, not proofs of biological mechanisms.

5. **Not a medical device in v1**
   The system must be positioned as research decision-support / workflow aid unless and until validated and regulated accordingly.

---

## 12) Deployment plan (offline / hospital on-prem)

### 12.1 Deployment modes

**Mode A: "Single-box workstation"**

* Everything runs on DGX Spark.
* Local web UI accessible on hospital intranet.

**Mode B: "Department server"**

* Same container stack runs on a small on-prem GPU server.
* Multiple users access via browser.

### 12.2 Packaging

* Docker Compose with services:

  * `atlas-ui`
  * `atlas-wsi-worker`
  * `atlas-embedder`
  * `atlas-faiss`
  * `atlas-report`
* Models stored on encrypted disk; loaded at runtime.

### 12.3 Offline-first enforcement

* Provide a "no network" runtime flag:

  * disables outbound HTTP
  * requires models already present
* In UI show status: "Offline mode enabled: No outbound connections."

### 12.4 Why DGX Spark is a credible target

DGX Spark user guide enumerates the hardware features (20-core Arm, 128GB unified memory, etc.). ([NVIDIA Docs][4])
NVIDIA product materials describe unified memory and model scale targets. ([NVIDIA][9])

Your pitch: hospitals can run the entire pipeline locally without needing a compute cluster.

---

## 13) Security, privacy, compliance posture (practical engineering)

Not legal advice; this is engineering framing.

### 13.1 Data boundary principle

* The hospital owns and operates the compute.
* No PHI is transmitted to your servers by default.
* Outputs are stored on hospital-controlled storage.

### 13.2 Logging policy

* Default: do not log filenames, MRNs, patient identifiers.
* Log only:

  * content hashes,
  * model versions,
  * runtime performance metrics.

### 13.3 Access control

* Local authentication (OIDC integration optional).
* Role-based access:

  * viewer (view results)
  * operator (run jobs)
  * admin (configure models)

### 13.4 Auditability

Every report export includes:

* model versions (Path Foundation, task head version, MedGemma version),
* patch sampling configuration,
* timestamp,
* whether offline mode was enabled.

---

## 14) Evaluation plan (ML + usability + safety)

### 14.1 Dataset for v1 demo: ovarian bevacizumab response

The dataset includes **288 de-identified H&E WSIs from 78 patients**, with counts of effective vs invalid slides and clinical info. ([Nature][1])

**Important evaluation requirement:** patient-level split.

### 14.2 ML metrics

* AUC, PR-AUC
* calibration (reliability curve)
* subgroup checks if metadata supports it

### 14.3 Evidence quality checks (human-centered)

Even 1-2 pathologists reviewing:

* Are top patches "reasonable"?
* Do the similar-patch results make sense?
* Does the report summary stay within evidence?

### 14.4 Safety checks for MedGemma output

* Enforce schema validation.
* Block prohibited statements:

  * "Start/stop drug X"
  * "This patient will respond"
* Require limitations section and safety statement.

---

## 15) Roadmap and "swap-in Enso FM later" plan

### Phase 0: Kaggle-winning MVP (2-4 weeks sprint style) [DONE]

* [DONE] Implement patching + embedding + MIL + heatmap + evidence patches
* [DONE] Add FAISS retrieval
* [DONE] Add MedGemma structured report generation
* [WIP] Package into local docker stack
* [TODO] Produce 3-minute demo + writeup with safety positioning

### Phase 1: Pilot-ready research tool [WIP]

* [TODO] Add stain robustness improvements (Macenko normalization)
* [TODO] Add quality control gate (artifact detection)
* [TODO] Add cohort management UI (local reference library)
* [TODO] Add "case comparison" view

### Phase 2: Enso foundation model integration [TODO]

Because the architecture is encoder-agnostic:

* [TODO] Replace `PathFoundationEmbedder` with `EnsoEmbedder`
* Keep:

  * sampling,
  * caching,
  * MIL head interface,
  * retrieval,
  * UI,
  * reporting.

---

## 16) Appendix: the exact tech stack (recommended)

### Languages

* Python (core)
* TypeScript/React optional (if you want a slick UI; Gradio is enough for Kaggle)

### Core libraries

**WSI I/O**

* OpenSlide (broad compatibility)
* cuCIM (if available for acceleration)

**Embeddings**

* Path Foundation (open-weight, HAI-DEF). ([Google for Developers][5])
* Optional: MedSigLIP for text↔patch retrieval. ([Google for Developers][7])

**Indexing**

* FAISS (local vector search)

**Slide head**

* PyTorch attention MIL implementation (write it yourself to avoid license headaches)

**LLM reporting**

* MedGemma 1.5 4B inference (Transformers + quantization/TensorRT-LLM depending on environment). ([Google for Developers][3])

**UI**

* Gradio or Streamlit
* OpenSeadragon for slide viewing overlays (if you want better WSI UX)

**Deployment**

* Docker + NVIDIA Container Toolkit
* Optional: Triton Inference Server (if you want production-ish serving)

### Task-head plug-in interface (keeps you non-locked-in)

Define a simple contract:

* `TaskHead.fit(embeddings, labels, metadata) -> model`
* `TaskHead.predict(embeddings) -> prob, attention_weights`
* `TaskHead.label_space -> {labels}`
* `TaskHead.report_template -> schema + prompt`

Now you can pivot targets without changing UI or evidence engine.

---

# Final note: what makes this "winning" isn't just the model

A lot of teams will show a MedGemma chatbot or a classifier. Few will show:

* WSI evidence heatmaps,
* similarity search,
* structured report generation with guardrails,
* and a credible on-prem/offline deployment story.

Your "moat" for the challenge is the **evidence workflow** and the **local-first packaging**, not chasing state-of-the-art AUC.

If you want, I can also produce:

* a **repo structure** (folders, modules, interfaces),
* a **demo storyboard** (exact 3-min script),
* and a **writeup outline** that directly maps each feature to the judging criteria and special "Edge AI" category requirements.

[1]: https://www.nature.com/articles/s41597-022-01127-6?utm_source=chatgpt.com "Histopathological whole slide image dataset for ..."
[2]: https://developers.google.com/health-ai-developer-foundations/path-foundation?utm_source=chatgpt.com "Path Foundation Model | Health AI Developer Foundations"
[3]: https://developers.google.com/health-ai-developer-foundations/medgemma/model-card "MedGemma 1.5 model card  |  Health AI Developer Foundations  |  Google for Developers"
[4]: https://docs.nvidia.com/dgx/dgx-spark/hardware.html "Hardware Overview - DGX Spark User Guide"
[5]: https://developers.google.com/health-ai-developer-foundations/path-foundation/model-card "Path Foundation model card  |  Health AI Developer Foundations  |  Google for Developers"
[6]: https://www.edtechinnovationhub.com/news/google-launches-medgemma-impact-challenge-to-advance-human-centered-health-ai "Google MedGemma Impact Challenge opens on Kaggle | ETIH EdTech News - EdTech Innovation Hub"
[7]: https://developers.google.com/health-ai-developer-foundations/medsiglip/model-card?utm_source=chatgpt.com "MedSigLIP model card | Health AI Developer Foundations"
[8]: https://pubmed.ncbi.nlm.nih.gov/39961889/?utm_source=chatgpt.com "Benchmarking histopathology foundation models for ..."
[9]: https://www.nvidia.com/en-us/products/workstations/dgx-spark/ "A Grace Blackwell AI supercomputer on your desk | NVIDIA DGX Spark"
