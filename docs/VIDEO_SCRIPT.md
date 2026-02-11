# Enso Atlas -- 3-Minute Video Demo Script

**Target length:** 3:00 max
**Frontend URL:** https://spark.tailcf1f80.ts.net (public via Tailscale Funnel)
**Recording:** 1920x1080, 30fps, MP4
**Browser:** Chrome, full screen, 80% zoom for density

---

## Core Message

Enso Atlas is a **modular, model-agnostic pathology platform** -- not a single-purpose demo. The HAI-DEF models showcase the system; the architecture supports any cancer type, any foundation model, any classification head.

**Lead with the clinical result, not the UI.**

---

## Timeline

| Segment | Time | Duration | Focus |
|---------|------|----------|-------|
| Hook + Problem | 0:00-0:15 | 15s | Clinical need |
| Dashboard + Case Selection | 0:15-0:30 | 15s | 208 slides, 3-panel layout |
| Run Analysis + Prediction | 0:30-0:55 | 25s | AUC 0.907, multi-model |
| Heatmap + Sensitivity | 0:55-1:20 | 25s | Attention overlay, model switching, sensitivity slider |
| Evidence + Navigation | 1:20-1:35 | 15s | Top patches, click-to-navigate |
| Outlier Detector | 1:35-1:50 | 15s | Flag unusual tissue regions |
| Semantic Search (MedSigLIP) | 1:50-2:05 | 15s | Text-to-patch query |
| AI Report (MedGemma) | 2:05-2:20 | 15s | Structured tumor board report |
| Modularity + Deployment | 2:20-2:45 | 25s | YAML config, plug-and-play, Mac mini feasibility |
| Closing | 2:45-3:00 | 15s | On-prem, evidence-based, open platform |

---

## Script

### 0:00-0:15 -- HOOK AND PROBLEM

**Narration:**
"30% of ovarian cancer patients receive platinum chemotherapy that won't work for them. By the time clinicians know, the patient has endured months of toxic side effects. What if a standard tissue slide -- already collected during biopsy -- could predict who will respond before treatment begins?"

**Screen:**
- Black screen with text: "30% of patients receive ineffective chemotherapy"
- Fade to Enso Atlas dashboard

---

### 0:15-0:30 -- DASHBOARD AND CASE SELECTION

**Narration:**
"Enso Atlas processes whole-slide histopathology images into actionable evidence. Here we have 208 ovarian cancer cases from TCGA. Selecting a slide loads the full-resolution image with over 6,000 tissue patches."

**Screen:**
- Show 3-panel layout: case list (left), WSI viewer (center), results (right)
- Scroll through case list showing thumbnails and metadata
- Click slide_000 -- WSI loads in viewer with minimap

---

### 0:30-0:55 -- RUN ANALYSIS AND PREDICTION

**Narration:**
"Running analysis triggers five specialized TransMIL models in parallel. Our platinum sensitivity model achieves AUC 0.907 -- trained on 199 TCGA slides using Path Foundation embeddings. Each model returns a score, confidence estimate, and the attention weights that produced it."

**Screen:**
- Click "Run Analysis" (or show cached results loading instantly)
- Prediction panel: "Sensitive -- 100% -- High Confidence"
- Scroll to Multi-Model Analysis: all 5 models with scores
- Point out AUC badge on Platinum Sensitivity model card

---

### 0:55-1:20 -- ATTENTION HEATMAP AND SENSITIVITY CONTROL

**Narration:**
"The attention heatmap reveals which tissue regions drove the prediction. Each pixel represents one 224-by-224 patch. The sensitivity slider controls visibility of low-attention regions -- pulling it left reveals the full attention landscape, while pushing right isolates the most discriminative tissue."

**Screen:**
- Toggle "Show overlay" ON -- heatmap appears over tissue
- Toggle "Heatmap only" -- black background, attention map visible
- Switch model dropdown from Platinum Sensitivity to 3-Year Survival -- heatmap updates
- Drag sensitivity slider from 0.7 to 0.2 -- low-attention patches become visible
- Drag back to 1.0 -- only hot spots remain
- Toggle patch grid ON briefly to show 224px boundaries

---

### 1:20-1:35 -- EVIDENCE PATCHES AND NAVIGATION

**Narration:**
"The evidence panel extracts the highest-attention patches as a clickable gallery. Each shows its normalized weight. Clicking a patch navigates directly to that region on the slide at high magnification."

**Screen:**
- Scroll to Evidence Patches in right panel
- Click a top-attention patch -- WSI zooms to that location
- Show the scale bar and magnification updating

---

### 1:35-1:50 -- OUTLIER TISSUE DETECTOR

**Narration:**
"The outlier detector uses Path Foundation embeddings to flag morphologically unusual patches -- potential artifacts, rare tissue patterns, or regions worth a second look. No additional model training required; it operates directly on the foundation model's feature space."

**Screen:**
- Scroll to Outlier Tissue Detector panel
- Click "Detect Outliers" with threshold at 2.0 SD
- Show outlier count and statistics
- Toggle "Show as Heatmap" -- amber/red overlay on WSI
- Click an outlier patch to navigate

---

### 1:50-2:05 -- SEMANTIC SEARCH (MedSigLIP)

**Narration:**
"MedSigLIP enables natural language tissue search. Type 'tumor infiltrating lymphocytes' and retrieve matching regions ranked by semantic similarity. Pathologists can validate model predictions against their own expertise."

**Screen:**
- Type "tumor infiltrating lymphocytes" in Semantic Search
- Results appear with similarity scores
- Click a match to navigate on WSI

---

### 2:05-2:20 -- AI REPORT (MedGemma)

**Narration:**
"MedGemma generates a structured tumor board report describing observed morphology, clinical significance, and limitations. Reports include mandatory research-only disclaimers and export as PDF."

**Screen:**
- Click "Generate Report" or show cached report
- Scroll through report sections: morphology, clinical significance, limitations
- Click "Export PDF"

---

### 2:20-2:45 -- MODULARITY AND DEPLOYMENT

**Narration:**
"Enso Atlas is a platform, not a single-purpose tool. Adding a new cancer type requires one YAML configuration block -- define the project, assign models, toggle features. Foundation models are swappable: replace Path Foundation with UNI, CONCH, or any custom encoder. Classification heads are pluggable: drop in weights, register in config. The entire stack deploys via Docker Compose and runs on hardware as modest as a Mac mini with 16 gigabytes of RAM."

**Screen:**
- Show `config/projects.yaml` in a code editor: foundation_models section, classification_models section, project definition
- Highlight: changing `foundation_model: path_foundation` to `foundation_model: uni_v2`
- Show terminal: `docker compose up -d` (3 services starting)
- Show health check JSON: model_loaded: true, slides_available: 208
- Quick flash of the architecture: Path Foundation -> TransMIL -> Evidence -> MedGemma Report

---

### 2:45-3:00 -- CLOSING

**Narration:**
"Every prediction comes with evidence. Every model is replaceable. Every byte of patient data stays on-premise. Enso Atlas turns pathology AI into something clinicians can actually trust."

**Screen:**
- Return to main dashboard with heatmap visible and report in right panel
- Fade to:
  - "Enso Atlas" title
  - "github.com/Hilo-Hilo/med-gemma-hackathon"
  - "Built with Path Foundation, MedGemma, MedSigLIP"

---

## Recording Checklist

- [ ] Pre-load slide_000 with cached results (avoid cold-start wait)
- [ ] Pre-warm MedSigLIP cache for "tumor infiltrating lymphocytes" query
- [ ] Pre-generate at least one clinical report
- [ ] Clear heatmap cache for slide_000 if demonstrating live generation
- [ ] Test sensitivity slider with 3-Year Survival model (sparse attention = dramatic effect)
- [ ] Chrome full-screen, 80% zoom, dark mode OFF (light mode photographs better)
- [ ] Verify public URL works: https://spark.tailcf1f80.ts.net
- [ ] Record narration separately for clean audio, overlay in post
- [ ] Total runtime target: 2:50-3:00 (leave buffer for cuts)

## Key Numbers to Mention

- AUC 0.907 (platinum sensitivity, full dataset)
- 208 slides, 5 models, 3 HAI-DEF foundation models
- 6,934 tissue patches per slide (level 0, 224x224)
- Runs on Mac mini (16 GB minimum)
- Docker Compose: 3 containers, single command deployment

## What NOT to Say

- Don't call it a "wrapper" or "interface" -- it's an evidence engine
- Don't spend time on UI polish details -- focus on clinical workflow and modularity
- Don't mention specific limitations unless asked (save for writeup)
- Don't use "groundbreaking", "revolutionary", "game-changing"
