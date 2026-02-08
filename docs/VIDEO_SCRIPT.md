# Enso Atlas -- 3-Minute Video Demo Script

**Target length:** 3:00 max
**Frontend URL:** http://100.111.126.23:3002
**Recording:** 1920x1080, 30fps, MP4

---

## Timeline

| Segment | Time | Duration |
|---------|------|----------|
| Intro | 0:00--0:20 | 20s |
| Select Case | 0:20--0:35 | 15s |
| Run Analysis | 0:35--1:05 | 30s |
| Evidence Heatmap | 1:05--1:25 | 20s |
| Evidence Patches | 1:25--1:40 | 15s |
| Semantic Search | 1:40--2:00 | 20s |
| AI Report | 2:00--2:15 | 15s |
| Batch + Projects | 2:15--2:35 | 20s |
| Architecture | 2:35--2:50 | 15s |
| Closing | 2:50--3:00 | 10s |

---

## Script

### 0:00--0:20 -- INTRO

**Narration:**
"Oncologists need evidence-based AI, not black-box predictions. Enso Atlas is an on-premise pathology evidence engine that predicts ovarian cancer treatment response from whole-slide images -- showing clinicians exactly why a model reaches its conclusion."

**Screen:**
- Open Enso Atlas landing screen
- Show 3-panel layout: Case Selection, WSI Viewer, Analysis Results

---

### 0:20--0:35 -- SELECT CASE

**Narration:**
"We start by selecting a case from the TCGA ovarian cancer cohort -- 208 slides with platinum sensitivity labels. The Slide Manager shows thumbnails, patient metadata, and embedding status."

**Screen:**
- Click through case list in left panel
- Show slide thumbnails and metadata
- Select a specific TCGA case

---

### 0:35--1:05 -- RUN ANALYSIS

**Narration:**
"Clicking Run Analysis triggers the full pipeline. Path Foundation extracts patch embeddings. Five specialized TransMIL models run in parallel -- platinum sensitivity, tumor grade, and survival endpoints. Our best model achieves AUC 0.907 for platinum sensitivity prediction."

**Screen:**
- Click "Run Analysis" button
- Show progress steps: embedding, prediction, evidence generation
- Results appear: prediction scores for each model with confidence bands

---

### 1:05--1:25 -- EVIDENCE HEATMAP

**Narration:**
"The attention heatmap shows which tissue regions drove the prediction. Powered by Path Foundation embeddings and TransMIL attention weights, hot regions correspond to morphologically significant areas -- tumor-stroma interface, immune infiltrates, and nuclear features."

**Screen:**
- Toggle heatmap overlay on WSI viewer
- Zoom into high-attention regions
- Show jet colormap overlay with attention scores

---

### 1:25--1:40 -- EVIDENCE PATCHES

**Narration:**
"Evidence patches extract the highest-attention regions as a clickable gallery. Each patch shows its normalized attention weight. Clicking a patch navigates to that location on the slide."

**Screen:**
- Show evidence patches panel with attention weights
- Click a patch to navigate WSI viewer to that region

---

### 1:40--2:00 -- SEMANTIC SEARCH (MedSigLIP)

**Narration:**
"MedSigLIP enables semantic search. Type a clinical query -- here, 'tumor infiltrating lymphocytes' -- and instantly retrieve matching tissue regions with similarity scores. This lets pathologists validate AI predictions against their own expertise."

**Screen:**
- Type "tumor infiltrating lymphocytes" in search bar
- Show matching patches with similarity scores
- Click a result to jump to that location on the slide

---

### 2:00--2:15 -- AI REPORT (MedGemma)

**Narration:**
"MedGemma 1.5 4B generates a structured clinical report, describing morphological features observed in the evidence patches and citing specific regions. Reports export as PDF or JSON for the clinical record."

**Screen:**
- Click "Generate Report"
- Show structured report with morphology descriptions
- Click Export PDF

---

### 2:15--2:35 -- BATCH MODE AND PROJECT MANAGEMENT

**Narration:**
"Batch mode processes multiple slides in parallel for cohort-level analysis. The Project Management system organizes cases, models, and configurations -- supporting multiple studies on the same deployment."

**Screen:**
- Switch to Batch view mode
- Show multi-slide selection and parallel processing
- Switch to Project Management UI
- Show project with associated slides and models

---

### 2:35--2:50 -- ARCHITECTURE

**Narration:**
"Under the hood, Enso Atlas runs entirely on-premise. All three HAI-DEF models -- Path Foundation for embeddings, MedSigLIP for semantic search, MedGemma for reports -- run locally on a single GPU workstation. No patient data ever leaves the hospital network. The system deploys via Docker Compose and is tested on NVIDIA DGX Spark."

**Screen:**
- Show architecture diagram: WSI to Path Foundation to TransMIL to Evidence + Report
- Show Docker deployment and health check

---

### 2:50--3:00 -- CLOSING

**Narration:**
"Enso Atlas turns pathology AI into evidence clinicians can trust -- accelerating treatment decisions today and enabling better outcomes tomorrow."

**Screen:**
- Final dashboard view with heatmap and report visible
- Fade to project name and repository URL

---

## Recording Notes

- Keep narration calm and clinical; avoid jargon overload
- Ensure every model name appears on screen at least once: Path Foundation, MedSigLIP, MedGemma
- Pre-load a case to avoid cold-start wait during recording
- Use Chrome browser in full-screen mode
- Total runtime target: 2:55--3:00
