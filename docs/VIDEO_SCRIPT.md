# Enso Atlas — 3-Minute Demo Script (MedGemma Hackathon)

**Target length:** 3:00 max

---

## Timeline Overview
- **0:00–0:20** Intro (20s)
- **0:20–2:20** Demo (2:00)
- **2:20–2:50** Tech (30s)
- **2:50–3:00** Closing (10s)

---

## Script (Narration + Screen Actions + Visuals)

### 0:00–0:20 — INTRO (20s)
**Narration**
"Oncologists need evidence‑based AI, not black‑box predictions. Enso Atlas brings on‑prem, explainable pathology intelligence to predict treatment response from whole‑slide images—so clinical teams can see *why* a model recommends a therapy."  

**Screen actions**
- Open app landing screen: “Enso Atlas — On‑prem Pathology Evidence Engine”
- Quick cut to a WSI thumbnail gallery and a heatmap preview

**Suggested visuals/transitions**
- Cold open on slide scan → dissolve to product logo
- Subtle overlay text: “Evidence, not black boxes”

---

### 0:20–2:20 — DEMO (2:00)

**0:20–0:35 — Select WSI**
**Narration**
"We start by selecting a whole‑slide image from the local PACS or uploading a new scan."  

**Screen actions**
- Click **Upload / Select WSI**
- Choose a sample H&E slide (show metadata: patient ID, stain, magnification)

**Suggested visuals/transitions**
- Slide zoom animation into tissue region

**0:35–1:05 — Multi‑model analysis (TransMIL endpoints)**
**Narration**
"Enso Atlas runs multiple TransMIL models in parallel—each tuned to a specific endpoint like response probability, risk stratification, or recurrence."  

**Screen actions**
- Click **Run Analysis**
- Show three endpoint tiles with progress: *Response*, *Risk*, *Recurrence*
- Results appear with confidence bands

**Suggested visuals/transitions**
- Split‑panel showing three model cards filling in

**1:05–1:30 — Evidence heatmap (Path Foundation)**
**Narration**
"Next, we open the evidence heatmap. Powered by a Path Foundation model, it highlights the regions that most influenced each prediction."  

**Screen actions**
- Toggle **Evidence Heatmap**
- Hover on hot regions to show patch‑level scores and explanations

**Suggested visuals/transitions**
- Heatmap wipe over the slide; callouts on high‑signal regions

**1:30–1:55 — Semantic search (MedSigLIP)**
**Narration**
"Need corroboration? Use semantic search with MedSigLIP. We can query the slide with clinical text and instantly retrieve matching patches."  

**Screen actions**
- Type query: “tumor‑infiltrating lymphocytes”
- Show patch results grid with similarity scores
- Click a result to jump to location on the slide

**Suggested visuals/transitions**
- Search bar focus glow; patches animate into view

**1:55–2:10 — AI report (MedGemma)**
**Narration**
"Finally, MedGemma generates a concise, evidence‑linked report that cites the most relevant regions and model outputs."  

**Screen actions**
- Click **Generate Report**
- Show report sections: Summary, Evidence, Model Outputs, Recommendations

**Suggested visuals/transitions**
- Type‑in effect for key bullet points

**2:10–2:20 — Export PDF/JSON**
**Narration**
"Everything exports cleanly for the clinical record or downstream research."  

**Screen actions**
- Click **Export → PDF** then **Export → JSON**

**Suggested visuals/transitions**
- Quick icon pop: PDF, JSON

---

### 2:20–2:50 — TECH (30s)
**Narration**
"Under the hood, Enso Atlas is local‑first: slides never leave the hospital network. Path Foundation handles patch embeddings and evidence maps, TransMIL models deliver endpoint predictions, MedSigLIP powers semantic retrieval, and MedGemma composes structured, auditable reports. The pipeline runs in containers on standard GPU workstations—feasible for real deployments."  

**Screen actions**
- Show architecture diagram: WSI → Patch Embeddings → TransMIL / MedSigLIP → Evidence + Report
- Brief glimpse of on‑prem deployment checklist

**Suggested visuals/transitions**
- Animated flow arrows; “On‑prem / HIPAA‑friendly” badge

---

### 2:50–3:00 — CLOSING (10s)
**Narration**
"Enso Atlas turns pathology AI into evidence clinicians can trust—accelerating treatment decisions today and enabling better outcomes tomorrow."  

**Screen actions**
- Show final dashboard view with heatmap + report preview
- End on logo and tagline

**Suggested visuals/transitions**
- Slow zoom out; fade to logo

---

## Notes for Recording
- Keep narration calm and clinical; avoid jargon overload.
- Ensure every model name appears on screen at least once: **Path Foundation**, **MedSigLIP**, **MedGemma**.
- Total runtime target: **2:55–3:00**.
