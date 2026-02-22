# Enso Atlas -- 3-Minute Video Demo Script

**Target length:** 2:50-3:00
**Challenge:** MedGemma Impact Challenge (Kaggle)
**Output format:** MP4, 1920x1080, 30fps

---

## Requirement Coverage Matrix

| Challenge requirement | Covered at |
|---|---|
| 1) Problem statement (on-prem pathology evidence engine for treatment response prediction) | 0:00-0:20 |
| 2) Live demo (select/upload WSI -> view tiles -> run analysis -> prediction) | 0:20-0:55 |
| 3) Semantic search with MedSigLIP | 1:20-1:40 |
| 4) Clinical report generation with MedGemma | 2:00-2:25 |
| 5) Similar case retrieval | 1:40-2:00 |
| 6) Heatmap visualization | 0:55-1:20 |
| 7) Technical architecture overview (Path Foundation + MedGemma + MedSigLIP) | 2:25-2:50 |

---

## Timeline (3:00 max)

| Segment | Time | Duration |
|---|---|---|
| Hook + Problem statement | 0:00-0:20 | 20s |
| Live workflow: project switch (ovarian -> lung), WSI selection, tile viewing, run analysis | 0:20-0:55 | 35s |
| Heatmap visualization | 0:55-1:20 | 25s |
| Semantic search (MedSigLIP) | 1:20-1:40 | 20s |
| Similar case retrieval | 1:40-2:00 | 20s |
| Clinical report generation (MedGemma) | 2:00-2:25 | 25s |
| Technical architecture overview | 2:25-2:50 | 25s |
| Closing | 2:50-3:00 | 10s |

---

## Script + On-Screen Actions

### 0:00-0:20 -- Hook + problem statement

**Narration**
"Platinum chemotherapy is standard for ovarian cancer, but many patients still receive treatment that does not work for them. Enso Atlas is an on-prem pathology evidence engine that predicts treatment response from whole-slide images and shows exactly what evidence drove each prediction."

**Screen actions**
- Start on title slide: "Enso Atlas -- On-Prem Pathology Evidence Engine"
- Cut to dashboard with one case already selected

---

### 0:20-0:55 -- Live workflow (select/upload WSI -> tiles -> run analysis -> prediction)

**Narration**
"Here is a live case workflow. We switch projects from ovarian to lung to show multi-cancer support, select a whole-slide image, load its tiled viewer, and run analysis. The model returns a project-specific prediction with confidence and supporting evidence."

**Screen actions**
- Open project selector and switch from **ovarian-platinum** to **lung-stage**
- Open **Slide Manager** or left case list and select a lung WSI (or show quick upload if available)
- Pan/zoom in OpenSeadragon to show tile loading
- Click **Run Analysis**
- Highlight prediction card (class + confidence) and mention project-specific labels

---

### 0:55-1:20 -- Heatmap visualization

**Narration**
"The attention heatmap highlights tissue regions that most influenced the prediction. This gives clinicians spatial evidence instead of a black-box score."

**Screen actions**
- Toggle heatmap overlay ON
- Adjust sensitivity slider once (low to high)
- Briefly switch model to show heatmap updates per endpoint

---

### 1:20-1:40 -- Semantic search (MedSigLIP)

**Narration**
"With MedSigLIP, users can search tissue morphology in natural language. For example, we query 'tumor infiltrating lymphocytes' and retrieve matching patches ranked by similarity."

**Screen actions**
- Enter text query in semantic search box
- Show top matches with similarity scores
- Click one result to jump viewer to that region

---

### 1:40-2:00 -- Similar case retrieval

**Narration**
"Enso Atlas also retrieves morphologically similar historical cases, helping contextualize each prediction against reference patients."

**Screen actions**
- Open similar cases panel
- Show top retrieved cases and similarity/confidence metadata
- Click a retrieved case briefly

---

### 2:00-2:25 -- Clinical report generation (MedGemma)

**Narration**
"MedGemma generates a structured clinical summary grounded in model outputs and visual evidence. The report is exportable for downstream review workflows."

**Screen actions**
- Click **Generate Report** (or show cached report if pre-generated)
- Scroll report sections: morphology, interpretation, limitations/disclaimer
- Click **Export PDF**

---

### 2:25-2:50 -- Technical architecture overview

**Narration**
"Under the hood, Path Foundation creates patch embeddings, MedSigLIP powers text-to-patch semantic retrieval, and MedGemma handles report generation. These components run behind a FastAPI plus Next.js stack for local, auditable deployment."

**Screen actions**
- Show architecture slide or `README.md` architecture diagram
- Call out: **Path Foundation**, **MedSigLIP**, **MedGemma**
- Briefly show local deployment command (`docker compose -f docker/docker-compose.yaml up -d`)

---

### 2:50-3:00 -- Closing

**Narration**
"Enso Atlas keeps pathology AI on-prem, evidence-based, and clinically reviewable -- from slide-level prediction to interpretable heatmaps, semantic search, similar-case context, and structured reporting."

**Screen actions**
- Return to dashboard with prediction + heatmap visible
- End card with repo URL: `github.com/Hilo-Hilo/enso-atlas`

---

## Recording Checklist

### Pre-record
- [ ] Backend healthy (`GET /api/health`)
- [ ] Frontend loaded and responsive
- [ ] Ovarian and lung demo slides pre-selected and cached
- [ ] One MedGemma report pre-generated (backup for timing)
- [ ] Semantic search query tested (`tumor infiltrating lymphocytes`)
- [ ] Similar-case panel verified
- [ ] Architecture slide/window prepared

### Recording
- [ ] Keep narration pace at ~130-145 wpm
- [ ] Avoid dead time (pre-warm heavy endpoints)
- [ ] Keep total runtime <= 3:00

### Export
- [ ] MP4 (H.264), 1080p, 30fps
- [ ] Audio clear and normalized
- [ ] Filename: `enso-atlas-3min-demo.mp4`

---

## Notes

- If live inference is slow, use cached results and state that the flow is identical in production.
- Prioritize required challenge coverage over extra UI features.
- Keep claims factual and aligned with repo metrics.
- Explicitly show project switching (ovarian -> lung) at least once to demonstrate multi-project isolation and multi-cancer capability.
