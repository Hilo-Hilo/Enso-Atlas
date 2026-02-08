# Ovarian Cancer WSI Datasets

This document describes the datasets used and considered for Enso Atlas.

---

## Primary Dataset: TCGA Ovarian Cancer (TCGA-OV)

**This is the dataset used for all training and evaluation.**

### Overview

- **Source:** Genomic Data Commons (GDC) / The Cancer Genome Atlas
- **Project ID:** TCGA-OV
- **Slides Used:** 208 whole-slide images with platinum sensitivity labels
- **Format:** SVS (open access)
- **License:** CC BY 3.0

### Labels

- **Platinum Sensitivity:** Binary (Sensitive vs. Resistant), derived from PLATINUM_STATUS field in cBioPortal clinical data
- **Tumor Grade:** High vs. Low grade
- **Survival:** 1-year, 3-year, 5-year overall survival (binary)

### Usage in Enso Atlas

- 208 slides with platinum sensitivity labels used for the primary TransMIL model (AUC 0.907)
- Up to 1,135 slides used for survival prediction models (larger subset with available survival data)
- Level 0 (full resolution) patch extraction: 224x224 pixels, yielding 6,000--30,000 patches per slide
- Path Foundation embeddings: 384-dimensional vectors cached as FP16
- 5-fold stratified cross-validation with patient-level splits

### Clinical Data Fields (via cBioPortal)

- PLATINUM_STATUS: Sensitive / Resistant / TooEarly
- PRIMARY_THERAPY_OUTCOME_SUCCESS: Complete Response / Partial Response / Stable Disease / Progressive Disease
- OS_MONTHS, OS_STATUS: Overall survival
- DFS_MONTHS, DFS_STATUS: Disease-free survival
- Tumor type, stage, grade

### Access

```bash
# GDC Data Portal API for slides
curl 'https://api.gdc.cancer.gov/files?filters=...'

# cBioPortal API for clinical data
curl 'https://www.cbioportal.org/api/studies/ov_tcga_pub/clinical-data?clinicalDataType=PATIENT'

# Download with gdc-client
gdc-client download -m manifest.txt
```

---

## Blocked Dataset: Ovarian Bevacizumab Response (PathDB/TCIA)

**Originally planned as primary dataset. Download was blocked.**

### Overview

- **Source:** The Cancer Imaging Archive (TCIA) / PathDB
- **DOI:** 10.7937/TCIA.985G-EY35
- **Subjects:** 78 patients
- **Slides:** 288 H&E stained WSIs (162 effective, 126 invalid)
- **Labels:** Bevacizumab treatment response (Effective vs. Invalid)

### Why It Was Blocked

The PathDB download server returned **0-byte files for 217 out of 286 slides**. Only 69 slides downloaded successfully, which was insufficient for training. Multiple download attempts over several days produced the same result. The server appeared to be experiencing persistent issues with file serving.

This dataset would have been ideal for the treatment response prediction use case due to its direct bevacizumab response labels and well-balanced classes (56% effective, 44% invalid).

### Reference

Wang et al. (2022) "Histopathological whole slide image dataset for classification of treatment effectiveness to ovarian cancer." Scientific Data. DOI: 10.1038/s41597-022-01127-6

---

## Other Considered Datasets

### PTRC-HGSOC (TCIA)

- 158 patients, 348 WSIs, platinum chemotherapy sensitivity labels
- Not used due to time constraints; would complement TCGA-OV for future multi-cohort validation

### CPTAC-OV

- 102 patients, proteomic + histopathology data
- Supplementary dataset for potential multi-modal analysis

---

## Summary

| Dataset | Status | Slides | Labels | Used |
|---------|--------|--------|--------|------|
| TCGA-OV | Available | 208+ | Platinum sensitivity, survival, grade | Yes (primary) |
| Bevacizumab Response | BLOCKED | 288 | Treatment response | No (PathDB server issues) |
| PTRC-HGSOC | Available | 348 | Platinum sensitivity | No (future work) |
| CPTAC-OV | Available | ~200 | Survival, proteomics | No (future work) |
