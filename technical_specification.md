# Enso Atlas: Technical Specification and Implementation Plan

## Implementation Status Legend
- [DONE] - Fully implemented and working
- [WIP] - Work in progress
- [TODO] - Not yet started

This document provides a detailed technical specification and implementation plan for the Enso Atlas project, an on-premise pathology evidence engine for treatment-response insight. The recommendations are based on extensive research into the latest open-source technologies, frameworks, and models in computational pathology.

## Table of Contents

1.  [**Executive Summary**](#1-executive-summary)
2.  [**System Architecture**](#2-system-architecture)
3.  [**Core Components & Technology Stack**](#3-core-components--technology-stack)
    1.  [WSI Processing & Tiling](#31-wsi-processing--tiling)
    2.  [Feature Extraction & Embedding](#32-feature-extraction--embedding)
    3.  [Multiple Instance Learning (MIL) Head](#33-multiple-instance-learning-mil-head)
    4.  [Evidence Generation & Visualization](#34-evidence-generation--visualization)
    5.  [LLM-Powered Reporting](#35-llm-powered-reporting)
    6.  [User Interface (UI)](#36-user-interface-ui)
    7.  [Deployment & Infrastructure](#37-deployment--infrastructure)
4.  [**Implementation Plan & Roadmap**](#4-implementation-plan--roadmap)
5.  [**References**](#5-references)

## 1. Executive Summary

Enso Atlas is designed as an on-premise, local-first pathology evidence engine to predict treatment response from whole-slide images (WSIs). The system will provide clinicians with a patient-level score, an evidence-based heatmap, and a similarity search feature to compare with a local reference cohort. A key component is the integration of MedGemma for generating structured, cautious, and auditable reports grounded in the visual evidence. The entire system is designed to be foundation-model-agnostic, allowing for future integration of Enso's proprietary models. The proposed architecture prioritizes privacy, offline capability, and a seamless workflow for pathologists and oncologists, running on a single workstation-class machine like the NVIDIA DGX Spark. This document outlines the technical specifications, component choices, and implementation roadmap to build a robust and impactful prototype for-research-use prototype.

## 2. System Architecture

The system is designed with a modular, service-oriented architecture to ensure scalability, maintainability, and the ability to swap components as new models and technologies become available. The core design principles are embedding-first, evidence-first, and local-first.

### 2.1. High-Level Architecture Diagram

```
[WSI File (.svs, .ndpi, etc.)]
           |
           v
+------------------------------------+
|      WSI Processing Service        |
| (OpenSlide/cuCIM, PyTorch)         |
| - Tissue Masking (Otsu, Morphology)|
| - Patch Sampling (Grid, Random)    |
+------------------------------------+
           |
           v (Patches: 224x224, Coordinates)
+------------------------------------+
|      Feature Extraction Service    |
| (PyTorch, Path Foundation Model)   |
| - Generate 384-dim Embeddings      |
| - Cache Embeddings (HDF5/LanceDB)  |
+------------------------------------+
           |
           v (Embeddings, Coordinates)
+------------------------------------+
|      MIL & Evidence Engine         |
| (PyTorch, CLAM/Slideflow, FAISS)   |
| - Attention MIL Head (Prediction)  |
| - Heatmap Generation               |
| - Evidence Patch Selection         |
| - Similarity Search (FAISS)        |
+------------------------------------+
           |
           v (Score, Heatmap, Evidence Patches, Similar Cases)
+------------------------------------+
|      LLM Reporting Service         |
| (Hugging Face Transformers, MedGemma) |
| - Generate Structured JSON Report  |
| - Create Human-Readable Summary    |
+------------------------------------+
           |
           v (Report, Visualization Data)
+------------------------------------+
|      User Interface Service        |
| (Gradio/Streamlit, OpenSeadragon)  |
| - WSI Viewer with Heatmap Overlay  |
| - Evidence Gallery & Comparison    |
| - Report Display & Export (PDF)    |
+------------------------------------+
           |
           v (User Interaction)
+------------------------------------+
|      Deployment Environment        |
| (Docker, NVIDIA Container Toolkit) |
| - Runs on DGX Spark or similar     |
| - Offline-first, on-premise        |
+------------------------------------+
```

### 2.2. Key Design Choices

*   **Embedding-First**: Patch embeddings are computed once and cached. This allows for rapid experimentation with different MIL heads and downstream tasks without reprocessing the entire WSI. The Path Foundation model is ideal for this, providing high-quality, general-purpose embeddings for histopathology [1].
*   **Evidence-First**: The system is not a black box. Heatmaps, attention scores, and similarity search results are first-class outputs, providing clinicians with auditable and interpretable evidence to support the model's prediction.
*   **Local-First**: All components are designed to run on-premise, with no requirement for cloud connectivity at inference time. This is crucial for handling sensitive patient data (PHI) and for deployment in hospital environments with strict network policies.
*   **Modularity**: Each component is a distinct service, allowing for independent development, testing, and upgrading. For example, the `PathFoundationEmbedder` can be swapped with a future `EnsoEmbedder` with minimal changes to the surrounding architecture.

## 3. Core Components & Technology Stack

This section details the recommended technology stack for each component of the Enso Atlas pipeline, based on the comprehensive research conducted.

### Component Implementation Status

| Component | Status | File Location |
|-----------|--------|---------------|
| WSI Processing | [DONE] | `src/enso_atlas/wsi/processor.py` |
| Path Foundation Embedder | [DONE] | `src/enso_atlas/embedding/embedder.py` |
| CLAM MIL Classifier | [DONE] | `src/enso_atlas/mil/clam.py` |
| Evidence Generator | [DONE] | `src/enso_atlas/evidence/generator.py` |
| MedGemma Reporter | [DONE] | `src/enso_atlas/reporting/medgemma.py` |
| FastAPI Backend | [DONE] | `src/enso_atlas/api/main.py` |
| Gradio Demo UI | [DONE] | `src/enso_atlas/ui/demo_app.py` |
| Next.js Frontend | [DONE] | `frontend/` |
| Docker Deployment | [WIP] | `docker/` |

### 3.1. WSI Processing & Tiling [DONE]

**Objective:** Efficiently read whole-slide images, identify tissue regions, and extract patches for feature extraction.

**Implementation:** `src/enso_atlas/wsi/processor.py`

**Recommended Technologies:**

*   **WSI Reading:** A combination of **OpenSlide** and **cuCIM** is recommended. OpenSlide provides broad format compatibility and is a well-established standard in the field [2]. cuCIM, part of the NVIDIA RAPIDS suite, offers significant GPU-accelerated performance for reading and processing WSIs, which can dramatically speed up the pipeline [3]. A hybrid approach can be used where cuCIM is the default for supported formats, with a fallback to OpenSlide for broader compatibility.

| Library | Pros | Cons | Primary Use Case |
| :--- | :--- | :--- | :--- |
| **OpenSlide** | Broad format support, stable, widely used | CPU-based, slower I/O | Fallback for formats not supported by cuCIM |
| **cuCIM** | GPU-accelerated, extremely fast I/O | Fewer formats supported, requires NVIDIA GPU | Primary WSI reader for maximum performance |

*   **Tissue Detection:** A classic computer vision pipeline using **Otsu's thresholding** on a downsampled version of the WSI, followed by **morphological operations** (e.g., closing, opening) to refine the tissue mask, is a robust and computationally efficient starting point [4]. This avoids the need for a separate deep learning model for segmentation, simplifying the initial implementation.

*   **Patch Sampling:** A **two-phase sampling strategy** is recommended. First, a coarse grid-based sampling of the entire tissue region is performed. After an initial prediction and attention map generation, a second, more targeted sampling is performed in high-attention regions. This adaptive approach balances computational efficiency with the need to focus on diagnostically relevant areas.

### 3.2. Feature Extraction & Embedding

**Objective:** Convert image patches into meaningful, low-dimensional feature vectors (embeddings) that can be used for downstream machine learning tasks.

**Recommended Technologies:**

*   **Embedding Model:** The **Path Foundation** model from Google Health AI is the recommended starting point [1]. It is a Vision Transformer (ViT-S) based model specifically trained on histopathology images and is designed to produce high-quality 384-dimensional embeddings from 224x224 pixel H&E patches. Its availability on Hugging Face makes it easy to integrate and use [5].

*   **Implementation:** The model can be loaded using the `huggingface_hub` and `tensorflow` libraries. The feature extraction process should be run as a separate service that processes batches of patches and saves the resulting embeddings.

*   **Embedding Cache:** To support the "embedding-first" architecture, a robust caching mechanism is critical. Storing embeddings in **HDF5 files** (one per slide) is a simple and effective approach. For more advanced querying and management, a dedicated vector database like **LanceDB** or **FAISS** can be used. LanceDB is a modern, serverless vector database that is easy to embed in a Python application, while FAISS is a highly optimized library for efficient similarity search [6] [7].

| Caching Method | Pros | Cons | Recommendation |
| :--- | :--- | :--- | :--- |
| **HDF5** | Simple, file-based, good for per-slide storage | Not optimized for fast querying across slides | Start with this for simplicity in v1 |
| **LanceDB** | Embedded, serverless, fast filtering & search | Newer, less mature than FAISS | Excellent choice for v2+ for cohort-level search |
| **FAISS** | Highly optimized, industry standard, GPU support | More complex to set up and manage | Best for large-scale, high-performance similarity search |

### 3.3. Multiple Instance Learning (MIL) Head

**Objective:** To aggregate patch-level embeddings into a slide-level prediction and identify the most influential patches (evidence).

**Recommended Technologies:**

*   **MIL Architecture:** **CLAM (Clustering-constrained Attention Multiple Instance Learning)** is the recommended architecture [8]. It is a powerful and widely-used attention-based MIL method that has demonstrated strong performance on various WSI classification tasks. CLAM's attention mechanism naturally provides interpretability by assigning an attention score to each patch, which can be used to generate heatmaps and identify evidence.

*   **Implementation Framework:** The **Slideflow** library is highly recommended for implementing the MIL pipeline [9]. Slideflow is a comprehensive, open-source framework for digital pathology that integrates several state-of-the-art MIL models, including CLAM. It simplifies the process of training, evaluation, and deployment, and it supports both PyTorch and TensorFlow. Using Slideflow will significantly accelerate development by providing a robust, pre-built foundation for the MIL head.

| Framework | Pros | Cons | Recommendation |
| :--- | :--- | :--- | :--- |
| **Slideflow** | Integrated MIL models (CLAM), end-to-end pipeline, supports PyTorch/TF, active development | Can be a large dependency | **Primary choice**. Provides a complete and flexible framework for the entire MIL workflow. |
| **Original CLAM Repo** | Direct implementation from the authors, good for understanding the core algorithm | Less of an end-to-end framework, may require more integration work | Use as a reference for the core CLAM algorithm, but implement using Slideflow. |

*   **Training:** The MIL head will be trained on the cached patch embeddings. This is computationally efficient, as the expensive feature extraction step is only performed once. The training process will involve standard cross-validation with patient-level splits to prevent data leakage.

### 3.4. Evidence Generation & Visualization

**Objective:** To translate the model's internal state (attention scores) into human-interpretable evidence for clinicians.

**Recommended Technologies:**

*   **Heatmap Generation:** The attention scores from the CLAM model will be used to generate a heatmap overlay on the WSI. The attention score of each patch is mapped to its corresponding location on the slide, and interpolation is used to create a smooth, visually interpretable heatmap. Libraries like **OpenCV** and **Pillow** can be used for this process.

*   **Evidence Patch Selection:** The top-K patches with the highest attention scores will be selected as the primary evidence. These patches will be displayed in a gallery in the UI for detailed inspection.

*   **Similarity Search:** **FAISS (Facebook AI Similarity Search)** is the recommended library for implementing the similarity search feature [7]. A FAISS index will be built on the embeddings of a reference cohort of WSIs. When a user clicks on an evidence patch, a k-nearest neighbor search will be performed in the FAISS index to retrieve the most similar patches from the reference cohort. This provides powerful precedent-based evidence for the clinician.

### 3.5. LLM-Powered Reporting

**Objective:** To automatically generate a structured, cautious, and clinically relevant report from the model's evidence.

**Recommended Technologies:**

*   **LLM Model:** **MedGemma 1.5 4B** is the ideal choice for this task [10]. It is a multimodal model specifically designed for the medical domain and can process both text and images (the evidence patches). Its 4-billion parameter size makes it feasible to run on a local, on-premise machine like the DGX Spark, which is a core requirement of the project.

*   **Implementation:** The model can be run locally using the **Hugging Face `transformers`** library. The reporting service will take the top evidence patches, the final prediction score, and a predefined prompt template as input. The prompt will be carefully engineered to constrain the model's output and ensure it is grounded in the provided evidence.

*   **Grounding and Safety:** To prevent hallucinations and ensure the clinical safety of the output, a strict grounding strategy will be implemented:
    *   **Constrained Input:** The model will only be provided with the evidence patches, the model's score, and a predefined schema for the output. It will not have access to any other patient information or external knowledge.
    *   **Schema Enforcement:** The output of MedGemma will be validated against a strict JSON schema. If the output is invalid, the generation can be retried with a more constrained prompt.
    *   **Prohibited Statements:** The prompt will explicitly forbid the model from making any direct clinical recommendations, such as advising for or against a specific treatment.

*   **Output:** The service will produce two outputs: a **structured JSON file** containing all the evidence and the model's interpretation, and a **human-readable summary** formatted for inclusion in a tumor board packet. The JSON output ensures the results are machine-readable and auditable, while the summary provides a concise overview for clinicians.

### 3.6. User Interface (UI)

**Objective:** To provide an intuitive and interactive interface for clinicians to upload WSIs, view results, and explore the evidence.

**Recommended Technologies:**

*   **UI Framework:** **Gradio** is the recommended framework for building the user interface [11]. It is a Python library that makes it incredibly easy to create and share web apps for machine learning models. Its simplicity and focus on data science workflows make it ideal for rapid prototyping and building a functional UI for Enso Atlas.

*   **WSI Viewer:** For the core WSI viewing experience, **OpenSeadragon** is the best choice [12]. It is a powerful, open-source JavaScript library for deep zoom image viewing. It can be integrated into a Gradio application to provide a seamless experience for navigating large WSIs and displaying heatmap overlays.

*   **UI Components:**
    *   **WSI Upload:** A simple file upload interface for selecting the WSI to be analyzed.
    *   **Main View:** An OpenSeadragon viewer displaying the WSI with a toggleable heatmap overlay.
    *   **Evidence Gallery:** A gallery of the top-K evidence patches, which can be clicked to zoom to the corresponding region in the main viewer.
    *   **Similarity View:** A display of the most similar patches from the reference cohort, retrieved using FAISS.
    *   **Report View:** A formatted display of the MedGemma-generated report, with options to export as PDF or JSON.

### 3.7. Deployment & Infrastructure

**Objective:** To package and deploy the entire Enso Atlas application as a single, self-contained unit for on-premise installation.

**Recommended Technologies:**

*   **Containerization:** **Docker** is the standard for containerizing applications. The entire Enso Atlas pipeline will be packaged into a set of Docker containers, one for each service (WSI processing, feature extraction, etc.). This ensures a consistent and reproducible environment, regardless of the host system.

*   **GPU Support:** The **NVIDIA Container Toolkit** is essential for enabling Docker containers to access the host's NVIDIA GPUs [13]. This is a hard requirement for the feature extraction and MIL training components.

*   **Orchestration:** **Docker Compose** will be used to define and manage the multi-container application. A `docker-compose.yml` file will specify the services, networks, and volumes required to run Enso Atlas.

*   **Target Hardware:** The application will be optimized to run on a single **NVIDIA DGX Spark** workstation or a similarly specced machine with a powerful NVIDIA GPU and at least 128GB of RAM. The DGX Spark's unified memory architecture is particularly well-suited for this type of workload [14].

*   **Model Serving (Optional v2+):** For a more production-oriented deployment in the future, **Triton Inference Server** can be used for serving the Path Foundation and MedGemma models [15]. Triton provides features like dynamic batching and concurrent model execution, which can significantly improve inference throughput and resource utilization.

## 4. Implementation Plan & Roadmap

This project will be implemented in a phased approach, starting with a core MVP and progressively adding more advanced features.

### Phase 0: Core MVP (4-6 Weeks) [DONE]

**Goal:** Build a functional end-to-end prototype that demonstrates the core value proposition of Enso Atlas.

*   [DONE] **WSI Pipeline:** Implement the basic WSI processing pipeline using OpenSlide, Otsu thresholding, and grid-based patch sampling.
*   [DONE] **Feature Extraction:** Set up the Path Foundation model for feature extraction and implement numpy-based embedding caching.
*   [DONE] **MIL Head:** Implemented full CLAM architecture with gated attention in `src/enso_atlas/mil/clam.py`.
*   [DONE] **Basic UI:** Created Gradio demo interface in `src/enso_atlas/ui/demo_app.py`.
*   [WIP] **Deployment:** Docker configuration exists but needs testing.

### Phase 1: Enhanced Evidence & Reporting (4 Weeks) [DONE]

**Goal:** Improve the evidence generation capabilities and integrate the LLM reporting service.

*   [DONE] **Advanced MIL:** Full CLAM architecture with gated attention and instance clustering.
*   [DONE] **Similarity Search:** FAISS integration in `src/enso_atlas/evidence/generator.py`.
*   [DONE] **Interactive Viewer:** OpenSeadragon integrated in Next.js frontend.
*   [DONE] **MedGemma Integration:** Full reporting service in `src/enso_atlas/reporting/medgemma.py`.
*   [DONE] **Professional Frontend:** Next.js 14 app with TypeScript in `frontend/`.

### Phase 2: Performance & Robustness (Ongoing) [WIP]

**Goal:** Optimize the pipeline for performance and improve its robustness to variations in data.

*   [TODO] **GPU Acceleration:** cuCIM integration for WSI I/O.
*   [TODO] **Stain Normalization:** Macenko normalization.
*   [TODO] **Advanced Caching:** LanceDB or vector database integration.
*   [TODO] **Model Serving:** Triton Inference Server deployment.
*   [TODO] **Cohort Management:** UI for managing reference cohort.

### Phase 3: Enso Foundation Model Integration [TODO]

**Goal:** Swap the open-source Path Foundation model with Enso's proprietary foundation model.

*   [TODO] **Adapter Implementation:** Create a new `EnsoEmbedder` service that conforms to the same interface as the `PathFoundationEmbedder`.
*   [TODO] **Retraining & Evaluation:** Retrain the MIL head on the new embeddings and perform a comprehensive evaluation to quantify the performance improvement.
*   [TODO] **System Integration:** Integrate the new embedder into the main pipeline.

## 5. References

[1] Google Health AI. "Path Foundation Model." [Online]. Available: https://developers.google.com/health-ai-developer-foundations/path-foundation

[2] OpenSlide. "OpenSlide: A C library for reading whole-slide images." [Online]. Available: https://openslide.org/

[3] NVIDIA. "cuCIM: An open-source, accelerated I/O, and computer vision library for N-dimensional images." [Online]. Available: https://developer.nvidia.com/cucim

[4] N. Otsu, "A threshold selection method from gray-level histograms," in IEEE Transactions on Systems, Man, and Cybernetics, vol. 9, no. 1, pp. 62-66, Jan. 1979.

[5] Hugging Face. "google/path-foundation." [Online]. Available: https://huggingface.co/google/path-foundation

[6] LanceDB. "LanceDB: The open-source, serverless vector database for AI." [Online]. Available: https://lancedb.com/

[7] Facebook AI Research. "FAISS: A library for efficient similarity search and clustering of dense vectors." [Online]. Available: https://github.com/facebookresearch/faiss

[8] M. Y. Lu et al., "Data-efficient and weakly supervised computational pathology on whole-slide images," Nature Biomedical Engineering, vol. 5, no. 6, pp. 591-602, 2021.

[9] J. M. Dolezal et al., "Slideflow: deep learning for digital histopathology with real-time whole-slide visualization," BMC Bioinformatics, vol. 25, no. 1, p. 134, 2024.

[10] Google. "MedGemma." [Online]. Available: https://developers.google.com/health-ai-developer-foundations/medgemma

[11] Gradio. "Gradio: Build & Share Delightful ML Apps." [Online]. Available: https://www.gradio.app/

[12] OpenSeadragon. "OpenSeadragon: An open-source, web-based viewer for high-resolution zoomable images." [Online]. Available: https://openseadragon.github.io/

[13] NVIDIA. "NVIDIA Container Toolkit." [Online]. Available: https://github.com/NVIDIA/nvidia-docker

[14] NVIDIA. "NVIDIA DGX Spark." [Online]. Available: https://www.nvidia.com/en-us/data-center/dgx-spark/

[15] NVIDIA. "Triton Inference Server." [Online]. Available: https://developer.nvidia.com/nvidia-triton-inference-server

[16] A. M. Macenko et al., "A method for normalizing histology slides for quantitative analysis," in 2009 IEEE International Symposium on Biomedical Imaging: From Nano to Macro, 2009, pp. 1107-1110.
