# Enso Atlas Frontend

Next.js 14.2 frontend for the Enso Atlas pathology evidence engine.

## Overview

The frontend provides a clinical-grade interface for whole-slide image analysis, treatment-response prediction, and evidence-based clinical reporting. It connects to the FastAPI backend (port 8003) and PostgreSQL database for slide management, model execution, and result caching.

## Competition Context

This frontend is part of the submission for the **Kaggle-Google Med Gemma Impact Challenge**:

- Writeups page: https://www.kaggle.com/competitions/med-gemma-impact-challenge/writeups
- Challenge prompt: **The MedGemma Impact Challenge — Build human-centered AI applications with MedGemma and other open models from Google’s Health AI Developer Foundations (HAI-DEF).**

## Technology Stack

- **Framework**: Next.js 14.2 with App Router
- **Language**: TypeScript
- **Styling**: Tailwind CSS
- **WSI Viewer**: OpenSeadragon with heatmap overlays
- **Layout**: react-resizable-panels v4
- **UI Components**: Custom clinical-grade components with dark mode support

## Screenshots

See the [main README](../README.md#screenshots) for full application screenshots, including the 3-panel dashboard, oncologist and pathologist views, batch analysis, slide manager, and project management interfaces.

---

## Getting Started

### Prerequisites

- Node.js 18+
- npm
- Backend API running at localhost:8003

### Installation

```bash
cd frontend
npm install
```

### Development

```bash
npm run dev
# Available at http://localhost:3000
```

### Production Build

```bash
npm run build
npx next start -p 3002
# Available at http://localhost:3002
```

## Features

### 3-Panel Layout
- **Case Selection** (left): Slide browser with thumbnails, filtering, embedding status chips (Level 0/1), and model picker filtered by project
- **WSI Viewer** (center): OpenSeadragon deep-zoom viewer with TransMIL attention heatmap overlays (jet colormap), minimap, annotation tools
- **Analysis Results** (right): Prediction scores, evidence patches with normalized attention weights, similar cases, clinical reports

### View Modes
- **Oncologist**: Treatment-focused view with prediction summary and clinical report
- **Pathologist**: Detailed evidence view with annotation tools (always visible), attention heatmaps, and patch-level analysis
- **Batch**: Multi-slide batch processing with parallel execution

### Pages
- **/** - Main analysis workspace (3-panel layout)
- **/slides** - Slide Manager with thumbnails, filtering, and pagination
- **/projects** - Project Management with CRUD operations and slide upload

### Analysis Capabilities
- Unified "Run Analysis" button with result caching in PostgreSQL
- Multi-model ensemble analysis
- MedGemma clinical report generation with PDF export
- Semantic search via MedSigLIP (text-to-patch retrieval)
- Similar case retrieval via FAISS
- AI Assistant with 7-step agentic workflow
- Batch processing with parallel execution

## Backend API Requirements

The frontend expects the backend at `NEXT_PUBLIC_API_URL` (default: `http://localhost:8003`).

### Key Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /api/health | Health check |
| GET | /api/slides | List slides (with ?project_id= filtering) |
| GET | /api/slides/{id}/dzi | DZI metadata for OpenSeadragon |
| GET | /api/slides/{id}/thumbnail | Slide thumbnail |
| GET | /api/slides/{id}/cached-results | Cached analysis results |
| POST | /api/analyze | Single slide analysis |
| POST | /api/analyze/batch | Multi-slide batch analysis |
| POST | /api/analyze/multi-model | Multi-model ensemble |
| GET | /api/models | List models (with ?project_id= filtering) |
| GET/POST/PUT/DELETE | /api/projects | Project CRUD |
| GET/POST/DELETE | /api/projects/{id}/slides | Assign/unassign slides |
| GET/POST/DELETE | /api/projects/{id}/models | Assign/unassign models |
| POST | /api/semantic-search | MedSigLIP text-to-patch search |
| GET | /api/heatmap/{slide_id}/{model_id} | Attention heatmap tiles |
| POST | /api/report | Generate MedGemma clinical report |
| GET | /api/slides/{id}/report/pdf | Export report as PDF |

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| NEXT_PUBLIC_API_URL | Backend API base URL | http://localhost:8003 |

## Design Principles

1. **Clinical-grade UI**: Professional design suitable for medical settings
2. **Evidence-first**: All AI predictions include supporting evidence and attention maps
3. **Safety-conscious**: Clear disclaimers and limitations displayed throughout
4. **Offline-capable**: Designed for on-premise deployment with no external dependencies
5. **Accessible**: Dark mode, keyboard navigation, focus states

## License

MIT License
