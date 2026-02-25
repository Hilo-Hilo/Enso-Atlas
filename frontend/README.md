# Enso Atlas Frontend

Next.js 14.2 frontend for the Enso Atlas pathology evidence engine.

## Overview

The frontend provides a research/demo interface for whole-slide image analysis, treatment-response prediction, and evidence-aware report workflows. It connects to the FastAPI backend and PostgreSQL database for slide management, model execution, and result caching.

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
- **UI Components**: Custom components with dark mode support

## Screenshots

See the [main README](../README.md#screenshots) for full application screenshots, including the 3-panel dashboard, oncologist and pathologist views, batch analysis, slide manager, and project management interfaces.

---

## Getting Started

### Prerequisites

- Node.js 18+
- npm
- Backend API running at:
  - `http://localhost:8000` for local backend development, or
  - `http://localhost:8003` when using Docker backend

### Installation

```bash
cd frontend
npm install
```

### API base URL / ports

Set `NEXT_PUBLIC_API_URL` in `frontend/.env.local`:

```bash
# Local backend (recommended local dev)
NEXT_PUBLIC_API_URL=http://localhost:8000

# Docker backend (docker/docker-compose.yaml)
# NEXT_PUBLIC_API_URL=http://localhost:8003
```

If unset, Next.js rewrites default to `http://127.0.0.1:8000`.

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
- **Analysis Results** (right): Prediction scores, evidence patches with normalized attention weights, similar cases, generated reports

### View Modes
- **Oncologist**: Treatment-focused view with prediction summary and generated report
- **Pathologist**: Detailed evidence view with annotation tools (always visible), attention heatmaps, and patch-level analysis
- **Batch**: Multi-slide batch processing with parallel execution

### Pages
- **/** - Main analysis workspace (3-panel layout)
- **/slides** - Slide Manager with thumbnails, filtering, and pagination
- **/projects** - Project Management with CRUD operations and slide upload

### Analysis Capabilities
- Unified "Run Analysis" button with result caching in PostgreSQL
- Multi-model ensemble analysis
- MedGemma report generation with PDF export paths
- Semantic search via MedSigLIP (text-to-patch retrieval)
- Similar case retrieval via FAISS
- AI Assistant with 7-step agentic workflow
- Batch processing with parallel execution

## Backend API Requirements

The frontend expects the backend base URL from `NEXT_PUBLIC_API_URL`:
- local backend: `http://localhost:8000`
- Docker backend: `http://localhost:8003`

### Key Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /api/health | Health check |
| GET | /api/slides | List slides (with `?project_id=` filtering) |
| GET | /api/slides/{id}/dzi | DZI metadata for OpenSeadragon |
| GET | /api/slides/{id}/thumbnail | Slide thumbnail |
| GET | /api/slides/{id}/cached-results | Cached analysis results |
| POST | /api/analyze | Single slide analysis |
| POST | /api/analyze-batch | Multi-slide batch analysis |
| POST | /api/analyze-multi | Multi-model ensemble |
| GET | /api/models | List models (with `?project_id=` filtering) |
| GET/POST | /api/projects | List/create projects |
| GET/PUT/DELETE | /api/projects/{project_id} | Read/update/delete one project |
| GET/POST/DELETE | /api/projects/{project_id}/slides | Assign/unassign slides |
| GET/POST/DELETE | /api/projects/{project_id}/models | Assign/unassign models |
| POST | /api/semantic-search | MedSigLIP text-to-patch search |
| GET | /api/heatmap/{slide_id}/{model_id} | Attention heatmap tiles |
| POST | /api/report | Generate MedGemma report |
| POST | /api/report/async | Start async report generation |
| GET | /api/report/status/{task_id} | Async report status |
| POST | /api/report/pdf | Lightweight PDF from report JSON |
| POST | /api/export/pdf | Full PDF export endpoint |

## Environment Variables

| Variable | Description | Typical Local Value |
|----------|-------------|---------------------|
| NEXT_PUBLIC_API_URL | Backend API base URL | `http://localhost:8000` (local backend) or `http://localhost:8003` (Docker backend) |

## Design Principles

1. **Research-first UI**: Professional design for demo/research pathology workflows
2. **Evidence-first**: AI predictions include supporting evidence and attention maps
3. **Safety-conscious**: Clear disclaimers and limitations displayed throughout
4. **Offline-capable**: Designed for on-premise deployment with no external dependencies
5. **Accessible**: Dark mode, keyboard navigation, focus states

## License

MIT License
