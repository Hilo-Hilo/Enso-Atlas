# Enso Atlas Frontend

Professional React/Next.js frontend for the Enso Atlas pathology evidence engine.

## Overview

This frontend provides a clinical-grade user interface for:
- Whole-slide image (WSI) viewing with OpenSeadragon
- AI prediction results display with confidence scores
- Evidence patch visualization with attention-based heatmaps
- Similar case retrieval from FAISS embeddings
- Structured clinical report generation powered by MedGemma

## Technology Stack

- **Framework**: Next.js 14+ with App Router
- **Language**: TypeScript
- **Styling**: Tailwind CSS
- **WSI Viewer**: OpenSeadragon
- **UI Components**: Custom professional medical UI components

## Getting Started

### Prerequisites

- Node.js 18+
- npm or yarn
- Backend API running at localhost:8000 (or configure NEXT_PUBLIC_API_URL)

### Installation

```bash
cd frontend
npm install
```

### Development

```bash
npm run dev
```

The application will be available at http://localhost:3000

### Production Build

```bash
npm run build
npm start
```

## Project Structure

```
frontend/
  src/
    app/                    # Next.js App Router pages
      layout.tsx            # Root layout with metadata
      page.tsx              # Main application page
      globals.css           # Global styles and Tailwind directives
    components/
      ui/                   # Reusable UI components
        Button.tsx          # Clinical button variants
        Card.tsx            # Card and panel components
        Badge.tsx           # Status badges
        Slider.tsx          # Range sliders
        Toggle.tsx          # Toggle switches
        Spinner.tsx         # Loading indicators
      viewer/
        WSIViewer.tsx       # OpenSeadragon WSI viewer with heatmap overlay
      panels/
        PredictionPanel.tsx # Model prediction display
        EvidencePanel.tsx   # Evidence patch gallery
        SimilarCasesPanel.tsx # FAISS similarity results
        ReportPanel.tsx     # Structured clinical report
        SlideSelector.tsx   # Slide selection interface
      layout/
        Header.tsx          # Application header
        Footer.tsx          # Application footer with disclaimers
    hooks/
      useAnalysis.ts        # Analysis workflow state management
      useViewer.ts          # OpenSeadragon viewer state
    lib/
      api.ts                # Backend API client
      utils.ts              # Utility functions
      mock-data.ts          # Development mock data
    types/
      index.ts              # TypeScript type definitions
```

## Backend API Requirements

The frontend expects the following API endpoints:

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /api/health | Health check |
| GET | /api/slides | List available slides |
| GET | /api/slides/:id | Get slide details |
| GET | /api/slides/:id/dzi | Get DZI metadata for OpenSeadragon |
| GET | /api/slides/:id/thumbnail | Get slide thumbnail |
| GET | /api/slides/:id/heatmap | Get attention heatmap |
| POST | /api/analyze | Run analysis on a slide |
| POST | /api/report | Generate structured report |
| GET | /api/slides/:id/report/pdf | Export report as PDF |

## Features

### WSI Viewer
- Deep zoom navigation with OpenSeadragon
- Mini-map navigator
- Attention heatmap overlay with adjustable opacity
- Click-to-navigate to evidence regions
- Fullscreen mode

### Prediction Panel
- Model prediction with probability score
- Confidence level indicator (high/moderate/low)
- Visual probability bar with threshold marker
- Calibration notes

### Evidence Gallery
- Top-K patches by attention weight
- Grid and list view modes
- Click to navigate to patch location
- Morphology descriptions

### Similar Cases
- Reference cohort matches
- Distance-based similarity scores
- Case labels (responder/non-responder)
- Expandable case details

### Clinical Report
- Structured JSON output
- Human-readable summary
- Evidence citations
- Limitations and next steps
- Safety statement
- PDF and JSON export

## Design Principles

1. **Clinical-grade UI**: Professional, clean design suitable for medical settings
2. **Evidence-first**: All AI predictions include supporting evidence
3. **Safety-conscious**: Clear disclaimers and limitations displayed
4. **Offline-capable**: Designed for on-premise deployment
5. **Accessibility**: Focus states and keyboard navigation

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| NEXT_PUBLIC_API_URL | Backend API base URL | http://localhost:8000 |

## Development Notes

- No emojis in code or documentation (clinical standard)
- All components use TypeScript with strict typing
- Tailwind classes follow a clinical color palette
- Components are designed for reuse and extension

## License

Proprietary - Enso Labs
