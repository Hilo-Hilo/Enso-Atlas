"""PDF export route handlers."""

from __future__ import annotations

import io
import logging
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from PIL import Image

from .schemas import PdfExportRequest, ReportPdfRequest

AuditLogger = Callable[[str, str | None, str, dict[str, Any] | None], None]


def _pdf_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def create_pdf_router(
    *,
    pdf_export_available_provider: Callable[[], bool],
    generate_report_pdf_provider: Callable[[], Any],
    generate_pdf_report_provider: Callable[[], Any],
    embeddings_dir_provider: Callable[[], Path],
    classifier_provider: Callable[[], Any],
    evidence_generator_provider: Callable[[], Any],
    logger: logging.Logger,
    log_audit_event: AuditLogger,
) -> APIRouter:
    """Create lightweight and full PDF export routes."""
    router = APIRouter()

    @router.post("/api/report/pdf")
    async def report_pdf(request: ReportPdfRequest):
        """Generate a lightweight PDF from a report JSON payload."""
        generate_report_pdf = generate_report_pdf_provider()
        if generate_report_pdf is None:
            raise HTTPException(
                status_code=503,
                detail="PDF export not available. Install fpdf2: pip install fpdf2",
            )

        report_data = request.report
        cid = request.case_id or report_data.get("case_id", report_data.get("caseId", "UNKNOWN"))

        try:
            pdf_bytes = generate_report_pdf(report_data, cid)
        except Exception as exc:
            logger.error("PDF generation failed: %s", exc)
            raise HTTPException(
                status_code=500, detail=f"PDF generation failed: {str(exc)}"
            ) from exc

        filename = f"enso-atlas-report-{cid}-{_pdf_timestamp()}.pdf"

        log_audit_event("pdf_exported", cid, "clinician", {"endpoint": "/api/report/pdf"})

        pdf_content = bytes(pdf_bytes) if not isinstance(pdf_bytes, bytes) else pdf_bytes
        return Response(
            content=pdf_content,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "Content-Length": str(len(pdf_content)),
            },
        )

    @router.post("/api/export/pdf")
    async def export_pdf(request: PdfExportRequest):
        """Generate a full PDF report for tumor-board presentation."""
        if not pdf_export_available_provider():
            raise HTTPException(
                status_code=503,
                detail="PDF export not available. Install reportlab: pip install reportlab>=4.0.0",
            )

        generate_pdf_report = generate_pdf_report_provider()
        if generate_pdf_report is None:
            raise HTTPException(
                status_code=503,
                detail="PDF export not available. Install reportlab: pip install reportlab>=4.0.0",
            )

        slide_id = request.slide_id

        heatmap_image = None
        if request.include_heatmap:
            try:
                embeddings_dir = embeddings_dir_provider()
                classifier = classifier_provider()
                evidence_gen = evidence_generator_provider()
                emb_path = embeddings_dir / f"{slide_id}.npy"
                coord_path = embeddings_dir / f"{slide_id}_coords.npy"

                if emb_path.exists() and classifier is not None and evidence_gen is not None:
                    embeddings = np.load(emb_path)

                    patch_size = 224
                    if coord_path.exists():
                        coords_arr = np.load(coord_path).astype(np.int64, copy=False)
                    else:
                        raise FileNotFoundError(
                            f"Patch coordinates missing for {slide_id}; cannot generate truthful PDF heatmap."
                        )

                    coords = [tuple(map(int, c)) for c in coords_arr]

                    import torch

                    from ..mil.clam import LegacyCLAMModel

                    x = torch.from_numpy(embeddings).float()
                    model = LegacyCLAMModel(input_dim=384, hidden_dim=256)

                    model_path = (
                        Path(__file__).parent.parent.parent.parent / "models" / "clam_attention.pt"
                    )
                    if model_path.exists():
                        checkpoint = torch.load(
                            model_path, map_location=torch.device("cpu"), weights_only=False
                        )
                        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                            state_dict = checkpoint["model_state_dict"]
                        else:
                            state_dict = checkpoint
                        model.load_state_dict(state_dict)

                    model.eval()
                    with torch.no_grad():
                        _, attention = model(x, return_attention=True)

                    if coords_arr.size > 0:
                        x_max = int(coords_arr[:, 0].max()) + patch_size
                        y_max = int(coords_arr[:, 1].max()) + patch_size
                        slide_dims = (x_max, y_max)
                    else:
                        slide_dims = (patch_size, patch_size)

                    heatmap = evidence_gen.create_heatmap(
                        attention.numpy(),
                        coords,
                        slide_dims,
                        (512, 512),
                        smooth=True,
                        blur_kernel=31,
                    )

                    img = Image.fromarray(heatmap)
                    buf = io.BytesIO()
                    img.save(buf, format="PNG")
                    heatmap_image = buf.getvalue()

            except Exception as exc:
                logger.warning("Failed to generate heatmap for PDF: %s", exc)

        evidence_patches = None
        if request.include_evidence_patches:
            evidence_patches = []
            evidence_items = request.report_data.get("evidence", [])[:9]

            for item in evidence_items:
                patch_data = {
                    "attention": item.get("attentionWeight", 0),
                    "image": None,
                }

                patch_id = item.get("patchId", item.get("patch_id"))
                if patch_id:
                    try:
                        patch_cache_dir = (
                            Path(__file__).parent.parent.parent.parent
                            / "outputs"
                            / "patches"
                            / slide_id
                        )
                        patch_path = patch_cache_dir / f"{patch_id}.png"

                        if patch_path.exists():
                            with open(patch_path, "rb") as f:
                                patch_data["image"] = f.read()
                    except Exception as exc:
                        logger.warning("Failed to load patch %s: %s", patch_id, exc)

                evidence_patches.append(patch_data)

        try:
            pdf_bytes = generate_pdf_report(
                slide_id=slide_id,
                report_data=request.report_data,
                prediction_data=request.prediction_data,
                heatmap_image=heatmap_image,
                evidence_patches=evidence_patches,
                institution_name="Enso Labs",
                patient_context=request.patient_context,
            )
        except Exception as exc:
            logger.error("PDF generation failed: %s", exc)
            raise HTTPException(
                status_code=500, detail=f"PDF generation failed: {str(exc)}"
            ) from exc

        filename = f"enso-atlas-report-{slide_id}-{_pdf_timestamp()}.pdf"

        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "Content-Length": str(len(pdf_bytes)),
            },
        )

    return router
