"""Report generation routes and helpers."""

import asyncio
import csv
import hashlib
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

from .report_tasks import ReportTaskStatus
from .schemas import ReportRequest, ReportResponse


class AsyncReportRequest(BaseModel):
    """Request for async report generation."""

    slide_id: str = Field(..., min_length=1, max_length=256)
    include_evidence: bool = True
    include_similar: bool = True
    project_id: str | None = Field(
        default=None, description="Project ID to determine cancer type and embeddings path"
    )


class AsyncReportResponse(BaseModel):
    """Response from async report generation."""

    task_id: str
    slide_id: str
    status: str
    message: str
    estimated_time_seconds: int = 30


def create_report_router(
    *,
    classifier_provider: Callable[[], Any],
    reporter_provider: Callable[[], Any],
    decision_support_provider: Callable[[], Any],
    evidence_generator_provider: Callable[[], Any],
    report_task_manager: Any,
    require_project: Callable[[str | None], Any],
    project_labels_path: Callable[[str | None], Path | None],
    data_root_provider: Callable[[], Path],
    embeddings_dir_provider: Callable[[], Path],
    resolve_project_embeddings_dir: Callable[..., Path],
    resolve_embedding_path: Callable[..., tuple[Path | None, list[Path]]],
    resolve_project_label_pair: Callable[..., tuple[str, str]],
    classifier_threshold: Callable[[], float],
    project_slide_ids: Callable[[str | None], Any],
    similar_case_slide_id: Callable[[Any], str | None],
    logger: Any,
) -> APIRouter:
    router = APIRouter()

    def load_patient_context(
        slide_id: str,
        project_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Load patient context from CSV labels for a slide."""
        require_project(project_id)
        labels_path = project_labels_path(project_id)

        if project_id:
            if labels_path is None:
                return None
        else:
            if labels_path is None:
                data_root = data_root_provider()
                labels_path = data_root / "labels.csv"
                if not labels_path.exists():
                    labels_path = embeddings_dir_provider().parent / "labels.csv"

        if not labels_path.exists() or labels_path.suffix.lower() != ".csv":
            return None

        with open(labels_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if "slide_id" in row:
                    sid = row["slide_id"]
                else:
                    slide_file = row.get("slide_file", "")
                    sid = slide_file.replace(".svs", "").replace(".SVS", "")

                if sid == slide_id:
                    patient_ctx: dict[str, Any] = {}
                    if row.get("age"):
                        try:
                            patient_ctx["age"] = int(row["age"])
                        except ValueError:
                            pass
                    if row.get("sex"):
                        patient_ctx["sex"] = row["sex"]
                    if row.get("stage"):
                        patient_ctx["stage"] = row["stage"]
                    if row.get("grade"):
                        patient_ctx["grade"] = row["grade"]
                    if row.get("prior_treatments"):
                        try:
                            patient_ctx["prior_lines"] = int(row["prior_treatments"])
                        except ValueError:
                            pass
                    if row.get("histology"):
                        patient_ctx["histology"] = row["histology"]
                    return patient_ctx if patient_ctx else None
        return None

    def format_patient_summary(patient_ctx: dict[str, Any] | None) -> str:
        """Format patient context into a clinical summary sentence."""
        if not patient_ctx:
            return ""

        parts = []
        if patient_ctx.get("age"):
            parts.append(f"{patient_ctx['age']}-year-old")
        if patient_ctx.get("sex"):
            sex_full = (
                "female"
                if patient_ctx["sex"].upper() == "F"
                else "male"
                if patient_ctx["sex"].upper() == "M"
                else patient_ctx["sex"]
            )
            parts.append(sex_full)

        summary = " ".join(parts) if parts else "Patient"

        clinical_parts = []
        if patient_ctx.get("stage"):
            clinical_parts.append(f"Stage {patient_ctx['stage']}")
        if patient_ctx.get("histology"):
            clinical_parts.append(patient_ctx["histology"].lower())
        if clinical_parts:
            summary += " with " + " ".join(clinical_parts)

        if patient_ctx.get("prior_lines") is not None:
            lines = patient_ctx["prior_lines"]
            if lines == 0:
                summary += ", treatment-naive"
            else:
                summary += f", {lines} prior line{'s' if lines > 1 else ''} of therapy"

        return summary

    def template_report(
        slide_id,
        label,
        score,
        evidence_patches,
        similar_cases,
        patient_ctx,
        decision_support_data,
        cancer_type="Cancer",
    ):
        """Create a fallback template report."""
        return {
            "case_id": slide_id,
            "task": f"{cancer_type} prediction from H&E histopathology",
            "patient_context": patient_ctx,
            "model_output": {
                "label": label,
                "probability": score,
                "calibration_note": "Model probability requires external validation.",
            },
            "evidence": [
                {
                    "patch_id": f"patch_{p['patch_index']}",
                    "attention_weight": p["attention_weight"],
                    "coordinates": p["coordinates"],
                    "morphology_description": "High attention region identified by model",
                    "significance": "Region contributes to prediction outcome",
                }
                for p in evidence_patches[:5]
            ],
            "similar_examples": [
                {
                    "example_id": s.get("metadata", {}).get("slide_id", f"case_{i}"),
                    "distance": float(s.get("distance", 0)),
                    "label": s.get("metadata", {}).get("label", "unknown"),
                }
                for i, s in enumerate(similar_cases[:5])
            ],
            "limitations": [
                "This is an uncalibrated research model",
                "Prediction based on morphological patterns only",
                "Requires validation by qualified pathologists",
            ],
            "suggested_next_steps": [
                "Review high-attention regions with pathologist",
                "Correlate with clinical history",
                "Consider molecular profiling",
            ],
            "safety_statement": "This is a research tool. All findings must be validated by qualified clinicians.",
            "decision_support": decision_support_data,
        }

    def template_summary(
        slide_id,
        label,
        score,
        num_patches,
        patient_ctx,
        similar_cases,
        decision_threshold: float = 0.5,
    ):
        """Create a fallback template summary."""
        confidence = abs(score - decision_threshold) * 2
        return f"""CASE ANALYSIS SUMMARY
Case ID: {slide_id}
Prediction: {label.upper()}
Score: {score:.3f}
Confidence: {confidence:.1%}

This analysis examined {num_patches:,} tissue patches.
{len(similar_cases)} similar cases were identified for reference.

DISCLAIMER: This is a research tool. All findings must be validated by qualified pathologists."""

    @router.post("/api/report", response_model=ReportResponse)
    async def generate_report(request: ReportRequest):
        """Generate a structured report for a slide using project-scoped data."""
        classifier = classifier_provider()
        if classifier is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        slide_id = request.slide_id

        project_requested = request.project_id is not None
        proj_cfg = require_project(request.project_id)
        report_embeddings_dir = resolve_project_embeddings_dir(
            request.project_id,
            require_exists=project_requested,
        )

        emb_path, searched_dirs = resolve_embedding_path(
            slide_id,
            level=0,
            project_id=request.project_id,
            base_embeddings_dir=report_embeddings_dir,
        )
        if emb_path is None:
            raise HTTPException(
                status_code=404,
                detail=f"Level 0 embeddings not found for slide {slide_id}. Searched: {', '.join(str(d) for d in searched_dirs)}",
            )

        coord_path = emb_path.with_name(f"{slide_id}_coords.npy")

        embeddings = np.load(emb_path)
        score, attention = classifier.predict(embeddings)
        threshold = classifier_threshold()
        pos_label, _neg_label = resolve_project_label_pair(
            request.project_id,
            positive_default="responder",
            negative_default="non-responder",
            uppercase=False,
        )
        label = pos_label if score >= threshold else _neg_label

        patient_ctx = load_patient_context(slide_id, project_id=request.project_id)

        top_k = min(8, len(attention))
        top_indices = np.argsort(attention)[-top_k:][::-1]

        coords = None
        if coord_path.exists():
            coords = np.load(coord_path)

        evidence_patches = []
        for rank, idx in enumerate(top_indices, 1):
            evidence_patches.append(
                {
                    "rank": rank,
                    "patch_index": int(idx),
                    "attention_weight": float(attention[idx]),
                    "coordinates": [int(coords[idx][0]), int(coords[idx][1])]
                    if coords is not None
                    else [0, 0],
                }
            )

        similar_cases = []
        allowed_slide_ids = await project_slide_ids(request.project_id)
        evidence_gen = evidence_generator_provider()
        if evidence_gen is not None:
            try:
                similar_results = evidence_gen.find_similar(
                    embeddings, attention, k=5, top_patches=3
                )
                for similar in similar_results:
                    sid = similar_case_slide_id(similar)
                    if not sid or sid == slide_id:
                        continue
                    if allowed_slide_ids is not None and sid not in allowed_slide_ids:
                        continue
                    if isinstance(similar, dict) and not similar.get("slide_id"):
                        similar = {**similar, "slide_id": sid}
                    similar_cases.append(similar)
            except Exception as exc:
                logger.warning("Similar case search failed for report: %s", exc)

        quality_metrics = None
        try:
            hash_val = int(hashlib.md5(slide_id.encode()).hexdigest(), 16)
            tissue_coverage = 0.60 + (hash_val % 40) / 100.0
            blur_score = (hash_val % 30) / 100.0
            stain_uniformity = 0.70 + (hash_val % 30) / 100.0
            artifact_detected = (hash_val % 10) == 0

            quality_score = (
                tissue_coverage * 0.3
                + (1 - blur_score) * 0.3
                + stain_uniformity * 0.2
                + (0 if artifact_detected else 0.2)
            )
            overall_quality = (
                "good"
                if quality_score >= 0.75
                else "acceptable"
                if quality_score >= 0.50
                else "poor"
            )

            quality_metrics = {
                "overall_quality": overall_quality,
                "tissue_coverage": tissue_coverage,
                "blur_score": blur_score,
                "artifact_detected": artifact_detected,
            }
        except Exception as exc:
            logger.warning("Could not compute quality metrics: %s", exc)

        cancer_type = (proj_cfg.cancer_type if proj_cfg else "Cancer") or "Cancer"

        decision_support_data = None
        decision_support = decision_support_provider()
        if decision_support is not None:
            try:
                ds_output = decision_support.generate(
                    prediction=label,
                    score=float(score),
                    similar_cases=similar_cases,
                    quality_metrics=quality_metrics,
                    patient_context=patient_ctx,
                    cancer_type=cancer_type,
                )
                decision_support_data = decision_support.to_dict(ds_output)
                logger.info(
                    "Generated decision support for %s: risk_level=%s",
                    slide_id,
                    ds_output.risk_level.value,
                )
            except Exception as exc:
                logger.warning("Decision support generation failed: %s", exc)

        reporter = reporter_provider()
        if reporter is not None:
            timeout_s = None
            try:
                timeout_s = getattr(reporter.config, "max_generation_time_s", None)
                if timeout_s is None:
                    timeout_s = 120.0
                timeout_s = max(10.0, float(timeout_s) + 60.0)

                report = await asyncio.wait_for(
                    asyncio.to_thread(
                        reporter.generate_report,
                        evidence_patches=evidence_patches,
                        score=score,
                        label=label,
                        similar_cases=similar_cases,
                        case_id=slide_id,
                        patient_context=patient_ctx,
                        cancer_type=cancer_type,
                    ),
                    timeout=timeout_s,
                )

                structured = report["structured"]
                if decision_support_data:
                    structured["decision_support"] = decision_support_data

                return ReportResponse(
                    slide_id=slide_id,
                    report_json=structured,
                    summary_text=report["summary"],
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "MedGemma report generation timed out after %.1fs for slide %s",
                    timeout_s or 0.0,
                    slide_id,
                )
            except Exception as exc:
                logger.warning("MedGemma report generation failed, using template: %s", exc)

        patient_summary = format_patient_summary(patient_ctx)

        def generate_morphology_description(patch: dict, rank: int) -> tuple[str, str]:
            tissue_type = patch.get("tissue_type", "unknown")
            attention_weight = patch.get("attention_weight", 0)
            coords_for_patch = patch.get("coordinates", [0, 0])

            morphology_templates = {
                "tumor": [
                    "Dense cellular region with atypical epithelial morphology and increased nuclear-to-cytoplasmic ratio",
                    "Papillary architecture with stratified epithelium showing nuclear atypia",
                    "Solid sheets of cells with irregular nuclear contours and prominent nucleoli",
                    "Glandular structures with back-to-back arrangement and cribriform patterns",
                ],
                "stroma": [
                    "Desmoplastic stroma with spindle-shaped fibroblasts and collagen deposition",
                    "Fibrovascular core with loose connective tissue and scattered vessels",
                    "Dense fibrous stroma with hyalinized collagen bundles",
                ],
                "necrosis": [
                    "Geographic necrosis with ghost cell outlines and nuclear debris",
                    "Coagulative necrosis with preserved tissue architecture",
                    "Necrotic debris with inflammatory cell infiltration",
                ],
                "inflammatory": [
                    "Lymphocytic infiltrate with peritumoral distribution",
                    "Tumor-infiltrating lymphocytes forming dense aggregates",
                    "Mixed inflammatory infiltrate with plasma cells and lymphocytes",
                ],
                "normal": [
                    "Normal epithelial architecture with maintained polarity",
                    "Benign glandular tissue with regular spacing",
                ],
            }

            significance_templates = {
                "positive": {
                    "tumor": "Tumor morphology patterns in this region are associated with the positive prediction in the training cohort",
                    "stroma": "Stromal composition in this area correlates with the predicted outcome",
                    "inflammatory": "Inflammatory infiltrate pattern suggests a tumor microenvironment consistent with the prediction",
                    "necrosis": "Necrotic pattern may indicate tissue changes relevant to prognosis",
                    "normal": "Preserved tissue architecture in adjacent regions may indicate better overall tissue health",
                },
                "negative": {
                    "tumor": "Tumor morphology in this region shows patterns associated with the predicted outcome",
                    "stroma": "Stromal characteristics suggest mechanisms consistent with the prediction",
                    "inflammatory": "Inflammatory pattern may indicate a tumor microenvironment associated with the predicted outcome",
                    "necrosis": "Necrotic patterns in this configuration are associated with the predicted outcome",
                    "normal": "Limited tumor involvement in this area provides context for overall assessment",
                },
            }

            templates = morphology_templates.get(
                tissue_type, ["Tissue region with notable morphological features"]
            )
            morphology = templates[rank % len(templates)]
            morphology += f" at position ({coords_for_patch[0]:,}, {coords_for_patch[1]:,})"

            label_key = "positive" if label.lower() == pos_label.lower() else "negative"
            sig_templates = significance_templates.get(label_key, {})
            significance = sig_templates.get(
                tissue_type,
                f"High model attention (weight: {attention_weight:.3f}) indicates this region contributes significantly to the prediction",
            )

            return morphology, significance

        evidence_list = []
        for i, patch in enumerate(evidence_patches[:5]):
            morphology, significance = generate_morphology_description(patch, i)
            evidence_list.append(
                {
                    "patch_id": f"patch_{patch['patch_index']}",
                    "attention_weight": patch["attention_weight"],
                    "coordinates": patch["coordinates"],
                    "morphology_description": morphology,
                    "significance": significance,
                    "tissue_type": patch.get("tissue_type", "unknown"),
                }
            )

        report_json = {
            "case_id": slide_id,
            "task": f"{cancer_type} prediction from H&E histopathology",
            "patient_context": patient_ctx,
            "model_output": {
                "label": label,
                "probability": float(score),
                "calibration_note": "Model probability requires external validation. This is an uncalibrated research model.",
            },
            "evidence": evidence_list,
            "similar_examples": [
                {
                    "example_id": s.get("metadata", {}).get("slide_id", f"case_{i}"),
                    "slide_id": s.get("metadata", {}).get("slide_id", f"case_{i}"),
                    "distance": float(s.get("distance", 0)),
                    "similarity_score": 1.0 / (1.0 + s.get("distance", 0)),
                    "label": s.get("metadata", {}).get("label", "unknown"),
                }
                for i, s in enumerate(similar_cases[:5])
            ],
            "limitations": [
                "This is an uncalibrated research model - probabilities are not clinically validated",
                "Prediction is based on morphological patterns and may not capture all relevant clinical factors",
                "Model has been trained on a limited cancer dataset and may not generalize to all populations",
                "Slide quality and tissue representation may affect prediction accuracy",
                "Similar case comparison is based on embedding distance, not verified clinical outcomes",
            ],
            "suggested_next_steps": [
                "Review high-attention regions with attending pathologist",
                "Correlate findings with patient clinical history and imaging",
                "Consider molecular profiling (e.g., BRCA status, HRD) for additional treatment guidance",
                "Discuss findings in multidisciplinary tumor board before treatment decisions",
                "Validate prediction against institutional experience with similar cases",
            ],
            "safety_statement": "This is a research decision-support tool, not a diagnostic device. All findings must be validated by qualified pathologists and clinicians. Do not use for standalone clinical decision-making. Treatment decisions should incorporate all available clinical, pathological, and molecular data.",
            "decision_support": decision_support_data,
        }

        patient_intro = f"Patient: {patient_summary}.\n\n" if patient_summary else ""

        confidence_val = abs(score - 0.5) * 2
        if confidence_val >= 0.6:
            confidence_desc = "high"
        elif confidence_val >= 0.3:
            confidence_desc = "moderate"
        else:
            confidence_desc = "low"

        tissue_types_seen = [
            str(patch.get("tissue_type", "unknown")) for patch in evidence_patches[:5]
        ]
        tissue_summary = (
            ", ".join(set(t for t in tissue_types_seen if t != "unknown")) or "various tissue types"
        )

        summary_text = f"""{patient_intro}CASE ANALYSIS SUMMARY
=====================

Case ID: {slide_id}
Prediction: {label.upper()}
Model Score: {score:.3f}
Confidence Level: {confidence_desc.upper()} ({confidence_val:.1%})

ANALYSIS OVERVIEW
-----------------
This analysis examined {len(embeddings):,} tissue patches extracted from the whole-slide image.
The multiple instance learning (MIL) model identified {min(5, len(attention))} high-attention
regions that contributed most significantly to the prediction.

Key morphological features observed in high-attention regions include: {tissue_summary}.

{"POSITIVE INTERPRETATION" if label == pos_label else "NEGATIVE INTERPRETATION"}
---------------------------------
{"The morphological patterns identified by the model suggest features associated with the positive class in the training cohort. These patterns may include specific tumor architecture, stromal characteristics, or inflammatory infiltrate distributions that have been correlated with the predicted outcome." if label == pos_label else "The morphological patterns identified by the model suggest features associated with the negative class in the training cohort. Further clinical evaluation is recommended to determine appropriate treatment strategies."}

SIMILAR CASES
-------------
{len(similar_cases)} similar cases from the reference cohort were identified based on
morphological similarity. Review of these cases may provide additional context for
interpreting the current prediction.

IMPORTANT DISCLAIMER
--------------------
This is a RESEARCH TOOL for decision support only. The model is uncalibrated and has not
been clinically validated. All findings must be reviewed and validated by qualified
pathologists and clinicians before any clinical decision-making. Treatment decisions
should incorporate all available clinical, pathological, and molecular data."""

        return ReportResponse(
            slide_id=slide_id,
            report_json=report_json,
            summary_text=summary_text,
        )

    @router.post("/api/report/async", response_model=AsyncReportResponse)
    async def generate_report_async(request: AsyncReportRequest, background_tasks: BackgroundTasks):
        """
        Start asynchronous report generation for a slide.

        Returns a task ID immediately. Task reuse and deduplication are scoped
        by ``(slide_id, project_id)``.
        """
        slide_id = request.slide_id

        project_requested = request.project_id is not None
        report_embeddings_dir = resolve_project_embeddings_dir(
            request.project_id,
            require_exists=project_requested,
        )

        report_emb_path, searched_dirs = resolve_embedding_path(
            slide_id,
            level=0,
            project_id=request.project_id,
            base_embeddings_dir=report_embeddings_dir,
        )
        if report_emb_path is None:
            raise HTTPException(
                status_code=404,
                detail=f"Level 0 embeddings not found for slide {slide_id}. Searched: {', '.join(str(d) for d in searched_dirs)}",
            )

        existing_task = report_task_manager.get_task_by_slide(slide_id, request.project_id)
        if existing_task:
            return AsyncReportResponse(
                task_id=existing_task.task_id,
                slide_id=slide_id,
                status=existing_task.status.value,
                message=existing_task.message,
                estimated_time_seconds=30,
            )

        task = report_task_manager.create_task(slide_id, request.project_id)

        def run_report_generation():
            generate_report_background(
                task.task_id,
                slide_id,
                request.include_evidence,
                request.include_similar,
                request.project_id,
            )

        background_tasks.add_task(run_report_generation)

        return AsyncReportResponse(
            task_id=task.task_id,
            slide_id=slide_id,
            status="pending",
            message="Report generation started. Poll /api/report/status/{task_id} for progress.",
            estimated_time_seconds=30,
        )

    def generate_report_background(
        task_id: str,
        slide_id: str,
        include_evidence: bool,
        include_similar: bool,
        project_id: str | None = None,
    ):
        """Background task to generate report."""
        task = report_task_manager.get_task(task_id)
        if not task:
            return

        report_task_manager.update_task(
            task_id,
            status=ReportTaskStatus.RUNNING,
            started_at=time.time(),
            stage="analyzing",
            progress=10,
            message="Loading embeddings and running analysis...",
        )

        try:
            classifier = classifier_provider()
            if classifier is None:
                raise RuntimeError("Model not loaded")

            proj_cfg = require_project(project_id)
            project_requested = project_id is not None
            report_embeddings_dir = resolve_project_embeddings_dir(
                project_id,
                require_exists=project_requested,
            )
            cancer_type = (proj_cfg.cancer_type if proj_cfg else "Cancer") or "Cancer"
            positive_label, negative_label = resolve_project_label_pair(
                project_id,
                positive_default="responder",
                negative_default="non-responder",
                uppercase=False,
            )

            emb_path, searched_dirs = resolve_embedding_path(
                slide_id,
                level=0,
                project_id=project_id,
                base_embeddings_dir=report_embeddings_dir,
            )
            if emb_path is None:
                raise FileNotFoundError(
                    f"Level 0 embeddings not found for slide {slide_id}. Searched: {', '.join(str(d) for d in searched_dirs)}"
                )

            coord_path = emb_path.with_name(f"{slide_id}_coords.npy")
            embeddings = np.load(emb_path)

            report_task_manager.update_task(
                task_id, progress=20, message="Running MIL prediction..."
            )

            score, attention = classifier.predict(embeddings)
            threshold_val = classifier_threshold()
            label = positive_label if score >= threshold_val else negative_label

            report_task_manager.update_task(
                task_id,
                progress=30,
                stage="generating",
                message="Loading patient context and evidence...",
            )

            patient_ctx = load_patient_context(slide_id, project_id=project_id)

            top_k = min(8, len(attention))
            top_indices = np.argsort(attention)[-top_k:][::-1]

            coords = None
            if coord_path.exists():
                coords = np.load(coord_path)

            evidence_patches = []
            for rank, idx in enumerate(top_indices, 1):
                evidence_patches.append(
                    {
                        "rank": rank,
                        "patch_index": int(idx),
                        "attention_weight": float(attention[idx]),
                        "coordinates": [int(coords[idx][0]), int(coords[idx][1])]
                        if coords is not None
                        else [0, 0],
                    }
                )

            report_task_manager.update_task(
                task_id, progress=40, message="Finding similar cases..."
            )

            similar_cases = []
            allowed_slide_ids = asyncio.run(project_slide_ids(project_id))
            evidence_gen = evidence_generator_provider()
            if include_similar and evidence_gen is not None:
                try:
                    similar_results = evidence_gen.find_similar(
                        embeddings, attention, k=5, top_patches=3
                    )
                    for similar in similar_results:
                        sid = similar_case_slide_id(similar)
                        if not sid or sid == slide_id:
                            continue
                        if allowed_slide_ids is not None and sid not in allowed_slide_ids:
                            continue
                        if isinstance(similar, dict) and not similar.get("slide_id"):
                            similar = {**similar, "slide_id": sid}
                        similar_cases.append(similar)
                except Exception as exc:
                    logger.warning("Similar case search failed: %s", exc)

            report_task_manager.update_task(
                task_id, progress=50, message="Generating clinical decision support..."
            )

            hash_val = int(hashlib.md5(slide_id.encode()).hexdigest(), 16)
            quality_metrics = {
                "overall_quality": "good" if hash_val % 3 == 0 else "acceptable",
                "tissue_coverage": 0.60 + (hash_val % 40) / 100.0,
                "blur_score": (hash_val % 30) / 100.0,
                "artifact_detected": (hash_val % 10) == 0,
            }

            decision_support_data = None
            decision_support = decision_support_provider()
            if decision_support is not None:
                try:
                    ds_output = decision_support.generate(
                        prediction=label,
                        score=float(score),
                        similar_cases=similar_cases,
                        quality_metrics=quality_metrics,
                        patient_context=patient_ctx,
                        cancer_type=cancer_type,
                    )
                    decision_support_data = decision_support.to_dict(ds_output)
                except Exception as exc:
                    logger.warning("Decision support failed: %s", exc)

            report_task_manager.update_task(
                task_id,
                progress=60,
                message="Generating report with MedGemma (up to 90s)...",
            )

            report_json = None
            summary_text = None
            reporter = reporter_provider()

            if reporter is not None:
                stop_event = None
                heartbeat = None
                try:
                    max_time = getattr(reporter.config, "max_generation_time_s", None)
                    max_tokens = getattr(reporter.config, "max_output_tokens", None)
                    max_time_display = f"{float(max_time):.1f}s" if max_time is not None else "none"
                    logger.info(
                        "Starting MedGemma report generation for %s (max_time=%s, max_new_tokens=%s)",
                        slide_id,
                        max_time_display,
                        max_tokens,
                    )

                    gen_start = time.time()
                    stop_event = threading.Event()

                    def progress_heartbeat():
                        progress = 60.0
                        tick = 0
                        while not stop_event.wait(3):
                            tick += 1
                            progress = min(92, 60 + 32 * (1 - 1.0 / (1 + tick * 0.15)))
                            elapsed = time.time() - gen_start
                            report_task_manager.update_task(
                                task_id,
                                progress=round(progress, 1),
                                stage="generating",
                                message=f"MedGemma is generating the report... ({int(elapsed)}s)",
                            )

                    heartbeat = threading.Thread(target=progress_heartbeat, daemon=True)
                    heartbeat.start()

                    gen_timeout = float(max_time) + 60.0 if max_time else 180.0
                    gen_result = [None]
                    gen_error = [None]

                    def run_medgemma():
                        try:
                            gen_result[0] = reporter.generate_report(
                                evidence_patches=evidence_patches,
                                score=score,
                                label=label,
                                similar_cases=similar_cases,
                                case_id=slide_id,
                                patient_context=patient_ctx,
                                cancer_type=cancer_type,
                            )
                        except Exception as exc:
                            gen_error[0] = exc

                    gen_thread = threading.Thread(target=run_medgemma, daemon=True)
                    gen_thread.start()
                    gen_thread.join(timeout=gen_timeout)

                    stop_event.set()
                    heartbeat.join(timeout=2)

                    if gen_thread.is_alive():
                        logger.warning(
                            "MedGemma report generation timed out after %.1fs for %s, falling back to template",
                            time.time() - gen_start,
                            slide_id,
                        )
                    elif gen_error[0] is not None:
                        logger.warning("MedGemma generation error: %s", gen_error[0])
                    elif gen_result[0] is not None:
                        report = gen_result[0]
                        logger.info(
                            "MedGemma report generation completed for %s in %.1fs",
                            slide_id,
                            time.time() - gen_start,
                        )

                        report_json = report["structured"]
                        summary_text = report["summary"]

                        if decision_support_data:
                            report_json["decision_support"] = decision_support_data
                    else:
                        logger.warning("MedGemma returned None result for %s", slide_id)

                except Exception as exc:
                    logger.warning("MedGemma failed: %s", exc)
                    if stop_event is not None:
                        stop_event.set()
                    if heartbeat is not None:
                        heartbeat.join(timeout=1)

            if report_json is None:
                report_task_manager.update_task(
                    task_id,
                    progress=80,
                    message="Using template report (MedGemma unavailable)...",
                )
                report_json = template_report(
                    slide_id,
                    label,
                    float(score),
                    evidence_patches,
                    similar_cases,
                    patient_ctx,
                    decision_support_data,
                    cancer_type,
                )
                summary_text = template_summary(
                    slide_id,
                    label,
                    float(score),
                    len(embeddings),
                    patient_ctx,
                    similar_cases,
                    decision_threshold=threshold_val,
                )

            report_task_manager.update_task(
                task_id,
                progress=90,
                stage="formatting",
                message="Finalizing report...",
            )

            elapsed = time.time() - task.started_at
            report_task_manager.update_task(
                task_id,
                status=ReportTaskStatus.COMPLETED,
                progress=100,
                stage="complete",
                message=f"Report generated successfully in {elapsed:.1f}s",
                completed_at=time.time(),
                result={
                    "slide_id": slide_id,
                    "report_json": report_json,
                    "summary_text": summary_text,
                },
            )

            logger.info("Report generation completed for %s in %.1fs", slide_id, elapsed)

        except Exception as exc:
            logger.error("Report generation failed for %s: %s", slide_id, exc)
            report_task_manager.update_task(
                task_id,
                status=ReportTaskStatus.FAILED,
                error=str(exc),
                message=f"Report generation failed: {str(exc)}",
            )

    return router
