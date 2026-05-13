"""Slide and project-membership resolution helpers."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path


def slide_id_base(slide_id: str) -> str:
    """Return deterministic slide-id base used for UUID-tolerant matching."""
    sid = (slide_id or "").strip()
    if not sid:
        return ""
    return sid.split(".", 1)[0]


def resolve_slide_path_in_dirs(
    slide_id: str,
    candidate_dirs: list[Path],
    exts: list[str],
    *,
    logger_obj: logging.Logger | None = None,
) -> Path | None:
    """Resolve slide path with exact matching before unambiguous base fallback."""
    for d in candidate_dirs:
        if not d.exists():
            continue
        for ext in exts:
            cand = d / f"{slide_id}{ext}"
            if cand.exists():
                return cand

    base = slide_id_base(slide_id)
    if not base:
        return None

    wildcard_matches: list[Path] = []
    seen: set[Path] = set()

    for d in candidate_dirs:
        if not d.exists():
            continue
        for ext in exts:
            for cand in sorted(d.glob(f"{base}.*{ext}")):
                if not cand.is_file() or cand in seen:
                    continue
                wildcard_matches.append(cand)
                seen.add(cand)

    if len(wildcard_matches) == 1:
        return wildcard_matches[0]

    if len(wildcard_matches) > 1 and logger_obj is not None:
        logger_obj.warning(
            "Ambiguous WSI fallback for slide_id=%s (base=%s): %s",
            slide_id,
            base,
            [str(p) for p in wildcard_matches],
        )

    return None


def load_slide_ids_from_labels_file(labels_path: Path | None) -> set[str]:
    """Load slide IDs from a labels file (CSV or JSON)."""
    if labels_path is None or not labels_path.exists():
        return set()

    slide_ids: set[str] = set()

    try:
        if labels_path.suffix.lower() == ".json":
            with open(labels_path) as f:
                payload = json.load(f)
            if isinstance(payload, dict):
                for sid in payload.keys():
                    sid_str = str(sid).strip()
                    if sid_str:
                        slide_ids.add(sid_str)
            return slide_ids

        if labels_path.suffix.lower() == ".csv":
            with open(labels_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    sid = (row.get("slide_id") or "").strip()
                    if not sid:
                        slide_file = (row.get("slide_file") or "").strip()
                        sid = slide_file.replace(".svs", "").replace(".SVS", "")
                    if sid:
                        slide_ids.add(sid)
    except Exception:
        return set()

    return slide_ids


def filter_project_candidate_slide_ids(
    candidate_slide_ids: set[str],
    allowed_slide_ids: set[str],
) -> list[str]:
    """Filter candidate embedding IDs using authoritative project membership."""
    cleaned_candidates = sorted({str(s).strip() for s in candidate_slide_ids if str(s).strip()})
    cleaned_allowed = {str(s).strip() for s in allowed_slide_ids if str(s).strip()}

    if not cleaned_candidates or not cleaned_allowed:
        return []

    filtered: set[str] = {sid for sid in cleaned_candidates if sid in cleaned_allowed}

    candidates_by_base: dict[str, list[str]] = {}
    for sid in cleaned_candidates:
        base = slide_id_base(sid)
        candidates_by_base.setdefault(base, []).append(sid)

    for allowed_sid in sorted(cleaned_allowed):
        base = slide_id_base(allowed_sid)
        if not base:
            continue
        unresolved = [sid for sid in candidates_by_base.get(base, []) if sid not in filtered]
        if len(unresolved) == 1:
            filtered.add(unresolved[0])

    return sorted(filtered)
