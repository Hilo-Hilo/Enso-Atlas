"""
PostgreSQL Database Module for Enso Atlas.

Provides:
- Connection pool management (asyncpg)
- Schema creation / migration
- Data population from existing flat files (labels.csv, .npy scans, SVS dimensions)
- Query helpers for slides, patients, metadata, and analysis results

Design: raw SQL with asyncpg for maximum performance. No ORM overhead.
"""

from __future__ import annotations

import asyncio
import csv
import hashlib
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# asyncpg is imported lazily to allow import even when not installed
_pool = None  # asyncpg.Pool | None

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://enso:enso_atlas_2024@atlas-db:5432/enso_atlas",
)

# ---------------------------------------------------------------------------
# Schema DDL
# ---------------------------------------------------------------------------

SCHEMA_SQL = """
-- Patients table
CREATE TABLE IF NOT EXISTS patients (
    patient_id   TEXT PRIMARY KEY,
    age          INTEGER,
    sex          TEXT,
    stage        TEXT,
    grade        TEXT,
    prior_lines  INTEGER,
    histology    TEXT,
    treatment_response TEXT,
    diagnosis    TEXT,
    vital_status TEXT,
    created_at   TIMESTAMPTZ DEFAULT now(),
    updated_at   TIMESTAMPTZ DEFAULT now()
);

-- Slides table (the big performance win: dimensions cached here)
CREATE TABLE IF NOT EXISTS slides (
    slide_id              TEXT PRIMARY KEY,
    patient_id            TEXT REFERENCES patients(patient_id) ON DELETE SET NULL,
    filename              TEXT,
    width                 INTEGER DEFAULT 0,
    height                INTEGER DEFAULT 0,
    mpp                   DOUBLE PRECISION,
    magnification         TEXT DEFAULT '40x',
    num_patches           INTEGER,
    has_embeddings        BOOLEAN DEFAULT FALSE,
    has_level0_embeddings BOOLEAN DEFAULT FALSE,
    label                 TEXT,
    embedding_date        TIMESTAMPTZ,
    file_path             TEXT,
    file_size_bytes       BIGINT,
    created_at            TIMESTAMPTZ DEFAULT now(),
    updated_at            TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_slides_patient ON slides(patient_id);
CREATE INDEX IF NOT EXISTS idx_slides_label   ON slides(label);

-- Extensible key/value metadata (tags, stars, groups, notes)
CREATE TABLE IF NOT EXISTS slide_metadata (
    id        BIGSERIAL PRIMARY KEY,
    slide_id  TEXT NOT NULL REFERENCES slides(slide_id) ON DELETE CASCADE,
    key       TEXT NOT NULL,
    value     TEXT,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now(),
    UNIQUE(slide_id, key, value)
);

CREATE INDEX IF NOT EXISTS idx_slide_metadata_slide ON slide_metadata(slide_id);
CREATE INDEX IF NOT EXISTS idx_slide_metadata_key   ON slide_metadata(key);

-- Analysis results from MIL models
CREATE TABLE IF NOT EXISTS analysis_results (
    id             BIGSERIAL PRIMARY KEY,
    slide_id       TEXT NOT NULL REFERENCES slides(slide_id) ON DELETE CASCADE,
    model_id       TEXT,
    score          DOUBLE PRECISION,
    label          TEXT,
    confidence     DOUBLE PRECISION,
    threshold      DOUBLE PRECISION,
    attention_hash TEXT,
    created_at     TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_analysis_results_slide ON analysis_results(slide_id);

-- Embedding task tracking
CREATE TABLE IF NOT EXISTS embedding_tasks (
    task_id       TEXT PRIMARY KEY,
    slide_id      TEXT REFERENCES slides(slide_id) ON DELETE CASCADE,
    level         INTEGER DEFAULT 0,
    status        TEXT DEFAULT 'pending',   -- pending, running, completed, failed
    progress      DOUBLE PRECISION DEFAULT 0,
    created_at    TIMESTAMPTZ DEFAULT now(),
    completed_at  TIMESTAMPTZ,
    error         TEXT
);

CREATE INDEX IF NOT EXISTS idx_embedding_tasks_slide ON embedding_tasks(slide_id);
CREATE INDEX IF NOT EXISTS idx_embedding_tasks_status ON embedding_tasks(status);

-- Projects table (config-driven multi-cancer support)
CREATE TABLE IF NOT EXISTS projects (
    id              TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    cancer_type     TEXT,
    prediction_target TEXT,
    config_json     JSONB,
    created_at      TIMESTAMPTZ DEFAULT now(),
    updated_at      TIMESTAMPTZ DEFAULT now()
);

-- Project-to-model assignments (many-to-many)
CREATE TABLE IF NOT EXISTS project_models (
    project_id  TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    model_id    TEXT NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (project_id, model_id)
);

-- Project-to-slide assignments (many-to-many)
CREATE TABLE IF NOT EXISTS project_slides (
    project_id  TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    slide_id    TEXT NOT NULL REFERENCES slides(slide_id) ON DELETE CASCADE,
    created_at  TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (project_id, slide_id)
);

CREATE INDEX IF NOT EXISTS idx_project_slides_project ON project_slides(project_id);
CREATE INDEX IF NOT EXISTS idx_project_slides_slide ON project_slides(slide_id);
CREATE INDEX IF NOT EXISTS idx_project_models_project ON project_models(project_id);

-- Track schema version for future migrations
CREATE TABLE IF NOT EXISTS schema_version (
    version    INTEGER PRIMARY KEY,
    applied_at TIMESTAMPTZ DEFAULT now()
);

-- Insert initial version if not present
INSERT INTO schema_version (version) VALUES (1) ON CONFLICT DO NOTHING;
"""

# ---------------------------------------------------------------------------
# Connection pool management
# ---------------------------------------------------------------------------


async def get_pool():
    """Get or create the asyncpg connection pool."""
    global _pool
    if _pool is None:
        import asyncpg
        # Retry connection a few times (DB might still be starting)
        for attempt in range(10):
            try:
                _pool = await asyncpg.create_pool(
                    DATABASE_URL,
                    min_size=2,
                    max_size=10,
                    command_timeout=60,
                )
                logger.info("Database connection pool created")
                break
            except Exception as e:
                if attempt < 9:
                    logger.warning(f"DB connection attempt {attempt+1}/10 failed: {e}, retrying in 2s...")
                    await asyncio.sleep(2)
                else:
                    raise
    return _pool


async def close_pool():
    """Close the connection pool."""
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
        logger.info("Database connection pool closed")


async def init_schema():
    """Create tables if they don't exist."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(SCHEMA_SQL)
    logger.info("Database schema initialized")
    # Run migrations for new columns
    await _migrate_project_columns()
    await _migrate_project_scoped_tables()
    await _migrate_display_name()


async def _migrate_project_columns():
    """Add project_id column to slides table if not present (v2 migration)."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        col_exists = await conn.fetchval(
            """
            SELECT COUNT(*) FROM information_schema.columns
            WHERE table_name = 'slides' AND column_name = 'project_id'
            """
        )
        if not col_exists:
            await conn.execute(
                "ALTER TABLE slides ADD COLUMN project_id TEXT DEFAULT 'ovarian-platinum'"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_slides_project ON slides(project_id)"
            )
            logger.info("Added project_id column to slides table")
        # Mark migration
        await conn.execute(
            "INSERT INTO schema_version (version) VALUES (2) ON CONFLICT DO NOTHING"
        )


async def _migrate_project_scoped_tables():
    """v3 migration: create project_models and project_slides junction tables
    and seed initial data for the ovarian-platinum project.

    Safe to run multiple times (idempotent).
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        # Check if v3 migration already applied
        v3_done = await conn.fetchval(
            "SELECT COUNT(*) FROM schema_version WHERE version = 3"
        )
        if v3_done:
            return

        # Tables are already created in SCHEMA_SQL, but ensure they exist
        # in case this migration runs against an older schema.
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS project_models (
                project_id  TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                model_id    TEXT NOT NULL,
                created_at  TIMESTAMPTZ DEFAULT now(),
                PRIMARY KEY (project_id, model_id)
            )
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS project_slides (
                project_id  TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                slide_id    TEXT NOT NULL REFERENCES slides(slide_id) ON DELETE CASCADE,
                created_at  TIMESTAMPTZ DEFAULT now(),
                PRIMARY KEY (project_id, slide_id)
            )
        """)
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_project_slides_project ON project_slides(project_id)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_project_slides_slide ON project_slides(slide_id)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_project_models_project ON project_models(project_id)"
        )

        # Seed: assign ALL existing slides to ovarian-platinum
        project_exists = await conn.fetchval(
            "SELECT COUNT(*) FROM projects WHERE id = 'ovarian-platinum'"
        )
        if project_exists:
            await conn.execute("""
                INSERT INTO project_slides (project_id, slide_id)
                SELECT 'ovarian-platinum', slide_id FROM slides
                ON CONFLICT DO NOTHING
            """)
            logger.info("Seeded project_slides with all existing slides for ovarian-platinum")

            # Seed: assign all 5 models to ovarian-platinum
            model_ids = [
                "platinum_sensitivity",
                "tumor_grade",
                "survival_5y",
                "survival_3y",
                "survival_1y",
            ]
            await conn.executemany(
                """
                INSERT INTO project_models (project_id, model_id)
                VALUES ('ovarian-platinum', $1)
                ON CONFLICT DO NOTHING
                """,
                [(mid,) for mid in model_ids],
            )
            logger.info("Seeded project_models with 5 models for ovarian-platinum")

        # Mark migration complete
        await conn.execute(
            "INSERT INTO schema_version (version) VALUES (3) ON CONFLICT DO NOTHING"
        )
        logger.info("v3 migration complete: project_models and project_slides tables ready")


async def _migrate_display_name():
    """v4 migration: add display_name column to slides table."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        v4_done = await conn.fetchval(
            "SELECT COUNT(*) FROM schema_version WHERE version = 4"
        )
        if v4_done:
            return
        col_exists = await conn.fetchval(
            """
            SELECT COUNT(*) FROM information_schema.columns
            WHERE table_name = 'slides' AND column_name = 'display_name'
            """
        )
        if not col_exists:
            await conn.execute(
                "ALTER TABLE slides ADD COLUMN display_name TEXT"
            )
            logger.info("Added display_name column to slides table")
        await conn.execute(
            "INSERT INTO schema_version (version) VALUES (4) ON CONFLICT DO NOTHING"
        )
        logger.info("v4 migration complete: display_name column added")


async def update_slide_display_name(slide_id: str, display_name: str | None) -> bool:
    """Update the display_name for a slide. Returns True if the slide exists."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        result = await conn.execute(
            """
            UPDATE slides SET display_name = $2, updated_at = now()
            WHERE slide_id = $1
            """,
            slide_id, display_name,
        )
        return result != "UPDATE 0"


async def get_slide_embedding_status(slide_id: str) -> dict:
    """Get embedding and analysis status for a slide.

    Returns which foundation models have been used to embed this slide,
    and which classification models have cached results.
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        slide = await conn.fetchrow(
            """
            SELECT has_embeddings, has_level0_embeddings, num_patches, embedding_date
            FROM slides WHERE slide_id = $1
            """,
            slide_id,
        )
        if not slide:
            return {"error": "slide not found"}

        # Get distinct model_ids that have cached results
        cached_models = await conn.fetch(
            """
            SELECT DISTINCT model_id FROM analysis_results
            WHERE slide_id = $1
            """,
            slide_id,
        )

        # Check MedSigLIP embedding status by looking for the .npy file
        # (we track this via has_embeddings flag currently)
        return {
            "slide_id": slide_id,
            "has_level0_embeddings": slide["has_level0_embeddings"],
            "has_level1_embeddings": slide["has_embeddings"],
            "num_patches": slide["num_patches"],
            "embedding_date": slide["embedding_date"].isoformat() if slide["embedding_date"] else None,
            "cached_model_ids": [r["model_id"] for r in cached_models],
        }


async def populate_projects_from_registry(registry) -> None:
    """
    Populate the projects table from a ProjectRegistry instance.

    Called during startup to sync YAML config → database.
    """
    pool = await get_pool()
    import json as _json
    async with pool.acquire() as conn:
        for pid, proj in registry.list_projects().items():
            config_json = _json.dumps(proj.to_dict())
            await conn.execute(
                """
                INSERT INTO projects (id, name, cancer_type, prediction_target, config_json)
                VALUES ($1, $2, $3, $4, $5::jsonb)
                ON CONFLICT (id) DO UPDATE SET
                    name = EXCLUDED.name,
                    cancer_type = EXCLUDED.cancer_type,
                    prediction_target = EXCLUDED.prediction_target,
                    config_json = EXCLUDED.config_json,
                    updated_at = now()
                """,
                pid, proj.name, proj.cancer_type, proj.prediction_target, config_json,
            )
    logger.info(f"Synced {len(registry.list_projects())} project(s) to database")


# ---------------------------------------------------------------------------
# Population from flat files
# ---------------------------------------------------------------------------


async def populate_from_flat_files(
    data_root: Path,
    embeddings_dir: Path,
    slide_dirs: List[Path] | None = None,
):
    """
    One-time population of the database from existing flat-file data.

    Steps:
    1. Parse labels.csv files → patients + slides (labels)
    2. Scan .npy files → num_patches, has_embeddings
    3. Read SVS files with openslide → dimensions, mpp (the slow part we only do once!)
    4. Import slide_metadata.json → slide_metadata table

    This function is idempotent — it uses INSERT ... ON CONFLICT DO UPDATE.
    """
    pool = await get_pool()
    t0 = time.time()

    # ------ Step 1: Parse labels.csv → patients + slides ------
    label_files = [
        data_root / "labels.csv",
        data_root / "tcga_full" / "labels.csv",
    ]

    patients_data: Dict[str, Dict[str, Any]] = {}
    slides_data: Dict[str, Dict[str, Any]] = {}

    for labels_path in label_files:
        if not labels_path.exists():
            continue
        logger.info(f"Parsing labels from {labels_path}")
        with open(labels_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Format 1: data/labels.csv (patient_id, slide_file, treatment_response, age, sex, ...)
                if "slide_file" in row:
                    patient_id = row.get("patient_id", "")
                    slide_file = row.get("slide_file", "")
                    slide_id = slide_file.replace(".svs", "").replace(".SVS", "")
                    response = row.get("treatment_response", "")
                    label = "1" if response == "responder" else "0" if response == "non-responder" else ""

                    if patient_id:
                        patients_data[patient_id] = {
                            "patient_id": patient_id,
                            "age": _safe_int(row.get("age")),
                            "sex": row.get("sex") or None,
                            "stage": row.get("stage") or None,
                            "grade": row.get("grade") or None,
                            "prior_lines": _safe_int(row.get("prior_treatments")),
                            "histology": row.get("histology") or None,
                            "treatment_response": response or None,
                            "diagnosis": row.get("diagnosis") or None,
                            "vital_status": row.get("vital_status") or None,
                        }

                    if slide_id:
                        slides_data[slide_id] = {
                            "slide_id": slide_id,
                            "patient_id": patient_id or None,
                            "filename": slide_file,
                            "label": label,
                        }

                # Format 2: tcga_full/labels.csv (slide_id, patient_id, label, platinum_status)
                elif "slide_id" in row:
                    slide_id = row["slide_id"]
                    patient_id = row.get("patient_id", "")
                    label = row.get("label", "")
                    platinum = row.get("platinum_status", "")

                    # Normalize label
                    if not label and platinum:
                        label = "1" if platinum == "sensitive" else "0" if platinum == "resistant" else ""

                    if patient_id and patient_id not in patients_data:
                        patients_data[patient_id] = {
                            "patient_id": patient_id,
                            "treatment_response": "responder" if label == "1" else "non-responder" if label == "0" else None,
                        }

                    if slide_id:
                        # Don't overwrite if we already have richer data
                        if slide_id not in slides_data:
                            slides_data[slide_id] = {
                                "slide_id": slide_id,
                                "patient_id": patient_id or None,
                                "filename": f"{slide_id}.svs",
                                "label": label,
                            }

    logger.info(f"Parsed {len(patients_data)} patients, {len(slides_data)} slides from CSV")

    # ------ Step 2: Scan .npy embeddings ------
    embedding_slides = set()
    level0_slides = set()

    if embeddings_dir.exists():
        for f in sorted(embeddings_dir.glob("*.npy")):
            if f.name.endswith("_coords.npy"):
                continue
            sid = f.stem
            embedding_slides.add(sid)
            try:
                import numpy as np
                emb = np.load(f)
                num_patches = len(emb)
            except Exception:
                num_patches = None

            if sid not in slides_data:
                slides_data[sid] = {"slide_id": sid}
            slides_data[sid]["has_embeddings"] = True
            slides_data[sid]["num_patches"] = num_patches

    # Check level0 subdirectory
    level0_dir = embeddings_dir / "level0" if embeddings_dir.name != "level0" else embeddings_dir
    if level0_dir.exists():
        for f in level0_dir.glob("*.npy"):
            if not f.name.endswith("_coords.npy"):
                sid = f.stem
                level0_slides.add(sid)
                if sid in slides_data:
                    slides_data[sid]["has_level0_embeddings"] = True

    # If embeddings_dir IS the level0 dir, all embedding slides are level0
    if embeddings_dir.name == "level0":
        for sid in embedding_slides:
            if sid in slides_data:
                slides_data[sid]["has_level0_embeddings"] = True

    logger.info(f"Found {len(embedding_slides)} slides with embeddings, {len(level0_slides)} with level0")

    # ------ Step 3: Read SVS dimensions (the slow part — only on first run) ------
    # Build candidate slide directories
    if slide_dirs is None:
        slide_dirs = [
            data_root / "slides",
            data_root / "tcga_full" / "slides",
            data_root / "ovarian_bev" / "slides",
            data_root / "demo" / "slides",
        ]

    # Check which slides already have dimensions in DB
    async with pool.acquire() as conn:
        existing = await conn.fetch(
            "SELECT slide_id, width FROM slides WHERE width > 0"
        )
    existing_dims = {r["slide_id"] for r in existing}

    slides_needing_dims = {
        sid for sid in slides_data if sid not in existing_dims
    }

    if slides_needing_dims:
        logger.info(f"Reading dimensions from SVS files for {len(slides_needing_dims)} slides...")
        try:
            import openslide
            exts = {'.svs', '.tiff', '.tif', '.ndpi', '.mrxs', '.vms', '.scn'}
            dims_read = 0
            for slide_dir in slide_dirs:
                if not slide_dir.exists():
                    continue
                for fp in sorted(slide_dir.iterdir()):
                    if not fp.is_file() or fp.suffix.lower() not in exts:
                        continue
                    sid = fp.stem
                    if sid not in slides_needing_dims:
                        continue
                    try:
                        with openslide.OpenSlide(str(fp)) as slide:
                            w, h = slide.dimensions
                            mpp_x = slide.properties.get(openslide.PROPERTY_NAME_MPP_X)
                            mpp = float(mpp_x) if mpp_x else None
                        file_size = fp.stat().st_size

                        slides_data.setdefault(sid, {"slide_id": sid})
                        slides_data[sid]["width"] = w
                        slides_data[sid]["height"] = h
                        slides_data[sid]["mpp"] = mpp
                        slides_data[sid]["file_path"] = str(fp)
                        slides_data[sid]["file_size_bytes"] = file_size
                        dims_read += 1
                        if dims_read % 20 == 0:
                            logger.info(f"  Read dimensions for {dims_read} slides...")
                    except Exception as e:
                        logger.warning(f"Could not read SVS {fp.name}: {e}")

            logger.info(f"Read dimensions from {dims_read} SVS files")
        except ImportError:
            logger.warning("openslide not available, skipping dimension reading")

    # ------ Step 4: Write to database ------
    async with pool.acquire() as conn:
        # Insert patients
        if patients_data:
            await conn.executemany(
                """
                INSERT INTO patients (patient_id, age, sex, stage, grade, prior_lines, histology,
                                      treatment_response, diagnosis, vital_status)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                ON CONFLICT (patient_id) DO UPDATE SET
                    age = COALESCE(EXCLUDED.age, patients.age),
                    sex = COALESCE(EXCLUDED.sex, patients.sex),
                    stage = COALESCE(EXCLUDED.stage, patients.stage),
                    grade = COALESCE(EXCLUDED.grade, patients.grade),
                    prior_lines = COALESCE(EXCLUDED.prior_lines, patients.prior_lines),
                    histology = COALESCE(EXCLUDED.histology, patients.histology),
                    treatment_response = COALESCE(EXCLUDED.treatment_response, patients.treatment_response),
                    diagnosis = COALESCE(EXCLUDED.diagnosis, patients.diagnosis),
                    vital_status = COALESCE(EXCLUDED.vital_status, patients.vital_status),
                    updated_at = now()
                """,
                [
                    (
                        p["patient_id"],
                        p.get("age"),
                        p.get("sex"),
                        p.get("stage"),
                        p.get("grade"),
                        p.get("prior_lines"),
                        p.get("histology"),
                        p.get("treatment_response"),
                        p.get("diagnosis"),
                        p.get("vital_status"),
                    )
                    for p in patients_data.values()
                ],
            )
            logger.info(f"Upserted {len(patients_data)} patients")

        # Insert slides
        if slides_data:
            await conn.executemany(
                """
                INSERT INTO slides (slide_id, patient_id, filename, width, height, mpp,
                                    num_patches, has_embeddings, has_level0_embeddings,
                                    label, file_path, file_size_bytes)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                ON CONFLICT (slide_id) DO UPDATE SET
                    patient_id = COALESCE(EXCLUDED.patient_id, slides.patient_id),
                    filename = COALESCE(EXCLUDED.filename, slides.filename),
                    width = CASE WHEN EXCLUDED.width > 0 THEN EXCLUDED.width ELSE slides.width END,
                    height = CASE WHEN EXCLUDED.height > 0 THEN EXCLUDED.height ELSE slides.height END,
                    mpp = COALESCE(EXCLUDED.mpp, slides.mpp),
                    num_patches = COALESCE(EXCLUDED.num_patches, slides.num_patches),
                    has_embeddings = EXCLUDED.has_embeddings OR slides.has_embeddings,
                    has_level0_embeddings = EXCLUDED.has_level0_embeddings OR slides.has_level0_embeddings,
                    label = COALESCE(EXCLUDED.label, slides.label),
                    file_path = COALESCE(EXCLUDED.file_path, slides.file_path),
                    file_size_bytes = COALESCE(EXCLUDED.file_size_bytes, slides.file_size_bytes),
                    updated_at = now()
                """,
                [
                    (
                        s["slide_id"],
                        s.get("patient_id"),
                        s.get("filename"),
                        s.get("width", 0),
                        s.get("height", 0),
                        s.get("mpp"),
                        s.get("num_patches"),
                        s.get("has_embeddings", False),
                        s.get("has_level0_embeddings", False),
                        s.get("label"),
                        s.get("file_path"),
                        s.get("file_size_bytes"),
                    )
                    for s in slides_data.values()
                ],
            )
            logger.info(f"Upserted {len(slides_data)} slides")

    # ------ Step 5: Import slide_metadata.json ------
    metadata_json_path = data_root / "slide_metadata.json"
    if metadata_json_path.exists():
        try:
            import json
            with open(metadata_json_path) as f:
                meta_store = json.load(f)
            slide_metas = meta_store.get("slides", {})
            groups = meta_store.get("groups", {})

            rows = []
            for sid, meta in slide_metas.items():
                # Tags
                for tag in meta.get("tags", []):
                    rows.append((sid, "tag", tag))
                # Groups
                for gid in meta.get("groups", []):
                    group_name = groups.get(gid, {}).get("name", gid)
                    rows.append((sid, "group", group_name))
                # Starred
                if meta.get("starred"):
                    rows.append((sid, "starred", "true"))
                # Notes
                if meta.get("notes"):
                    rows.append((sid, "notes", meta["notes"]))

            if rows:
                async with pool.acquire() as conn:
                    await conn.executemany(
                        """
                        INSERT INTO slide_metadata (slide_id, key, value)
                        VALUES ($1, $2, $3)
                        ON CONFLICT (slide_id, key, value) DO NOTHING
                        """,
                        rows,
                    )
                logger.info(f"Imported {len(rows)} metadata entries from slide_metadata.json")
        except Exception as e:
            logger.warning(f"Failed to import slide_metadata.json: {e}")

    elapsed = time.time() - t0
    logger.info(f"Database population completed in {elapsed:.1f}s")


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------


async def get_all_slides() -> List[Dict[str, Any]]:
    """
    Get all slides with patient context — the fast replacement for the old
    /api/slides endpoint. Should return in <100ms instead of 30-60s.
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT
                s.slide_id,
                s.patient_id,
                s.filename,
                s.display_name,
                s.width,
                s.height,
                s.mpp,
                s.magnification,
                s.num_patches,
                s.has_embeddings,
                s.has_level0_embeddings,
                s.label,
                s.file_path,
                s.file_size_bytes,
                p.age,
                p.sex,
                p.stage,
                p.grade,
                p.prior_lines,
                p.histology,
                p.treatment_response
            FROM slides s
            LEFT JOIN patients p ON s.patient_id = p.patient_id
            ORDER BY s.slide_id
            """
        )
    return [dict(r) for r in rows]


async def get_slide(slide_id: str) -> Optional[Dict[str, Any]]:
    """Get a single slide with patient context."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT
                s.*, p.age, p.sex, p.stage, p.grade,
                p.prior_lines, p.histology, p.treatment_response
            FROM slides s
            LEFT JOIN patients p ON s.patient_id = p.patient_id
            WHERE s.slide_id = $1
            """,
            slide_id,
        )
    return dict(row) if row else None


async def get_slide_labels() -> Dict[str, str]:
    """Get slide_id -> label mapping for all slides. Used by FAISS index etc."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT slide_id, label FROM slides WHERE label IS NOT NULL AND label != ''"
        )
    result = {}
    for r in rows:
        label = r["label"]
        # Normalize: "1" -> "responder", "0" -> "non-responder"
        if label == "1":
            label = "responder"
        elif label == "0":
            label = "non-responder"
        result[r["slide_id"]] = label
    return result


async def get_slide_metadata_kv(slide_id: str) -> Dict[str, List[str]]:
    """Get all key-value metadata for a slide, grouped by key."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT key, value FROM slide_metadata WHERE slide_id = $1",
            slide_id,
        )
    result: Dict[str, List[str]] = {}
    for r in rows:
        result.setdefault(r["key"], []).append(r["value"])
    return result


async def set_slide_metadata(slide_id: str, key: str, value: str):
    """Set a metadata key-value pair. Upserts on (slide_id, key, value)."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO slide_metadata (slide_id, key, value)
            VALUES ($1, $2, $3)
            ON CONFLICT (slide_id, key, value) DO UPDATE SET updated_at = now()
            """,
            slide_id, key, value,
        )


async def delete_slide_metadata(slide_id: str, key: str, value: Optional[str] = None):
    """Delete metadata. If value is None, delete all entries for that key."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        if value is not None:
            await conn.execute(
                "DELETE FROM slide_metadata WHERE slide_id = $1 AND key = $2 AND value = $3",
                slide_id, key, value,
            )
        else:
            await conn.execute(
                "DELETE FROM slide_metadata WHERE slide_id = $1 AND key = $2",
                slide_id, key,
            )


async def toggle_star(slide_id: str) -> bool:
    """Toggle starred status. Returns new status."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        existing = await conn.fetchval(
            "SELECT value FROM slide_metadata WHERE slide_id = $1 AND key = 'starred'",
            slide_id,
        )
        if existing:
            await conn.execute(
                "DELETE FROM slide_metadata WHERE slide_id = $1 AND key = 'starred'",
                slide_id,
            )
            return False
        else:
            await conn.execute(
                "INSERT INTO slide_metadata (slide_id, key, value) VALUES ($1, 'starred', 'true') ON CONFLICT DO NOTHING",
                slide_id,
            )
            return True


async def save_analysis_result(
    slide_id: str,
    model_id: str,
    score: float,
    label: str,
    confidence: float,
    threshold: float | None = None,
    attention_hash: str | None = None,
):
    """Save an analysis result to the database."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO analysis_results (slide_id, model_id, score, label, confidence, threshold, attention_hash)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            """,
            slide_id, model_id, score, label, confidence, threshold, attention_hash,
        )


async def get_cached_results(
    slide_id: str,
    model_id: str | None = None,
) -> list[dict]:
    """Fetch cached analysis results for a slide (optionally filtered by model).

    Returns a list of dicts with keys: model_id, score, label, confidence,
    threshold, attention_hash, created_at.
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        if model_id:
            rows = await conn.fetch(
                """
                SELECT model_id, score, label, confidence, threshold, attention_hash, created_at
                FROM analysis_results
                WHERE slide_id = $1 AND model_id = $2
                ORDER BY created_at DESC
                LIMIT 1
                """,
                slide_id, model_id,
            )
        else:
            # Get latest result per model using DISTINCT ON
            rows = await conn.fetch(
                """
                SELECT DISTINCT ON (model_id)
                    model_id, score, label, confidence, threshold, attention_hash, created_at
                FROM analysis_results
                WHERE slide_id = $1
                ORDER BY model_id, created_at DESC
                """,
                slide_id,
            )
    return [dict(r) for r in rows]


async def get_all_cached_results(slide_id: str) -> list[dict]:
    """Return all cached analysis results for a slide (latest per model)."""
    return await get_cached_results(slide_id)


async def update_slide_embeddings(slide_id: str, num_patches: int, level: int = 1):
    """Update embedding status for a slide after embedding completes."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        if level == 0:
            await conn.execute(
                """
                UPDATE slides SET has_level0_embeddings = TRUE, num_patches = $2,
                       embedding_date = now(), updated_at = now()
                WHERE slide_id = $1
                """,
                slide_id, num_patches,
            )
        else:
            await conn.execute(
                """
                UPDATE slides SET has_embeddings = TRUE, num_patches = $2,
                       embedding_date = now(), updated_at = now()
                WHERE slide_id = $1
                """,
                slide_id, num_patches,
            )


async def is_populated() -> bool:
    """Check if the database has been populated (has any slides)."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        count = await conn.fetchval("SELECT COUNT(*) FROM slides")
    return count > 0


# ---------------------------------------------------------------------------
# Project-scoped slide and model queries
# ---------------------------------------------------------------------------


async def get_project_slides(project_id: str) -> List[str]:
    """Return slide_ids assigned to a project."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT slide_id FROM project_slides WHERE project_id = $1 ORDER BY slide_id",
            project_id,
        )
    return [r["slide_id"] for r in rows]


async def assign_slides_to_project(project_id: str, slide_ids: List[str]) -> int:
    """Assign slides to project. Returns count added."""
    if not slide_ids:
        return 0
    pool = await get_pool()
    async with pool.acquire() as conn:
        # Use a single statement with unnest for efficiency
        result = await conn.execute(
            """
            INSERT INTO project_slides (project_id, slide_id)
            SELECT $1, unnest($2::text[])
            ON CONFLICT DO NOTHING
            """,
            project_id, slide_ids,
        )
        # result is like "INSERT 0 N"
        count = int(result.split()[-1])
    return count


async def unassign_slides_from_project(project_id: str, slide_ids: List[str]) -> int:
    """Remove slide assignments. Returns count removed."""
    if not slide_ids:
        return 0
    pool = await get_pool()
    async with pool.acquire() as conn:
        result = await conn.execute(
            """
            DELETE FROM project_slides
            WHERE project_id = $1 AND slide_id = ANY($2::text[])
            """,
            project_id, slide_ids,
        )
        count = int(result.split()[-1])
    return count


async def get_project_models(project_id: str) -> List[str]:
    """Return model_ids assigned to a project."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT model_id FROM project_models WHERE project_id = $1 ORDER BY model_id",
            project_id,
        )
    return [r["model_id"] for r in rows]


async def assign_models_to_project(project_id: str, model_ids: List[str]) -> int:
    """Assign models to project. Returns count added."""
    if not model_ids:
        return 0
    pool = await get_pool()
    async with pool.acquire() as conn:
        result = await conn.execute(
            """
            INSERT INTO project_models (project_id, model_id)
            SELECT $1, unnest($2::text[])
            ON CONFLICT DO NOTHING
            """,
            project_id, model_ids,
        )
        count = int(result.split()[-1])
    return count


async def unassign_models_from_project(project_id: str, model_ids: List[str]) -> int:
    """Remove model assignments. Returns count removed."""
    if not model_ids:
        return 0
    pool = await get_pool()
    async with pool.acquire() as conn:
        result = await conn.execute(
            """
            DELETE FROM project_models
            WHERE project_id = $1 AND model_id = ANY($2::text[])
            """,
            project_id, model_ids,
        )
        count = int(result.split()[-1])
    return count


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _safe_int(val) -> Optional[int]:
    """Safely convert a value to int, returning None on failure."""
    if val is None or val == "":
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None
