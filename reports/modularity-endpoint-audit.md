# Modularity Endpoint Audit (Project-Scoped API Behavior)

Date: 2026-02-22  
Repo: `/Users/hansonwen/clawd/med-gemma-hackathon`

## Scope
Read-only code audit focused on:
1. Endpoints that accept `project_id` and whether they enforce project-scoped slide/model access.
2. Paths where `slide_id` / `model_id` can bypass project boundaries.
3. Global fallback behavior inside project-scoped flows.

---

## Endpoint review summary (`project_id` consumers)

### Stronger enforcement observed
- `GET /api/slides` and `GET /api/slides/search`: project validity + scoped slide filtering in list flow (`src/enso_atlas/api/main.py:1534-1836`).
- `GET/HEAD /api/slides/{slide_id}/dzi`, `GET /api/slides/{slide_id}/dzi_files/...`, `GET /api/slides/{slide_id}/thumbnail`: `_require_project(project_id)` + `resolve_slide_path(..., project_id=...)` (`src/enso_atlas/api/main.py:4917-5039`).
- `GET /api/models`: model allowlist filtered by project (`src/enso_atlas/api/main.py:5504-5523`).

### Partial / bypass-prone enforcement
- `POST /api/analyze`, `POST /api/report`, `POST /api/report/async`, `GET /api/similar`, `POST /api/semantic-search`, `POST /api/analyze-batch/async`, `POST /api/analyze-multi`, `GET /api/heatmap/{slide_id}`, `GET /api/heatmap/{slide_id}/{model_id}`.
- Common pattern: project-scoped path resolution is used, but authoritative membership check for requested `slide_id` is missing (details in Findings #2/#4).

---

## Findings

## 1) **CRITICAL** — Project slide route fallback leaks unassigned or all slides
**Evidence**
- `src/enso_atlas/api/project_routes.py:429-457`
  - Fallback query includes `WHERE s.project_id = $1 OR s.project_id IS NULL` (`:444`), leaking globally/unassigned slides.
  - If legacy column is absent, fallback returns all slides (`:450-456`).

**Impact**
`GET /api/projects/{project_id}/slides` can return slides outside the requested project boundary.

**Recommendation**
- Fail closed when `project_slides` has no rows (return empty list unless explicit migration/compat flag is set).
- Remove `OR s.project_id IS NULL`.
- Remove/guard the `SELECT ... FROM slides ORDER BY ...` all-slides fallback.

---

## 2) **HIGH** — Project-scoped level-0 embedding resolution falls back to global dirs
**Evidence**
- `src/enso_atlas/api/main.py:878-916` (`_candidate_embedding_dirs_for_level`)
  - For `level == 0`, global candidates are always added (`:904-907`) even when `project_id` is provided.
- Reachable in project-scoped endpoints:
  - `GET /api/heatmap/{slide_id}` (`src/enso_atlas/api/main.py:4107-4112`)
  - `POST /api/analyze-multi` (`src/enso_atlas/api/main.py:6350-6355`)
  - `GET /api/heatmap/{slide_id}/{model_id}` (`src/enso_atlas/api/main.py:6501-6506`)
  - Async batch worker path (`src/enso_atlas/api/main.py:2449-2454`)

**Impact**
A request with `project_id` can still resolve embeddings from global storage at level 0, violating project isolation.

**Recommendation**
- In `_candidate_embedding_dirs_for_level`, when `project_requested` is true, do **not** append global dirs.
- Add a post-resolution invariant check: resolved embedding path must be under the project’s embedding root.

---

## 3) **HIGH** — `analyze-multi` cache path bypasses project model allowlist when allowlist is empty
**Evidence**
- `src/enso_atlas/api/main.py:6296`, `6305-6310`
  - `effective_model_ids = sorted(allowed_model_ids)` can be `[]`.
  - `requested_models = set(effective_model_ids) if effective_model_ids else None` turns empty allowlist into `None`.
  - Filter only applies when `requested_models` is truthy (`if requested_models and ...`), so all cached models pass through.

**Impact**
For a project with no assigned models, cached predictions for **all** models on the slide can still be returned.

**Recommendation**
- Treat empty allowlist as explicit deny-all, not “no filter”.
- Use `requested_models = set(effective_model_ids) if effective_model_ids is not None else None` and filter on `requested_models is not None`.

---

## 4) **HIGH** — Project-aware endpoints do not verify requested `slide_id` is project-assigned
**Evidence (representative)**
- `POST /api/analyze`: only checks file existence in project embedding dir (`src/enso_atlas/api/main.py:2001-2010`).
- `POST /api/report`: same pattern (`src/enso_atlas/api/main.py:3092-3103`).
- `POST /api/report/async`: same preflight (`src/enso_atlas/api/main.py:3454-3463`).
- `GET /api/similar`: query slide existence check, but assignment filter is only for returned neighbors (`src/enso_atlas/api/main.py:4323-4327`, filter at `4349-4350`).
- `POST /api/semantic-search`: no project-membership check before serving (`src/enso_atlas/api/main.py:4408-4418`).
- `POST /api/analyze-batch/async`: accepts arbitrary `slide_ids` and background worker gates by emb-file existence, not project membership (`src/enso_atlas/api/main.py:2372-2388`, `2459-2466`).

**Impact**
If project embeddings become contaminated (shared path/symlink/manual copy), these endpoints can process or disclose unassigned slides under a project context.

**Recommendation**
- Add a single reusable guard (e.g., `_ensure_slide_in_project(slide_id, project_id)`) and call it for every project-scoped `slide_id` endpoint.
- Use DB `project_slides` as authority; avoid filesystem-presence as authorization.

---

## 5) **HIGH** — Annotation update/delete endpoints ignore `slide_id` path parameter
**Evidence**
- Endpoint layer passes only annotation ID:
  - `src/enso_atlas/api/main.py:5442-5451`, `5469-5473`
- DB layer mutates by annotation ID only:
  - `src/enso_atlas/api/database.py:435-455` (`WHERE id = $1`)
  - `src/enso_atlas/api/database.py:467-473` (`DELETE ... WHERE id = $1`)

**Impact**
Knowing an `annotation_id` allows cross-slide mutation/deletion regardless of `{slide_id}` in URL (IDOR-style boundary bypass; by extension can cross projects).

**Recommendation**
- Scope DB writes by both IDs: `WHERE id = $1 AND slide_id = $2`.
- Return 404 on mismatch; optionally add project-scoped check for annotation operations.

---

## 6) **MEDIUM** — Many slide-data endpoints are global-only (no project scoping), enabling direct boundary bypass
**Evidence (examples)**
- Patch image endpoint uses global embedding/WSI lookup: `src/enso_atlas/api/main.py:5113-5158`.
- Slide info endpoint uses global lookup: `src/enso_atlas/api/main.py:5251-5267`.
- Cached results and embedding status by bare `slide_id`: `src/enso_atlas/api/main.py:5304-5347`.
- Additional global slide ops: QC / patch classify / outlier / patch coords (`src/enso_atlas/api/main.py:4821-4836`, `6825-6853`, `6958-6973`, `7041-7051`).

**Impact**
A caller can query slide artifacts outside the currently selected project if they know or guess `slide_id`.

**Recommendation**
- Add `project_id` (or nest under `/api/projects/{project_id}/...`) and enforce membership guard consistently.
- If legacy global endpoints must remain, gate them behind explicit non-project mode.

---

## Prioritized remediation plan
1. **Immediate (P0):** Fix Findings #1, #2, #3, #5.
2. **Short-term (P1):** Implement `_ensure_slide_in_project` and apply to all project-aware slide endpoints (Finding #4).
3. **Medium-term (P2):** Normalize endpoint surface so slide operations are explicitly project-scoped or explicitly global-only with guardrails (Finding #6).
4. **Regression tests:** Add negative tests for cross-project `slide_id`/`annotation_id`/`model_id` access attempts across all affected endpoints.
