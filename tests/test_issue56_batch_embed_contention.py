from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def test_backend_analyze_multi_returns_server_busy_contract_when_batch_embed_active():
    src = _read("src/enso_atlas/api/main.py")

    assert "active_batch_embedding = _active_batch_embed_info()" in src
    assert '"error": "SERVER_BUSY"' in src
    assert 'headers={"Retry-After": "30"}' in src


def test_health_endpoint_exposes_active_batch_embed_and_uses_cached_cuda_probe():
    src = _read("src/enso_atlas/api/main.py")

    assert "_CUDA_AVAILABLE_AT_STARTUP" in src
    assert '"cuda_available": bool(_CUDA_AVAILABLE_AT_STARTUP)' in src
    assert '"active_batch_embedding": active_batch_embedding' in src


def test_frontend_parses_fastapi_detail_shape_and_shows_busy_message():
    api_src = _read("frontend/src/lib/api.ts")
    page_src = _read("frontend/src/app/page.tsx")

    assert "const detail = rawError?.detail;" in api_src
    assert "if (detail && typeof detail === \"object\")" in api_src
    assert "const isServerBusy =" in page_src
    assert "lowerError.includes(\"server_busy\")" in page_src
