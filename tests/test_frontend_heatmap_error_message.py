from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def test_wsi_viewer_extracts_backend_heatmap_error_messages():
    src = _read("frontend/src/components/viewer/WSIViewer.tsx")

    assert "extractHeatmapErrorMessage" in src
    assert "setHeatmapErrorMessage(extractHeatmapErrorMessage(response.status, errorBody))" in src
    assert "Patch coordinates are missing for this slide." in src


def test_wsi_viewer_renders_heatmap_error_message_in_toolbar():
    src = _read("frontend/src/components/viewer/WSIViewer.tsx")

    assert "{heatmapError && heatmapErrorMessage && (" in src
    assert "text-2xs text-red-600" in src
