from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_SRC = REPO_ROOT / "frontend" / "src"

CLIENT_CALLSITE_FILES = [
    "frontend/src/lib/api.ts",
    "frontend/src/contexts/ProjectContext.tsx",
    "frontend/src/app/projects/page.tsx",
    "frontend/src/components/panels/AIAssistantPanel.tsx",
    "frontend/src/components/modals/SystemStatusModal.tsx",
    "frontend/src/app/page.tsx",
]


def _read(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def test_client_api_base_helper_guards_loopback_urls_for_remote_browsers():
    src = _read("frontend/src/lib/clientApiBase.ts")

    assert "LOOPBACK_HOSTS" in src
    for host in ("localhost", "127.0.0.1", "0.0.0.0", "::1"):
        assert f'"{host}"' in src

    assert "targetIsLoopback && !browserIsLoopback" in src
    assert 'return "";' in src


def test_client_callsites_use_api_base_helper_not_raw_env_url():
    for path in CLIENT_CALLSITE_FILES:
        src = _read(path)
        assert "getClientApiBaseUrl" in src, f"{path} should use getClientApiBaseUrl()"
        assert "process.env.NEXT_PUBLIC_API_URL" not in src, (
            f"{path} should not read NEXT_PUBLIC_API_URL directly in client code"
        )


def test_next_public_api_url_is_not_referenced_directly_in_client_files():
    offenders: list[str] = []

    for path in FRONTEND_SRC.rglob("*.ts*"):
        src = path.read_text(encoding="utf-8")
        if "process.env.NEXT_PUBLIC_API_URL" not in src:
            continue

        rel = str(path.relative_to(REPO_ROOT)).replace("\\", "/")
        if rel == "frontend/src/lib/clientApiBase.ts":
            continue
        if "/src/app/api/" in rel:
            continue

        offenders.append(rel)

    assert offenders == [], (
        "Direct NEXT_PUBLIC_API_URL reads in client files can leak loopback URLs "
        f"to browsers: {offenders}"
    )
