"""Граница пакета проверяется тестом, а не дисциплиной (R2)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import dsx
from dsx.quality import find_boundary_findings

FORBIDDEN = frozenset({"projects"})
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src" / "dsx"


def _kinds(findings: list) -> list[str]:
    return [f.kind for f in findings]


def test_package_imports_from_clean_environment() -> None:
    assert dsx.__version__


def test_core_has_no_boundary_findings() -> None:
    findings = find_boundary_findings(SRC_ROOT, FORBIDDEN)
    assert findings == [], "\n".join(str(f) for f in findings)


def test_plain_import_violation_is_detected(tmp_path: Path) -> None:
    (tmp_path / "leaky.py").write_text("import projects.olist\n", encoding="utf-8")

    findings = find_boundary_findings(tmp_path, FORBIDDEN)

    assert _kinds(findings) == ["import"]
    assert findings[0].lineno == 1


def test_from_import_violation_is_detected(tmp_path: Path) -> None:
    (tmp_path / "leaky.py").write_text("from projects.olist import config\n", encoding="utf-8")

    assert _kinds(find_boundary_findings(tmp_path, FORBIDDEN)) == ["import"]


def test_relative_import_is_not_a_violation(tmp_path: Path) -> None:
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "inner.py").write_text("from . import projects\n", encoding="utf-8")

    assert find_boundary_findings(tmp_path, FORBIDDEN) == []


def test_similarly_named_module_is_not_a_violation(tmp_path: Path) -> None:
    (tmp_path / "ok.py").write_text("import projections\n", encoding="utf-8")

    assert find_boundary_findings(tmp_path, FORBIDDEN) == []


def test_unparseable_file_is_reported_not_raised(tmp_path: Path) -> None:
    (tmp_path / "broken.py").write_text("def f(:\n", encoding="utf-8")

    findings = find_boundary_findings(tmp_path, FORBIDDEN)

    assert _kinds(findings) == ["unparseable"]


def test_undecodable_file_is_reported_not_raised(tmp_path: Path) -> None:
    (tmp_path / "binary.py").write_bytes(b"\xff\xfe\x00import projects\n")

    assert _kinds(find_boundary_findings(tmp_path, FORBIDDEN)) == ["unparseable"]


def test_importlib_is_flagged_as_unverifiable(tmp_path: Path) -> None:
    (tmp_path / "sneaky.py").write_text(
        "import importlib\nm = importlib.import_module('projects.olist')\n", encoding="utf-8"
    )

    assert "dynamic" in _kinds(find_boundary_findings(tmp_path, FORBIDDEN))


def test_dunder_import_is_flagged_as_unverifiable(tmp_path: Path) -> None:
    (tmp_path / "sneaky.py").write_text("m = __import__('projects')\n", encoding="utf-8")

    assert "dynamic" in _kinds(find_boundary_findings(tmp_path, FORBIDDEN))


def test_exec_is_flagged_as_unverifiable(tmp_path: Path) -> None:
    (tmp_path / "sneaky.py").write_text("exec('import projects')\n", encoding="utf-8")

    assert "dynamic" in _kinds(find_boundary_findings(tmp_path, FORBIDDEN))


def test_linter_config_actually_bans_the_import(tmp_path: Path) -> None:
    """Правило должно быть включено в select, иначе banned-api ничего не проверяет."""
    probe = REPO_ROOT / "src" / "dsx" / "_boundary_probe.py"
    probe.write_text("import projects.olist\n\nUSED = projects.olist\n", encoding="utf-8")
    try:
        result = subprocess.run(
            ["ruff", "check", "--output-format=concise", str(probe)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    finally:
        probe.unlink(missing_ok=True)

    assert result.returncode != 0, "линтер пропустил запрещённый импорт"
    assert "TID251" in result.stdout, result.stdout
