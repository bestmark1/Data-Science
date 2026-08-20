"""Граница пакета проверяется тестом, а не дисциплиной (R2)."""

from __future__ import annotations

from pathlib import Path

import dsx
from dsx.quality import find_forbidden_imports

FORBIDDEN = frozenset({"projects"})
SRC_ROOT = Path(__file__).resolve().parent.parent / "src" / "dsx"


def test_package_imports_from_clean_environment() -> None:
    assert dsx.__version__


def test_core_does_not_import_projects() -> None:
    findings = find_forbidden_imports(SRC_ROOT, FORBIDDEN)
    assert findings == [], "\n".join(str(f) for f in findings)


def test_plain_import_violation_is_detected(tmp_path: Path) -> None:
    (tmp_path / "leaky.py").write_text("import projects.olist\n", encoding="utf-8")

    findings = find_forbidden_imports(tmp_path, FORBIDDEN)

    assert [f.module for f in findings] == ["projects.olist"]
    assert findings[0].lineno == 1


def test_from_import_violation_is_detected(tmp_path: Path) -> None:
    (tmp_path / "leaky.py").write_text("from projects.olist import config\n", encoding="utf-8")

    findings = find_forbidden_imports(tmp_path, FORBIDDEN)

    assert [f.module for f in findings] == ["projects.olist"]


def test_relative_import_is_not_a_violation(tmp_path: Path) -> None:
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "inner.py").write_text("from . import projects\n", encoding="utf-8")

    assert find_forbidden_imports(tmp_path, FORBIDDEN) == []


def test_similarly_named_module_is_not_a_violation(tmp_path: Path) -> None:
    (tmp_path / "ok.py").write_text("import projections\n", encoding="utf-8")

    assert find_forbidden_imports(tmp_path, FORBIDDEN) == []
