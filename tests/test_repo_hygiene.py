"""Правила, нарушаемые однократно и необратимо: данные и локальный журнал (R3, R4)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

pytestmark = pytest.mark.skipif(
    shutil.which("git") is None or not (REPO_ROOT / ".git").exists(),
    reason="проверка правил игнорирования требует git и рабочего репозитория",
)


def _is_ignored(relative_path: str) -> bool:
    result = subprocess.run(
        ["git", "check-ignore", "-q", "--no-index", relative_path],
        cwd=REPO_ROOT,
        check=False,
    )
    if result.returncode not in (0, 1):
        pytest.fail(f"git check-ignore завершился с кодом {result.returncode}")
    return result.returncode == 0


@pytest.mark.parametrize(
    "path",
    [
        "projects/olist-delivery-delay/data/orders.csv",
        "projects/olist-delivery-delay/artifacts/profile.parquet",
        "knowledge/local/friction-log.md",
        "mlruns/0/meta.yaml",
        "data/raw.csv",
        ".env",
        ".env.local",
        "kaggle.json",
        "notes/analysis.ipynb",
        "projects/olist-delivery-delay/data/orders.sqlite",
        "projects/olist-delivery-delay/data/export.xlsx",
    ],
)
def test_sensitive_paths_are_ignored(path: str) -> None:
    assert _is_ignored(path), f"{path} не игнорируется — данные утекут в git"


@pytest.mark.parametrize(
    "path",
    [
        "projects/olist-delivery-delay/config.yaml",
        "projects/olist-delivery-delay/manifest.yaml",
        "knowledge/portable/friction-log.md",
        "src/dsx/__init__.py",
    ],
)
def test_tracked_paths_are_not_ignored(path: str) -> None:
    assert not _is_ignored(path), f"{path} игнорируется, хотя должен попадать в git"
