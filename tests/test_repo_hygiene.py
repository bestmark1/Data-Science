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


# --- Команда проверок не смеет запускать исследование ------------------------

CHECK = REPO_ROOT / "tools" / "check.py"

FORBIDDEN = (
    "acquire",  # сбор данных: сеть и объём
    "model.py",  # обучение: долго и не про контракт репозитория
    "blind_control",  # слепой контроль: тратит лимит второй стороны
    "threshold_mutation",  # мутация: ПРАВИТ файлы ядра
    "codex",  # внешняя модель
    "urllib",  # сеть в любом виде
    "requests",
)


def test_the_check_command_runs_no_research() -> None:
    """Команда проверок читает репозиторий и ничего не исследует.

    Соблазн дописать в неё мутационный прогон или запуск кейса велик: всё это
    «тоже проверки». Но мутация ПРАВИТ файлы ядра, слепой контроль тратит
    внешний лимит, а сбор данных лезет в сеть — и тогда команда, которую
    запускают перед коммитом, начнёт менять то, что проверяет.
    """
    text = CHECK.read_text(encoding="utf-8")
    code = "\n".join(line for line in text.splitlines() if not line.strip().startswith("#")).split(
        '"""'
    )[-1]

    found = [word for word in FORBIDDEN if word in code]

    assert not found, f"команда проверок обзавелась исследовательскими вызовами: {found}"


def test_the_check_command_reports_pending_separately() -> None:
    """`pending` не должен теряться среди `passed`.

    Обещание, чей формуляр ещё не написан, не нарушено и не исполнено. Слить его
    с успехом значило бы сказать, что проверять нечего, — а проверять просто
    ещё нечего.
    """
    text = CHECK.read_text(encoding="utf-8")

    assert "pending" in text
    assert "НЕ успех" in text, "команда обязана сказать, что pending успехом не является"
