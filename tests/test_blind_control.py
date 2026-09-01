"""Слепой контроль достижим с рабочего пути.

Инструмент был написан, закоммичен и НИ РАЗУ не запущен: запуск блокировался
разрешением. При первом настоящем запуске он отказал на первом же шаге — искал
перечень находок по имени файла `finding*.py`, которого не существует.

Это шестой случай класса «механизм построен и не проверен на достижимость».
Здесь он закрывается тем, чем только и закрывается: вызовом с рабочего пути.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

TOOL = Path(__file__).resolve().parents[1] / "tools" / "blind_control.py"


def _tool():
    spec = importlib.util.spec_from_file_location("blind_control", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_findings_catalogue_is_found() -> None:
    """Каталог находок обязан находиться, иначе выбирать второй стороне не из чего."""
    catalogue = _tool()._findings_catalogue()

    assert catalogue is not None, "перечень находок не найден: инструмент откажет на первом шаге"
    assert "class Finding(" in catalogue.read_text(encoding="utf-8")


def test_patch_is_extracted_from_a_noisy_stream() -> None:
    """Вывод второй стороны несёт счётчик токенов и прочий шум."""
    module = _tool()
    stream = "думаю...\ndiff --git a/x b/x\n--- a/x\n+++ b/x\n@@ -1 +1 @@\n-a\n+b\ntokens used\n17"

    patch = module._extract_patch(stream)

    assert patch.startswith("diff --git")
    assert "tokens used" not in patch
    assert patch.endswith("+b\n")


def test_nothing_is_extracted_when_there_is_no_patch() -> None:
    """Отрицательный контроль: молчание не должно выглядеть патчем."""
    assert _tool()._extract_patch("никакого патча тут нет") == ""
