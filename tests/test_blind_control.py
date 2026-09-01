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


NEW_FILE_PATCH = """diff --git a/sealed.md b/sealed.md
new file mode 100644
index 0000000..e69de29
--- /dev/null
+++ b/sealed.md
@@ -0,0 +1,2 @@
+ОЖИДАЕМАЯ НАХОДКА: duplicate_rows
+объяснение
"""


def test_a_patch_that_creates_a_file_survives_extraction() -> None:
    """Патч, создающий файл, несёт `new file mode` второй строкой.

    Первая версия извлекателя знала три начала строки, обрывалась на этой и
    оставляла ровно 111 байт заголовка. `git apply` отказывал, и слепой
    контроль не мог состояться ни разу.
    """
    patch = _tool()._extract_patch("думаю...\n" + NEW_FILE_PATCH + "tokens used\n42")

    assert "new file mode 100644" in patch
    assert "+ОЖИДАЕМАЯ НАХОДКА: duplicate_rows" in patch
    assert "tokens used" not in patch


def test_an_empty_context_line_does_not_truncate_the_patch() -> None:
    """Пустая строка — тело патча: контекстная строка пустой строки файла."""
    stream = "diff --git a/x b/x\n--- a/x\n+++ b/x\n@@ -1,3 +1,3 @@\n a\n\n-b\n+c\n"

    patch = _tool()._extract_patch(stream)

    assert patch.endswith("+c\n")
    assert len(patch.splitlines()) == len(stream.splitlines())


def test_an_extracted_patch_actually_applies(tmp_path) -> None:
    """Извлечение проверяется применением, а не похожестью на патч.

    Оба прежних дефекта прошли мимо тестов именно потому, что тесты сверяли
    строки, а не результат.
    """
    import subprocess

    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    patch_file = tmp_path / "p.patch"
    patch_file.write_text(_tool()._extract_patch(NEW_FILE_PATCH), encoding="utf-8")

    done = subprocess.run(
        ["git", "apply", "--recount", "p.patch"], cwd=tmp_path, capture_output=True, text=True
    )

    assert done.returncode == 0, done.stderr
    assert (tmp_path / "sealed.md").read_text(encoding="utf-8").startswith("ОЖИДАЕМАЯ")
