"""Одна команда проверок: тесты, стиль и счёт ненаступивших обещаний.

    .venv/bin/python tools/check.py

ЧТО ОНА ДЕЛАЕТ. Запускает уже существующие проверки — `pytest` и `ruff`, — и
ничего не изобретает поверх них. Готовой такой команды в проекте не было:
`pytest` и `ruff` вызывались порознь и по памяти, а `pending` не показывался
вовсе.

ЧЕГО ОНА НЕ ДЕЛАЕТ, и это главное. Не запускает кейсы, не собирает данные, не
зовёт внешние модели, не мутирует исходники. Мутационный прогон порогов и слепой
контроль остаются отдельными инструментами: первый правит файлы ядра, второй
тратит внешний лимит, и смешивать их с проверкой репозитория нельзя.

PENDING — ЭТО НЕ УСПЕХ. Обещание пре-регистрации, чей формуляр ещё не написан,
не нарушено и не исполнено: оно НЕ НАСТУПИЛО. Такие обещания считаются отдельно
и печатаются отдельно — и ни одно из них не является разрешением запускать кейс.
Порядок кейса проверяет `protocol.preflight`, а не эта команда.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "bin" / "python"

PENDING = re.compile(r"ещё не наступило: (\d+)")
PASSED = re.compile(r"(\d+) passed")
FAILED = re.compile(r"(\d+) failed")


def _run(argv: list[str]) -> tuple[int, str]:
    done = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True)
    return done.returncode, done.stdout + done.stderr


def main() -> int:
    if not PYTHON.is_file():
        print(f"ОТКАЗ: окружения нет — {PYTHON}. Соберите его: uv sync --extra dev")
        return 1

    # `-s` нужен, чтобы счёт ненаступивших обещаний дошёл до вывода: тест
    # печатает его, а не возвращает. `-p no:cacheprovider` — чтобы проверка не
    # оставляла следов в рабочем дереве.
    code, out = _run([str(PYTHON), "-m", "pytest", "tests", "-q", "-s", "-p", "no:cacheprovider"])
    passed = int(m.group(1)) if (m := PASSED.search(out)) else 0
    failed = int(m.group(1)) if (m := FAILED.search(out)) else 0
    pending = sum(int(number) for number in PENDING.findall(out))

    print(f"тесты:   passed {passed}, failed {failed}, pending {pending}")
    if failed:
        print("\n--- провалившиеся ---")
        for line in out.splitlines():
            if line.startswith("FAILED"):
                print(f"  {line}")
    elif code != 0:
        # Ненулевой код без единого `failed` — это отказ ДО тестов: ошибка сборки
        # (ImportError, синтаксис, отсутствующий модуль). Прежняя версия печатала
        # «passed 0, failed 0» и «ЕСТЬ ПРОВАЛЫ», уничтожая первопричину: имя
        # отсутствующего модуля в вывод не попадало вовсе.
        print(
            f"\n--- pytest завершился с кодом {code}, не дав статистики ---\n"
            "Похоже на ошибку сборки тестов, а не на провал проверки.\n"
            "Вывод целиком:"
        )
        print(out.strip()[-4000:] or "(пусто)")

    style, styled = _run([str(PYTHON), "-m", "ruff", "check", "src", "tests", "tools", "protocol"])
    print(f"стиль:   {'чисто' if style == 0 else 'есть замечания'}")
    if style != 0:
        print(styled.strip()[:2000])

    if pending:
        print(
            f"\nобещаний ещё не наступило: {pending}. Это НЕ успех и НЕ разрешение\n"
            "запускать кейс: формуляр для них ещё не написан. Порядок кейса\n"
            "проверяет protocol.preflight, а не эта команда."
        )

    ok = code == 0 and style == 0
    print(f"\nитог: {'ПРОВЕРКИ ПРОЙДЕНЫ' if ok else 'ЕСТЬ ПРОВАЛЫ'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
