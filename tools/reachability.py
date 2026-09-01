"""Какие функции проекта ни разу не вызываются за прогон тестов.

Это НЕ детектор мёртвого кода. Тот в проекте уже испробован и отвергнут: из 195
публичных имён он назвал 52, настоящим оказался один. Здесь меряется другое и
точнее — не «есть ли на имя ссылка», а «исполнялось ли оно». Три последних
дефекта проекта лежали именно там: код вызывался из другого кода и ни разу не
запускался с настоящими входами.

Способ: профилировщик стандартной библиотеки ловит события вызова, имена
сопоставляются с определениями, найденными разбором исходников.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path("/Users/bestmark1/Data Science")
WATCHED = ("src/dsx", "tools")

called: set[tuple[str, int]] = set()


def _record(frame, event, _arg):
    if event == "call":
        code = frame.f_code
        called.add((code.co_filename, code.co_firstlineno))
    return None


def _defined() -> dict[tuple[str, int], str]:
    found: dict[tuple[str, int], str] = {}
    for base in WATCHED:
        for path in sorted((ROOT / base).rglob("*.py")):
            if "__pycache__" in str(path):
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                    # У декорированной функции профилировщик показывает строку
                    # ПЕРВОГО ДЕКОРАТОРА, а AST — строку `def`. Первая версия
                    # ключевалась по `def` и теряла каждое свойство и каждый
                    # валидатор: `Schema.decision_time` попал в «ни разу не
                    # исполнено», хотя его зовёт `compute` сотни раз за прогон.
                    first = min([node.lineno, *(d.lineno for d in node.decorator_list)])
                    found[(str(path), first)] = f"{path.relative_to(ROOT)}:{node.name}"
    return found


def main() -> int:
    import pytest

    defined = _defined()
    sys.setprofile(_record)
    try:
        pytest.main(["tests", "-q", "--no-header", "-p", "no:cacheprovider"])
    finally:
        sys.setprofile(None)

    ran = {key for key in defined if key in called}
    never = sorted(defined[key] for key in defined if key not in called)

    print(
        f"\nфункций определено: {len(defined)}, исполнено: {len(ran)}, "
        f"НИ РАЗУ не исполнено: {len(never)}"
    )
    for name in never:
        print(f"  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
