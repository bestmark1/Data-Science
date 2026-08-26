"""Соединения в обход объявленной грануляции.

`guarded_join` требует назвать ключи и ожидаемую грануляцию и отказывается
соединять без них. Механизм существует с первого кейса, где неверная
грануляция раздула таблицу в четыре раза.

За второй кейс я обошёл его трижды — писал `frame.join(...)` напрямую и
замечал беду по числу строк, а не по отказу. Механизм, который надо не забыть
позвать, защищает ровно настолько, насколько хороша память зовущего.

Проверка синтаксическая. Соединение, собранное через переменную-метод, она не
увидит; для рабочего кода проектов этого достаточно, и предел назван прямо.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

JOIN_KEYWORDS = frozenset({"on", "how", "left_on", "right_on", "by", "strategy"})
"""Именованные аргументы, встречающиеся только у соединения таблиц.

Отличают `frame.join(other, on=...)` от `", ".join(items)`: у строкового
метода таких аргументов нет, и путаницы не возникает.
"""

JOIN_METHODS = frozenset({"join", "join_asof", "join_where"})
"""Методы соединения таблиц.

`join_asof` добавлен шестым кейсом, где он прошёл мимо детектора незамеченным.
Он опаснее обычного соединения тем, что число строк левой таблицы сохраняет
всегда: ошибка направления не раздувает таблицу, а молча подставляет не ту
строку, и признак момента решения получает будущее.
"""


@dataclass(frozen=True)
class UnguardedJoin:
    """Соединение таблиц мимо объявленной грануляции."""

    path: Path
    lineno: int

    method: str = "join"

    def __str__(self) -> str:
        guard = "guarded_asof_join" if self.method == "join_asof" else "guarded_join"
        return f"{self.path}:{self.lineno}: соединение мимо {guard}"


def _is_dataframe_join(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    if not isinstance(node.func, ast.Attribute) or node.func.attr not in JOIN_METHODS:
        return False
    return any(keyword.arg in JOIN_KEYWORDS for keyword in node.keywords)


def find_unguarded_joins(*paths: Path) -> list[UnguardedJoin]:
    """Найти соединения таблиц, написанные в обход guarded_join."""
    findings: list[UnguardedJoin] = []
    for path in paths:
        files = sorted(path.rglob("*.py")) if path.is_dir() else [path]
        for file in files:
            try:
                tree = ast.parse(file.read_text(encoding="utf-8"))
            except SyntaxError:
                continue
            findings += [
                UnguardedJoin(path=file, lineno=node.lineno, method=node.func.attr)
                for node in ast.walk(tree)
                if _is_dataframe_join(node)
            ]
    return findings
