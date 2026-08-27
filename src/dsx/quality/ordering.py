"""Отбор строк, чей порядок не определён.

`unique` в polars порядок строк НЕ сохраняет, если об этом не попросить. Пока
результат идёт в справочник, это безразлично. Но стоит ему попасть в таблицу
решений — и отчёт перестаёт воспроизводиться: точечные оценки совпадают, а
доверительные интервалы расходятся, потому что пересчёт берёт НОМЕРА строк, и
при другом порядке это другие строки.

Тринадцатый кейс потерял на этом предсказание P-8. Класс был известен с
седьмого, где порядок был не определён после `group_by` В ЯДРЕ, — там его
закрыли, а в коде проектов не закрыли, и он вернулся.

Проверка синтаксическая и намеренно узкая: `unique(subset=...)` без
`maintain_order`. Просить сохранения порядка почти ничего не стоит, а не
просить — значит оставить скрытую опасность там, где её нельзя увидеть глазами.

Первая версия ловила всякий `unique` и дала двадцать девять находок, ни одна
из которых не была дефектом. Сужена ДО принятия: проверка, кричащая на
законном, по правилам этого проекта хуже своего отсутствия.

Предел назван прямо: `group_by`, чей порядок тоже не определён, здесь НЕ
проверяется. Его результат почти всегда идёт в соединение, где порядок
безразличен, и проверка кричала бы на законном — а такая, по правилам этого
проекта, хуже своего отсутствия.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class UnorderedSelection:
    """Отбор строк, оставляющий порядок на усмотрение библиотеки."""

    path: Path
    lineno: int

    def __str__(self) -> str:
        return f"{self.path}:{self.lineno}: unique без maintain_order"


def _is_row_deduplication(node: ast.AST) -> bool:
    """Отбор СТРОК по подмножеству колонок, а не перечень значений.

    Проверка намеренно узка. Первая версия ловила всякий `unique` и дала
    двадцать девять находок, из которых ни одна не была дефектом: `series
    .unique()` собирает словарь значений, и порядок там безразличен, а
    результат почти всегда сортируется следом.

    Опасна одна форма — `frame.unique(subset=[...])`: она выбирает, КАКИЕ
    СТРОКИ останутся, и её результат идёт в таблицу решений. Ровно на ней
    тринадцатый кейс потерял воспроизводимость отчёта.
    """
    if not isinstance(node, ast.Call):
        return False
    if not isinstance(node.func, ast.Attribute) or node.func.attr != "unique":
        return False
    if not any(k.arg == "subset" for k in node.keywords):
        return False
    return not any(k.arg == "maintain_order" for k in node.keywords)


def find_unordered_selections(*paths: Path) -> list[UnorderedSelection]:
    """Найти отборы строк, чей порядок не объявлен."""
    findings: list[UnorderedSelection] = []
    for path in paths:
        files = sorted(path.rglob("*.py")) if path.is_dir() else [path]
        for file in files:
            try:
                tree = ast.parse(file.read_text(encoding="utf-8"))
            except SyntaxError:
                continue
            findings += [
                UnorderedSelection(path=file, lineno=node.lineno)
                for node in ast.walk(tree)
                if _is_row_deduplication(node)
            ]
    return findings
