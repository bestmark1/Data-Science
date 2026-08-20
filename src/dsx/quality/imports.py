"""Граница пакета: ядро не зависит от проектов-экземпляров.

Зависимость односторонняя. Проект импортирует dsx; dsx о проектах не знает.
Нарушение означает, что абстракция протекла, — это критерий H1, а не стилистика.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ForbiddenImport:
    """Одно нарушение границы."""

    path: Path
    lineno: int
    module: str

    def __str__(self) -> str:
        return f"{self.path}:{self.lineno}: импорт {self.module!r}"


def _top_level(dotted_name: str) -> str:
    return dotted_name.split(".", 1)[0]


def find_forbidden_imports(root: Path, forbidden: frozenset[str]) -> list[ForbiddenImport]:
    """Найти импорты запрещённых пакетов верхнего уровня в дереве Python-файлов.

    Относительные импорты пропускаются: они не могут выйти за пределы пакета
    к постороннему модулю верхнего уровня.
    """
    findings: list[ForbiddenImport] = []

    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if _top_level(alias.name) in forbidden:
                        findings.append(ForbiddenImport(path, node.lineno, alias.name))

            elif isinstance(node, ast.ImportFrom):
                # level > 0 — относительный импорт, до чужого верхнего уровня не дотянется.
                if node.level == 0 and node.module and _top_level(node.module) in forbidden:
                    findings.append(ForbiddenImport(path, node.lineno, node.module))

    return findings
