"""Граница пакета: ядро не зависит от проектов-экземпляров.

Зависимость односторонняя. Проект импортирует dsx; dsx о проектах не знает.
Нарушение означает, что абстракция протекла, — это критерий H1, а не стилистика.

Проверка синтаксическая и потому не является гарантией: имя модуля, собранное
из данных, она не увидит. Поэтому динамический импорт в ядре считается
находкой сам по себе — не потому, что он нарушает границу, а потому, что
делает границу непроверяемой.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

FindingKind = Literal["import", "dynamic", "unparseable"]

_DYNAMIC_CALLS = frozenset({"__import__", "exec", "eval", "compile"})


@dataclass(frozen=True)
class BoundaryFinding:
    """Одна находка при проверке границы."""

    kind: FindingKind
    path: Path
    lineno: int
    detail: str

    def __str__(self) -> str:
        return f"{self.path}:{self.lineno}: [{self.kind}] {self.detail}"


def _top_level(dotted_name: str) -> str:
    return dotted_name.split(".", 1)[0]


def _dynamic_import_detail(node: ast.Call) -> str | None:
    """Имя конструкции, делающей импорт непроверяемым, либо None."""
    func = node.func

    if isinstance(func, ast.Name) and func.id in _DYNAMIC_CALLS:
        return func.id

    if isinstance(func, ast.Attribute) and func.attr == "import_module":
        return "importlib.import_module"

    return None


def find_boundary_findings(root: Path, forbidden: frozenset[str]) -> list[BoundaryFinding]:
    """Проверить дерево Python-файлов на нарушения границы пакета.

    Относительные импорты пропускаются: они не могут выйти за пределы пакета
    к постороннему модулю верхнего уровня.

    Файл, который не удалось разобрать, порождает находку, а не исключение:
    непроверенный файл — это дыра в границе, и молчать о нём нельзя.
    """
    findings: list[BoundaryFinding] = []

    for path in sorted(root.rglob("*.py")):
        try:
            source = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError) as exc:
            findings.append(BoundaryFinding("unparseable", path, 0, f"файл не читается: {exc}"))
            continue

        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError as exc:
            findings.append(
                BoundaryFinding("unparseable", path, exc.lineno or 0, f"не разбирается: {exc.msg}")
            )
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if _top_level(alias.name) in forbidden:
                        findings.append(
                            BoundaryFinding("import", path, node.lineno, f"импорт {alias.name!r}")
                        )

            elif isinstance(node, ast.ImportFrom):
                # level > 0 — относительный импорт, до чужого верхнего уровня не дотянется.
                if node.level == 0 and node.module and _top_level(node.module) in forbidden:
                    findings.append(
                        BoundaryFinding("import", path, node.lineno, f"импорт {node.module!r}")
                    )

            elif isinstance(node, ast.Call):
                detail = _dynamic_import_detail(node)
                if detail is not None:
                    findings.append(
                        BoundaryFinding(
                            "dynamic",
                            path,
                            node.lineno,
                            f"{detail} делает границу непроверяемой",
                        )
                    )

    return findings
