"""Сравнения, у которых пустое значение даёт не ложь, а пустоту.

Класс, повторявшийся трижды за десять кейсов и записанный в журнал повторов:

* пятый кейс — `pl.col(status) == None` даёт null, а не истину: доля считалась
  по пустой выборке и выходила нулевой, после чего порог «больше нуля в
  полтора раза» выполнялся всегда;
* седьмой кейс — `frame.filter(~broken)`, где `broken` собран из сравнений дат:
  при пустой дате сравнение даёт null, отрицание null остаётся null, и строка
  выбрасывается молча. Правило баланса поймало 737 715 потерянных строк вместо
  объявленных 3 994;
* восьмой кейс — то же в другой сборке.

Каждый раз чинился экземпляр. Здесь чинится класс.

Проверка синтаксическая, и предел назван прямо: выражение, собранное через
переменную в другой функции или переданное аргументом, она не увидит. Для
кода этого репозитория этого достаточно — все три случая были локальны.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

COMPARISONS = (ast.Lt, ast.Gt, ast.LtE, ast.GtE)
"""Сравнения порядка. Равенство и неравенство разбираются отдельно: у них
опасен не только пустой операнд, но и сама сверка с `None`."""


@dataclass(frozen=True)
class NullComparison:
    """Сравнение, чей результат при пустом значении не ложь, а пустота."""

    path: Path
    lineno: int
    kind: str

    def __str__(self) -> str:
        return f"{self.path}:{self.lineno}: {self.kind}"


def _mentions_polars_column(node: ast.AST) -> bool:
    """Есть ли в выражении обращение к колонке polars."""
    for inner in ast.walk(node):
        if (
            isinstance(inner, ast.Call)
            and isinstance(inner.func, ast.Attribute)
            and inner.func.attr == "col"
        ):
            return True
    return False


GUARDS = ("fill_null", "is_null", "is_not_null")
"""Способы назвать пустое значение вместо того, чтобы получить его молча."""


def _guarded_comparisons(root: ast.AST) -> set[int]:
    """Сравнения, ОБЁРНУТЫЕ в fill_null и потому безопасные.

    Первая версия искала `fill_null` ВНУТРИ узла сравнения и не находила
    никогда: обёртка снаружи, сравнение внутри неё, а не наоборот. Детектор
    выдал четыре находки из четырёх, и все были ложными. Ровно тот случай,
    ради которого проект отверг общий детектор недостижимого кода: проверка,
    кричащая на законном, хуже её отсутствия.
    """
    safe: set[int] = set()
    for node in ast.walk(root):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in GUARDS
        ):
            for inner in ast.walk(node):
                if isinstance(inner, ast.Compare):
                    safe.add(id(inner))
    return safe


def _columns_declared_null_safe(root: ast.AST) -> set[str]:
    """Колонки, у которых пустое значение названо отдельным `is_null` рядом.

    Верный оборот выглядит так:

        pl.col("x").is_null() | (pl.col("x") <= 0)

    Пустое значение здесь отсеяно первым слагаемым, и второе безопасно. Не
    увидев этого, детектор возражал бы на правильном коде — а проверка,
    кричащая на законном, запрещена правилами проекта.
    """
    safe: set[str] = set()
    for node in ast.walk(root):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in ("is_null", "is_not_null")
            and isinstance(node.func.value, ast.Call)
            and isinstance(node.func.value.func, ast.Attribute)
            and node.func.value.func.attr == "col"
            and node.func.value.args
            and isinstance(node.func.value.args[0], ast.Constant)
        ):
            safe.add(str(node.func.value.args[0].value))
    return safe


def _columns_in(node: ast.AST) -> set[str]:
    names = set()
    for inner in ast.walk(node):
        if (
            isinstance(inner, ast.Call)
            and isinstance(inner.func, ast.Attribute)
            and inner.func.attr == "col"
            and inner.args
            and isinstance(inner.args[0], ast.Constant)
        ):
            names.add(str(inner.args[0].value))
    return names


def _compared_with_none(tree: ast.AST, path: Path) -> list[NullComparison]:
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare) or not isinstance(node.ops[0], (ast.Eq, ast.NotEq)):
            continue
        operands = [node.left, *node.comparators]
        if not any(isinstance(o, ast.Constant) and o.value is None for o in operands):
            continue
        if not any(_mentions_polars_column(o) for o in operands):
            continue
        found.append(
            NullComparison(path, node.lineno, "сравнение колонки с None даёт пустоту, а не истину")
        )
    return found


def _negated_without_guard(tree: ast.AST, path: Path) -> list[NullComparison]:
    """Отрицание выражения, чьи сравнения могут дать пустоту.

    Отслеживается и через переменную: `broken = (...)`, ниже `filter(~broken)` —
    именно так выглядели оба случая в сборках.
    """
    risky: dict[str, int] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        safe = _guarded_comparisons(node.value)
        named = _columns_declared_null_safe(node.value)
        bare = [
            inner
            for inner in ast.walk(node.value)
            if isinstance(inner, ast.Compare)
            and isinstance(inner.ops[0], COMPARISONS)
            and _mentions_polars_column(inner)
            and id(inner) not in safe
            and not _columns_in(inner) <= named
        ]
        if bare:
            risky[target.id] = bare[0].lineno

    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.UnaryOp) or not isinstance(node.op, ast.Invert):
            continue
        operand = node.operand
        if isinstance(operand, ast.Name) and operand.id in risky:
            found.append(
                NullComparison(
                    path,
                    risky[operand.id],
                    f"отрицается {operand.id!r}, собранное из сравнений без fill_null: "
                    "строка с пустым значением выпадет молча",
                )
            )
        elif not isinstance(operand, ast.Name) and _mentions_polars_column(operand):
            safe = _guarded_comparisons(operand)
            bare = [
                inner
                for inner in ast.walk(operand)
                if isinstance(inner, ast.Compare)
                and isinstance(inner.ops[0], COMPARISONS)
                and id(inner) not in safe
            ]
            if bare:
                found.append(
                    NullComparison(
                        path,
                        node.lineno,
                        "отрицается сравнение без fill_null: строка с пустым "
                        "значением выпадет молча",
                    )
                )
    return found


def find_null_comparisons(*paths: Path) -> list[NullComparison]:
    """Найти сравнения, у которых пустое значение даёт пустоту вместо лжи."""
    findings: list[NullComparison] = []
    for path in paths:
        files = sorted(path.rglob("*.py")) if path.is_dir() else [path]
        for file in files:
            try:
                tree = ast.parse(file.read_text(encoding="utf-8"))
            except SyntaxError:
                continue
            findings += _compared_with_none(tree, file) + _negated_without_guard(tree, file)
    return findings
