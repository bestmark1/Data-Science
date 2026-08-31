"""Область смысла объявлена у каждого порога и соблюдается.

Список областей лежит отдельно от проверок, и без этих тестов он разошёлся бы
с кодом при первом же новом пороге. Механизм, удовлетворяемый необеспеченным
объявлением, защищает хуже своего отсутствия — поэтому объявление здесь
связано с кодом, а не приложено к нему.
"""

from __future__ import annotations

import ast
import dataclasses
from pathlib import Path

import pytest

from dsx.checks import ALL_CHECKS
from dsx.checks.domains import DOMAINS, Domain

CONSTANT_MODULES = (
    Path("src/dsx/checks/drift.py"),
    Path("src/dsx/checks/empirical.py"),
    Path("src/dsx/measure.py"),
)
SEED_NAMES = ("SEED", "seed")


def _module_constants(path: Path) -> list[tuple[str, float]]:
    """Числовые константы уровня модуля, кроме зёрен.

    Зерно не порог: его значение не сравнивается ни с чем в данных.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = []
    for item in tree.body:
        if not isinstance(item, ast.Assign) or len(item.targets) != 1:
            continue
        target = item.targets[0]
        if not isinstance(target, ast.Name) or not isinstance(item.value, ast.Constant):
            continue
        value = item.value.value
        if not isinstance(value, int | float) or isinstance(value, bool):
            continue
        if any(part in target.id for part in SEED_NAMES):
            continue
        found.append((f"{path.stem}.{target.id}", value))
    return found


def _thresholds() -> list[tuple[str, float]]:
    """Все числовые пороги ядра: поля проверок и константы модулей."""
    found: list[tuple[str, float]] = []
    for check in ALL_CHECKS:
        if not dataclasses.is_dataclass(check):
            continue
        for field in dataclasses.fields(check):
            value = getattr(check, field.name)
            if isinstance(value, int | float) and not isinstance(value, bool):
                found.append((f"{check.requirement}.{field.name}", value))
    for path in CONSTANT_MODULES:
        found.extend(_module_constants(path))
    return found


@pytest.mark.parametrize("name,value", _thresholds(), ids=lambda item: str(item))
def test_every_threshold_declares_its_domain(name: str, value: float) -> None:
    """Порог без объявленной области — число, о котором не сказано ничего."""
    assert name in DOMAINS, (
        f"у порога {name}={value} не объявлена область смысла: "
        "добавьте её в dsx.checks.domains вместе с причиной"
    )


@pytest.mark.parametrize("name,value", _thresholds(), ids=lambda item: str(item))
def test_every_threshold_lies_inside_its_domain(name: str, value: float) -> None:
    """Объявление, которому не удовлетворяет собственное значение, — не защита."""
    domain = DOMAINS.get(name)
    if domain is None:
        pytest.skip("область не объявлена: об этом говорит соседний тест")
    assert domain.admits(value), f"{name}={value} лежит вне объявленной области: {domain.reason}"


def test_no_domain_is_declared_for_a_threshold_that_does_not_exist() -> None:
    """Отрицательный контроль: список не должен обрастать мёртвыми записями.

    Запись о пороге, которого нет, создаёт видимость покрытия и переживает
    удаление самого порога.
    """
    live = {name for name, _ in _thresholds()}
    dead = sorted(set(DOMAINS) - live)

    assert not dead, f"область объявлена для несуществующих порогов: {dead}"


def test_domain_bounds_are_strict_on_both_sides() -> None:
    """Граница включённая означала бы, что «всегда» и «никогда» — тоже пороги."""
    domain = Domain(0.0, 1.0, "проверочная")

    assert not domain.admits(0.0)
    assert not domain.admits(1.0)
    assert domain.admits(0.5)
