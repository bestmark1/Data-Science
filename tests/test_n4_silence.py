"""DS-007: молчание N4 называет свой предел (класс 18 журнала повторов).

N4 засчитывает смену знака связи, только если связь в ОБОИХ окнах не слабее
max(floor, 3σ), σ = 0.29/√(строк меньшего класса). На окнах в 300 строк порог
около 0.13–0.14: смену знака слабее N4 не видит, и её молчание — не
доказательство устойчивости.
"""

from __future__ import annotations

import math

import polars as pl
from tests.test_project import form

from dsx.checks.base import Context, Report, Signal, run_checks
from dsx.evals.case import Finding
from dsx.evals.registry import BY_ID
from dsx.label import LABEL
from dsx.runner import run
from dsx.task import Premise


def _independent_threshold(evaluate) -> float:
    """Порог окна по формуле backlog, без функций ядра."""
    labels = evaluate.filter(evaluate[LABEL].is_not_null())[LABEL]
    positives = int(labels.sum())
    smaller = min(positives, labels.len() - positives)
    return max(0.05, 3 * 0.29 / math.sqrt(smaller))


def test_silent_n4_names_the_weakest_sign_flip_it_would_see() -> None:
    result = run(form(), BY_ID["clean-baseline"].build().main)
    assert not [s for s in result.checks.signals if s.finding is Finding.UNSTABLE_FEATURE_RELATION]

    thresholds = sorted(_independent_threshold(p.evaluate) for p in result.split.parts)
    weakest = thresholds[1]  # смена знака видна, если обе стороны не слабее порога своих окон
    assert round(weakest, 3) == 0.143

    def down(value: float) -> str:
        return f"{math.floor(value * 1000) / 1000:.3f}"

    [limit] = [line for line in result.checks.limits if line.startswith("N4")]
    assert f"слабее {down(weakest)}" in limit
    for part in result.split.parts:
        assert f"{part.name}: {down(_independent_threshold(part.evaluate))}" in limit


def test_the_limit_reaches_the_report() -> None:
    result = run(form(), BY_ID["clean-baseline"].build().main)
    rendered = result.study.render()

    assert "### Предел молчания" in rendered
    assert "N4: молчит" in rendered


class _Talks:
    requirement = "T1"
    premises = frozenset({Premise.UNIVERSAL})
    detects = frozenset({Finding.UNSTABLE_FEATURE_RELATION})

    def __init__(self, talks: bool) -> None:
        self.talks = talks
        self.asked = 0

    def run(self, context: Context) -> list[Signal]:
        return [Signal(Finding.UNSTABLE_FEATURE_RELATION, "есть")] if self.talks else []

    def silence(self, context: Context) -> str:
        self.asked += 1
        return "предел"


def test_the_limit_is_asked_only_of_a_silent_check() -> None:
    """Предел молчания говорит о молчании: сработавшей проверке он не нужен."""
    result = run(form(), BY_ID["clean-baseline"].build().main)
    context = Context(result.world, form().outcome.to_definition(), form().task.to_spec())

    silent, talking = _Talks(False), _Talks(True)
    report: Report = run_checks([silent, talking], context)

    assert report.limits == ["T1: предел"]
    assert silent.asked == 1 and talking.asked == 0


def _context(frames: dict[str, pl.DataFrame]) -> Context:
    """Контекст N4 с окнами, заданными тестом: проверке нужны только окна и схема."""
    import datetime as dt

    from dsx.evals.world import World, base_schema
    from dsx.split import Part, SplitResult, Window

    start = dt.datetime(2024, 1, 1)
    parts = [
        Part(Window(name, start, start + dt.timedelta(days=1)), None, frame)
        for name, frame in frames.items()
    ]
    world = World({"main": next(iter(frames.values()))}, base_schema())
    return Context(world, None, None, SplitResult(parts=parts))


def _window(labels: list[int], values: list[float]) -> pl.DataFrame:
    return pl.DataFrame({LABEL: labels, "lead_days": [float(v) for v in values]})


def _flip(rows: int = 200) -> tuple:
    """Два окна по 100/100 с идеальной связью +0.5 и −0.5."""
    labels = [0] * (rows // 2) + [1] * (rows // 2)
    values = list(range(rows))
    return _window(labels, values), _window(labels, values[::-1])


def test_a_window_where_no_relation_is_computable_leaves_no_numeric_limit() -> None:
    """Ревью `0c7b397`: в третьем окне один класс, связь не вычисляется, и N4
    пропускает признак целиком — а предел печатался 0.087, хотя та же пара окон
    без третьего даёт находку при силе 0.5."""
    from dsx.checks.drift import FeatureRelationStability

    w0, w1 = _flip()
    one_class = _window([0] * 200, list(range(200)))
    n4 = FeatureRelationStability()

    assert n4.run(_context({"w0": w0, "w1": w1}))  # без третьего окна — находка
    context = _context({"w0": w0, "w1": w1, "w2": one_class})
    assert n4.run(context) == []

    limit = n4.silence(context)
    assert "слабее" not in limit
    assert "не сравнила ни одного признака" in limit
    assert "lead_days" in limit and "w2" in limit


def test_a_feature_excluded_by_n5_leaves_no_numeric_limit() -> None:
    from dsx.checks.drift import FeatureRelationStability

    w0, _ = _flip()
    labels = [0] * 100 + [1] * 100
    far = _window(labels, [v + 10_000 for v in range(200)][::-1])
    context = _context({"w0": w0, "w1": far})
    n4 = FeatureRelationStability()

    assert n4.run(context) == []
    limit = n4.silence(context)
    assert "слабее" not in limit and "N5" in limit


def test_an_unreachable_threshold_is_named_unreachable() -> None:
    """Меньший класс из одной строки: порог 0.87, а связь не бывает сильнее 0.5 —
    смену знака N4 не заметит ни при какой силе."""
    from dsx.checks.drift import FeatureRelationStability

    labels = [0] * 199 + [1]
    context = _context(
        {"w0": _window(labels, list(range(200))), "w1": _window(labels, list(range(200))[::-1])}
    )
    limit = FeatureRelationStability().silence(context)

    assert "недостижим" in limit and "слабее" not in limit


def test_the_printed_limit_is_rounded_down() -> None:
    """Ревью `0c7b397`: порог 0.87/√45 = 0.12969 печатался как 0.130, а N4 ловила
    разворот силой 0.12971 < 0.130 — «слабее 0.130 не заметила бы» было неверно.
    Нижняя граница округляется вниз."""
    import re

    from dsx.checks.drift import FeatureRelationStability

    y = [0] * 601 + [1] * 28 + [0] + [1] * 17 + [0] * 353
    x = list(range(1000))
    context = _context({"w0": _window(y, x), "w1": _window(y, x[::-1])})
    n4 = FeatureRelationStability()

    assert n4.run(context)  # разворот силой 0.12971 N4 замечает
    limit = n4.silence(context)
    printed = float(re.search(r"слабее (\d\.\d+)", limit).group(1))
    assert printed <= 0.87 / math.sqrt(45) and printed == 0.129
