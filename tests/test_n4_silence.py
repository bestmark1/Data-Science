"""DS-007: молчание N4 называет свой предел (класс 18 журнала повторов).

N4 засчитывает смену знака связи, только если связь в ОБОИХ окнах не слабее
max(floor, 3σ), σ = 0.29/√(строк меньшего класса). На окнах в 300 строк порог
около 0.13–0.14: смену знака слабее N4 не видит, и её молчание — не
доказательство устойчивости.
"""

from __future__ import annotations

import math

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

    [limit] = [line for line in result.checks.limits if line.startswith("N4")]
    assert f"слабее {weakest:.3f}" in limit
    for part in result.split.parts:
        assert f"{part.name}: {_independent_threshold(part.evaluate):.3f}" in limit


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
