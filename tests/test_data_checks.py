"""Проверки данных против набора кейсов."""

from __future__ import annotations

import pytest

from dsx.checks import ALL_CHECKS, CONTRACT_CHECKS, DATA_CHECKS, Context, run_checks
from dsx.evals.case import Finding
from dsx.evals.registry import ALL, BY_ID, NEGATIVE_CONTROLS
from dsx.task import OutcomeTiming, TargetKind, TaskSpec

FULL = TaskSpec(
    target_kind=TargetKind.BINARY,
    outcome_timing=OutcomeTiming.DELAYED,
    has_process=True,
    is_stream=True,
)
COVERED = frozenset(f for c in ALL_CHECKS for f in c.detects)

DECLARATIVE = [*CONTRACT_CHECKS, *DATA_CHECKS]
"""Проверки, читающие объявления и данные, но не сверяющие одно с другим."""


from harness import context_for  # noqa: E402


def report_for(bundle, task: TaskSpec = FULL):
    context = context_for(bundle)
    if task is not FULL:
        context = Context(context.world, context.outcome, task, context.split)
    return run_checks(list(ALL_CHECKS), context)


@pytest.mark.parametrize("bundle", NEGATIVE_CONTROLS, ids=lambda b: b.id)
def test_no_alarm_on_clean_worlds(bundle) -> None:
    report = report_for(bundle)

    assert report.findings & COVERED == frozenset(), [str(s) for s in report.signals]


@pytest.mark.parametrize("bundle", ALL, ids=lambda b: b.id)
def test_injected_defect_is_found_and_nothing_else(bundle) -> None:
    """Инжектор ломает ровно одно; проверки обязаны увидеть ровно это."""
    found = report_for(bundle).findings & COVERED

    assert found == bundle.case.expectation.findings & COVERED


def test_lying_declaration_escapes_declarative_checks() -> None:
    """Проверка объявлений читает то же ложное утверждение и бессильна."""
    bundle = BY_ID["feature-falsely-declared-available"]

    report = run_checks(DECLARATIVE, context_for(bundle))

    assert report.findings == frozenset()
    assert bundle.case.expectation.caught_by == frozenset({"N6"})


def test_lying_declaration_is_caught_by_the_empirical_check() -> None:
    bundle = BY_ID["feature-falsely-declared-available"]

    found = report_for(bundle).findings & COVERED

    assert found == bundle.case.expectation.findings


def test_stream_checks_are_skipped_on_static_data() -> None:
    """Проверки полноты периода бессмысленны там, где потока сбора нет."""
    bundle = BY_ID["truncated-tail"]
    static = TaskSpec(
        target_kind=TargetKind.BINARY,
        outcome_timing=OutcomeTiming.DELAYED,
        has_process=True,
        is_stream=False,
    )

    report = report_for(bundle, static)

    assert {"A6", "A7"} <= {s.requirement for s in report.skipped}
    # Ложное объявление отсутствия потока теперь само является находкой (F-4):
    # выключить проверку молча больше нельзя.
    assert report.findings == frozenset({Finding.PREMISE_MISMATCH})


def test_process_checks_are_skipped_without_a_process() -> None:
    bundle = BY_ID["status-timestamp-conflict"]
    no_process = TaskSpec(
        target_kind=TargetKind.BINARY,
        outcome_timing=OutcomeTiming.DELAYED,
        has_process=False,
        is_stream=True,
    )

    report = report_for(bundle, no_process)

    assert {"A5", "A11"} <= {s.requirement for s in report.skipped}


def test_every_data_check_declares_premises_and_findings() -> None:
    for check in DATA_CHECKS:
        assert check.premises, check.requirement
        assert check.detects, check.requirement


def test_signals_explain_the_consequence_not_just_the_fact() -> None:
    """Сообщение, называющее только факт, не помогает решить, что делать."""
    signals = report_for(BY_ID["surrogate-key-as-entity"]).signals

    assert any("история по ней будет короче" in s.detail for s in signals)


def test_surrogate_key_check_stays_quiet_without_a_natural_key() -> None:
    """Уникальность идентификатора по строкам сама по себе нормальна."""
    bundle = BY_ID["clean-baseline"]

    assert not [
        s for s in report_for(bundle).signals if s.finding.value == "surrogate_key_as_entity"
    ]
