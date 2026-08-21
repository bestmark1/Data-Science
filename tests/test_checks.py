"""Проверки контракта против набора кейсов (этап 2)."""

from __future__ import annotations

import pytest

from dsx.checks import CONTRACT_CHECKS, Context, run_checks
from dsx.evals.registry import ALL, BY_ID, NEGATIVE_CONTROLS
from dsx.policy import OverrideLedger
from dsx.task import OutcomeTiming, Premise, TargetKind, TaskSpec

TASK = TaskSpec(
    target_kind=TargetKind.BINARY,
    outcome_timing=OutcomeTiming.DELAYED,
    has_process=True,
    is_stream=True,
)
CONTRACT_REQUIREMENTS = frozenset(c.requirement for c in CONTRACT_CHECKS)
COVERED = frozenset(f for c in CONTRACT_CHECKS for f in c.detects)


def report_for(bundle, task: TaskSpec = TASK, ledger: OverrideLedger | None = None):
    return run_checks(list(CONTRACT_CHECKS), Context(bundle.build(), bundle.outcome, task), ledger)


@pytest.mark.parametrize("bundle", NEGATIVE_CONTROLS, ids=lambda b: b.id)
def test_no_contract_alarm_on_clean_worlds(bundle) -> None:
    """Ложная тревога на чистом мире хуже пропуска: ей перестают верить."""
    assert report_for(bundle).findings & COVERED == frozenset()


@pytest.mark.parametrize(
    "bundle",
    [b for b in ALL if b.case.expectation.caught_by & CONTRACT_REQUIREMENTS],
    ids=lambda b: b.id,
)
def test_contract_checks_catch_what_they_claim(bundle) -> None:
    found = report_for(bundle).findings & COVERED

    assert found == bundle.case.expectation.findings & COVERED


def test_lying_declaration_is_not_caught_by_contract_checks() -> None:
    """Проверка деклараций читает то же ложное утверждение и бессильна.

    Кейс существует, чтобы это ограничение было зафиксировано, а не забыто.
    """
    bundle = BY_ID["feature-falsely-declared-available"]

    assert "N6" in bundle.case.expectation.caught_by
    assert not bundle.case.expectation.caught_by & CONTRACT_REQUIREMENTS
    assert report_for(bundle).findings & COVERED == frozenset()


def test_blocking_signals_are_marked_as_such() -> None:
    signals = report_for(BY_ID["outcome-component-as-feature"]).blocking

    assert signals and all(s.blocking for s in signals)


def test_recorded_override_downgrades_a_block_to_a_warning() -> None:
    ledger = OverrideLedger()
    ledger.override("C6", reason="компонент исхода используется осознанно", author="автор")

    report = report_for(BY_ID["outcome-component-as-feature"], ledger=ledger)

    assert report.blocking == []
    assert report.findings, "сигнал не должен исчезать — он лишь перестаёт блокировать"


def test_override_appears_in_the_report() -> None:
    ledger = OverrideLedger()
    ledger.override("C6", reason="компонент исхода используется осознанно", author="автор")

    report = report_for(BY_ID["outcome-component-as-feature"], ledger=ledger)

    assert len(report.overrides) == 1
    assert report.overrides[0].requirement == "C6"


# --- предпосылки ----------------------------------------------------------


def test_checks_declare_their_premises() -> None:
    for check in CONTRACT_CHECKS:
        assert check.premises, f"{check.requirement} не объявила предпосылок"


def test_check_is_skipped_when_its_premise_is_unmet() -> None:
    """Проверка выключается вместе с предпосылкой, а не срабатывает вхолостую."""

    class OnlyForStreams:
        requirement = "N7"
        premises = frozenset({Premise.STREAM})
        detects = frozenset()

        def run(self, context):  # pragma: no cover - не должна вызываться
            raise AssertionError("проверка запустилась при невыполненной предпосылке")

    bundle = BY_ID["clean-baseline"]
    static_task = TaskSpec(
        target_kind=TargetKind.BINARY, outcome_timing=OutcomeTiming.DELAYED, is_stream=False
    )

    report = run_checks([OnlyForStreams()], Context(bundle.build(), bundle.outcome, static_task))

    assert [s.requirement for s in report.skipped] == ["N7"]


def test_skipped_check_names_the_unmet_premise() -> None:
    """Молча выключенная проверка неотличима от проверки, которая ничего не нашла."""

    class OnlyForProcess:
        requirement = "A5"
        premises = frozenset({Premise.PROCESS})
        detects = frozenset()

        def run(self, context):  # pragma: no cover
            return []

    bundle = BY_ID["clean-baseline"]
    task = TaskSpec(
        target_kind=TargetKind.BINARY, outcome_timing=OutcomeTiming.DELAYED, has_process=False
    )

    report = run_checks([OnlyForProcess()], Context(bundle.build(), bundle.outcome, task))

    assert "process" in str(report.skipped[0])
