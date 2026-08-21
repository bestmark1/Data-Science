"""Предпосылки проверок (S6) и механизм обхода блокировок (S5)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from dsx.policy import Blocked, OverrideLedger
from dsx.task import (
    OutcomeTiming,
    Premise,
    TargetKind,
    TaskSpec,
    UnsupportedTask,
    require_supported,
)

OLIST_LIKE = TaskSpec(
    target_kind=TargetKind.BINARY,
    outcome_timing=OutcomeTiming.DELAYED,
    has_process=True,
    is_stream=True,
)
PLAIN_REGRESSION = TaskSpec(
    target_kind=TargetKind.REGRESSION, outcome_timing=OutcomeTiming.IMMEDIATE
)


def test_universal_premise_holds_for_any_task() -> None:
    assert OLIST_LIKE.satisfies(Premise.UNIVERSAL)
    assert PLAIN_REGRESSION.satisfies(Premise.UNIVERSAL)


def test_premises_reflect_the_declared_task() -> None:
    assert OLIST_LIKE.satisfies_all(
        frozenset({Premise.PROCESS, Premise.DELAYED_OUTCOME, Premise.BINARY_TARGET})
    )
    assert not PLAIN_REGRESSION.satisfies(Premise.BINARY_TARGET)
    assert not PLAIN_REGRESSION.satisfies(Premise.DELAYED_OUTCOME)


def test_unmet_premises_are_named_not_just_counted() -> None:
    """Проверка должна уметь объяснить, почему она не применилась."""
    unmet = PLAIN_REGRESSION.unmet(
        frozenset({Premise.BINARY_TARGET, Premise.PROCESS, Premise.UNIVERSAL})
    )

    assert unmet == frozenset({Premise.BINARY_TARGET, Premise.PROCESS})


def test_unsupported_task_is_refused_explicitly() -> None:
    """Отказать явно лучше, чем выдать неверный результат молча."""
    with pytest.raises(UnsupportedTask, match="реализации для неё нет"):
        require_supported(PLAIN_REGRESSION)


def test_supported_task_passes() -> None:
    require_supported(OLIST_LIKE)


# --- обход блокировок -----------------------------------------------------


def test_blocking_without_override_raises() -> None:
    ledger = OverrideLedger()

    with pytest.raises(Blocked, match="A8"):
        ledger.enforce("A8", "join без объявленной грануляции")


def test_block_message_explains_how_to_override() -> None:
    ledger = OverrideLedger()

    with pytest.raises(Blocked) as excinfo:
        ledger.enforce("P1", "выборка уже участвовала в выборе")

    assert "ledger.override" in str(excinfo.value)


def test_recorded_override_lets_the_work_continue() -> None:
    ledger = OverrideLedger()
    ledger.override("A8", reason="грануляция подтверждена владельцем данных", author="автор")

    entry = ledger.enforce("A8", "join без объявленной грануляции")

    assert entry is not None
    assert entry.requirement == "A8"


def test_override_returns_the_entry_so_it_reaches_the_report() -> None:
    """Обход, о котором не узнал отчёт, ничем не лучше тайного обхода."""
    ledger = OverrideLedger()
    ledger.override("A8", reason="грануляция подтверждена владельцем", author="автор")

    entry = ledger.enforce("A8", "деталь")

    assert str(entry) in ledger.report_section()


def test_override_of_one_requirement_does_not_cover_another() -> None:
    ledger = OverrideLedger()
    ledger.override("A8", reason="грануляция подтверждена владельцем", author="автор")

    with pytest.raises(Blocked, match="A10"):
        ledger.enforce("A10", "сравнение даты с моментом")


def test_short_reason_is_refused() -> None:
    """Отписка вместо причины делает обход неотличимым от обмана."""
    ledger = OverrideLedger()

    with pytest.raises(ValidationError):
        ledger.override("A8", reason="ок", author="автор")


def test_author_is_required() -> None:
    ledger = OverrideLedger()

    with pytest.raises(ValidationError):
        ledger.override("A8", reason="достаточно длинная причина", author="")


def test_requirement_id_must_look_like_one() -> None:
    ledger = OverrideLedger()

    with pytest.raises(ValidationError):
        ledger.override("что угодно", reason="достаточно длинная причина", author="автор")


def test_empty_ledger_is_reported_explicitly() -> None:
    """Отсутствие обходов — тоже факт, и он должен быть виден."""
    assert "Не зафиксировано" in OverrideLedger().report_section()


def test_every_override_appears_in_the_report() -> None:
    ledger = OverrideLedger()
    ledger.override("A8", reason="грануляция подтверждена владельцем", author="автор")
    ledger.override("P3", reason="дрейф признан несущественным осознанно", author="автор")

    section = ledger.report_section()

    assert section.count("- ") == 2
    assert "A8" in section and "P3" in section
