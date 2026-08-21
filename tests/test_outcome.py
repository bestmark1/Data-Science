"""Контракт исхода (C3, C4, C6, A10)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from dsx.evals.world import build_world
from dsx.outcome import (
    ComparisonMode,
    MissingEventMeaning,
    OutcomeContractError,
    OutcomeDefinition,
    validate_outcome,
)
from dsx.roles import ColumnSpec, Role, Schema, TemporalKind

MISSING = {"completed": MissingEventMeaning.NOT_OCCURRED}


def definition(**overrides) -> OutcomeDefinition:
    payload = {
        "event_column": "event_at",
        "deadline_column": "deadline_on",
        "comparison": ComparisonMode.BY_DATE,
        "missing_event": MISSING,
        "estimand": "событие позже назначенного срока",
    }
    return OutcomeDefinition(**{**payload, **overrides})


def test_by_date_comparison_is_accepted() -> None:
    validate_outcome(definition(), build_world().schema)


def test_direct_comparison_of_date_and_instant_is_refused() -> None:
    with pytest.raises(OutcomeContractError, match="хранит дату"):
        validate_outcome(definition(comparison=ComparisonMode.DIRECT), build_world().schema)


def test_refusal_names_the_consequence_not_just_the_types() -> None:
    with pytest.raises(OutcomeContractError) as excinfo:
        validate_outcome(definition(comparison=ComparisonMode.DIRECT), build_world().schema)

    assert "в тот же день" in str(excinfo.value)


def test_direct_comparison_is_fine_when_granularity_matches() -> None:
    schema = Schema(
        columns=[
            ColumnSpec(name="decided_at", role=Role.DECISION_TIME, temporal=TemporalKind.INSTANT),
            ColumnSpec(name="event_on", role=Role.OUTCOME_COMPONENT, temporal=TemporalKind.DATE),
            ColumnSpec(name="deadline_on", role=Role.DEADLINE, temporal=TemporalKind.DATE),
        ]
    )

    validate_outcome(definition(event_column="event_on", comparison=ComparisonMode.DIRECT), schema)


def test_unknown_column_is_refused() -> None:
    with pytest.raises(OutcomeContractError, match="отсутствует в схеме"):
        validate_outcome(definition(event_column="nope"), build_world().schema)


def test_component_without_granularity_is_refused() -> None:
    schema = Schema(
        columns=[
            ColumnSpec(name="decided_at", role=Role.DECISION_TIME, temporal=TemporalKind.INSTANT),
            ColumnSpec(name="event_at", role=Role.OUTCOME_COMPONENT),
            ColumnSpec(name="deadline_on", role=Role.DEADLINE, temporal=TemporalKind.DATE),
        ]
    )

    with pytest.raises(OutcomeContractError, match="обязана объявить"):
        validate_outcome(definition(), schema)


def test_empty_missing_event_meaning_is_refused() -> None:
    """Склейка разных причин отсутствия в один класс добавила 11.3% ложных положительных."""
    with pytest.raises(ValidationError, match="явно"):
        definition(missing_event={})


def test_estimand_is_required() -> None:
    with pytest.raises(ValidationError):
        definition(estimand="")


def test_components_are_listed_for_leakage_control() -> None:
    """Колонки, участвующие в вычислении метки, известны явно и не могут быть признаками."""
    assert definition().components() == frozenset({"event_at", "deadline_on"})
