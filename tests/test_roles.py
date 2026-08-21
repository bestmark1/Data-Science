"""Роли колонок и объявляемая семантика (S1, C1, C2, A10)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from dsx.roles import Availability, ColumnSpec, Role, Schema, TemporalKind

DECISION = ColumnSpec(name="decided_at", role=Role.DECISION_TIME, temporal=TemporalKind.INSTANT)


def feature(name: str, **kwargs) -> ColumnSpec:
    return ColumnSpec(name=name, role=Role.FEATURE, **kwargs)


def test_temporal_role_must_declare_granularity() -> None:
    with pytest.raises(ValidationError, match="грануляцию"):
        ColumnSpec(name="deadline", role=Role.DEADLINE)


def test_non_temporal_role_may_declare_granularity() -> None:
    """Компонент исхода часто является меткой времени, и его грануляция важна."""
    column = ColumnSpec(
        name="happened_at", role=Role.OUTCOME_COMPONENT, temporal=TemporalKind.INSTANT
    )

    assert column.temporal is TemporalKind.INSTANT


@pytest.mark.parametrize("availability", [Availability.AT_DECISION, Availability.AFTER])
def test_declared_availability_requires_a_source(availability: Availability) -> None:
    with pytest.raises(ValidationError, match="источника"):
        feature("amount", availability=availability)


def test_unknown_availability_needs_no_source() -> None:
    column = feature("amount")

    assert column.availability is Availability.UNKNOWN
    assert column.source_of_claim is None


def test_schema_requires_exactly_one_decision_time() -> None:
    with pytest.raises(ValidationError, match="ровно один"):
        Schema(columns=[feature("amount")])

    with pytest.raises(ValidationError, match="ровно один"):
        Schema(
            columns=[
                DECISION,
                ColumnSpec(name="other", role=Role.DECISION_TIME, temporal=TemporalKind.INSTANT),
            ]
        )


def test_schema_rejects_duplicate_names() -> None:
    with pytest.raises(ValidationError, match="[Дд]убликаты"):
        Schema(columns=[DECISION, feature("amount"), feature("amount")])


def test_usable_features_exclude_undeclared_availability() -> None:
    known = feature("amount", availability=Availability.AT_DECISION, source_of_claim="схема API")
    later = feature("score", availability=Availability.AFTER, source_of_claim="владелец")
    unknown = feature("limit")

    schema = Schema(columns=[DECISION, known, later, unknown])

    assert [c.name for c in schema.usable_features()] == ["amount"]


def test_outcome_component_is_never_a_usable_feature() -> None:
    component = ColumnSpec(name="delivered_on", role=Role.OUTCOME_COMPONENT)
    schema = Schema(columns=[DECISION, component])

    assert schema.usable_features() == []


def test_lookup_by_name_and_role() -> None:
    schema = Schema(columns=[DECISION, feature("amount")])

    assert schema.decision_time.name == "decided_at"
    assert schema.get("amount") is not None
    assert schema.get("missing") is None
    assert [c.name for c in schema.by_role(Role.FEATURE)] == ["amount"]
