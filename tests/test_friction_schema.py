"""Схема журнала не даёт заполнить запись бессодержательно (R6, R7)."""

from __future__ import annotations

import datetime as dt

import pytest
from pydantic import ValidationError

from dsx.contracts.friction import FrictionRecord

VALID = {
    "step": "03_grain_joins",
    "did_manually": "Считал заказы после join с позициями, получил задвоение сумм.",
    "minutes": 40,
    "tool_should_have": "Предупредить о fanout при join таблиц разной грануляции.",
    "outcome": "Обошёл агрегацией до join.",
}


def test_valid_record_passes() -> None:
    record = FrictionRecord(**VALID)

    assert record.step == "03_grain_joins"
    assert record.scope == "portable"
    assert record.created_at == dt.date.today()


def test_record_without_tool_should_have_is_rejected() -> None:
    payload = {k: v for k, v in VALID.items() if k != "tool_should_have"}

    with pytest.raises(ValidationError) as excinfo:
        FrictionRecord(**payload)

    assert "tool_should_have" in str(excinfo.value)


@pytest.mark.parametrize("field", ["step", "did_manually", "tool_should_have", "outcome"])
def test_empty_text_field_is_rejected(field: str) -> None:
    with pytest.raises(ValidationError):
        FrictionRecord(**{**VALID, field: ""})


@pytest.mark.parametrize("minutes", [0, -5])
def test_non_positive_duration_is_rejected(minutes: int) -> None:
    with pytest.raises(ValidationError):
        FrictionRecord(**{**VALID, "minutes": minutes})


def test_workplace_specifics_must_be_local() -> None:
    with pytest.raises(ValidationError, match="scope='local'"):
        FrictionRecord(**{**VALID, "contains_workplace_specifics": True})


def test_workplace_specifics_accepted_when_local() -> None:
    record = FrictionRecord(**{**VALID, "contains_workplace_specifics": True, "scope": "local"})

    assert record.scope == "local"


def test_unknown_field_is_rejected() -> None:
    with pytest.raises(ValidationError):
        FrictionRecord(**{**VALID, "severity": "high"})


def test_schema_exports_to_json_schema() -> None:
    schema = FrictionRecord.model_json_schema()

    assert "tool_should_have" in schema["required"]
    assert "step" in schema["required"]
