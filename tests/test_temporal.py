"""Совместимость временных колонок (A10)."""

from __future__ import annotations

import pytest

from dsx.roles import ColumnSpec, Role, TemporalKind
from dsx.temporal import IncompatibleComparison, check_comparable

INSTANT = ColumnSpec(name="delivered_at", role=Role.EVENT_TIME, temporal=TemporalKind.INSTANT)
DATE = ColumnSpec(name="promised_on", role=Role.DEADLINE, temporal=TemporalKind.DATE)
OTHER_DATE = ColumnSpec(name="due_on", role=Role.DEADLINE, temporal=TemporalKind.DATE)


def test_same_granularity_is_comparable() -> None:
    check_comparable(DATE, OTHER_DATE)
    check_comparable(INSTANT, INSTANT)


@pytest.mark.parametrize("pair", [(INSTANT, DATE), (DATE, INSTANT)])
def test_mixed_granularity_is_refused_both_ways(pair) -> None:
    with pytest.raises(IncompatibleComparison, match="хранит дату"):
        check_comparable(*pair)


def test_error_names_both_columns_and_the_consequence() -> None:
    with pytest.raises(IncompatibleComparison) as excinfo:
        check_comparable(INSTANT, DATE)

    message = str(excinfo.value)
    assert "promised_on" in message and "delivered_at" in message
    assert "в тот же день" in message
