"""Вычисление исхода по объявленному правилу и эмпирическая сверка (N6)."""

from __future__ import annotations

import pytest

from dsx.checks.empirical import separation
from dsx.evals.registry import BY_ID
from dsx.evals.world import build_world
from dsx.label import LABEL, LabelError, compute, positive_rate
from dsx.outcome import ComparisonMode, MissingEventMeaning, OutcomeDefinition
from dsx.roles import ColumnSpec, Role, Schema, TemporalKind

CLEAN = BY_ID["clean-baseline"]


def test_label_is_computed_for_every_observable_row() -> None:
    world = CLEAN.build()

    frame = compute(world, CLEAN.outcome)

    assert frame.height == world.main.height
    assert frame[LABEL].null_count() == 0


def test_positive_rate_is_plausible() -> None:
    rate = positive_rate(compute(CLEAN.build(), CLEAN.outcome))

    assert 0.0 < rate < 0.5


def test_by_date_and_direct_comparison_differ() -> None:
    """Ровно эта разница испортила 16.5% меток на этапе 0."""
    schema = Schema(
        columns=[
            ColumnSpec(name="decided_at", role=Role.DECISION_TIME, temporal=TemporalKind.INSTANT),
            ColumnSpec(name="event_at", role=Role.OUTCOME_COMPONENT, temporal=TemporalKind.DATE),
            ColumnSpec(name="deadline_on", role=Role.DEADLINE, temporal=TemporalKind.DATE),
            ColumnSpec(name="status", role=Role.STATUS),
        ]
    )
    world = build_world()
    world = type(world)(frames=world.frames, schema=schema)

    def definition(mode: ComparisonMode) -> OutcomeDefinition:
        return OutcomeDefinition(
            event_column="event_at",
            deadline_column="deadline_on",
            comparison=mode,
            missing_event={"completed": MissingEventMeaning.NOT_OCCURRED},
            estimand="событие позже срока",
        )

    by_date = positive_rate(compute(world, definition(ComparisonMode.BY_DATE)))
    direct = positive_rate(compute(world, definition(ComparisonMode.DIRECT)))

    assert direct > by_date, "прямое сравнение обязано пометить больше как поздние"


def test_unknown_status_is_refused() -> None:
    """Склейка разных причин отсутствия в один класс запрещена (C5)."""
    world = BY_ID["post-treatment-missingness"].build()
    definition = OutcomeDefinition(
        event_column="event_at",
        deadline_column="deadline_on",
        comparison=ComparisonMode.BY_DATE,
        missing_event={"completed": MissingEventMeaning.NOT_OCCURRED},
        estimand="событие позже срока",
    )

    with pytest.raises(LabelError, match="не объявлено"):
        compute(world, definition)


def test_excluded_status_leaves_the_population() -> None:
    world = BY_ID["post-treatment-missingness"].build()
    frame = compute(world, BY_ID["post-treatment-missingness"].outcome)

    assert frame.height == world.main.height, "события есть у всех, никто не выпадает"


def test_label_is_returned_separately_from_the_data() -> None:
    """Исход — производная величина; хранить его рядом с фактами значит смешивать."""
    world = CLEAN.build()

    compute(world, CLEAN.outcome)

    assert LABEL not in world.main.columns


# --- сила связи -----------------------------------------------------------


def test_separation_is_zero_for_noise() -> None:
    import polars as pl

    frame = compute(CLEAN.build(), CLEAN.outcome)
    noise = pl.Series("noise", range(frame.height))

    assert separation(noise, frame[LABEL]) < 0.1


def test_separation_is_maximal_for_a_feature_containing_the_answer() -> None:
    bundle = BY_ID["feature-falsely-declared-available"]
    frame = compute(bundle.build(), bundle.outcome)

    assert separation(frame["overrun_days"], frame[LABEL]) > 0.45


def test_separation_needs_enough_rows() -> None:
    import polars as pl

    tiny = pl.Series("x", [1, 2, 3])
    labels = pl.Series("y", [0, 1, 0])

    assert separation(tiny, labels) is None


def test_separation_is_none_for_a_constant() -> None:
    import polars as pl

    frame = compute(CLEAN.build(), CLEAN.outcome)
    constant = pl.Series("c", [1] * frame.height)

    assert separation(constant, frame[LABEL]) is None
