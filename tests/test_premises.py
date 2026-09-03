"""Сверка объявленных предпосылок с данными (F-4)."""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

from dsx.checks import ALL_CHECKS, Context, run_checks
from dsx.evals.case import Finding
from dsx.evals.registry import BY_ID
from dsx.evals.world import World
from dsx.premises import Premise, unverifiable, verify
from dsx.roles import ColumnSpec, Role, Schema, TemporalKind
from dsx.task import OutcomeTiming, TargetKind, TaskSpec
from harness import task_for

CLEAN = BY_ID["clean-baseline"]


def task(**overrides) -> TaskSpec:
    payload = {
        "target_kind": TargetKind.BINARY,
        "outcome_timing": OutcomeTiming.DELAYED,
        "has_process": True,
        "is_stream": True,
    }
    return TaskSpec(**{**payload, **overrides})


def test_honest_declaration_has_no_discrepancies() -> None:
    world = CLEAN.build()

    assert verify(world, task_for(world)) == []


def test_denying_an_existing_process_is_caught() -> None:
    """Объявив отсутствие процесса, можно выключить проверки ради тишины."""
    world = CLEAN.build()

    discrepancies = verify(world, task(has_process=False))

    assert [d.premise for d in discrepancies] == [Premise.PROCESS]
    assert discrepancies[0].observed is True


def test_claiming_a_stream_that_is_not_there_is_caught() -> None:
    """Признак потока — ритм, а не плотность: разрежаем НЕРАВНОМЕРНО."""
    import numpy as np

    world = CLEAN.build()
    days = world.main.select(pl.col("decided_at").dt.date().alias("day")).unique()["day"]
    rng = np.random.default_rng(11)
    kept = set(rng.choice(days.to_list(), size=days.len() // 6, replace=False).tolist())
    thinned = world.replace_main(
        world.main.filter(pl.col("decided_at").dt.date().is_in(list(kept)))
    )

    discrepancies = verify(thinned, task(is_stream=True))

    assert any(d.premise is Premise.STREAM for d in discrepancies)


def test_regular_but_sparse_collection_is_still_a_stream() -> None:
    """Сбор по первым числам месяца ритмичен, и проверки полноты периода к нему
    применимы. Прежняя эвристика требовала девяноста процентов календарных дней
    и выключала эти проверки на любом ритме реже ежедневного."""
    world = CLEAN.build()
    monthly = world.replace_main(world.main.filter(pl.col("decided_at").dt.day() == 1))

    assert not [d for d in verify(monthly, task(is_stream=True)) if d.premise is Premise.STREAM]


def test_absent_status_column_means_no_process() -> None:
    """Второй кейс: статуса нет ни в одной из пяти таблиц."""
    schema = Schema(
        columns=[
            ColumnSpec(name="decided_at", role=Role.DECISION_TIME, temporal=TemporalKind.INSTANT),
            ColumnSpec(name="value", role=Role.FEATURE),
        ]
    )
    frame = pl.DataFrame(
        {"decided_at": [dt.datetime(2024, 1, 1), dt.datetime(2024, 1, 2)], "value": [1, 2]}
    )
    world = World(frames={"main": frame}, schema=schema)

    assert verify(world, task(has_process=True, is_stream=True))[0].premise is Premise.PROCESS


def test_single_valued_status_still_counts_as_a_process() -> None:
    """Первая версия правила требовала двух значений и переставала видеть
    конфликт статуса с событием там, где статус один."""
    world = CLEAN.build()

    assert task_for(world).has_process


def test_discrepancy_message_states_both_sides() -> None:
    world = CLEAN.build()

    message = str(verify(world, task(has_process=False))[0])

    assert "объявлено False" in message and "по данным True" in message


def test_unverifiable_premises_are_named_not_hidden() -> None:
    """Тип таргета из данных не выводится, и это сказано прямо."""
    remaining = unverifiable()

    assert Premise.BINARY_TARGET in remaining
    assert Premise.PROCESS not in remaining


def test_check_blocks_on_mismatch() -> None:
    bundle = CLEAN
    world = bundle.build()

    report = run_checks(
        list(ALL_CHECKS), Context(world, bundle.outcome, task(has_process=False), None)
    )

    assert Finding.PREMISE_MISMATCH in report.findings
    assert report.blocking


def test_check_is_silent_when_declaration_matches() -> None:
    bundle = CLEAN
    world = bundle.build()

    report = run_checks(list(ALL_CHECKS), Context(world, bundle.outcome, task_for(world), None))

    assert Finding.PREMISE_MISMATCH not in report.findings


@pytest.mark.parametrize("premise", [Premise.PROCESS, Premise.STREAM])
def test_verifiable_premises_are_actually_verified(premise: Premise) -> None:
    assert premise not in unverifiable()
