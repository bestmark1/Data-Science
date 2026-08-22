"""Конкурирующие исходы: цензура вместо нуля (остаток F-8)."""

from __future__ import annotations

import pytest

from dsx.checks.base import Context
from dsx.checks.drift import CompetingKindsDeclared
from dsx.competing import CompetingError, compute_by_kind, kinds_in
from dsx.evals import injectors as inj
from dsx.evals.case import Finding
from dsx.evals.registry import BY_ID
from dsx.evals.world import build_world
from dsx.label import LABEL
from dsx.task import Simultaneity, TargetKind

BUNDLE = BY_ID["competing-kinds-collapsed"]


def world():
    return BUNDLE.build()


def test_kinds_are_read_from_the_declared_column() -> None:
    assert kinds_in(world()) == ("kind1", "kind2", "kind3", "kind4")


def test_binary_world_has_no_kinds() -> None:
    assert kinds_in(build_world()) == ()


def test_outcome_is_computed_for_every_kind() -> None:
    outcomes = compute_by_kind(world(), BUNDLE.outcome)

    assert [o.kind for o in outcomes] == ["kind1", "kind2", "kind3", "kind4"]


def test_a_competing_event_censors_instead_of_being_a_negative() -> None:
    """Отказ одного компонента не наблюдение об исправности другого."""
    outcomes = compute_by_kind(world(), BUNDLE.outcome)

    assert all(o.censored_by_others > 0 for o in outcomes)
    for outcome in outcomes:
        assert outcome.observable < outcome.frame.height


def test_competing_event_inside_the_deadline_censors() -> None:
    """Конкурирующее событие в горизонте обрывает наблюдение за целевым видом."""
    import polars as pl

    first = compute_by_kind(world(), BUNDLE.outcome)[0]
    inside = first.frame.filter(
        pl.col("event_kind").is_not_null()
        & (pl.col("event_kind") != first.kind)
        & (pl.col("event_at") <= pl.col("deadline_on"))
    )

    assert inside.height
    assert inside[LABEL].null_count() == inside.height


def test_competing_event_after_the_deadline_does_not_censor() -> None:
    """К концу срока уже видно, что целевого события не было."""
    import polars as pl

    first = compute_by_kind(world(), BUNDLE.outcome)[0]
    outside = first.frame.filter(
        pl.col("event_kind").is_not_null()
        & (pl.col("event_kind") != first.kind)
        & (pl.col("event_at") > pl.col("deadline_on"))
    )

    assert outside.height, "кейс обязан содержать конкурентов за горизонтом"
    assert outside[LABEL].null_count() == 0


def test_collapse_loses_more_than_it_looks() -> None:
    """Склейка прячет и конкуренцию, и цензуру: обе видны только по видам."""
    outcomes = compute_by_kind(world(), BUNDLE.outcome)

    total_positive = sum(o.positives for o in outcomes)
    censored = sum(o.censored_by_others for o in outcomes)

    assert censored > total_positive, "цензуры больше, чем положительных исходов"


def test_single_kind_is_not_competition() -> None:
    single = inj.competing_event_kinds(build_world(), kinds=1)

    with pytest.raises(CompetingError, match="конкуренции нет"):
        compute_by_kind(single, BUNDLE.outcome)


def test_undeclared_kinds_column_is_refused() -> None:
    with pytest.raises(CompetingError, match="не названа"):
        compute_by_kind(build_world(), BUNDLE.outcome)


# --- проверка N12 ----------------------------------------------------------


def _context(**task_overrides) -> Context:
    import sys

    sys.path.insert(0, "tests")
    from harness import context_for

    base = context_for(BUNDLE)
    return Context(
        base.world, base.outcome, base.task.model_copy(update=task_overrides), base.split
    )


def test_declared_collapse_silences_the_check() -> None:
    """Ответ дан: вмешательство одно на все виды, различать незачем."""
    assert CompetingKindsDeclared().run(_context(kinds_collapsed=True)) == []


def test_declaring_the_collapse_as_unintended_does_not_silence_it() -> None:
    """Признание «склейка не задумана» — не ответ, а описание находки."""
    signals = CompetingKindsDeclared().run(_context(kinds_collapsed=False))

    assert [s.finding for s in signals] == [Finding.COMPETING_KINDS_COLLAPSED]
    assert "НЕнамеренной" in signals[0].detail


def test_competing_target_kind_still_needs_the_simultaneity_answer() -> None:
    """Склейки нет, но допущение «побеждает ровно один» тоже надо объявить (F-12)."""
    signals = CompetingKindsDeclared().run(_context(target_kind=TargetKind.COMPETING))

    assert [s.finding for s in signals] == [Finding.SIMULTANEITY_UNDECLARED]


def test_competing_with_declared_simultaneity_is_silent() -> None:
    context = _context(target_kind=TargetKind.COMPETING, simultaneous_kinds=Simultaneity.OWN_KIND)

    assert CompetingKindsDeclared().run(context) == []


def test_undeclared_collapse_blocks() -> None:
    signals = CompetingKindsDeclared().run(_context())

    assert [s.finding for s in signals] == [Finding.COMPETING_KINDS_COLLAPSED]
    assert signals[0].blocking


# --- одновременность: объявление, которое можно сверить --------------------


def test_naming_an_absent_kind_as_simultaneous_is_refused() -> None:
    """Объявление описывает то, чего не происходит."""
    context = _context(
        target_kind=TargetKind.COMPETING,
        simultaneous_kinds=Simultaneity.OWN_KIND,
        simultaneous_kind_value="kind_which_is_absent",
    )

    signals = CompetingKindsDeclared().run(context)

    assert [s.finding for s in signals] == [Finding.SIMULTANEITY_UNDECLARED]
    assert "в данных его нет" in signals[0].detail


def test_claiming_impossibility_against_a_present_kind_is_refused() -> None:
    context = _context(
        target_kind=TargetKind.COMPETING,
        simultaneous_kinds=Simultaneity.NOT_POSSIBLE,
        simultaneous_kind_value="kind1",
    )

    signals = CompetingKindsDeclared().run(context)

    assert [s.finding for s in signals] == [Finding.SIMULTANEITY_UNDECLARED]
    assert "противоречит данным" in signals[0].detail


def test_naming_a_present_kind_as_simultaneous_passes() -> None:
    context = _context(
        target_kind=TargetKind.COMPETING,
        simultaneous_kinds=Simultaneity.OWN_KIND,
        simultaneous_kind_value="kind1",
    )

    assert CompetingKindsDeclared().run(context) == []


def test_excluded_simultaneity_removes_those_rows_from_the_population() -> None:
    """Объявление, ничего не меняющее в вычислении, — украшение."""
    with_all = compute_by_kind(world(), BUNDLE.outcome)
    without = compute_by_kind(world(), BUNDLE.outcome, exclude_kind="kind1")

    assert [o.kind for o in without] == ["kind2", "kind3", "kind4"]
    assert without[0].frame.height < with_all[0].frame.height
