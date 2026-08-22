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
from dsx.task import TargetKind

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


def test_censored_rows_are_empty_not_zero() -> None:
    outcomes = compute_by_kind(world(), BUNDLE.outcome)
    first = outcomes[0]
    others = first.frame.filter(
        (frame_kind := first.frame["event_kind"]).is_not_null() & (frame_kind != first.kind)
    )

    assert others[LABEL].null_count() == others.height


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


def test_declaring_the_collapse_as_unintended_also_silences_it() -> None:
    """Объявление «склейка не задумана» тоже ответ: дальше решает аналитик."""
    assert CompetingKindsDeclared().run(_context(kinds_collapsed=False)) == []


def test_competing_target_kind_silences_the_check() -> None:
    """Виды различаются по существу задачи — склейки нет."""
    context = _context(target_kind=TargetKind.COMPETING)

    assert CompetingKindsDeclared().run(context) == []


def test_undeclared_collapse_blocks() -> None:
    signals = CompetingKindsDeclared().run(_context())

    assert [s.finding for s in signals] == [Finding.COMPETING_KINDS_COLLAPSED]
    assert signals[0].blocking
