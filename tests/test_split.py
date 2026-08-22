"""Временной сплит, незрелость метки и пересечение сущностей (C7, A12, N2)."""

from __future__ import annotations

import datetime as dt

import pytest

from dsx.checks.base import Context
from dsx.evals.registry import BY_ID
from dsx.label import LABEL, compute, observable
from dsx.split import (
    KNOWN_AT,
    SplitResult,
    Window,
    entity_overlap,
    positive_rates,
    split_by_windows,
    with_label_known_at,
)

CLEAN = BY_ID["clean-baseline"]


def windows_for(world, count: int = 3) -> list[Window]:
    lo = world.main["decided_at"].min()
    return [
        Window(f"w{i}", lo + dt.timedelta(days=300 + i * 45), lo + dt.timedelta(days=345 + i * 45))
        for i in range(count)
    ]


def snapshot_for(world) -> dt.datetime:
    return max(world.main["decided_at"].max(), world.main["event_at"].max())


def test_label_is_known_at_the_event_when_it_happened_in_time() -> None:
    world = CLEAN.build()

    frame = with_label_known_at(world, CLEAN.outcome, snapshot_for(world))
    on_time = frame.filter(frame[LABEL] == 0)

    assert (on_time[KNOWN_AT] == on_time["event_at"]).all()


def test_label_is_known_at_the_deadline_when_the_event_was_late() -> None:
    """Ждать фактического наступления не нужно: срок истёк — ответ известен."""
    world = CLEAN.build()

    frame = with_label_known_at(world, CLEAN.outcome, snapshot_for(world))
    late = frame.filter(frame[LABEL] == 1)

    assert (late[KNOWN_AT] > late["deadline_on"]).all()
    assert (late[KNOWN_AT] <= late["event_at"]).all()


def test_training_excludes_rows_known_only_after_the_cutoff() -> None:
    """Объект, решённый до отсечки, но узнанный после, — знание будущего."""
    world = CLEAN.build()
    windows = windows_for(world)

    result = split_by_windows(world, CLEAN.outcome, windows, snapshot_for(world))
    train = result.part("w0").train

    assert (train[KNOWN_AT] < windows[0].start).all()
    assert result.dropped_not_yet_known > 0


def test_evaluation_window_holds_only_decisions_inside_it() -> None:
    world = CLEAN.build()
    windows = windows_for(world)

    part = split_by_windows(world, CLEAN.outcome, windows, snapshot_for(world)).part("w1")

    assert (part.evaluate["decided_at"] >= windows[1].start).all()
    assert (part.evaluate["decided_at"] < windows[1].stop).all()


def test_unobservable_outcome_stays_empty_rather_than_counted_as_late() -> None:
    """Отсутствие события означает исход только когда срок уже истёк."""
    bundle = BY_ID["label-immaturity"]
    world = bundle.build()

    frame = compute(world, bundle.outcome, snapshot_for(world))

    assert frame[LABEL].null_count() > 0
    assert observable(frame).height < frame.height


def test_immature_rows_are_counted_per_window() -> None:
    bundle = BY_ID["label-immaturity"]
    world = bundle.build()

    result = split_by_windows(world, bundle.outcome, windows_for(world), snapshot_for(world))

    assert result.immature, "незрелость должна быть замечена, а не отброшена молча"


def test_clean_world_has_no_immaturity_in_well_placed_windows() -> None:
    world = CLEAN.build()

    result = split_by_windows(world, CLEAN.outcome, windows_for(world), snapshot_for(world))

    assert result.immature == {}


def test_entity_overlap_is_absent_when_each_object_appears_once() -> None:
    world = CLEAN.build()
    result = split_by_windows(world, CLEAN.outcome, windows_for(world), snapshot_for(world))

    assert entity_overlap(result.parts, world) == {}


def test_entity_overlap_is_found_when_objects_repeat() -> None:
    bundle = BY_ID["entity-overlap"]
    world = bundle.build()

    result = split_by_windows(world, bundle.outcome, windows_for(world), snapshot_for(world))

    assert entity_overlap(result.parts, world)


def test_positive_rate_is_reported_per_window() -> None:
    """На этапе 0 она менялась от 1.95% до 20.8% и ломала калибровку и порог."""
    world = CLEAN.build()
    result = split_by_windows(world, CLEAN.outcome, windows_for(world), snapshot_for(world))

    rates = positive_rates(result.parts)

    assert len(rates) == 3
    assert all(0.0 < r < 1.0 for r in rates.values())


def test_windows_without_data_are_tolerated() -> None:
    world = CLEAN.build()
    far = [Window("empty", dt.datetime(2100, 1, 1), dt.datetime(2100, 2, 1))]

    result = split_by_windows(world, CLEAN.outcome, far, snapshot_for(world))

    assert result.part("empty").evaluate.height == 0


def test_overlap_of_decision_units_is_leakage_even_for_recurring_objects() -> None:
    """Долгоживущий объект по обе стороны нормален; повтор решения — нет."""
    from dsx.roles import Role
    from harness import context_for

    context = context_for(BY_ID["entity-overlap"])
    assert context.split is not None

    assert entity_overlap(context.split.parts, context.world, role=Role.ENTITY_ID)


def test_reserved_sample_is_disjoint_from_every_window() -> None:
    """F-9: окна вложены и пересекаются, резерв не пересекается ни с чем."""
    from harness import context_for

    context = context_for(BY_ID["clean-baseline"])
    split = context.split
    assert split.reserved is not None and split.reserved.height

    reserved = set(split.reserved["entity_id"].to_list())
    for part in split.parts:
        assert not reserved & set(part.train["entity_id"].to_list())
        assert not reserved & set(part.evaluate["entity_id"].to_list())


def test_windows_reaching_past_the_reserve_are_refused() -> None:
    """Резерв, в который заходит окно, независимым не является."""
    import datetime as dt

    from harness import context_for

    bundle = BY_ID["clean-baseline"]
    world = bundle.build()
    lo = world.main["decided_at"].min()
    windows = [Window("w", lo + dt.timedelta(days=300), lo + dt.timedelta(days=500))]

    with pytest.raises(ValueError, match="заходят за границу резерва"):
        split_by_windows(
            world,
            bundle.outcome,
            windows,
            world.main["decided_at"].max(),
            lo + dt.timedelta(days=480),
        )
    assert context_for(bundle).split is not None


def test_missing_reserve_blocks() -> None:
    from dsx.checks.split_checks import ReservedMeasurementSample
    from dsx.evals.case import Finding
    from harness import context_for

    context = context_for(BY_ID["clean-baseline"])
    without = SplitResult(parts=context.split.parts)
    stripped = Context(context.world, context.outcome, context.task, without)

    signals = ReservedMeasurementSample().run(stripped)

    assert [s.finding for s in signals] == [Finding.NO_RESERVED_MEASUREMENT_SAMPLE]
    assert signals[0].blocking
