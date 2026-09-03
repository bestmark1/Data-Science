"""Проверки данных против набора кейсов."""

from __future__ import annotations

import pytest

from dsx.checks import ALL_CHECKS, CONTRACT_CHECKS, DATA_CHECKS, Context, run_checks
from dsx.evals.case import Finding
from dsx.evals.registry import ALL, BY_ID, NEGATIVE_CONTROLS
from dsx.task import ObjectLifetime, OutcomeTiming, TargetKind, TaskSpec

FULL = TaskSpec(
    target_kind=TargetKind.BINARY,
    outcome_timing=OutcomeTiming.DELAYED,
    has_process=True,
    is_stream=True,
)
RECURRING = FULL.model_copy(update={"object_lifetime": ObjectLifetime.RECURRING})
ONE_SHOT = FULL.model_copy(update={"object_lifetime": ObjectLifetime.ONE_SHOT})
COVERED = frozenset(f for c in ALL_CHECKS for f in c.detects)

DECLARATIVE = [*CONTRACT_CHECKS, *DATA_CHECKS]
"""Проверки, читающие объявления и данные, но не сверяющие одно с другим."""


from harness import context_for  # noqa: E402


def report_for(bundle, task: TaskSpec = FULL):
    context = context_for(bundle)
    if task is not FULL:
        # Горизонт наблюдения переносится вместе с остальным: без него N18
        # отказывается работать, и подмена ЗАДАЧИ молча выключала бы проверку,
        # не имеющую к задаче отношения.
        context = Context(
            context.world, context.outcome, task, context.split, context.observed_until
        )
    return run_checks(list(ALL_CHECKS), context)


@pytest.mark.parametrize("bundle", NEGATIVE_CONTROLS, ids=lambda b: b.id)
def test_no_alarm_on_clean_worlds(bundle) -> None:
    report = report_for(bundle)

    assert report.findings & COVERED == frozenset(), [str(s) for s in report.signals]


@pytest.mark.parametrize("bundle", ALL, ids=lambda b: b.id)
def test_injected_defect_is_found_and_nothing_else(bundle) -> None:
    """Инжектор ломает ровно одно; проверки обязаны увидеть ровно это."""
    found = report_for(bundle).findings & COVERED

    assert found == bundle.case.expectation.findings & COVERED


def test_lying_declaration_escapes_declarative_checks() -> None:
    """Проверка объявлений читает то же ложное утверждение и бессильна."""
    bundle = BY_ID["feature-falsely-declared-available"]

    report = run_checks(DECLARATIVE, context_for(bundle))

    assert report.findings == frozenset()
    assert bundle.case.expectation.caught_by == frozenset({"N6"})


def test_lying_declaration_is_caught_by_the_empirical_check() -> None:
    bundle = BY_ID["feature-falsely-declared-available"]

    found = report_for(bundle).findings & COVERED

    assert found == bundle.case.expectation.findings


def test_stream_checks_are_skipped_on_static_data() -> None:
    """Проверки полноты периода бессмысленны там, где потока сбора нет."""
    bundle = BY_ID["truncated-tail"]
    static = TaskSpec(
        target_kind=TargetKind.BINARY,
        outcome_timing=OutcomeTiming.DELAYED,
        has_process=True,
        is_stream=False,
    )

    report = report_for(bundle, static)

    assert {"A6", "A7"} <= {s.requirement for s in report.skipped}
    # Ложное объявление отсутствия потока теперь само является находкой (F-4):
    # выключить проверку молча больше нельзя.
    assert report.findings == frozenset({Finding.PREMISE_MISMATCH})


def test_process_checks_are_skipped_without_a_process() -> None:
    bundle = BY_ID["status-timestamp-conflict"]
    no_process = TaskSpec(
        target_kind=TargetKind.BINARY,
        outcome_timing=OutcomeTiming.DELAYED,
        has_process=False,
        is_stream=True,
    )

    report = report_for(bundle, no_process)

    assert {"A5", "A11"} <= {s.requirement for s in report.skipped}


def test_every_data_check_declares_premises_and_findings() -> None:
    for check in DATA_CHECKS:
        assert check.premises, check.requirement
        assert check.detects, check.requirement


def test_signals_explain_the_consequence_not_just_the_fact() -> None:
    """Сообщение, называющее только факт, не помогает решить, что делать."""
    signals = report_for(BY_ID["surrogate-key-as-entity"]).signals

    assert any("история по ней будет короче" in s.detail for s in signals)


def test_surrogate_key_check_stays_quiet_without_a_natural_key() -> None:
    """Уникальность идентификатора по строкам сама по себе нормальна."""
    bundle = BY_ID["clean-baseline"]

    assert not [
        s for s in report_for(bundle).signals if s.finding.value == "surrogate_key_as_entity"
    ]


# --- жизненный цикл объекта (F-2, F-3) -------------------------------------


def test_recurring_object_does_not_trigger_the_surrogate_key_check() -> None:
    """Визит намеренно мельче машины: это устройство задачи, а не подмена ключа."""
    report = report_for(BY_ID["surrogate-key-as-entity"], task=RECURRING)

    assert not any(s.finding is Finding.SURROGATE_KEY_AS_ENTITY for s in report.signals)


def test_one_shot_declaration_contradicted_by_data_is_refused() -> None:
    """Объявление, расходящееся с данными, хуже отсутствия объявления."""
    report = report_for(BY_ID["surrogate-key-as-entity"], task=ONE_SHOT)
    signals = [s for s in report.signals if s.finding is Finding.SURROGATE_KEY_AS_ENTITY]

    assert signals and "противоречит" in signals[0].detail


def test_feature_window_overlap_ignores_the_lag() -> None:
    """Отступ сдвигает оба интервала одинаково и при сравнении сокращается."""
    from dsx.checks.split_checks import FeatureWindowOverlap
    from dsx.evals import injectors as inj
    from dsx.evals.registry import Bundle
    from dsx.evals.world import build_world
    from dsx.task import ObjectLifetime

    base = BY_ID["clean-recurring-narrow-windows"]
    bundle = Bundle(
        case=base.case,
        build=lambda: inj.declare_feature_windows(inj.repeated_object(build_world()), 0, 90),
        outcome=base.outcome,
        lifetime=ObjectLifetime.RECURRING,
    )
    context = context_for(bundle)

    assert FeatureWindowOverlap().run(context) == [], "мгновенные измерения не пересекаются"


# --- отсутствие, записанное строкой (A14) ----------------------------------


def test_sentinel_string_is_caught() -> None:
    """Ошибка возникает при чтении файла и портит все проверки разом."""
    from dsx.checks.data import SentinelAsValue

    signals = SentinelAsValue().run(context_for(BY_ID["sentinel-as-value"]))

    assert [s.finding for s in signals] == [Finding.SENTINEL_AS_VALUE]
    assert "NA" in signals[0].detail
    assert signals[0].blocking


def test_clean_world_has_no_sentinels() -> None:
    from dsx.checks.data import SentinelAsValue

    assert SentinelAsValue().run(context_for(BY_ID["clean-baseline"])) == []


def test_override_lets_a_legitimate_unknown_through() -> None:
    """«UNKNOWN» бывает законной категорией, но объявленной, а не молчаливой."""
    from dsx.checks import run_checks
    from dsx.checks.data import SentinelAsValue
    from dsx.policy import OverrideLedger

    ledger = OverrideLedger()
    ledger.override("A14", reason="UNKNOWN — законная категория справочника", author="автор")
    report = run_checks([SentinelAsValue()], context_for(BY_ID["sentinel-as-value"]), ledger)

    assert not report.blocking
    assert report.signals, "находка обязана остаться видимой, а не исчезнуть"


def test_null_status_share_is_computed_not_zeroed() -> None:
    """Сравнение с пустым значением через `==` давало null вместо истины.

    Доля статуса считалась по пустой выборке и выходила нулевой, после чего
    порог «больше нуля в полтора раза» выполнялся всегда: проверка становилась
    генератором ложных тревог на любом пропуске с пустым статусом.
    """
    import polars as pl

    from dsx.checks.data import PostTreatmentMissingness

    context = context_for(BY_ID["clean-baseline"])
    frame = context.world.main
    # Половина строк без статуса; пропуск признака ровно у них.
    half = frame.height // 2
    spoiled = frame.with_columns(
        pl.when(pl.int_range(pl.len()) < half)
        .then(None)
        .otherwise(pl.col("status"))
        .alias("status"),
        pl.when(pl.int_range(pl.len()) < half).then(None).otherwise(pl.col("size")).alias("size"),
    )
    world = context.world.replace_main(spoiled)

    signals = PostTreatmentMissingness().run(
        Context(world, context.outcome, context.task, context.split)
    )

    assert signals, "концентрация пропуска на пустом статусе — настоящая находка"
    assert "в данных 50%" in signals[0].detail, "доля пустого статуса считается, а не зануляется"
    assert "в данных 0%" not in signals[0].detail


def test_identical_rates_are_not_reported_as_a_difference() -> None:
    """При доле ровно 100% разброс равен нулю, и строгое `0 < 0` было ложным."""
    from dsx.checks.drift import TargetRateStationarity
    from dsx.evals import injectors as inj
    from dsx.evals.registry import BY_ID, Bundle
    from dsx.evals.world import build_world

    base = BY_ID["clean-baseline"]
    bundle = Bundle(
        case=base.case,
        build=lambda: inj.degenerate_outcome(build_world()),
        outcome=base.outcome,
    )
    context = context_for(bundle)

    signals = TargetRateStationarity().run(context)

    assert signals == [], [s.detail for s in signals]


# --- месячная точность (A10) ------------------------------------------------


def _outcome_with(bundle, event_kind, deadline_kind):
    """Тот же мир, но с объявленной грануляцией у двух колонок исхода."""
    from dsx.evals.registry import Bundle
    from dsx.evals.world import build_world
    from dsx.roles import Schema

    world = build_world()
    columns = []
    for column in world.schema.columns:
        if column.name == bundle.outcome.event_column:
            columns.append(column.model_copy(update={"temporal": event_kind}))
        elif column.name == bundle.outcome.deadline_column:
            columns.append(column.model_copy(update={"temporal": deadline_kind}))
        else:
            columns.append(column)
    from dsx.evals.world import World

    return Bundle(
        case=bundle.case,
        build=lambda: World(frames=world.frames, schema=Schema(columns=columns)),
        outcome=bundle.outcome,
    )


def test_month_against_day_is_blocking_even_when_compared_by_date() -> None:
    """Приведение к дате лечит расхождение дата/момент и только его."""
    from dsx.checks.contract import MixedTemporalComparison
    from dsx.evals.registry import BY_ID
    from dsx.roles import TemporalKind

    base = BY_ID["clean-baseline"]
    context = context_for(_outcome_with(base, TemporalKind.DATE, TemporalKind.MONTH))

    signals = MixedTemporalComparison().run(context)

    assert [s.finding for s in signals] == [Finding.MIXED_TEMPORAL_COMPARISON]
    assert signals[0].blocking
    assert "месяц" in signals[0].detail


def test_two_month_columns_are_reported_but_not_blocked() -> None:
    """Одинаково грубые стороны не смешиваются, но люфт остаётся."""
    from dsx.checks.contract import MixedTemporalComparison
    from dsx.evals.registry import BY_ID
    from dsx.roles import TemporalKind

    base = BY_ID["clean-baseline"]
    context = context_for(_outcome_with(base, TemporalKind.MONTH, TemporalKind.MONTH))

    signals = MixedTemporalComparison().run(context)

    assert len(signals) == 1
    assert not signals[0].blocking, "запрещать тут нечего"
    assert "люфт" in signals[0].detail


def test_matching_day_granularity_stays_silent() -> None:
    """Отрицательный контроль: одинаковая дневная точность возражений не вызывает."""
    from dsx.checks.contract import MixedTemporalComparison
    from dsx.evals.registry import BY_ID
    from dsx.roles import TemporalKind

    base = BY_ID["clean-baseline"]
    context = context_for(_outcome_with(base, TemporalKind.DATE, TemporalKind.DATE))

    assert MixedTemporalComparison().run(context) == []
