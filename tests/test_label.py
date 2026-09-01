"""Вычисление исхода по объявленному правилу и эмпирическая сверка (N6)."""

from __future__ import annotations

import pytest

from dsx.checks.empirical import separation
from dsx.evals.registry import BY_ID
from dsx.evals.world import World, build_world
from dsx.label import LABEL, LabelError, compute, observable, positive_rate
from dsx.outcome import (
    ComparisonMode,
    MissingEventCause,
    MissingEventMeaning,
    OutcomeDefinition,
    PositiveClass,
)
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
            positive_class=PositiveClass.EVENT_AFTER_DEADLINE,
            missing_causes=[
                MissingEventCause(
                    name="события не было",
                    meaning=MissingEventMeaning.NOT_OCCURRED,
                    status_value="completed",
                )
            ],
            event_name="событие",
            deadline_name="назначенный срок",
        )

    by_date = positive_rate(compute(world, definition(ComparisonMode.BY_DATE)))
    direct = positive_rate(compute(world, definition(ComparisonMode.DIRECT)))

    assert direct > by_date, "прямое сравнение обязано пометить больше как поздние"


def test_unknown_status_is_refused() -> None:
    """Склейка разных причин отсутствия в один класс запрещена (C5).

    Мир взят с ОТСУТСТВУЮЩИМИ событиями: только на таких строках статус и
    определяет исход. На мире, где событие есть у всех, проверка неприменима,
    и прежняя версия теста этого не замечала.
    """
    world = BY_ID["label-immaturity"].build()
    definition = OutcomeDefinition(
        event_column="event_at",
        deadline_column="deadline_on",
        comparison=ComparisonMode.BY_DATE,
        positive_class=PositiveClass.EVENT_AFTER_DEADLINE,
        missing_causes=[
            MissingEventCause(
                name="события не было",
                meaning=MissingEventMeaning.NOT_OCCURRED,
                status_value="completed",
            )
        ],
        event_name="событие",
        deadline_name="назначенный срок",
    )

    with pytest.raises(LabelError, match="не объявлено"):
        compute(world, definition)


def test_status_of_a_row_with_an_observed_event_does_not_matter() -> None:
    """Исход такой строки вычисляется напрямую; статус на него не влияет."""
    import polars as pl

    world = BY_ID["label-immaturity"].build()
    frame = world.main.filter(pl.col("event_at").is_not_null())
    world = world.replace_main(frame.with_columns(pl.lit("невиданный").alias("status")))

    compute(world, BY_ID["label-immaturity"].outcome)


def test_indistinguishable_causes_do_not_cover_a_visible_status() -> None:
    """Раз значение видно в данных, оно различимо — и требует объявления."""
    world = BY_ID["label-immaturity"].build()
    definition = OutcomeDefinition(
        event_column="event_at",
        deadline_column="deadline_on",
        comparison=ComparisonMode.BY_DATE,
        positive_class=PositiveClass.EVENT_AFTER_DEADLINE,
        missing_causes=[
            MissingEventCause(
                name="причина, неотличимая по данным",
                meaning=MissingEventMeaning.NOT_OCCURRED,
                assumption="принимается, что таких строк немного",
            )
        ],
        event_name="событие",
        deadline_name="назначенный срок",
    )

    with pytest.raises(LabelError, match="не покрывают"):
        compute(world, definition)


def test_excluded_status_leaves_the_population() -> None:
    """Исключённый объект не получает метки и не входит в знаменатель.

    Прежняя версия проверяла лишь сохранение числа строк после with_columns и
    работала на мире, где событие есть у каждой строки. Она проходила при
    полностью сломанной семантике исключения.
    """
    import polars as pl

    from dsx.label import REASON, OutcomeReason

    bundle = BY_ID["clean-excluded-from-population"]
    frame = compute(bundle.build(), bundle.outcome)
    excluded = frame.filter(pl.col(REASON) == OutcomeReason.EXCLUDED.value)

    assert excluded.height, "кейс обязан содержать исключённых"
    assert excluded[LABEL].null_count() == excluded.height, "метки у них быть не должно"
    assert observable(frame).height == frame.height - excluded.height


def test_excluded_is_not_immature() -> None:
    """Исключение из популяции и несозревший исход — разные вещи (A12)."""

    from dsx.label import REASON, OutcomeReason

    bundle = BY_ID["clean-excluded-from-population"]
    frame = compute(bundle.build(), bundle.outcome)
    reasons = set(frame[REASON].unique().to_list())

    assert OutcomeReason.EXCLUDED.value in reasons
    assert OutcomeReason.IMMATURE.value not in reasons


def test_positive_rate_ignores_excluded_rows() -> None:
    import polars as pl

    from dsx.label import REASON, OutcomeReason

    bundle = BY_ID["clean-excluded-from-population"]
    frame = compute(bundle.build(), bundle.outcome)
    by_hand = frame.filter(pl.col(REASON) != OutcomeReason.EXCLUDED.value)[LABEL].mean()

    assert positive_rate(frame) == pytest.approx(float(by_hand))


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


# --- предсказываемый вид при конкурирующих исходах --------------------------
#
# Пятнадцатый кейс: честное объявление конкуренции выключало ВОСЕМЬ проверок из
# тридцати шести по предпосылке binary_target, и вместе с ними — дрейф в
# девятнадцать процентных пунктов. Двоичная метка при конкуренции существует,
# своя у каждого вида; не хватало объявления, какой вид предсказывается.


def _competing_definition(primary_kind: str | None = None) -> OutcomeDefinition:
    return OutcomeDefinition(
        event_column="event_at",
        deadline_column="deadline_on",
        comparison=ComparisonMode.BY_DATE,
        positive_class=PositiveClass.EVENT_WITHIN_DEADLINE,
        primary_kind=primary_kind,
        event_name="событие",
        deadline_name="назначенный срок",
        missing_causes=[
            MissingEventCause(
                name="события не было",
                meaning=MissingEventMeaning.NOT_OCCURRED,
                status_value="completed",
            )
        ],
    )


def _competing_world():
    """Мир с двумя видами события: целевым и конкурирующим."""
    import polars as pl

    world = build_world()
    kinds = ["target" if index % 2 else "rival" for index in range(world.main.height)]
    frame = world.main.with_columns(pl.Series("kind", kinds))
    schema = Schema(columns=(*world.schema.columns, ColumnSpec(name="kind", role=Role.EVENT_KIND)))
    return World(frames={**world.frames, "main": frame}, schema=schema)


def test_a_rival_kind_before_the_deadline_censors_the_row() -> None:
    """Конкурент оборвал наблюдение: исход неизвестен, а не отрицателен.

    Разметить такую строку нулём значит объявить наблюдением то, чего не
    наблюдали.
    """
    world = _competing_world()

    plain = compute(world, _competing_definition())
    targeted = compute(world, _competing_definition("target"))

    assert targeted[LABEL].null_count() > plain[LABEL].null_count()


def test_naming_a_kind_absent_from_the_data_is_refused() -> None:
    """Объявление, которому в данных ничто не отвечает, — не объявление."""
    with pytest.raises(LabelError, match="в данных не встречается"):
        compute(_competing_world(), _competing_definition("нетакого"))


def test_a_predicted_kind_without_a_kind_column_is_refused() -> None:
    """Отличить целевой вид от конкурирующего нечем — значит нечем и считать."""
    with pytest.raises(LabelError, match="колонка вида события не названа"):
        compute(build_world(), _competing_definition("target"))
