"""Вычисление исхода по объявленному правилу.

Метка считается так, как объявлено в контракте, а не выражением, спрятанным
внутри кода анализа. Пока сравнение скрыто в коде, проверить его нечем.

Строки с ненаблюдаемым исходом не отсеиваются здесь: их доля в оценочном окне —
самостоятельная находка, а молча отброшенное окно смещается в сторону объектов
с короткими сроками.
"""

from __future__ import annotations

import datetime as dt
from enum import StrEnum

import polars as pl

from dsx.evals.world import World
from dsx.outcome import (
    ComparisonMode,
    MissingEventMeaning,
    OutcomeDefinition,
    PositiveClass,
    validate_outcome,
)
from dsx.roles import Role


class LabelError(Exception):
    """Исход невозможно вычислить по объявленному контракту."""


LABEL = "__outcome"
"""Имя служебной колонки. Двойное подчёркивание, чтобы не столкнуться с данными."""

REASON = "__outcome_reason"
"""Почему исход не размечен. Пустая метка бывает трёх разных вещей."""


class OutcomeReason(StrEnum):
    """Отчего строка не получила метку.

    Три причины склеивались в один `null`, и проверка незрелости считала
    незрелыми исключённых из популяции. Различие существенное: незрелость
    означает короткий период наблюдения, исключение — что объект не
    предполагался к обработке, цензура — что наблюдение оборвалось.
    """

    OBSERVED = "observed"
    """Исход наблюдён, метка есть."""

    IMMATURE = "immature"
    """Срок ещё не истёк: исход появится позже."""

    EXCLUDED = "excluded"
    """Объект не предполагался к обработке."""

    CENSORED = "censored"
    """Наблюдение оборвалось, исход неизвестен навсегда."""

    CONFLATED = "conflated"
    """Неразличимые причины имеют разный смысл: разметить нельзя."""

    BEFORE_DECISION = "before_decision"
    """Событие датировано раньше момента решения.

    Исходом этого решения оно быть не может: решение ещё не принято. Правило
    сравнения со сроком проверяло только верхнюю границу, и такая строка молча
    становилась положительной.
    """


def compute(
    world: World, definition: OutcomeDefinition, snapshot: dt.datetime | None = None
) -> pl.DataFrame:
    """Вернуть таблицу с колонкой исхода. Ненаблюдаемый исход остаётся пустым.

    Отсутствие события означает исход только тогда, когда срок УЖЕ ИСТЁК. Пока
    срок не наступил, исход не наблюдаем. Без снимка границей считается
    последнее наблюдённое событие.

    Причина отсутствия определяется по объявленным причинам: различимые — по
    значению статуса, остальные попадают в общую группу. Если неразличимые
    причины имеют разный смысл, строки этой группы разметить нельзя.
    """
    validate_outcome(definition, world.schema)
    frame = world.main

    event = pl.col(definition.event_column)
    deadline = pl.col(definition.deadline_column)
    if definition.comparison is ComparisonMode.BY_DATE:
        event, deadline = event.dt.date(), deadline.dt.date()

    if snapshot is None:
        observed = frame[definition.event_column].max()
        fallback = frame[world.schema.decision_time.name].max()
        snapshot = observed if observed is not None else fallback
    horizon = pl.lit(snapshot).cast(pl.Datetime)
    if definition.comparison is ComparisonMode.BY_DATE:
        horizon = horizon.dt.date()

    statuses = world.schema.by_role(Role.STATUS)
    status = statuses[0].name if statuses else None

    if status is not None:
        # Проверяются только строки БЕЗ события: у остальных исход вычисляется
        # напрямую, и статус на него не влияет.
        missing = frame.filter(pl.col(definition.event_column).is_null())
        declared = set(definition.status_meaning())
        present = set(missing[status].drop_nulls().unique().to_list())
        unknown = present - declared
        if unknown:
            # Неразличимые причины не покрывают ВИДИМЫЙ статус: раз значение
            # присутствует в данных, оно по определению различимо, и молча
            # отправлять его в общую группу значит терять объявленное различие.
            raise LabelError(
                f"для статусов {sorted(unknown)!r} не объявлено, что означает "
                "отсутствие события. Статус виден в данных, поэтому неразличимые "
                "причины его не покрывают"
            )

    # Смысл для строк, не отнесённых ни к одной различимой причине.
    fallback_meaning = definition.fallback_meaning()

    after = definition.positive_class is PositiveClass.EVENT_AFTER_DEADLINE
    observed = (event > deadline) if after else (event <= deadline)

    def meaning_expr(meaning: MissingEventMeaning | None) -> pl.Expr:
        """Метка для строки без события при известном смысле отсутствия.

        Событие, не случившееся никогда, положительно при одном направлении и
        отрицательно при другом. Отсутствие доставки к сроку — опоздание;
        отсутствие отказа в горизонте — исправная работа.
        """
        if meaning is MissingEventMeaning.NOT_OCCURRED:
            # Отсутствие — наблюдение, но только после того, как срок истёк.
            return pl.when(deadline < horizon).then(1 if after else 0).otherwise(None)
        # Цензура и исключение из популяции: исход не наблюдаем.
        return pl.lit(None)

    def reason_expr(meaning: MissingEventMeaning | None) -> pl.Expr:
        """Отчего строка без события не получила метку."""
        if meaning is MissingEventMeaning.NOT_OCCURRED:
            return (
                pl.when(deadline < horizon)
                .then(pl.lit(OutcomeReason.OBSERVED.value))
                .otherwise(pl.lit(OutcomeReason.IMMATURE.value))
            )
        if meaning is MissingEventMeaning.EXCLUDED:
            return pl.lit(OutcomeReason.EXCLUDED.value)
        if meaning is MissingEventMeaning.UNOBSERVED:
            return pl.lit(OutcomeReason.CENSORED.value)
        return pl.lit(OutcomeReason.CONFLATED.value)

    missing_label = meaning_expr(fallback_meaning)
    missing_reason = reason_expr(fallback_meaning)
    if status is not None:
        for value, meaning in definition.status_meaning().items():
            matches = pl.col(status) == value
            missing_label = pl.when(matches).then(meaning_expr(meaning)).otherwise(missing_label)
            missing_reason = pl.when(matches).then(reason_expr(meaning)).otherwise(missing_reason)

    # Нижняя граница: событие не может быть исходом решения, принятого позже.
    decided = pl.col(world.schema.decision_time.name)
    premature = pl.col(definition.event_column).is_not_null() & (
        pl.col(definition.event_column) < decided
    )

    absent = pl.col(definition.event_column).is_null()
    labelled = frame.with_columns(
        pl.when(premature)
        .then(None)
        .when(absent)
        .then(missing_label)
        .otherwise(observed.cast(pl.Int8))
        .cast(pl.Int8)
        .alias(LABEL),
        pl.when(premature)
        .then(pl.lit(OutcomeReason.BEFORE_DECISION.value))
        .when(absent)
        .then(missing_reason)
        .otherwise(pl.lit(OutcomeReason.OBSERVED.value))
        .alias(REASON),
    )
    if definition.primary_kind is None:
        return labelled
    return censor_competing_kinds(labelled, definition, world, definition.primary_kind)


def censor_competing_kinds(
    frame: pl.DataFrame, definition: OutcomeDefinition, world: World, kind: str
) -> pl.DataFrame:
    """Обнулить в пустоту строки, где раньше срока наступил ДРУГОЙ вид события.

    Наступление конкурирующего вида обрывает наблюдение за целевым: животное
    усыпили — и что стало бы с его усыновлением, неизвестно. Разметить такую
    строку нулём значит объявить наблюдением то, чего не наблюдали.

    Конкурент ПОСЛЕ срока наблюдению не мешает: к концу срока уже видно, что
    целевого события не было.
    """
    columns = world.schema.by_role(Role.EVENT_KIND)
    if not columns:
        raise LabelError(
            f"объявлен предсказываемый вид {kind!r}, но колонка вида события "
            "не названа: отличить целевой вид от конкурирующего нечем"
        )
    kind_column = columns[0].name
    present = set(frame[kind_column].drop_nulls().unique().to_list())
    if kind not in present:
        raise LabelError(
            f"предсказываемый вид {kind!r} в данных не встречается; есть {sorted(present)!r}"
        )
    censored = (
        pl.col(kind_column).is_not_null()
        & (pl.col(kind_column) != kind)
        & (pl.col(definition.event_column) <= pl.col(definition.deadline_column))
    )
    return frame.with_columns(
        pl.when(censored).then(None).otherwise(pl.col(LABEL)).alias(LABEL),
        pl.when(censored)
        .then(pl.lit(OutcomeReason.CENSORED.value))
        .otherwise(pl.col(REASON))
        .alias(REASON),
    )


def observable(frame: pl.DataFrame) -> pl.DataFrame:
    """Строки с наблюдаемым исходом."""
    return frame.filter(pl.col(LABEL).is_not_null())


def positive_rate(frame: pl.DataFrame) -> float:
    return float(observable(frame)[LABEL].mean())
