"""Вычисление исхода по объявленному правилу.

Метка считается так, как объявлено в контракте, а не выражением, спрятанным
внутри кода анализа. Пока сравнение скрыто в коде, проверить его нечем.

Строки с ненаблюдаемым исходом не отсеиваются здесь: их доля в оценочном окне —
самостоятельная находка, а молча отброшенное окно смещается в сторону объектов
с короткими сроками.
"""

from __future__ import annotations

import datetime as dt

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
        declared = set(definition.status_meaning())
        present = set(frame[status].drop_nulls().unique().to_list())
        unknown = present - declared
        if unknown and not definition.indistinguishable:
            raise LabelError(
                f"для статусов {sorted(unknown)!r} не объявлено, что означает "
                "отсутствие события, и неразличимых причин тоже не объявлено"
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

    if status is None:
        missing_label = meaning_expr(fallback_meaning)
    else:
        missing_label = meaning_expr(fallback_meaning)
        for value, meaning in definition.status_meaning().items():
            missing_label = (
                pl.when(pl.col(status) == value)
                .then(meaning_expr(meaning))
                .otherwise(missing_label)
            )

    return frame.with_columns(
        pl.when(pl.col(definition.event_column).is_null())
        .then(missing_label)
        .otherwise(observed.cast(pl.Int8))
        .cast(pl.Int8)
        .alias(LABEL)
    )


def observable(frame: pl.DataFrame) -> pl.DataFrame:
    """Строки с наблюдаемым исходом."""
    return frame.filter(pl.col(LABEL).is_not_null())


def positive_rate(frame: pl.DataFrame) -> float:
    return float(observable(frame)[LABEL].mean())
