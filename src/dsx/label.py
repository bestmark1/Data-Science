"""Вычисление исхода по объявленному правилу.

Метка считается так, как объявлено в контракте, а не выражением, спрятанным
внутри кода анализа. Пока сравнение скрыто в коде, проверить его нечем — а
самая дорогая ошибка этапа 0 сидела именно в способе сравнения.

Второе следствие: строки, для которых исход не наблюдаем, отсеиваются здесь и
по объявленной причине, а не молча выпадают из-за пропуска в данных.
"""

from __future__ import annotations

import datetime as dt

import polars as pl

from dsx.evals.world import World
from dsx.outcome import ComparisonMode, MissingEventMeaning, OutcomeDefinition, validate_outcome
from dsx.roles import Role


class LabelError(Exception):
    """Исход невозможно вычислить по объявленному контракту."""


LABEL = "__outcome"
"""Имя служебной колонки. Двойное подчёркивание, чтобы не столкнуться с данными."""


def compute(
    world: World, definition: OutcomeDefinition, snapshot: dt.datetime | None = None
) -> pl.DataFrame:
    """Вернуть таблицу с колонкой исхода. Ненаблюдаемый исход остаётся пустым.

    Возвращается новая таблица, а не изменённый мир: исход — производная
    величина, и хранить её рядом с данными значит смешивать факт и вывод.

    Отсутствие события означает исход только тогда, когда срок УЖЕ ИСТЁК.
    Пока срок не наступил, исход не наблюдаем, и метка пуста. Без снимка
    границей считается последнее наблюдённое событие.

    Строки с пустой меткой не отсеиваются здесь: их доля в оценочном окне —
    самостоятельная находка (A12), а молча отброшенное окно смещается в
    сторону объектов с короткими сроками.
    """
    validate_outcome(definition, world.schema)
    frame = world.main

    event = pl.col(definition.event_column)
    deadline = pl.col(definition.deadline_column)
    if definition.comparison is ComparisonMode.BY_DATE:
        event, deadline = event.dt.date(), deadline.dt.date()

    statuses = world.schema.by_role(Role.STATUS)
    if not statuses:
        raise LabelError("не объявлена колонка статуса: трактовать отсутствие события не по чему")
    status = statuses[0].name

    unknown = set(frame[status].unique().to_list()) - set(definition.missing_event)
    if unknown:
        raise LabelError(
            f"для статусов {sorted(unknown)!r} не объявлено, что означает отсутствие события. "
            "Склейка разных причин в один класс добавила 11.3% ложных положительных "
            "на этапе 0"
        )

    excluded = [
        value
        for value, meaning in definition.missing_event.items()
        if meaning is not MissingEventMeaning.NOT_OCCURRED
    ]

    if snapshot is None:
        observed = frame[definition.event_column].max()
        snapshot = (
            observed if observed is not None else frame[world.schema.decision_time.name].max()
        )
    horizon = pl.lit(snapshot).cast(pl.Datetime).dt.date()

    return frame.with_columns(
        pl.when(pl.col(definition.event_column).is_null())
        .then(
            pl.when(pl.col(status).is_in(excluded))
            .then(None)
            .when(deadline < horizon)
            .then(1)
            .otherwise(None)  # срок ещё не наступил: исход не наблюдаем
        )
        .otherwise((event > deadline).cast(pl.Int8))
        .cast(pl.Int8)
        .alias(LABEL)
    )


def observable(frame: pl.DataFrame) -> pl.DataFrame:
    """Строки с наблюдаемым исходом."""
    return frame.filter(pl.col(LABEL).is_not_null())


def positive_rate(frame: pl.DataFrame) -> float:
    return float(observable(frame)[LABEL].mean())
