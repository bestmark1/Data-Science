"""Проверки данных.

В отличие от контрактных, эти смотрят на содержимое и потому несут предпосылки:
часть из них осмысленна только при событийном процессе или регулярном потоке
сбора. Предпосылки объявлены явно — проверка выключается вместе с ними, а не
срабатывает вхолостую.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import polars as pl

from dsx.checks.base import Context, Signal
from dsx.evals.case import Finding
from dsx.roles import Role
from dsx.task import ObjectLifetime, Premise

UNIVERSAL = frozenset({Premise.UNIVERSAL})
PROCESS = frozenset({Premise.PROCESS})
STREAM = frozenset({Premise.STREAM})


@dataclass(frozen=True)
class SurrogateKeyAsEntity:
    """A2. Единица решения мельче объекта, и не объявлено, намеренно ли.

    У маркетплейса ключ заказа выдавался за клиента — ошибка. В обслуживании
    оборудования визит намеренно мельче машины — верный дизайн. По данным эти
    случаи неотличимы, поэтому проверка требует ответа, а не запрещает ситуацию.

    Срабатывает только при объявленном естественном ключе: без него сравнивать
    не с чем.
    """

    requirement: str = "A2"
    premises: frozenset[Premise] = UNIVERSAL
    detects: frozenset[Finding] = frozenset({Finding.SURROGATE_KEY_AS_ENTITY})

    def run(self, context: Context) -> list[Signal]:
        schema = context.world.schema
        entities = schema.by_role(Role.ENTITY_ID)
        natural = schema.by_role(Role.NATURAL_KEY)
        if not entities or not natural:
            return []

        lifetime = context.task.object_lifetime
        if lifetime is ObjectLifetime.RECURRING:
            return []  # повторные решения по одному объекту — устройство задачи

        frame = context.world.main
        entity_cardinality = frame[entities[0].name].n_unique()

        signals = []
        for key in natural:
            if key.name not in frame.columns:
                continue
            natural_cardinality = frame[key.name].n_unique()
            if natural_cardinality < entity_cardinality:
                signals.append(
                    Signal(
                        Finding.SURROGATE_KEY_AS_ENTITY,
                        f"{entities[0].name!r} различает {entity_cardinality:,} значений, "
                        f"а {key.name!r} — {natural_cardinality:,}: на один объект "
                        f"приходится несколько решений. "
                        + (
                            "Объявлено, что объект одноразов: объявление противоречит "
                            "данным, и история по нему будет короче настоящей"
                            if lifetime is ObjectLifetime.ONE_SHOT
                            else "Объявите жизненный цикл объекта. Намеренное дробление "
                            "и подмена ключа по данным неотличимы, а во втором случае "
                            "история по ней будет короче настоящей"
                        ),
                        blocking=True,
                    )
                )
        return signals


@dataclass(frozen=True)
class DuplicateRows:
    """A3. Полные дубликаты строк.

    Трение 01: 26% дубликатов нашлись только потому, что проверка была
    дописана вручную.
    """

    requirement: str = "A3"
    premises: frozenset[Premise] = UNIVERSAL
    detects: frozenset[Finding] = frozenset({Finding.DUPLICATE_ROWS})

    def run(self, context: Context) -> list[Signal]:
        frame = context.world.main
        duplicates = frame.height - frame.unique().height
        if not duplicates:
            return []
        return [
            Signal(
                Finding.DUPLICATE_ROWS,
                f"полных дубликатов строк: {duplicates:,} "
                f"({duplicates / frame.height:.1%} таблицы)",
            )
        ]


@dataclass(frozen=True)
class StatusEventConflict:
    """A5. Статус не определяет наличие события.

    Трение 02: восемь объектов со статусом завершения не имели метки события.

    Формулировка не требует объявлять, какой статус что означает: достаточно,
    что при одном и том же значении статуса событие иногда есть, а иногда нет.
    Это либо противоречие, либо неполнота контракта — в обоих случаях вопрос.
    """

    requirement: str = "A5"
    premises: frozenset[Premise] = PROCESS
    detects: frozenset[Finding] = frozenset({Finding.STATUS_TIMESTAMP_CONFLICT})

    def run(self, context: Context) -> list[Signal]:
        schema = context.world.schema
        statuses = schema.by_role(Role.STATUS)
        frame = context.world.main
        event = context.outcome.event_column
        if not statuses or event not in frame.columns:
            return []

        column = statuses[0].name
        grouped = (
            frame.group_by(column)
            .agg(
                pl.len().alias("rows"),
                pl.col(event).null_count().alias("without_event"),
            )
            .filter((pl.col("without_event") > 0) & (pl.col("without_event") < pl.col("rows")))
        )

        return [
            Signal(
                Finding.STATUS_TIMESTAMP_CONFLICT,
                f"при статусе {row[column]!r} событие есть у "
                f"{row['rows'] - row['without_event']:,} объектов и отсутствует у "
                f"{row['without_event']:,}: статус не определяет наличие события",
            )
            for row in grouped.iter_rows(named=True)
        ]


@dataclass(frozen=True)
class TruncatedTail:
    """A6. Обрыв сбора данных в конце периода.

    Трение 04: объём падал с 245 наблюдений в день до 4, и это выглядело как
    цензурирование, хотя было обрывом выгрузки.
    """

    requirement: str = "A6"
    premises: frozenset[Premise] = STREAM
    detects: frozenset[Finding] = frozenset({Finding.TRUNCATED_TAIL})
    window_days: int = 14
    ratio: float = 0.35

    def run(self, context: Context) -> list[Signal]:
        daily = _daily_volume(context)
        if daily is None or daily.height < self.window_days * 3:
            return []

        # Окно по датам, а не по строкам: при разрежённом хвосте последние
        # строки таблицы приходятся на дни задолго до обрыва.
        edge = daily["day"].max() - dt.timedelta(days=self.window_days)
        tail_frame = daily.filter(pl.col("day") > edge)
        typical = daily.filter(pl.col("day") <= edge)["n"].median()
        tail = tail_frame["n"].sum() / self.window_days if tail_frame.height else 0.0

        if typical is None or typical == 0 or tail >= typical * self.ratio:
            return []

        return [
            Signal(
                Finding.TRUNCATED_TAIL,
                f"в последние {self.window_days} дней в среднем {tail:.0f} наблюдений в день "
                f"против медианных {typical:.0f}: похоже на обрыв сбора, а не на спад",
            )
        ]


@dataclass(frozen=True)
class MissingPeriod:
    """A7. Пропущенный отрезок во временном ряду объёма.

    Трение 01: месяц отсутствовал целиком, обнаружено случайно.
    """

    requirement: str = "A7"
    premises: frozenset[Premise] = STREAM
    detects: frozenset[Finding] = frozenset({Finding.MISSING_PERIOD})
    min_gap_days: int = 7

    def run(self, context: Context) -> list[Signal]:
        daily = _daily_volume(context)
        if daily is None or daily.height < 2:
            return []

        gaps = daily.with_columns((pl.col("day").diff().dt.total_days() - 1).alias("gap")).filter(
            pl.col("gap") >= self.min_gap_days
        )

        return [
            Signal(
                Finding.MISSING_PERIOD,
                f"перед {row['day']} нет наблюдений {int(row['gap'])} дней подряд",
            )
            for row in gaps.iter_rows(named=True)
        ]


@dataclass(frozen=True)
class PostTreatmentMissingness:
    """A11. Пропуск признака объясняется исходом.

    Трение 05: из 775 объектов без позиций 767 были отменены. Импутация нулём
    превратила бы пропуск в признак из будущего.
    """

    requirement: str = "A11"
    premises: frozenset[Premise] = PROCESS
    detects: frozenset[Finding] = frozenset({Finding.POST_TREATMENT_MISSINGNESS})
    concentration: float = 0.8

    min_missing: int = 30
    """Меньше этого пропусков — концентрация ничего не значит.

    Один пропуск у строки редкого статуса даёт стопроцентную концентрацию, и
    проверка объявляла случайность находкой.
    """

    def run(self, context: Context) -> list[Signal]:
        schema = context.world.schema
        statuses = schema.by_role(Role.STATUS)
        frame = context.world.main
        if not statuses:
            return []
        status = statuses[0].name

        signals = []
        for column in schema.by_role(Role.FEATURE):
            if column.name not in frame.columns:
                continue
            missing = frame.filter(pl.col(column.name).is_null())
            if missing.height < self.min_missing:
                continue

            share = (
                missing.group_by(status)
                .agg(pl.len().alias("n"))
                .with_columns((pl.col("n") / missing.height).alias("share"))
                .sort("share", descending=True)
            )
            top = share.row(0, named=True)
            overall = frame.filter(pl.col(status) == top[status]).height / frame.height
            if top["share"] >= self.concentration and top["share"] > overall * 1.5:
                signals.append(
                    Signal(
                        Finding.POST_TREATMENT_MISSINGNESS,
                        f"пропуск признака {column.name!r} на {top['share']:.0%} приходится "
                        f"на статус {top[status]!r} (доля статуса в данных {overall:.0%}): "
                        "пропуск объясняется исходом, а не свойством объекта",
                    )
                )
        return signals


def _daily_volume(context: Context) -> pl.DataFrame | None:
    """Число наблюдений по дням момента решения."""
    column = context.world.schema.decision_time.name
    frame = context.world.main
    if column not in frame.columns:
        return None
    return (
        frame.select(pl.col(column).dt.date().alias("day"))
        .group_by("day")
        .agg(pl.len().alias("n"))
        .sort("day")
    )


DATA_CHECKS = [
    SurrogateKeyAsEntity(),
    DuplicateRows(),
    StatusEventConflict(),
    TruncatedTail(),
    MissingPeriod(),
    PostTreatmentMissingness(),
]
