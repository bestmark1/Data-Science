"""Шаг 04 — протокол B: полная скользящая постановка.

Протокол A выбирал момент решения там, где он почти естественен — визит на
обслуживание. Здесь моменты решения назначаются календарём: каждая машина
оценивается каждый день. Признаки становятся агрегатами по окну назад,
и именно это протокол A не нагружал.

Цель прежняя: не модель, а места, где ядро ломается. Пре-регистрация
предсказывает три излома — P-9, P-10, P-11.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from dsx.evals.world import World
from dsx.roles import Availability, ColumnSpec, FeatureWindow, Role, Schema, TemporalKind

PROJECT = Path(__file__).resolve().parent.parent
RAW = PROJECT / "data" / "raw"

HORIZON_DAYS = 30
"""Горизонт исхода: отказ любого компонента в течение стольких дней."""

LOOKBACK_DAYS = 7
"""Окно назад, по которому считаются признаки."""

FEATURE_LAG_DAYS = 0.25
"""Отступ окна от момента решения, шесть часов.

Первая версия его не имела, и проверка S7 нашла утечку: суточный агрегат за
день решения включал показания, снятые ПОСЛЕ решения. Телеметрия идёт с 06:00
до 23:00, решение принималось в 06:00 того же дня — до 17 часов из будущего в
36 400 строках из 36 500.
"""

SOURCE = "владелец данных (гипотетический)"
WINDOW_SOURCE = "код построения признаков, шаг 04"


def daily_telemetry() -> pl.DataFrame:
    """Суточные агрегаты датчиков по каждой машине."""
    telemetry = pl.read_csv(RAW / "PdM_telemetry.csv").with_columns(
        pl.col("datetime").str.to_datetime()
    )
    sensors = ["volt", "rotate", "pressure", "vibration"]
    return (
        telemetry.with_columns(pl.col("datetime").dt.date().alias("day"))
        .group_by("machineID", "day")
        .agg(
            [pl.col(s).mean().alias(f"{s}_day") for s in sensors]
            # Самое позднее показание суток. Нужно, чтобы объявленное окно
            # можно было сверить с данными, а не принять на слово (F-11).
            + [pl.col("datetime").max().alias("last_reading")]
        )
        .sort("machineID", "day")
    )


def rolling_features(daily: pl.DataFrame) -> pl.DataFrame:
    """Средние и разброс по окну назад.

    Окно замыкается на дне решения включительно: измерения этого дня доступны
    к утру следующего. Проверить это ядро не может — окно нигде не объявлено.
    """
    sensors = ["volt", "rotate", "pressure", "vibration"]
    window = f"{LOOKBACK_DAYS}d"
    return (
        daily.rolling(index_column="day", period=window, group_by="machineID")
        .agg(
            [pl.col(f"{s}_day").mean().alias(f"{s}_mean_{LOOKBACK_DAYS}d") for s in sensors]
            + [pl.col(f"{s}_day").std().alias(f"{s}_std_{LOOKBACK_DAYS}d") for s in sensors]
            + [pl.col("last_reading").max().alias("measured_at")]
        )
        .sort("machineID", "day")
    )


def error_counts(days: pl.DataFrame) -> pl.DataFrame:
    """Число ошибок за окно назад."""
    errors = (
        pl.read_csv(RAW / "PdM_errors.csv")
        .with_columns(pl.col("datetime").str.to_datetime().dt.date().alias("day"))
        .group_by("machineID", "day")
        .agg(pl.len().alias("errors_day"))
    )
    joined = days.join(errors, on=["machineID", "day"], how="left").with_columns(
        pl.col("errors_day").fill_null(0)
    )
    return (
        joined.sort("machineID", "day")
        .rolling(index_column="day", period=f"{LOOKBACK_DAYS}d", group_by="machineID")
        .agg(pl.col("errors_day").sum().alias(f"errors_{LOOKBACK_DAYS}d"))
        .sort("machineID", "day")
    )


def build_case() -> World:
    """Собрать таблицу решений: одна строка на машину на день."""
    daily = daily_telemetry()
    features = rolling_features(daily)
    errors = error_counts(daily.select("machineID", "day"))
    machines = pl.read_csv(RAW / "PdM_machines.csv")

    failures = (
        pl.read_csv(RAW / "PdM_failures.csv")
        .with_columns(pl.col("datetime").str.to_datetime().alias("failed_at"))
        .select("machineID", "failed_at")
    )

    grid = features.join(errors, on=["machineID", "day"], how="inner").with_columns(
        # Решение принимается на следующее утро после последнего показания:
        # окно признаков обязано кончаться до момента решения, а не в нём.
        pl.col("day").cast(pl.Datetime).dt.offset_by("1d6h").alias("decided_at")
    )

    # Ближайший отказ строго после момента решения.
    next_failure = (
        grid.select("machineID", "decided_at")
        .join(failures, on="machineID", how="left")
        .filter(pl.col("failed_at") > pl.col("decided_at"))
        .group_by("machineID", "decided_at")
        .agg(pl.col("failed_at").min())
    )

    table = (
        grid.join(next_failure, on=["machineID", "decided_at"], how="left")
        .join(machines, on="machineID", how="left")
        .with_columns(
            (pl.col("decided_at") + pl.duration(days=HORIZON_DAYS)).alias("horizon_on"),
            pl.col("machineID").cast(pl.Utf8).alias("machine_key"),
            (pl.col("machineID").cast(pl.Utf8) + pl.lit("@") + pl.col("day").cast(pl.Utf8)).alias(
                "decision_key"
            ),
        )
        .drop_nulls(subset=[f"volt_std_{LOOKBACK_DAYS}d"])
        .sort("decided_at", "machine_key")
    )

    feature_names = [
        c
        for c in table.columns
        if c.endswith(f"_mean_{LOOKBACK_DAYS}d")
        or c.endswith(f"_std_{LOOKBACK_DAYS}d")
        or c == f"errors_{LOOKBACK_DAYS}d"
    ] + ["age", "model"]

    rolling = FeatureWindow(
        lookback_days=LOOKBACK_DAYS,
        lag_days=FEATURE_LAG_DAYS,
        source_of_claim=WINDOW_SOURCE,
    )
    static = FeatureWindow(lookback_days=0, source_of_claim=WINDOW_SOURCE)

    schema = Schema(
        columns=[
            ColumnSpec(name="decision_key", role=Role.ENTITY_ID),
            ColumnSpec(name="machine_key", role=Role.NATURAL_KEY),
            ColumnSpec(name="decided_at", role=Role.DECISION_TIME, temporal=TemporalKind.INSTANT),
            ColumnSpec(
                name="failed_at", role=Role.OUTCOME_COMPONENT, temporal=TemporalKind.INSTANT
            ),
            ColumnSpec(name="horizon_on", role=Role.DEADLINE, temporal=TemporalKind.INSTANT),
            ColumnSpec(name="measured_at", role=Role.MEASURED_AT, temporal=TemporalKind.INSTANT),
        ]
        + [
            ColumnSpec(
                name=name,
                role=Role.FEATURE,
                availability=Availability.AT_DECISION,
                source_of_claim=SOURCE,
                window=static if name in ("age", "model") else rolling,
                # У справочника машин времени измерения нет: окно остаётся
                # объявлением, и это видно в отчёте.
                measured_at=None if name in ("age", "model") else "measured_at",
            )
            for name in feature_names
        ]
    )
    return World(frames={"main": table}, schema=schema)


def main() -> int:
    world = build_case()
    frame = world.main
    print(f"решений: {frame.height:,}, машин: {frame['machine_key'].n_unique()}")
    print(f"период: {frame['decided_at'].min():%Y-%m-%d} .. {frame['decided_at'].max():%Y-%m-%d}")
    print(f"решений на машину: {frame.height / frame['machine_key'].n_unique():.0f}")
    print(
        f"окно признаков: {LOOKBACK_DAYS} дн с отступом {FEATURE_LAG_DAYS} дн, "
        f"горизонт исхода: {HORIZON_DAYS} дн"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
