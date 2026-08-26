"""Сборка таблицы решений шестого кейса.

Момент решения — **регистрация сводки**, а не время происшествия. Полиция
узнаёт о происшествии, когда его записали; взять временем решения момент
самого события значило бы дать знание, которого тогда не было.

Признак истории участка считается ДВАЖДЫ, и в этом весь кейс:

  `nearby_90d_known` — по срезу на момент решения: учтены только сводки,
  ЗАРЕГИСТРИРОВАННЫЕ не позже него;
  `nearby_90d_export` — по всей выгрузке: учтены все сводки, произошедшие в
  окне, включая записанные позже.

Второй способ — обычный и неверный. Разница между ними и есть изменчивость
истории: то, что аналитик считает «историей места», зависит от момента
расчёта.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from dsx.join import AsofDirection, Cardinality, guarded_asof_join, guarded_join

PROJECT = Path(__file__).resolve().parent
RAW = PROJECT / "data" / "raw"
SOURCE = RAW / "Police_Department_Incident_Reports__2018_to_Present.csv"

DATE_FORMAT = "%Y/%m/%d %I:%M:%S %p"
"""Формат объявляется явно. На пятом кейсе угадывание испортило бы 38% строк
переставленными месяцем и днём, и заметить это было бы нечем."""

HORIZONS = (3, 7, 14, 30)
"""Кандидаты на срок для НАСИЛЬСТВЕННОГО происшествия.

Первая постановка предсказывала любое новое происшествие за 90 дней и была
вырожденной: 95.3% одного класса. При медианном промежутке между сводками в
двое суток вырожден любой длинный горизонт. Поправка A-1.
"""

VIOLENT = (
    "Assault",
    "Robbery",
    "Homicide",
    "Rape",
    "Sex Offense",
    "Weapons Offense",
    "Weapons Carrying Etc",
)
"""Категории, считающиеся насильственными. Перечень объявлен, а не выведен из
данных подбором: подбор категорий под желаемую долю класса есть выбор,
расходующий выборку."""

WINDOW_DAYS = 90
"""Окно истории участка."""

KEEP = (
    "Incident Datetime",
    "Report Datetime",
    "Row ID",
    "Incident Category",
    "Incident Subcategory",
    "Report Type Code",
    "Filed Online",
    "Resolution",
    "CNN",
    "Police District",
    "Analysis Neighborhood",
)

RENAMED = {
    "Incident Datetime": "happened_at",
    "Report Datetime": "reported_at",
    "Row ID": "report_id",
    "Incident Category": "category",
    "Incident Subcategory": "subcategory",
    "Report Type Code": "report_type",
    "Filed Online": "filed_online",
    "Resolution": "resolution",
    "CNN": "segment_id",
    "Police District": "district",
    "Analysis Neighborhood": "neighborhood",
}


def _history(frame: pl.DataFrame, order_by: str, name: str) -> pl.DataFrame:
    """Сколько происшествий на участке попало в окно перед решением.

    `order_by` задаёт, какие сводки считаются известными: по времени
    регистрации либо по времени происшествия. Разница между двумя вызовами и
    есть предмет кейса.
    """
    window = f"{WINDOW_DAYS}d"
    counted = (
        frame.select("segment_id", order_by)
        .sort("segment_id", order_by)
        .rolling(index_column=order_by, period=window, group_by="segment_id", closed="left")
        .agg(pl.len().alias(name))
    )
    return counted.select(name)


def build() -> pl.DataFrame:
    """Одна строка на сводку, с историей участка, посчитанной двумя способами."""
    source = pl.read_csv(SOURCE, infer_schema_length=0)
    read = source.height

    frame = (
        source.select(KEEP)
        .rename(RENAMED)
        .with_columns(
            pl.col("happened_at").str.to_datetime(format=DATE_FORMAT, strict=False),
            pl.col("reported_at").str.to_datetime(format=DATE_FORMAT, strict=False),
        )
        # Выгрузка пишет отсутствующий район словом "null" — это пропуск,
        # записанный как значение. Проверка A14 возразила на 156 строках.
        # Превращаем в настоящий пропуск ЗДЕСЬ, при сборке, а не обходом.
        .with_columns(
            pl.when(pl.col(name) == "null").then(None).otherwise(pl.col(name)).alias(name)
            for name in ("neighborhood", "district", "category", "subcategory", "resolution")
        )
    )
    unusable = frame.filter(
        pl.col("happened_at").is_null()
        | pl.col("reported_at").is_null()
        | pl.col("segment_id").is_null()
    ).height
    frame = frame.drop_nulls(subset=["happened_at", "reported_at", "segment_id"])

    lost = read - frame.height
    if lost != unusable:
        raise ValueError(
            f"сборка потеряла {lost:,} строк, а объяснено {unusable:,}: "
            "необъяснённая потеря запрещена"
        )
    print(
        f"  прочитано {read:,}, потеряно {lost:,} (нет дат или участка), осталось {frame.height:,}"
    )

    # Известное на момент решения: сводки, ЗАРЕГИСТРИРОВАННЫЕ до него.
    by_report = frame.sort("segment_id", "reported_at")
    known = _history(by_report, "reported_at", "nearby_90d_known")
    by_report = by_report.with_columns(known)

    # Известное на момент выгрузки: сводки, ПРОИЗОШЕДШИЕ в окне, независимо от
    # того, когда о них записали. Так считает всякий, кто не думал о запаздывании.
    by_event = frame.sort("segment_id", "happened_at")
    export = _history(by_event, "happened_at", "nearby_90d_export")
    by_event = by_event.select("report_id").with_columns(export)

    # Грануляция объявляется: report_id уникален, соответствие однозначное.
    joined = guarded_join(
        by_report, by_event, on=["report_id"], expect=Cardinality.ONE_TO_ONE, how="left"
    )

    # Исход: следующее НАСИЛЬСТВЕННОЕ происшествие на том же участке. Событием
    # считается его РЕГИСТРАЦИЯ: узнать о происшествии раньше, чем о нём
    # сообщили, нельзя.
    ordered = joined.with_columns(pl.col("category").is_in(VIOLENT).alias("is_violent")).sort(
        "reported_at"
    )
    violent = (
        ordered.filter("is_violent")
        .select("segment_id", pl.col("reported_at").alias("violent_at"))
        .sort("violent_at")
    )
    # Ближайшее СЛЕДУЮЩЕЕ насильственное происшествие — соединение вперёд по
    # времени. Сдвиг на микросекунду делает границу строгой: сводка не считается
    # исходом сама для себя, а одновременная с ней — считается.
    with_next = guarded_asof_join(
        ordered.with_columns(
            (pl.col("reported_at") + pl.duration(microseconds=1)).alias("__after")
        ).sort("__after"),
        violent,
        left_on="__after",
        right_on="violent_at",
        by=["segment_id"],
        direction=AsofDirection.FORWARD,
    )
    with_next = with_next.rename({"violent_at": "next_violent_at"}).drop("__after")

    return with_next.with_columns(
        pl.col("next_violent_at").alias("next_incident_at"),
        (pl.col("reported_at") - pl.col("happened_at")).dt.total_hours().alias("report_lag_hours"),
        *[
            (pl.col("reported_at") + pl.duration(days=days)).alias(f"horizon_{days}d")
            for days in HORIZONS
        ],
    ).sort("reported_at")


def main() -> int:
    frame = build()
    print(f"строк: {frame.height:,}, колонок: {frame.width}")
    print(f"участков: {frame['segment_id'].n_unique():,}")
    print(
        f"период регистрации: {frame['reported_at'].min():%Y-%m-%d} .. "
        f"{frame['reported_at'].max():%Y-%m-%d}"
    )
    differ = frame.filter(pl.col("nearby_90d_known") != pl.col("nearby_90d_export")).height
    print()
    print("ПОРОГ 3 — изменчивость признака (нужно >=10% строк)")
    print(f"  различаются: {differ:,} ({100 * differ / frame.height:.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
