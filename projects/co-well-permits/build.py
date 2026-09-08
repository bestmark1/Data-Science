"""Сборка единицы решения двадцатого кейса: разрешение на бурение скважины.

Единица решения — строка выгрузки, и построитель её НЕ ТРОГАЕТ. Ни свёртки, ни
дедупликации: 22% строк набора (7 451 из 33 852) суть полные дубликаты — все
двадцать восемь колонок совпадают, включая номер квитанции.

Пре-регистрация §3 объявила это заранее: «чем именно различаются строки одного
разрешения, автор до чтения значений не знает; если окажется, что строки
полностью совпадают, это дефект данных, и ядро обязано назвать его
`duplicate_rows`». Убрать их построителем значило бы спрятать дефект источника
от проверки, которая для того и заведена.

СРОК НЕ ВЫЧИСЛЯЕТСЯ. `permit_expires` — колонка источника, и построитель её
только приводит к дате. Во всех девятнадцати прежних кейсах срок собирался как
«момент решения плюс N суток», и это дважды записано ошибкой суждения автора.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

PROJECT = Path(__file__).resolve().parent
SOURCE = PROJECT / "data" / "raw" / "well_permits_2018_2022.csv"

FORMAT = "%Y-%m-%dT%H:%M:%S%.f"
MOMENTS = (
    "permit_issued",
    "permit_expires",
    "well_constructed",
    "pump_installed",
    "well_plugged",
    "_1st_beneficial_use",
    "static_water_level_date",
    "modified",
)

NUMBERS = ("elev", "well_depth", "yield", "static_water_level", "div", "wd")


def build() -> pl.DataFrame:
    """Собрать таблицу разрешений."""
    frame = pl.read_csv(SOURCE, infer_schema_length=0).with_columns(
        [pl.col(name).str.to_datetime(FORMAT, strict=False) for name in MOMENTS]
        + [pl.col(name).cast(pl.Float64, strict=False) for name in NUMBERS]
    )
    read = frame.height

    kept = frame.filter(
        pl.col("receipt").is_not_null()
        & pl.col("permit_issued").is_not_null()
        & pl.col("permit_expires").is_not_null()
    )
    lost = read - kept.height
    duplicated = kept.height - kept.unique().height

    # Отсутствие, записанное значением. Ядро нашло четыре колонки: 'NA' в
    # `current_status` (124 строки — это был контроль К-1) и пустые строки в
    # `associated_uses`, `associated_aquifers`, `wdid`. Пустое поле CSV читается
    # пустой строкой, и дальше она идёт значением наравне с прочими.
    sentinels = {"current_status": "NA", "associated_uses": "", "associated_aquifers": "", "wdid": ""}
    cleaned = {
        name: kept.filter(pl.col(name) == value).height for name, value in sentinels.items()
    }
    kept = kept.with_columns(
        [
            pl.when(pl.col(name) == value).then(None).otherwise(pl.col(name)).alias(name)
            for name, value in sentinels.items()
        ]
    )

    print(f"  прочитано {read:,}, потеряно {lost:,}")
    print(f"  заглушек вычищено: {cleaned}")
    print(f"  полных дубликатов в наборе: {duplicated:,} — НЕ убраны, это дефект источника")
    print(f"  разрешений: {kept.height:,}")

    return kept


if __name__ == "__main__":
    table = build()
    built = table.filter(pl.col("well_constructed").is_not_null())
    within = (built["well_constructed"] <= built["permit_expires"]).mean()
    print(f"  построено до истечения среди построенных: {within:.1%}")
    print(f"  различных номеров разрешения: {table['permit'].n_unique():,}")
    print(f"строк: {table.height:,}, колонок: {len(table.columns)}")
