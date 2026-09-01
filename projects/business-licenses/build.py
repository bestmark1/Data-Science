"""Сборка единицы решения семнадцатого кейса: заявка на лицензию.

Единица решения — заявка, и строка источника ей уже соответствует: свёртки
здесь нет. Это первый кейс за три, где построитель ничего не собирает, и тем
он полезен: класс 13 журнала повторов говорит, что ядро не видит построителя, —
здесь видеть почти нечего.

Предсказывается поведение ТРЕТЬЕЙ СТОРОНЫ: донесёт ли заявитель требуемое за
месяц. Ведомство этим не управляет.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

PROJECT = Path(__file__).resolve().parent
SOURCE = PROJECT / "data" / "raw" / "applications_2015_2023.csv"

FORMAT = "%Y-%m-%dT%H:%M:%S%.f"
MOMENTS = (
    "application_created_date",
    "application_requirements_complete",
    "payment_date",
    "date_issued",
    "expiration_date",
    "license_status_change_date",
)

TARGET_DAYS = 30
"""Срок: календарный месяц от подачи. Объявлен до данных, а не подобран
правилом, — иначе ожидание относилось бы к сроку, выбранному под результат."""


def build() -> pl.DataFrame:
    """Собрать таблицу заявок."""
    frame = pl.read_csv(SOURCE, infer_schema_length=0).with_columns(
        [pl.col(name).str.to_datetime(FORMAT, strict=False) for name in MOMENTS]
    )
    read = frame.height

    # КОНТРОЛЬ К-2: отсутствие, записанное значением, намеренно не вычищается.

    broken = pl.col("id").is_null() | pl.col("application_created_date").is_null()
    kept = frame.filter(~broken)
    print(f"  прочитано {read:,}, потеряно {read - kept.height:,}, заявок {kept.height:,}")

    return kept.rename(
        {
            "application_created_date": "applied_at",
            "application_requirements_complete": "requirements_met_at",
            "date_issued": "issued_at",
        }
    ).with_columns(
        (pl.col("applied_at") + pl.duration(days=TARGET_DAYS)).alias("due_at"),
    )


if __name__ == "__main__":
    table = build()
    met = table.filter(pl.col("requirements_met_at").is_not_null())
    within = (
        (met["requirements_met_at"] - met["applied_at"]).dt.total_days() <= TARGET_DAYS
    ).mean()
    print(f"  выполнили требования за месяц: {within:.1%} (объявлено 0.70 ± 0.25)")
    print(f"строк: {table.height:,}, колонок: {len(table.columns)}")
