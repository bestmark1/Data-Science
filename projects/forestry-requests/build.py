"""Сборка единицы решения восемнадцатого кейса: наряд на работы с деревом.

Единица решения — наряд, и строка источника ей уже соответствует: свёртки
здесь нет. Построитель делает ровно два содержательных действия, и оба названы
в допущениях формуляра, потому что ядро построителя не видит (класс 13
журнала повторов).

ПЕРВОЕ: отменённые наряды выводятся из популяции. У них дата закрытия
проставлена — 11 771 из 24 204, — и объявления `Cancel -> excluded` в
контракте исхода недостаточно: ядро применяет смысл статуса ТОЛЬКО к строкам
без события. Отмена получила бы метку успеха.

ВТОРОЕ: `None`, записанное строкой, НЕ вычищается — это контроль К-2 §5
пре-регистрации, и ядро обязано о нём возразить.

Чего построитель НЕ делает: не выкидывает наряды с закрытием раньше заведения.
Их 32, и ядро зануляет такую метку само (`premature` в `label.py`). Решение,
принятое ядром, видно в отчёте; решение, принятое здесь, — нет.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

PROJECT = Path(__file__).resolve().parent
SOURCE = PROJECT / "data" / "raw" / "tree_work_orders_2021_2023.csv"

FORMAT = "%Y-%m-%dT%H:%M:%S%.f"
MOMENTS = (
    "createddate",
    "closeddate",
    "canceldate",
    "actualfinishdate",
    "projstartdate",
    "updateddate",
)

TARGET_DAYS = 90
"""Срок: квартал от заведения наряда. Объявлен до данных, а не подобран
правилом, — иначе ожидание относилось бы к сроку, выбранному под результат."""


def build() -> pl.DataFrame:
    """Собрать таблицу нарядов."""
    frame = pl.read_csv(SOURCE, infer_schema_length=0).with_columns(
        [pl.col(name).str.to_datetime(FORMAT, strict=False) for name in MOMENTS]
    )
    read = frame.height

    kept = frame.filter(pl.col("objectid").is_not_null() & pl.col("createddate").is_not_null())
    lost = read - kept.height

    cancelled = kept.filter(pl.col("wostatus") == "Cancel").height
    kept = kept.filter(pl.col("wostatus") != "Cancel")

    print(f"  прочитано {read:,}, потеряно {lost:,}, отменённых выведено {cancelled:,}")
    print(f"  нарядов {kept.height:,}")

    return kept.rename({"createddate": "created_at", "closeddate": "closed_at"}).with_columns(
        (pl.col("created_at") + pl.duration(days=TARGET_DAYS)).alias("due_at"),
    )


if __name__ == "__main__":
    table = build()
    closed = table.filter(pl.col("closed_at").is_not_null())
    within = ((closed["closed_at"] - closed["created_at"]).dt.total_days() <= TARGET_DAYS).mean()
    print(f"  закрыто за 90 суток среди закрытых: {within:.1%}")
    print(f"строк: {table.height:,}, колонок: {len(table.columns)}")
