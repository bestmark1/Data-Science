"""Сборка таблицы решений двенадцатого кейса: нарушения жилищного кодекса.

Единица решения — нарушение, выданное инспектором. Момент решения — день
инспекции: тогда нарушение зафиксировано и срок назначен. Срок —
`originalcorrectbydate`, назначенный ведомством при выдаче. Событие —
подтверждение устранения.

ВНИМАНИЕ. В первый прогон намеренно внесены положительные контроли §5
пре-регистрации; они помечены словом КОНТРОЛЬ.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

PROJECT = Path(__file__).resolve().parent
SOURCE = PROJECT / "data" / "raw" / "housing_violations_2022_2024.csv"

DATES = (
    "inspectiondate",
    "approveddate",
    "originalcertifybydate",
    "originalcorrectbydate",
    "newcertifybydate",
    "newcorrectbydate",
    "certifieddate",
    "novissueddate",
    "currentstatusdate",
)
"""Все временные колонки набора. Формат объявляется явно в `DATE_FORMAT` после
сверки с данными: угадывание в пятом кейсе обнулило 94 625 строк из 153 810."""

DATE_FORMAT = "%Y-%m-%dT%H:%M:%S%.f"

RENAMED = {
    "violationid": "violation_id",
    "buildingid": "building_id",
    "registrationid": "owner_id",
    "class": "severity",
    "inspectiondate": "inspected_at",
    "originalcorrectbydate": "due_at_original",
    "newcorrectbydate": "due_at_revised",
    "certifieddate": "certified_at",
    "currentstatus": "current_status",
    "violationstatus": "violation_status",
}


def build() -> pl.DataFrame:
    """Одна строка на нарушение, с назначенным сроком и подтверждением."""
    source = pl.read_csv(SOURCE, infer_schema_length=0)
    read = source.height

    frame = source.rename(RENAMED).with_columns(
        *[
            pl.col(RENAMED.get(name, name))
            .str.to_datetime(format=DATE_FORMAT, strict=False)
            .dt.date()
            .alias(RENAMED.get(name, name))
            for name in DATES
        ]
    )

    # Непригодна строка без момента решения, без назначенного срока либо без
    # номера: без них решение нельзя ни поставить во времени, ни судить.
    #
    # КОНТРОЛЬ К-2. Строки, где подтверждение датировано раньше инспекции,
    # намеренно НЕ отбрасываются: ядро обязано назвать их само.
    broken = (
        pl.col("inspected_at").is_null()
        | pl.col("due_at_original").is_null()
        | pl.col("violation_id").is_null()
    )
    unusable = frame.filter(broken).height
    kept = frame.filter(~broken)
    lost = read - kept.height
    if lost != unusable:
        raise AssertionError(
            f"баланс строк не сошёлся: прочитано {read:,}, потеряно {lost:,}, "
            f"непригодных {unusable:,}. Строки теряются не по объявленной причине"
        )
    print(
        f"  прочитано {read:,}, потеряно {lost:,} (нет даты инспекции, срока "
        f"или номера), осталось {kept.height:,}"
    )

    # Отсутствие, записанное значением: '9999' в номере здания (3 строки) и
    # '-', '-1', 'NA' в номере квартиры (50). Найдено проверкой A14 и
    # исправлено ЗДЕСЬ, при сборке, — по правилу самой проверки.
    return kept.with_columns(
        pl.when(pl.col("building_id") == "9999")
        .then(None)
        .otherwise(pl.col("building_id"))
        .alias("building_id"),
        pl.when(pl.col("apartment").is_in(["-", "-1", "NA"]))
        .then(None)
        .otherwise(pl.col("apartment"))
        .alias("apartment"),
    )


if __name__ == "__main__":
    table = build()
    print(f"строк: {table.height:,}, колонок: {len(table.columns)}")
    print("период инспекций:", table["inspected_at"].min(), "..", table["inspected_at"].max())
