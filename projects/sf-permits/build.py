"""Сборка таблицы решений тринадцатого кейса: заявки на строительные разрешения.

Единица решения — ЗАЯВКА. В выгрузке одна заявка занимает столько строк,
сколько участков она покрывает: 306 091 строка на 280 380 заявок, и различаются
такие строки только номером квартала и участка. Всё остальное — дата подачи,
дата выдачи, статус, тип, смета — внутри заявки совпадает; проверено счётом.

Поэтому строки сводятся к одной на заявку. Оставить их как есть значило бы
считать одну заявку несколько раз, причём тем чаще, чем больше участков она
затрагивает: у одной их сто один.

ВНИМАНИЕ. В первый прогон намеренно внесены положительные контроли §5
пре-регистрации; они помечены словом КОНТРОЛЬ.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

PROJECT = Path(__file__).resolve().parent
SOURCE = PROJECT / "data" / "raw" / "sf_permits_2015_2023.csv"

DATE_FORMAT = "%Y-%m-%dT%H:%M:%S%.f"
"""Формат объявлен явно и сверен с данными: значения вида
'2015-01-02T08:14:28.000'. Пятый кейс: угадывание обнулило 94 625 строк из
153 810 и переставило местами месяц с днём ещё в 54 927."""

DATES = (
    "filed_date",
    "issued_date",
    "approved_date",
    "completed_date",
    "status_date",
    "data_as_of",
)

TOLERANCES = (30, 60, 90, 180)
"""Кандидаты на допуск, в днях. Берётся наименьший с долей выданных к сроку в
[20%, 80%] — правило объявлено до контакта, приём восьмого кейса."""

NUMBERS = (
    "estimated_cost",
    "revised_cost",
    "number_of_existing_stories",
    "number_of_proposed_stories",
    "plansets",
    "existing_units",
    "proposed_units",
)


def build() -> pl.DataFrame:
    """Одна строка на заявку, со сроком и датой выдачи."""
    source = pl.read_csv(SOURCE, infer_schema_length=0)
    read = source.height

    # Сведение к заявке. Квартал и участок сохраняются первыми попавшимися:
    # заявка может покрывать несколько, и объявлять один из них единственным
    # было бы неправдой — поэтому они идут в признаки как «первый участок»,
    # а допущение об этом записано в форме.
    # maintain_order=True обязателен: без него порядок строк после `unique`
    # не определён, и он менялся между прогонами. Точечные оценки при этом
    # совпадали, а доверительные интервалы расходились — пересчёт брал те же
    # НОМЕРА строк, но другие строки. Отчёт переставал воспроизводиться.
    #
    # Тот же класс, что дефект седьмого кейса: там порядок был не определён
    # после group_by в ядре. Класс был закрыт в ядре и не был закрыт в коде
    # проектов; здесь он и вернулся.
    frame = source.unique(subset=["permit_number"], keep="first", maintain_order=True).with_columns(
        *[
            pl.col(name).str.to_datetime(format=DATE_FORMAT, strict=False).dt.date().alias(name)
            for name in DATES
        ],
        *[pl.col(name).cast(pl.Float64, strict=False).alias(name) for name in NUMBERS],
    )
    collapsed = frame.height

    # Непригодна строка без момента решения либо без номера заявки.
    #
    # КОНТРОЛЬ К-2. Строки, где разрешение выдано РАНЬШЕ подачи заявки,
    # намеренно НЕ отбрасываются: ядро обязано назвать их само.
    # Выдача раньше подачи невозможна: две такие строки оставлены в первом
    # прогоне намеренно (контроль К-2) и отброшены после того, как ядро их
    # назвало.
    broken = (
        pl.col("filed_date").is_null()
        | pl.col("permit_number").is_null()
        | (pl.col("issued_date") < pl.col("filed_date")).fill_null(False)
    )
    unusable = frame.filter(broken).height
    kept = frame.filter(~broken)
    lost = collapsed - kept.height
    if lost != unusable:
        raise AssertionError(
            f"баланс строк не сошёлся: после сведения {collapsed:,}, потеряно {lost:,}, "
            f"непригодных {unusable:,}. Строки теряются не по объявленной причине"
        )
    print(
        f"  прочитано {read:,}, сведено к заявкам {collapsed:,}, потеряно {lost:,} "
        f"(нет даты подачи или номера либо выдача раньше подачи), "
        f"осталось {kept.height:,}"
    )

    return kept.with_columns(
        # Отсутствие, записанное значением: 'unknown' в статусе двух заявок.
        # Найдено проверкой A14 и исправлено ЗДЕСЬ, при сборке.
        pl.when(pl.col("status") == "unknown")
        .then(None)
        .otherwise(pl.col("status"))
        .alias("status"),
        *[
            (pl.col("filed_date") + pl.duration(days=days)).alias(f"due_{days}d")
            for days in TOLERANCES
        ],
    )


if __name__ == "__main__":
    table = build()
    print(f"строк: {table.height:,}, колонок: {len(table.columns)}")
    print("период подачи:", table["filed_date"].min(), "..", table["filed_date"].max())
