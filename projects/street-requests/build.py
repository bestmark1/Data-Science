"""Сборка таблицы решений четырнадцатого кейса: обращения о состоянии улиц.

Единица решения — обращение жителя. Момент решения — приём обращения. Исход —
будет ли обращение закрыто в течение объявленного допуска.

Назначенного срока ведомство не заполняет: колонка `due_date` пуста во всех
871 117 записях, и это проверено ДО опечатывания. Поэтому допуск выбирается
объявленным правилом, а не берётся из данных.

ВНИМАНИЕ. В первый прогон намеренно внесены положительные контроли §5
пре-регистрации; они помечены словом КОНТРОЛЬ.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

PROJECT = Path(__file__).resolve().parent
SOURCE = PROJECT / "data" / "raw" / "dot_requests_2020_2023.csv"

DATE_FORMAT = "%Y-%m-%dT%H:%M:%S%.f"
"""Формат объявлен явно и будет сверен с данными до прогона."""

DATES = ("created_date", "closed_date", "resolution_action_updated_date")

TOLERANCES = (1, 3, 7, 30)
"""Кандидаты на допуск, в днях. Взяты короче, чем в прежних кейсах: закрывается
99.2% обращений, и при длинном сроке исход вырожден по построению. Берётся
наименьший с долей закрытых к сроку в [20%, 80%]."""


def build() -> pl.DataFrame:
    """Одна строка на обращение, со сроком и датой закрытия."""
    source = pl.read_csv(SOURCE, infer_schema_length=0)
    read = source.height

    frame = source.with_columns(
        *[
            pl.col(name).str.to_datetime(format=DATE_FORMAT, strict=False).dt.date().alias(name)
            for name in DATES
        ]
    )

    # Непригодна строка без момента решения либо без номера обращения.
    #
    # КОНТРОЛЬ К-2. Строки, где закрытие датировано РАНЬШЕ приёма, намеренно НЕ
    # отбрасываются: ядро обязано назвать их само.
    # Закрытие раньше приёма невозможно: 43 110 строк (4.9%). Либо обращение
    # заведено задним числом после решения вопроса, либо в поле закрытия стоит
    # отметка другого обращения. В обоих случаях исход такой строки не
    # относится к её приёму.
    #
    # Строки оставлены в первом прогоне намеренно (контроль К-2) и отброшены
    # после того, как ядро их назвало.
    broken = (
        pl.col("created_date").is_null()
        | pl.col("unique_key").is_null()
        | (pl.col("closed_date") < pl.col("created_date")).fill_null(False)
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
        f"  прочитано {read:,}, потеряно {lost:,} (нет даты приёма или номера "
        f"либо закрытие раньше приёма), осталось {kept.height:,}"
    )

    return kept.with_columns(
        # Отсутствие, записанное значением. 'UNKNOWN' у канала подачи стоит в
        # 532 054 строках — шестьдесят один процент, — и это именно запись
        # отсутствия: канал не зафиксирован. Превращение в пропуск делает эту
        # дыру видимой, а не оставляет её значением наравне с «телефон» и
        # «через сайт».
        pl.when(pl.col("open_data_channel_type") == "UNKNOWN")
        .then(None)
        .otherwise(pl.col("open_data_channel_type"))
        .alias("open_data_channel_type"),
        pl.when(pl.col("location_type") == "N/A")
        .then(None)
        .otherwise(pl.col("location_type"))
        .alias("location_type"),
        *[
            (pl.col("created_date") + pl.duration(days=days)).alias(f"due_{days}d")
            for days in TOLERANCES
        ],
    )


if __name__ == "__main__":
    table = build()
    print(f"строк: {table.height:,}, колонок: {len(table.columns)}")
    print("период приёма:", table["created_date"].min(), "..", table["created_date"].max())
