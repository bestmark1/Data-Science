"""Сборка единицы решения девятнадцатого кейса: жалоба против одного ответчика.

Строка источника — пара «жалоба × ответчик»: 64 580 строк на 61 149 номеров
жалоб, и пара уникальна (повторов ноль, сосчитано до опечатывания). Свёртки к
жалобе построитель НЕ делает: департамент ведёт каждого ответчика отдельно, и
закрытие проставлено у строки, а не у жалобы.

Отсюда единственное содержательное действие построителя: **ключ строки
собирается из двух колонок**, потому что готового в источнике нет. Ядро
построителя не видит (класс 13 журнала повторов), поэтому действие названо и
здесь, и в допущениях формуляра.

Номер жалобы остаётся отдельной колонкой и объявляется естественным ключом: на
него приходится по нескольку строк, и проверка A2 обязана об этом сказать, если
жизненный цикл объекта объявлен одноразовым. Это контроль К-2 §5.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

PROJECT = Path(__file__).resolve().parent
SOURCE = PROJECT / "data" / "raw" / "tdi_complaints_2023_2025.csv"

FORMAT = "%Y-%m-%dT%H:%M:%S%.f"
MOMENTS = ("received_date", "closed_date")

TARGET_DAYS = 45
"""Срок: 45 суток от получения жалобы. Объявлен до данных с опорой на
публикуемые TDI сроки ответа компании — 15 дней, продление 10, для авто- и
домашнего страхования 25, — а не подобран правилом под результат."""


def build() -> pl.DataFrame:
    """Собрать таблицу жалоб."""
    frame = pl.read_csv(SOURCE, infer_schema_length=0).with_columns(
        [pl.col(name).str.to_datetime(FORMAT, strict=False) for name in MOMENTS]
    )
    read = frame.height

    kept = frame.filter(
        pl.col("complaint_number").is_not_null()
        & pl.col("respondent_id").is_not_null()
        & pl.col("received_date").is_not_null()
    )
    lost = read - kept.height

    # Отсутствие, записанное строкой: 'UNKNOWN' стоит именем компании у четырёх
    # строк. Ядро нашло это само (`sentinel_as_value`), и правка сделана по его
    # сигналу: заглушка превращается в пустое значение при СБОРКЕ, а не читается
    # значением дальше.
    #
    # Найденное здесь стоит дороже четырёх строк: предсборочный счёт §5б, каким
    # проверялась применимость контроля на `sentinel_as_value`, эту колонку НЕ
    # проверял — автор счёл текстовыми только двенадцать колонок из семнадцати.
    # Контроль был снят как неприменимый, а он был применим.
    unknown = kept.filter(pl.col("respondent_name") == "UNKNOWN").height
    kept = kept.with_columns(
        pl.when(pl.col("respondent_name") == "UNKNOWN")
        .then(None)
        .otherwise(pl.col("respondent_name"))
        .alias("respondent_name")
    )

    print(f"  прочитано {read:,}, потеряно {lost:,}, заглушек вычищено {unknown}")
    print(f"  жалоб против ответчика: {kept.height:,}")

    return kept.rename(
        {"received_date": "received_at", "closed_date": "closed_at"}
    ).with_columns(
        (pl.col("complaint_number") + "-" + pl.col("respondent_id")).alias("case_id"),
        (pl.col("received_at") + pl.duration(days=TARGET_DAYS)).alias("due_at"),
    )


if __name__ == "__main__":
    table = build()
    closed = table.filter(pl.col("closed_at").is_not_null())
    within = ((closed["closed_at"] - closed["received_at"]).dt.total_days() <= TARGET_DAYS).mean()
    print(f"  закрыто за {TARGET_DAYS} суток среди закрытых: {within:.1%}")
    print(f"  уникальных ключей строки: {table['case_id'].n_unique():,}")
    print(f"  уникальных номеров жалоб: {table['complaint_number'].n_unique():,}")
    print(f"строк: {table.height:,}, колонок: {len(table.columns)}")
