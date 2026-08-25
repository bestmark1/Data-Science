"""Сборка таблицы решений пятого кейса.

Момент решения — инспекция. Вопрос ставится в форме; здесь только то, что
нельзя объявить: разбор дат, приведение имён, построение сроков-кандидатов и
признаков, известных ДО инспекции.

Одна таблица, соединять нечего. Зато впервые естественный ключ по-настоящему
повторяется: 153 810 инспекций на 32 851 лицензию.

**Признаки строятся по истории заведения ДО текущей инспекции.** Это первый
кейс, где такие признаки вообще возможны: у заказа, машины-визита, претензии и
подписки истории не было либо она не использовалась. Окно объявляется формой,
а время последнего наблюдения — колонкой, чтобы объявление сверялось (F-11).
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

PROJECT = Path(__file__).resolve().parent
RAW = PROJECT / "data" / "raw"

HORIZONS = (90, 180, 365)
"""Кандидаты на срок в днях от текущей инспекции.

Исход самой инспекции известен в её момент, то есть немедленный, а ядро
поддерживает только отложенный — и правильно: при немедленном исходе вся
машинерия сплита, созревания и зазоров вырождается.

Поэтому предсказывается СЛЕДУЮЩАЯ инспекция: провалит ли заведение её в
течение горизонта. Практический смысл прямой — после каждой инспекции город
решает, когда прийти снова.
"""

NULLS = ["NA", "N/A", "", "-", "?", "Unknown", "UNKNOWN"]
"""Заглушки, которыми в этой выгрузке записано отсутствие. Объявляются при
чтении: строка, прочитанная значением, портит все проверки разом (A14)."""

RENAMED = {
    "Inspection ID": "inspection_id",
    "License #": "license_id",
    "DBA Name": "business_name",
    "AKA Name": "trade_name",
    "Facility Type": "facility_type",
    "Risk": "risk",
    "Zip": "zip_code",
    "Inspection Date": "inspected_at",
    "Inspection Type": "inspection_type",
    "Results": "result",
}

DROPPED = ("Address", "City", "State", "Latitude", "Longitude", "Location", "Violations")
"""Адрес и координаты — та же информация, что индекс, только грязнее.
`Violations` — текст, записанный ПО ИТОГАМ инспекции: признаком быть не может,
а исходом является `Results`."""


DATE_FORMAT = "%m/%d/%Y"
"""Формат даты объявляется явно.

Без него разбор молча обнулил 94 625 строк из 153 810, а из разобравшихся
54 927 получили ПЕРЕСТАВЛЕННЫЕ месяц и день: «08/11/2017» стало восьмым ноября
вместо одиннадцатого августа. Порядок инспекций внутри заведения — тот самый,
на котором строится вся история признаков, — оказался бы перемешан.

Угадывание формата запрещено здесь по той же причине, по которой в ядре
запрещено угадывание временнóй грануляции: ошибка не видна глазами.
"""


def build() -> pl.DataFrame:
    """Одна строка на инспекцию, с историей заведения до неё.

    Сборка сводит баланс строк: сколько прочитано, сколько потеряно и почему.
    Молчаливая потеря запрещена так же, как молчаливое умолчание в объявлениях.
    """
    source = pl.read_csv(
        RAW / "Food_Inspections.csv",
        null_values=NULLS,
        infer_schema_length=50000,
        ignore_errors=True,
    )
    read = source.height

    frame = (
        source.drop(DROPPED)
        .rename(RENAMED)
        .with_columns(
            pl.col("inspected_at").str.to_datetime(format=DATE_FORMAT, strict=False),
            pl.col("license_id").cast(pl.Utf8),
            pl.col("inspection_id").cast(pl.Utf8),
        )
    )

    unusable = frame.filter(
        pl.col("inspected_at").is_null() | pl.col("license_id").is_null()
    ).height
    frame = frame.drop_nulls(subset=["inspected_at", "license_id"]).sort(
        "license_id", "inspected_at"
    )

    lost = read - frame.height
    if lost != unusable:
        raise ValueError(
            f"сборка потеряла {lost:,} строк, а объяснено {unusable:,}: "
            "необъяснённая потеря запрещена"
        )
    print(
        f"  прочитано {read:,}, потеряно {lost:,} (нет даты или лицензии), "
        f"осталось {frame.height:,}"
    )

    failed = pl.col("result").str.contains("(?i)fail").fill_null(False)

    # История заведения СТРОГО до текущей инспекции: сдвиг на одну строку назад
    # внутри лицензии. Без сдвига признак содержал бы текущий исход.
    with_history = frame.with_columns(
        pl.col("inspected_at").shift(1).over("license_id").alias("previous_inspected_at"),
        failed.shift(1).over("license_id").cast(pl.Int64).alias("previous_failed"),
        failed.cum_sum().shift(1).over("license_id").fill_null(0).alias("failures_before"),
        pl.int_range(pl.len()).over("license_id").alias("inspections_before"),
    )

    # Исход: следующая инспекция того же заведения, завершившаяся провалом.
    with_next = with_history.with_columns(
        pl.col("inspected_at").shift(-1).over("license_id").alias("next_inspected_at"),
        pl.col("result").shift(-1).over("license_id").alias("next_result"),
    )

    return with_next.with_columns(
        (pl.col("inspected_at") - pl.col("previous_inspected_at"))
        .dt.total_days()
        .alias("days_since_previous"),
        # Следующая инспекция с иным результатом событием не является: провала
        # не было, и это наблюдение, а не пропуск.
        pl.when(pl.col("next_result").str.contains("(?i)fail"))
        .then(pl.col("next_inspected_at"))
        .otherwise(None)
        .alias("next_failure_at"),
        *[
            (pl.col("inspected_at") + pl.duration(days=days)).alias(f"horizon_{days}d")
            for days in HORIZONS
        ],
    )


def _both_missing(source: pl.DataFrame) -> int:
    """Строки, где отсутствуют и дата, и лицензия: иначе они считались бы дважды."""
    return int(
        source.filter(
            pl.col("Inspection Date").str.to_datetime(format=DATE_FORMAT, strict=False).is_null()
            & pl.col("License #").is_null()
        ).height
    )


def main() -> int:
    frame = build()
    print(f"строк: {frame.height:,}, колонок: {frame.width}")
    print(f"заведений: {frame['license_id'].n_unique():,}")
    print(
        f"период: {frame['inspected_at'].min():%Y-%m-%d} .. {frame['inspected_at'].max():%Y-%m-%d}"
    )
    print(
        "первых инспекций (истории нет): "
        f"{frame.filter(pl.col('inspections_before') == 0).height:,}"
    )
    print()
    for name in frame.columns:
        print(f"  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
