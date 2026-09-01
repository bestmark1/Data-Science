"""Сборка единицы решения шестнадцатого кейса: вызов, а не выезд машины.

Строка источника — выезд ОДНОЙ машины: на 336 390 вызовов приходится 695 428
строк, по 2.07 на вызов. Единица решения — вызов, и исход определяет **первый
прибывший** расчёт: горожанину важно, приехал ли кто-нибудь, а не приехала ли
конкретная машина.

Свёртка живёт здесь, в коде проекта, и ядро её не видит — оно проверяет
объявленную схему. Это долг, записанный классом 13 журнала повторов, и здесь он
становится практическим: ошибка в свёртке даст время не того расчёта, и никакая
проверка об этом не скажет.

Порядок внутри вызова задан явно — по `unit_sequence_in_call_dispatch`. Без него
«первая строка вызова» зависела бы от того, как библиотека решила сгруппировать.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

PROJECT = Path(__file__).resolve().parent
SOURCE = PROJECT / "data" / "raw" / "fire_calls_2022_2023.csv"

FORMAT = "%Y-%m-%dT%H:%M:%S%.f"
MOMENTS = ("received_dttm", "dispatch_dttm", "on_scene_dttm", "available_dttm")

TARGET_SECONDS = 8 * 60
"""Норматив: восемь минут от приёма вызова до прибытия первого расчёта.

Взят доменным знанием, а не подобран правилом, — первый такой случай в проекте.
Происхождение числа записано в пре-регистрации честно: это ориентир экстренных
служб по памяти автора, а не цифра из отчёта города.
"""

# Признаки, известные в момент ПРИЁМА вызова. Всё остальное — результат
# диспетчеризации и потому будущее: какая машина поехала, сколько было тревог,
# какой приоритет ей присвоили в итоге.
AT_RECEIPT = (
    "call_type",
    "call_type_group",
    "original_priority",
    "battalion",
    "station_area",
    "zipcode_of_incident",
    "neighborhoods_analysis_boundaries",
)

NIGHT_FROM, NIGHT_UNTIL = 22, 6
RUSH_MORNING = (7, 10)
RUSH_EVENING = (16, 19)


def build() -> pl.DataFrame:
    """Собрать таблицу вызовов."""
    frame = pl.read_csv(SOURCE, infer_schema_length=0).with_columns(
        [pl.col(name).str.to_datetime(FORMAT, strict=False) for name in MOMENTS]
    )
    read = frame.height

    ordered = frame.sort(["call_number", "unit_sequence_in_call_dispatch"], maintain_order=True)
    calls = ordered.group_by("call_number", maintain_order=True).agg(
        pl.col("received_dttm").min().alias("received_at"),
        # Первый ПРИБЫВШИЙ, а не первый отправленный: отправленная машина могла
        # не доехать, и тогда исход вызова определяет следующая.
        pl.col("on_scene_dttm").min().alias("on_scene_at"),
        # КОНТРОЛЬ К-1: момент, когда расчёт освободился. Он наступает ПОСЛЕ
        # прибытия, и объявлять его признаком момента приёма — подлог.
        pl.col("available_dttm").min().alias("available_at"),
        pl.col("call_final_disposition").first().alias("disposition"),
        *[pl.col(name).first().alias(name) for name in AT_RECEIPT],
    )

    # Прибытие раньше приёма невозможно: 19 вызовов. Либо часы расчёта разошлись
    # с часами диспетчерской, либо запись заведена задним числом. Исходом своего
    # решения такая строка быть не может.
    broken = (
        pl.col("call_number").is_null()
        | pl.col("received_at").is_null()
        | (pl.col("on_scene_at") < pl.col("received_at")).fill_null(False)
    )
    kept = calls.filter(~broken)
    lost = calls.height - kept.height
    print(
        f"  строк выездов {read:,} -> вызовов {calls.height:,}, "
        f"потеряно {lost:,}, осталось {kept.height:,}"
    )

    hour = pl.col("received_at").dt.hour()
    return kept.with_columns(
        # Отсутствие, записанное значением: строка 'None' в районе у 121 вызова.
        # Оставлено в первом прогоне контролем К-3 и превращено в пропуск после
        # того, как ядро назвало колонку поимённо.
        pl.when(pl.col("neighborhoods_analysis_boundaries") == "None")
        .then(None)
        .otherwise(pl.col("neighborhoods_analysis_boundaries"))
        .alias("neighborhoods_analysis_boundaries"),
        (pl.col("received_at") + pl.duration(seconds=TARGET_SECONDS)).alias("due_at"),
        # Ночь: дороги свободны, и расчёт доезжает быстрее. Направление
        # объявляется ядру доменным знанием и проверяется им же (N8).
        ((hour >= NIGHT_FROM) | (hour < NIGHT_UNTIL)).cast(pl.Int8).alias("at_night"),
        # Часы пик: то же знание о дорогах, повёрнутое обратной стороной.
        # Два объявления, но ОДИН факт о городе, и независимыми свидетельствами
        # они не являются.
        (
            hour.is_between(*RUSH_MORNING, closed="left")
            | hour.is_between(*RUSH_EVENING, closed="left")
        )
        .cast(pl.Int8)
        .alias("at_rush_hour"),
        # Состояние вызова. Нужно, чтобы отличить «расчёт не доехал» от
        # «прибытие не записано»: исход вызова объясняет, почему момента
        # прибытия нет.
        pl.col("disposition").alias("call_status"),
    )


if __name__ == "__main__":
    table = build()
    arrived = table.filter(pl.col("on_scene_at").is_not_null())
    within = (
        (arrived["on_scene_at"] - arrived["received_at"]).dt.total_seconds() <= TARGET_SECONDS
    ).mean()
    print(f"  прибыли к нормативу: {within:.1%} (объявлено ожидание 0.85)")
    print(f"строк: {table.height:,}, колонок: {len(table.columns)}")
