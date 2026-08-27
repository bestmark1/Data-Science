"""Сборка таблицы решений одиннадцатого кейса: соблюдение расписания рейсов.

Единица решения — рейс. Момент решения — расписанное время вылета: обещание уже
дано, и известны перевозчик, аэропорты, расстояние, расписанная длительность.
Срок — расписанное время прибытия плюс допуск. Исход — прибудет ли рейс позже
срока.

ВРЕМЯ В ЭТОЙ ВЫГРУЗКЕ ЗАПИСАНО ЧЕТЫРЬМЯ ЦИФРАМИ — «1425» значит 14:25 — и
привязано к дате рейса, а не к календарю. Отсюда две ловушки, обе названы явно:

  * рейс, прибывающий после полуночи, имеет время прибытия МЕНЬШЕ времени
    вылета. Такой рейс прибывает на следующие сутки, и без поправки его
    прибытие оказалось бы за одиннадцать часов ДО вылета;
  * «2400» означает полночь следующих суток, а не 24-й час: разбор его как
    часа выдал бы ошибку либо, что хуже, тихий сдвиг.

ВНИМАНИЕ. В первый прогон намеренно внесены положительные контроли §5
пре-регистрации; они помечены словом КОНТРОЛЬ.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import polars as pl

PROJECT = Path(__file__).resolve().parent
RAW = PROJECT / "data" / "raw"

TOLERANCES = (0, 15, 30, 60)
"""Кандидаты на допуск, в минутах. Берётся наименьший с долей опозданий в
[20%, 80%] — правило объявлено до контакта, приём восьмого кейса."""

KEEP = [
    "FlightDate",
    "Reporting_Airline",
    "Tail_Number",
    "Flight_Number_Reporting_Airline",
    "Origin",
    "OriginState",
    "Dest",
    "DestState",
    "CRSDepTime",
    "DepTime",
    "DepDelay",
    "CRSArrTime",
    "ArrTime",
    "Cancelled",
    "CancellationCode",
    "Diverted",
    "CRSElapsedTime",
    "Distance",
    "DayOfWeek",
]

RENAMED = {
    "FlightDate": "flight_date",
    "Reporting_Airline": "carrier",
    "Tail_Number": "tail_number",
    "Flight_Number_Reporting_Airline": "flight_number",
    "Origin": "origin",
    "OriginState": "origin_state",
    "Dest": "dest",
    "DestState": "dest_state",
    "DepDelay": "departure_delay",
    "CancellationCode": "cancellation_code",
    "CRSElapsedTime": "scheduled_minutes",
    "Distance": "distance",
    "DayOfWeek": "day_of_week",
}


def _clock(name: str) -> pl.Expr:
    """Минуты от полуночи из записи вида «1425».

    «2400» — полночь СЛЕДУЮЩИХ суток, то есть 1440 минут. Оставить его 24-м
    часом значило бы получить недопустимое время; отбросить — потерять рейсы,
    прибывающие ровно в полночь.
    """
    digits = pl.col(name).cast(pl.Int64, strict=False)
    return (digits // 100 * 60 + digits % 100).alias(name)


def _moment(date: str, minutes: str) -> pl.Expr:
    return pl.col(date).dt.combine(pl.time(0, 0)) + pl.duration(minutes=pl.col(minutes))


def build() -> pl.DataFrame:
    """Одна строка на рейс, со сроком и фактическим прибытием."""
    frames = []
    for archive in sorted(RAW.glob("bts_*.zip")):
        with zipfile.ZipFile(archive) as bundle:
            name = next(i.filename for i in bundle.infolist() if i.filename.endswith(".csv"))
            with bundle.open(name) as handle:
                frames.append(pl.read_csv(handle, infer_schema_length=0, columns=KEEP).select(KEEP))
    source = pl.concat(frames, how="vertical")
    read = source.height

    frame = (
        source.rename(RENAMED)
        .with_columns(
            pl.col("flight_date").str.to_date(format="%Y-%m-%d", strict=False),
            _clock("CRSDepTime"),
            _clock("DepTime"),
            _clock("CRSArrTime"),
            _clock("ArrTime"),
            pl.col("departure_delay").cast(pl.Float64, strict=False),
            pl.col("scheduled_minutes").cast(pl.Float64, strict=False),
            pl.col("distance").cast(pl.Float64, strict=False),
        )
        .with_columns(
            _moment("flight_date", "CRSDepTime").alias("scheduled_departure"),
            # Прибытие раньше вылета по часам означает следующие сутки.
            (
                _moment("flight_date", "CRSArrTime")
                + pl.when(pl.col("CRSArrTime") < pl.col("CRSDepTime"))
                .then(pl.duration(days=1))
                .otherwise(pl.duration(days=0))
            ).alias("scheduled_arrival"),
            (
                _moment("flight_date", "ArrTime")
                + pl.when(pl.col("ArrTime") < pl.col("CRSDepTime"))
                .then(pl.duration(days=1))
                .otherwise(pl.duration(days=0))
            ).alias("arrived_at"),
        )
    )

    # Непригодна строка без момента решения либо без расписанного прибытия:
    # без них решение нельзя ни поставить во времени, ни судить.
    #
    # КОНТРОЛЬ К-2. Строки, где прибытие оказалось раньше расписанного вылета,
    # намеренно НЕ отбрасываются: ядро обязано назвать их само.
    broken = (
        pl.col("scheduled_departure").is_null()
        | pl.col("scheduled_arrival").is_null()
        | pl.col("flight_date").is_null()
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
        f"  прочитано {read:,}, потеряно {lost:,} (нет расписания рейса), осталось {kept.height:,}"
    )

    return kept.with_columns(
        # Отсутствие, записанное пустой строкой. Найдено проверкой A14 и
        # исправлено ЗДЕСЬ, при сборке, — по правилу самой проверки.
        #
        # У кода отмены пустая строка стоит у 6 982 746 рейсов и означает
        # «рейс не отменяли»: это законное отсутствие, и оно тоже становится
        # пропуском, а не остаётся значением наравне с 'A', 'B', 'C'.
        *[
            pl.when(pl.col(name).str.len_chars() == 0)
            .then(None)
            .otherwise(pl.col(name))
            .alias(name)
            for name in ("tail_number", "flight_number", "cancellation_code")
        ],
        # Статус рейса одной колонкой: причины отсутствия прибытия различимы.
        pl.when(pl.col("Cancelled") == "1.00")
        .then(pl.lit("cancelled"))
        .when(pl.col("Diverted") == "1.00")
        .then(pl.lit("diverted"))
        .otherwise(pl.lit("completed"))
        .alias("status"),
        pl.concat_str([pl.col("origin"), pl.col("dest")], separator="-").alias("route"),
        pl.concat_str(
            [
                pl.col("carrier"),
                pl.col("flight_number"),
                pl.col("flight_date").cast(pl.Utf8),
                pl.col("origin"),
            ],
            separator="|",
        ).alias("flight_id"),
        *[
            (pl.col("scheduled_arrival") + pl.duration(minutes=minutes)).alias(f"due_{minutes}m")
            for minutes in TOLERANCES
        ],
    )


if __name__ == "__main__":
    table = build()
    print(f"строк: {table.height:,}, колонок: {len(table.columns)}")
    print("период:", table["scheduled_departure"].min(), "..", table["scheduled_departure"].max())
