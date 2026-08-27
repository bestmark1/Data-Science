"""Сборка таблицы решений десятого кейса: проверки площадок по опасным отходам.

Единица решения — проверка. Момент решения — её начало. Исход — будет ли по
следам проверки установлено нарушение в течение объявленного горизонта.

Ось кейса — ПОВОД проверки. Источник записывает его кодом (`EVALUATION_TYPE`),
и разбиение кодов на вызванные сигналом и плановые объявлено в `triggers.py`
до вычисления любой доли исхода.

СВЯЗЬ НАРУШЕНИЯ С ПРОВЕРКОЙ. В выгрузке нарушение привязано к площадке, а не к
проверке: общего ключа нет. Поэтому исходом проверки считается ближайшее
нарушение, установленное НА ЭТОЙ ПЛОЩАДКЕ не раньше её начала. Допущение
названо в форме: две близкие проверки одной площадки получат одно и то же
нарушение, и различить их вклад нечем.

ВНИМАНИЕ. В первый прогон намеренно внесены положительные контроли §5
пре-регистрации; они помечены словом КОНТРОЛЬ.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import polars as pl

from dsx.join import AsofDirection, Cardinality, guarded_asof_join, guarded_join

PROJECT = Path(__file__).resolve().parent
RAW = PROJECT / "data" / "raw"

DATE_FORMAT = "%m/%d/%Y"
"""Формат дат объявляется явно. Пятый кейс: угадывание обнулило 94 625 строк
из 153 810 и переставило местами месяц с днём ещё в 54 927."""

SNAPSHOT = dt.date(2026, 8, 25)
"""Дата, на которую собрана выгрузка: отметка времени файлов в архиве EPA."""

HORIZONS = (30, 90, 180, 365)
"""Кандидаты на горизонт. Берётся наименьший с долей в [20%, 80%] — правило
объявлено до контакта, приём восьмого кейса."""


def _read(name: str, columns: list[str]) -> pl.DataFrame:
    return pl.read_csv(RAW / name, infer_schema_length=0, columns=columns)


def _date(name: str) -> pl.Expr:
    return pl.col(name).str.to_date(format=DATE_FORMAT, strict=False).alias(name)


def build() -> pl.DataFrame:
    """Одна строка на проверку, с поводом, сроком и ближайшим нарушением."""
    evaluations = _read(
        "RCRA_EVALUATIONS.csv",
        [
            "ID_NUMBER",
            "ACTIVITY_LOCATION",
            "EVALUATION_IDENTIFIER",
            "EVALUATION_TYPE",
            "EVALUATION_AGENCY",
            "EVALUATION_START_DATE",
            "FOUND_VIOLATION",
        ],
    ).with_columns(_date("EVALUATION_START_DATE"))
    read = evaluations.height

    facilities = _read(
        "RCRA_FACILITIES.csv",
        [
            "ID_NUMBER",
            "STATE_CODE",
            "FED_WASTE_GENERATOR",
            "TRANSPORTER",
            "ACTIVE_SITE",
            "OPERATING_TSDF",
        ],
    ).unique(subset=["ID_NUMBER"], keep="first", maintain_order=True)

    sectors = (
        _read("RCRA_NAICS.csv", ["ID_NUMBER", "NAICS_CODE"])
        .with_columns(pl.col("NAICS_CODE").str.slice(0, 2).alias("sector"))
        .unique(subset=["ID_NUMBER"], keep="first", maintain_order=True)
        .select("ID_NUMBER", "sector")
    )

    violations = (
        _read("RCRA_VIOLATIONS.csv", ["ID_NUMBER", "DATE_VIOLATION_DETERMINED"])
        .with_columns(_date("DATE_VIOLATION_DETERMINED"))
        .drop_nulls("DATE_VIOLATION_DETERMINED")
        .rename({"DATE_VIOLATION_DETERMINED": "violation_at"})
        .sort("violation_at")
    )

    # Грануляция объявляется у обоих справочников: одна строка на площадку.
    frame = guarded_join(
        evaluations, facilities, on=["ID_NUMBER"], expect=Cardinality.MANY_TO_ONE, how="left"
    )
    frame = guarded_join(
        frame, sectors, on=["ID_NUMBER"], expect=Cardinality.MANY_TO_ONE, how="left"
    )

    # Ближайшее нарушение НЕ РАНЬШЕ начала проверки. Соединение вперёд включает
    # и сам день проверки: нарушение, установленное в этот день, считается её
    # исходом.
    #
    # Первая версия сдвигала левый ключ на день назад «ради нестрогой границы»,
    # и этим втягивала нарушения, установленные НАКАНУНЕ проверки: 4 278 строк,
    # у которых событие оказывалось раньше решения. Поймала это проверка A13 —
    # дефект был мой, а не источника.
    ordered = frame.sort("EVALUATION_START_DATE")
    with_violation = guarded_asof_join(
        ordered,
        violations,
        left_on="EVALUATION_START_DATE",
        right_on="violation_at",
        by=["ID_NUMBER"],
        direction=AsofDirection.FORWARD,
    )

    # Закон, по которому эти проверки проводятся, принят в 1976 году. Даты
    # раньше него невозможны: в выгрузке встречаются годы 0005 и 1900-е — три
    # десятка записей с испорченной датой. Граница взята из закона, а не
    # подобрана по данным.
    # История площадки: сколько нарушений установлено на ней СТРОГО РАНЬШЕ этой
    # проверки. Считается слиянием двух потоков и накопительной суммой, а не
    # соединением: соединять тут нечего.
    #
    # Отбор идёт по времени установления нарушения — по тому, когда о нём стало
    # известно. Это и объявляется часами окна (window_clock).
    stream = pl.concat(
        [
            with_violation.select(
                "ID_NUMBER",
                pl.col("EVALUATION_START_DATE").alias("__at"),
                pl.lit(0, dtype=pl.Int32).alias("__is_violation"),
                pl.int_range(pl.len(), dtype=pl.Int64).alias("__row"),
            ),
            violations.select(
                "ID_NUMBER",
                pl.col("violation_at").alias("__at"),
                pl.lit(1, dtype=pl.Int32).alias("__is_violation"),
                pl.lit(None, dtype=pl.Int64).alias("__row"),
            ),
        ],
        how="vertical",
    ).sort("ID_NUMBER", "__at", "__is_violation", descending=[False, False, True])

    history = (
        stream.with_columns(
            (pl.col("__is_violation").cum_sum().over("ID_NUMBER") - pl.col("__is_violation")).alias(
                "prior_violations"
            )
        )
        .filter(pl.col("__row").is_not_null())
        .sort("__row")
        .select("prior_violations")
    )
    with_violation = with_violation.with_columns(history["prior_violations"])

    broken = (
        pl.col("EVALUATION_START_DATE").is_null()
        | pl.col("EVALUATION_IDENTIFIER").is_null()
        | pl.col("ID_NUMBER").is_null()
        | (pl.col("EVALUATION_START_DATE") < pl.date(1976, 1, 1)).fill_null(False)
    )
    unusable = with_violation.filter(broken).height
    kept = with_violation.filter(~broken)
    lost = read - kept.height
    if lost != unusable:
        raise AssertionError(
            f"баланс строк не сошёлся: прочитано {read:,}, потеряно {lost:,}, "
            f"непригодных {unusable:,}. Строки теряются не по объявленной причине"
        )
    print(
        f"  прочитано {read:,}, потеряно {lost:,} (нет даты проверки, её номера, "
        f"площадки либо дата раньше закона 1976 года), осталось {kept.height:,}"
    )

    import sys

    sys.path.insert(0, str(PROJECT))
    from triggers import trigger_class

    return kept.with_columns(
        # Единица решения. Номер проверки сам по себе НЕ уникален: он
        # уникален лишь внутри площадки и органа. Ключ собирается явно, потому
        # что ядро требует одну колонку, а «одно решение» здесь — это
        # проверка на площадке в конкретный день.
        pl.concat_str(
            [
                pl.col("ID_NUMBER"),
                pl.col("ACTIVITY_LOCATION"),
                pl.col("EVALUATION_IDENTIFIER"),
                pl.col("EVALUATION_START_DATE").cast(pl.Utf8),
            ],
            separator="|",
        ).alias("evaluation_id"),
        pl.col("EVALUATION_TYPE")
        .map_elements(trigger_class, return_dtype=pl.Utf8)
        .alias("trigger_kind"),
        # Момент, НА КОТОРЫЙ взяты свойства площадки. Справочник отдаёт
        # СЕГОДНЯШНЕЕ состояние: действует ли площадка, возит ли отходы. В
        # момент проверки эти значения могли быть иными, и колонка нужна,
        # чтобы объявление можно было сверить, а не принять на слово.
        pl.lit(SNAPSHOT).alias("snapshot_at"),
        *[
            (pl.col("EVALUATION_START_DATE") + pl.duration(days=days)).alias(f"horizon_{days}d")
            for days in HORIZONS
        ],
    )


if __name__ == "__main__":
    table = build()
    print(f"строк: {table.height:,}, колонок: {len(table.columns)}")
    print(
        "период проверок:",
        table["EVALUATION_START_DATE"].min(),
        "..",
        table["EVALUATION_START_DATE"].max(),
    )
