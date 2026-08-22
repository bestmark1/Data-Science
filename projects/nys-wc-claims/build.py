"""Сборка таблицы решений третьего кейса.

Здесь только то, что нельзя объявить: разбор дат, приведение имён и построение
колонок-кандидатов на срок. Смысловые решения — что считать исходом, какие
колонки доступны в момент решения, какой горизонт — принимаются в форме, а не
здесь.

Единица данных уже готова: одна строка на претензию, ключ уникален по всем
4 232 321 строкам. Соединять нечего, и это первый кейс, где грануляция не
является вопросом.

**Что важно знать о происхождении колонок.** Словарь данных описывает почти все
неврéменные поля как значения НА МОМЕНТ ВЫГРУЗКИ: `Current Claim Status`,
`Claim Injury Type`, `Closed Count`, `Hearing Count`, `IME-4 Count`,
`Highest Process`, `Attorney/Representative`. Претензия, собранная в 2005 году,
несёт статус 2021 года. Отличить такое поле от значения на момент решения по
данным нельзя — их надо объявлять недоступными, и это решение принимает
заполняющий форму.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

PROJECT = Path(__file__).resolve().parent
SOURCE = PROJECT / "data" / "raw" / "assembled-workers-compensation-claims-beginning-2000.csv"

HORIZONS = (30, 90, 180, 365)
"""Кандидаты на срок, в днях от сборки претензии.

Строятся все: какой из них является сроком, объявляется в форме через
`deadline_column`. Неиспользованные объявляются ролью `ignored` — это ответ,
а промолчать нельзя.
"""

DATES = {
    "Assembly Date": "assembled_at",
    "Accident Date": "accident_at",
    "ANCR Date": "established_at",
    "Controverted Date": "controverted_at",
    "Section 32 Date": "section32_at",
    "PPD Scheduled Loss Date": "ppd_scheduled_at",
    "PPD Non-Scheduled Loss Date": "ppd_nonscheduled_at",
    "PTD Date": "ptd_at",
    "First Appeal Date": "first_appeal_at",
    "C-2 Date": "c2_received_at",
    "C-3 Date": "c3_received_at",
    "First Hearing Date": "first_hearing_at",
}

RENAMED = {
    "Claim Identifier": "claim_id",
    "Claim Type": "claim_type",
    "District Name": "district",
    "Average Weekly Wage": "weekly_wage",
    "Current Claim Status": "current_status",
    "Claim Injury Type": "injury_type",
    "Age at Injury": "age_at_injury",
    "Alternative Dispute Resolution": "alternative_dispute_resolution",
    "Gender": "gender",
    "Birth Year": "birth_year",
    "Zip Code": "zip_code",
    "Medical Fee Region": "medical_fee_region",
    "Highest Process": "highest_process",
    "Hearing Count": "hearing_count",
    "Closed Count": "closed_count",
    "Attorney/Representative": "has_representative",
    "Carrier Name": "carrier_name",
    "Carrier Type": "carrier_type",
    "IME-4 Count": "ime4_count",
    "Interval Assembled to ANCR": "days_assembled_to_established",
    "Accident": "is_accident",
    "Occupational Disease": "is_occupational_disease",
    "County of Injury": "county",
    "COVID-19 Indicator": "covid_related",
    "Industry Code": "industry_code",
    "Industry Code Description": "industry",
    "WCIO Part Of Body Code": "wcio_body_code",
    "WCIO Part Of Body Description": "wcio_body",
    "WCIO Nature of Injury Code": "wcio_nature_code",
    "WCIO Nature of Injury Description": "wcio_nature",
    "WCIO Cause of Injury Code": "wcio_cause_code",
    "WCIO Cause of Injury Description": "wcio_cause",
}

DROPPED = (
    "OIICS Part Of Body Code",
    "OIICS Part Of Body Description",
    "OIICS Nature of Injury Code",
    "OIICS Nature of Injury Description",
    "OIICS Injury Source Code",
    "OIICS Injury Source Description",
    "OIICS Event Exposure Code",
    "OIICS Event Exposure Description",
    "OIICS Secondary Source Code",
    "OIICS Secondary Source Description",
)
"""Дублирующая классификация OIICS. Отбрасывается при СБОРКЕ, а не объявляется
ролью ignored: колонки, которой нет в таблице, ядро и не увидит, а решение о
ней остаётся здесь, в коде, где его видно."""


def build(since: str | None = None) -> pl.DataFrame:
    """Одна строка на претензию, даты разобраны, сроки построены.

    `since` отсекает ранние годы. Отсечение — решение о популяции, и оно
    принимается вызывающей стороной, а не прячется в умолчании.
    """
    frame = (
        pl.scan_csv(SOURCE, infer_schema_length=20000, ignore_errors=True)
        .drop(DROPPED)
        .with_columns(
            [pl.col(source).str.to_datetime(strict=False).alias(source) for source in DATES]
        )
        .rename({**DATES, **RENAMED})
    )
    if since is not None:
        frame = frame.filter(pl.col("assembled_at") >= pl.lit(since).str.to_datetime())

    return frame.with_columns(
        pl.col("claim_id").cast(pl.Utf8),
        *[
            (pl.col("assembled_at") + pl.duration(days=days)).alias(f"horizon_{days}d")
            for days in HORIZONS
        ],
    ).collect()


def main() -> int:
    frame = build()
    print(f"строк: {frame.height:,}, колонок: {frame.width}")
    print(
        f"период сборки претензий: {frame['assembled_at'].min()} .. {frame['assembled_at'].max()}"
    )
    print()
    print("колонки таблицы решений:")
    for name in frame.columns:
        print(f"  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
