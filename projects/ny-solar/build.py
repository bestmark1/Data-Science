"""Сборка таблицы решений восьмого кейса: заявки на солнечные установки NYSERDA.

Единица решения — заявка. Момент решения — её подача. Исход — будет ли проект
завершён в течение 180 дней после подачи.

Горизонт выбран ПРАВИЛОМ, объявленным до контакта с данными: наименьший из
кандидатов 180/365/545/730, у которого доля попадает в [20%, 80%]. Доли вышли
75.2 / 92.9 / 96.3 / 97.5 процента, и правило выбрало 180 без моего участия.
Шестой кейс выбирал горизонт по аналогии и получил вырожденные 95.3%.

Положительные контроли пре-регистрации §5 были внесены в первый прогон и
СНЯТЫ после того, как ядро назвало каждый из них само:

  К-1  days_to_complete объявлялся признаком момента подачи — поймано двумя
       проверками сразу: сила разделения 0.50 при типичной 0.08, и пропуск,
       объясняемый исходом;
  К-2  166 строк с завершением раньше заявки — поймано, худшая на 1 063 дня;
  К-3  часовые 'None' (5 строк) и 'N/A' (1) — поймано.

Здесь они сняты делом: строки отброшены с названной причиной, часовые
превращены в пропуск при СБОРКЕ, ложное объявление убрано из формы.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

PROJECT = Path(__file__).resolve().parent
SOURCE = PROJECT / "data" / "raw" / "ny_solar_projects.csv"

DATE_FORMAT = "%m/%d/%Y"
"""Формат объявлен явно. Пятый кейс: угадывание обнулило 94 625 строк из
153 810 и переставило местами месяц с днём ещё в 54 927."""

HORIZON_DAYS = 180

DATES = ("Reporting Period", "Date Application Received", "Date Completed")

NUMBERS = (
    "Project Cost",
    "Total NYSERDA Incentive",
    "Total Nameplate kW DC",
    "Expected KWh Annual Production",
    "Total Inverter Quantity",
    "Total PV Module Quantity",
)

RENAMED = {
    "Project Number": "project_id",
    "Reporting Period": "snapshot_at",
    "Date Application Received": "applied_at",
    "Date Completed": "completed_at",
    "Project Status": "status",
    "Contractor": "contractor",
    "Sector": "sector",
    "Program Type": "program_type",
    "Purchase Type": "purchase_type",
    "Electric Utility": "utility",
    "County": "county",
    "Municipality Type": "municipality_type",
    "NYS Disadvantaged Community Status": "disadvantaged_status",
    "Project Cost": "project_cost",
    "Total NYSERDA Incentive": "incentive",
    "Total Nameplate kW DC": "nameplate_kw",
    "Expected KWh Annual Production": "expected_kwh",
    "Total Inverter Quantity": "inverter_count",
    "Total PV Module Quantity": "module_count",
    "Remote Net Metering": "remote_net_metering",
    "Primary Inverter Manufacturer": "inverter_maker",
    "Primary PV Module Manufacturer": "module_maker",
}

KEEP_AS_IS = (
    "Legacy Project Number",
    "CESIR Number",
    "Street Address",
    "City",
    "State",
    "ZIP Code",
    "Incorporated Municipality",
    "Census Tract",
    "Climate and Economic Justice Screening Tool Status",
    "Solicitation",
    "Minority or Women Owned Business Enterprise (MWBE)",
    "Primary Inverter Model Number",
    "PV Module Model Number",
    "Affordable Solar Residential Adder",
    "Affordable Multifamily Housing Incentive",
    "Community Adder",
    "Inclusive Community Solar Adder",
    "Expanded Solar For All Adder",
    "Statewide Solar For All Adder",
    "Brownfield/Landfill Adder",
    "Canopy Adder",
    "Floating Solar Adder",
    "Prevailing Wage Adder",
    "Community Distributed Generation",
    "Green Jobs Green New York Participant",
    "Latitude",
    "Longitude",
    "Georeference",
)
"""Колонки, которые в постановке не участвуют. Переименовываются в простые
имена, чтобы схема могла объявить их ролью 'ignored': ядро видит только
объявленное, и промолчать о колонке нельзя."""


def build() -> pl.DataFrame:
    """Одна строка на заявку, со сроком, исходом и признаками момента подачи."""
    source = pl.read_csv(SOURCE, infer_schema_length=0)
    read = source.height

    plain = {
        name: name.lower().replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")
        for name in KEEP_AS_IS
    }
    frame = source.rename({**RENAMED, **plain}).with_columns(
        *[
            pl.col(RENAMED[name]).str.to_date(format=DATE_FORMAT, strict=False).alias(RENAMED[name])
            for name in DATES
        ],
        *[
            pl.col(RENAMED[name]).cast(pl.Float64, strict=False).alias(RENAMED[name])
            for name in NUMBERS
        ],
    )

    # Непригодна строка без момента решения, без номера либо с завершением
    # РАНЬШЕ подачи заявки: 166 строк, худшая на 1 063 дня. Последние были
    # оставлены в первом прогоне намеренно (контроль К-2) и отброшены после
    # того, как ядро их назвало.
    broken = (
        pl.col("applied_at").is_null()
        | pl.col("project_id").is_null()
        | (pl.col("completed_at") < pl.col("applied_at")).fill_null(False)
    )
    unusable = frame.filter(broken).height
    kept = frame.filter(~broken)

    lost = read - kept.height
    if lost != unusable:
        raise AssertionError(
            f"баланс строк не сошёлся: прочитано {read:,}, потеряно {lost:,}, "
            f"непригодных {unusable:,}. Расхождение означает, что строки теряются "
            "не по объявленной причине"
        )
    print(
        f"  прочитано {read:,}, потеряно {lost:,} (нет даты заявки, номера либо "
        f"завершение раньше заявки), осталось {kept.height:,}"
    )

    return kept.with_columns(
        # Отсутствие, записанное значением: 'None' у производителя инвертора
        # (5 строк) и 'N/A' в номере CESIR (1). Был контроль К-3; ядро нашло
        # оба само, и они превращены в пропуск ЗДЕСЬ, при сборке, — по правилу
        # самой проверки, а не обходом.
        pl.when(pl.col("inverter_maker") == "None")
        .then(None)
        .otherwise(pl.col("inverter_maker"))
        .alias("inverter_maker"),
        pl.when(pl.col("cesir_number") == "N/A")
        .then(None)
        .otherwise(pl.col("cesir_number"))
        .alias("cesir_number"),
        (pl.col("applied_at") + pl.duration(days=HORIZON_DAYS)).alias("due_at"),
        # Вычислена из даты завершения и известна только ПОСЛЕ исхода. Была
        # контролем К-1; в форме объявлена ролью 'ignored'. Колонка оставлена
        # намеренно: убрать её значило бы стереть след контроля.
        (pl.col("completed_at") - pl.col("applied_at")).dt.total_days().alias("days_to_complete"),
    )


if __name__ == "__main__":
    table = build()
    print(f"строк: {table.height:,}, колонок: {len(table.columns)}")
    print("период заявок:", table["applied_at"].min(), "..", table["applied_at"].max())
    print("срез источника:", table["snapshot_at"].max())
