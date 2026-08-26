"""Сборка таблицы решений седьмого кейса: кредиты SBA малому бизнесу.

Единица решения — кредит. Момент решения — ВЫДАЧА денег, а не одобрение: до
выдачи риска нет, а между двумя датами лежит запаздывание.

Исход — объявлен ли кредит дефолтным до истечения срока, где срок считается
от выдачи и равен `Term` месяцев.

Даты записаны двузначным годом, и это ловушка века: `%y` относит 69–99 к
прошлому веку, а 00–68 к нынешнему, отчего кредиты 1962–1968 годов уезжают в
2062–2068. Ловушка снята не догадкой, а сверкой с `ApprovalFY`, где год
записан четырьмя цифрами.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

PROJECT = Path(__file__).resolve().parent
SOURCE = PROJECT / "data" / "raw" / "SBAnational.csv"

DATE_FORMAT = "%d-%b-%y"
"""Формат дат объявлен явно. Пятый кейс: угадывание обнулило 94 625 строк из
153 810 и переставило местами месяц с днём ещё в 54 927."""

LAST_YEAR = 2014
"""Последний год выгрузки. Дата, разобранная позже него, — жертва ловушки
века, и ей возвращается прошлый век."""

DATES = ("ApprovalDate", "DisbursementDate", "ChgOffDate")

MONEY = ("DisbursementGross", "BalanceGross", "ChgOffPrinGr", "GrAppv", "SBA_Appv")

SENTINELS = {
    "Name": ("N/A", "NA"),
    "NoEmp": ("9999",),
    "RevLineCr": ("-",),
    "City": ("-",),
    "Zip": ("9999",),
}
"""Отсутствие, записанное значением. Найдено проверкой A14 на первом прогоне и
исправлено ЗДЕСЬ, при сборке, а не обходом: правило самой проверки."""

RENAMED = {
    "LoanNr_ChkDgt": "loan_id",
    "Name": "borrower",
    "Bank": "bank",
    "BankState": "bank_state",
    "State": "state",
    "NAICS": "naics",
    "Term": "term_months",
    "NoEmp": "employees",
    "NewExist": "new_business",
    "CreateJob": "jobs_created",
    "RetainedJob": "jobs_retained",
    "FranchiseCode": "franchise_code",
    "UrbanRural": "urban_rural",
    "RevLineCr": "revolving_credit",
    "LowDoc": "low_doc",
    "MIS_Status": "status",
    "ApprovalDate": "approved_at",
    "DisbursementDate": "disbursed_at",
    "ChgOffDate": "charged_off_at",
    "DisbursementGross": "disbursed_amount",
    "GrAppv": "approved_amount",
    "SBA_Appv": "guaranteed_amount",
    "BalanceGross": "balance_gross",
    "ChgOffPrinGr": "charged_off_amount",
}


def _date(name: str) -> pl.Expr:
    """Разобрать дату и вернуть ей век, если она уехала в будущее."""
    parsed = pl.col(name).str.to_date(format=DATE_FORMAT, strict=False)
    return (
        pl.when(parsed.dt.year() > LAST_YEAR)
        .then(parsed.dt.offset_by("-100y"))
        .otherwise(parsed)
        .alias(name)
    )


def _money(name: str) -> pl.Expr:
    """«$60,000.00 » — это число, записанное как оформление."""
    return pl.col(name).str.replace_all(r"[$, ]", "").cast(pl.Float64, strict=False).alias(name)


def build() -> pl.DataFrame:
    """Одна строка на кредит, со сроком, исходом и признаками момента выдачи."""
    source = pl.read_csv(SOURCE, infer_schema_length=0)
    read = source.height

    frame = (
        source.with_columns(
            *[
                pl.when(pl.col(name).is_in(tokens)).then(None).otherwise(pl.col(name)).alias(name)
                for name, tokens in SENTINELS.items()
            ],
        )
        .with_columns(
            *[_date(name) for name in DATES],
            *[_money(name) for name in MONEY],
            pl.col("Term").cast(pl.Int64, strict=False),
        )
        .rename(RENAMED)
    )

    # Строка непригодна, если неизвестен момент решения или срок: без них
    # нельзя ни поставить решение во времени, ни назначить срок. Отдельно —
    # выдача раньше одобрения: 810 строк, у которых порядок событий невозможен.
    # Часть из них — жертвы ловушки века («31-Mar-00» одобрено, «14-Apr-20»
    # выдано), но не все, и чинить их догадкой значило бы выдумывать данные.
    #
    # Дефолт раньше выдачи (7 строк) НЕ отбрасывается намеренно: его обязано
    # заметить ядро. Кейс седьмой проверяет ядро, а не мою внимательность.
    broken = (
        pl.col("disbursed_at").is_null()
        | pl.col("term_months").is_null()
        | (pl.col("term_months") <= 0)
        | pl.col("loan_id").is_null()
        # fill_null(False): сравнение с пустой датой даёт null, а не ложь, и
        # отрицание такого условия выбрасывает строку молча. Правило баланса
        # поймало это на первом же прогоне — 737 715 потерянных вместо 3 994.
        | (pl.col("disbursed_at") < pl.col("approved_at")).fill_null(False)
        # Дефолт раньше выдачи: 7 строк, худшая на 24 573 дня. Оставлены на
        # первом прогоне намеренно — ядро обязано было их заметить. Заметило
        # (проверка A13), и теперь они отбрасываются с названной причиной.
        | (pl.col("charged_off_at") < pl.col("disbursed_at")).fill_null(False)
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
        f"  прочитано {read:,}, потеряно {lost:,} (нет даты выдачи, срока или номера, "
        f"выдача раньше одобрения либо дефолт раньше выдачи), осталось {kept.height:,}"
    )

    return kept.with_columns(
        # Срок: столько месяцев от выдачи. Месяц — не тридцать дней, и считать
        # его днями значило бы отвечать не на заданный вопрос.
        pl.col("disbursed_at")
        .dt.offset_by(pl.concat_str([pl.col("term_months").cast(pl.Utf8), pl.lit("mo")]))
        .alias("due_at"),
        (pl.col("disbursed_at") - pl.col("approved_at")).dt.total_days().alias("approval_lag_days"),
        pl.col("naics").str.slice(0, 2).alias("sector"),
    )


if __name__ == "__main__":
    table = build()
    print(f"строк: {table.height:,}, колонок: {len(table.columns)}")
    print("период выдачи:", table["disbursed_at"].min(), "..", table["disbursed_at"].max())
