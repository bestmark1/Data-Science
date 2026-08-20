"""Шаг 06 — обучающая таблица и временной сплит.

Сплит имитирует продакшен на дату отсечки: в обучении участвуют только заказы,
исход которых был ИЗВЕСТЕН к этой дате. Это строже фиксированного зазора по
дате покупки — заказ, купленный давно, но узнанный поздно, тоже исключается.

Признаки берутся только из объявленных доступными в момент оформления.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import polars as pl

PROJECT = Path(__file__).resolve().parent.parent
PARQUET = PROJECT / "data" / "parquet"
ARTIFACTS = PROJECT / "artifacts"

PERIOD_START = dt.datetime(2017, 1, 1)
PERIOD_STOP = dt.datetime(2018, 8, 21)
TEST_START = dt.datetime(2018, 6, 1)
NOT_INTENDED_FOR_DELIVERY = ["canceled", "unavailable"]


def build_table() -> pl.DataFrame:
    orders = pl.read_parquet(PARQUET / "olist_orders_dataset.parquet").with_columns(
        [
            pl.col(c).str.to_datetime(strict=False)
            for c in (
                "order_purchase_timestamp",
                "order_delivered_customer_date",
                "order_estimated_delivery_date",
            )
        ]
    )
    snapshot_date = max(
        orders["order_purchase_timestamp"].max(), orders["order_delivered_customer_date"].max()
    ).date()

    delivered_on = pl.col("order_delivered_customer_date").dt.date()
    promised_on = pl.col("order_estimated_delivery_date").dt.date()
    has_delivery = pl.col("order_delivered_customer_date").is_not_null()

    labelled = (
        orders.filter(
            (pl.col("order_purchase_timestamp") >= PERIOD_START)
            & (pl.col("order_purchase_timestamp") < PERIOD_STOP)
            & ~pl.col("order_status").is_in(NOT_INTENDED_FOR_DELIVERY)
        )
        .with_columns(
            pl.when(has_delivery & (delivered_on <= promised_on))
            .then(0)
            .when(has_delivery | (promised_on < pl.lit(snapshot_date)))
            .then(1)
            .otherwise(None)
            .cast(pl.Int8)
            .alias("is_late"),
            pl.when(has_delivery & (delivered_on <= promised_on))
            .then(pl.col("order_delivered_customer_date"))
            .otherwise(pl.col("order_estimated_delivery_date").dt.offset_by("1d"))
            .alias("label_known_at"),
        )
        .filter(pl.col("is_late").is_not_null())
    )

    # Позиции: агрегируем ДО join, иначе задваиваем суммы (шаг 03).
    items = pl.read_parquet(PARQUET / "olist_order_items_dataset.parquet")
    products = pl.read_parquet(PARQUET / "olist_products_dataset.parquet")
    sellers = pl.read_parquet(PARQUET / "olist_sellers_dataset.parquet")

    items_enriched = items.join(products, on="product_id", how="left").join(
        sellers, on="seller_id", how="left"
    )
    items_agg = items_enriched.group_by("order_id").agg(
        pl.len().alias("items_count"),
        pl.col("price").sum().alias("items_total"),
        pl.col("freight_value").sum().alias("freight_total"),
        pl.col("seller_id").n_unique().alias("sellers_count"),
        pl.col("seller_state").mode().first().alias("seller_state"),
        pl.col("seller_state").n_unique().alias("seller_states_count"),
        pl.col("product_weight_g").sum().alias("weight_total_g"),
        (pl.col("product_length_cm") * pl.col("product_height_cm") * pl.col("product_width_cm"))
        .sum()
        .alias("volume_total_cm3"),
        pl.col("product_category_name").mode().first().alias("main_category"),
    )

    payments = pl.read_parquet(PARQUET / "olist_order_payments_dataset.parquet")
    payments_agg = payments.group_by("order_id").agg(
        pl.col("payment_value").sum().alias("payment_value"),
        pl.col("payment_installments").max().alias("payment_installments"),
        pl.col("payment_type").mode().first().alias("payment_type"),
        pl.len().alias("payments_count"),
    )

    customers = pl.read_parquet(PARQUET / "olist_customers_dataset.parquet").select(
        "customer_id", "customer_state", "customer_zip_code_prefix"
    )

    table = (
        labelled.join(items_agg, on="order_id", how="inner")
        .join(payments_agg, on="order_id", how="inner")
        .join(customers, on="customer_id", how="inner")
        .with_columns(
            (pl.col("order_estimated_delivery_date") - pl.col("order_purchase_timestamp"))
            .dt.total_days()
            .alias("promised_lead_days"),
            pl.col("order_purchase_timestamp").dt.month().alias("purchase_month"),
            pl.col("order_purchase_timestamp").dt.weekday().alias("purchase_weekday"),
            pl.col("order_purchase_timestamp").dt.hour().alias("purchase_hour"),
            (pl.col("seller_state") != pl.col("customer_state")).alias("cross_state"),
            (pl.col("freight_total") / (pl.col("items_total") + 1)).alias("freight_ratio"),
        )
    )
    return table


def main() -> int:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    table = build_table()

    # Обучение видит только то, что было известно к моменту отсечки.
    train = table.filter(pl.col("label_known_at") < TEST_START)
    test = table.filter(pl.col("order_purchase_timestamp") >= TEST_START)

    dropped = table.height - train.height - test.height

    report = ["# Шаг 06 — обучающая таблица и сплит", ""]
    report.append(f"Строк в таблице: {table.height:,}, признаков: {table.width}")
    report.append("")
    report += ["## Сплит", "", "| выборка | заказов | период покупки | доля опозданий |", "|---|---|---|---|"]
    for name, part in (("train", train), ("test", test)):
        lo = part["order_purchase_timestamp"].min()
        hi = part["order_purchase_timestamp"].max()
        report.append(
            f"| {name} | {part.height:,} | {lo:%Y-%m-%d} .. {hi:%Y-%m-%d} | "
            f"{part['is_late'].mean():.2%} |"
        )
    report.append("")
    report.append(
        f"Не попало ни в одну выборку: {dropped:,} — заказы, купленные до отсечки, "
        f"но с исходом, ставшим известным после неё. Использовать их в обучении "
        f"значило бы знать будущее."
    )
    report.append("")

    report += ["## Дрейф доли опозданий по месяцам", "", "| месяц | заказов | доля опозданий |", "|---|---|---|"]
    monthly = (
        table.with_columns(pl.col("order_purchase_timestamp").dt.truncate("1mo").alias("m"))
        .group_by("m")
        .agg(pl.len().alias("n"), pl.col("is_late").mean().alias("rate"))
        .sort("m")
    )
    for row in monthly.iter_rows(named=True):
        report.append(f"| {row['m']:%Y-%m} | {row['n']:,} | {row['rate']:.2%} |")

    train.write_parquet(ARTIFACTS / "train.parquet")
    test.write_parquet(ARTIFACTS / "test.parquet")
    (ARTIFACTS / "06_split.md").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
