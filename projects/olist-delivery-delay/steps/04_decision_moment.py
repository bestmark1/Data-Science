"""Шаг 04 — момент решения, таргет и момент узнавания метки.

Переписан после внешнего ревью. Три исправления против первой версии:

1. Опоздание сравнивается ПО КАЛЕНДАРНОЙ ДАТЕ. Обещанная дата хранится как
   полночь, поэтому сравнение timestamp помечало опоздавшими все доставки
   в обещанный день.
2. Отменённые и недоступные заказы исключены из популяции: их не собирались
   доставлять, и склейка с опозданиями меняет сам вопрос.
3. Момент узнавания метки — не дата доставки. Если к концу обещанного дня
   доставки нет, исход уже известен. От этого считается зазор для сплита.
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
NOT_INTENDED_FOR_DELIVERY = ["canceled", "unavailable"]


def main() -> int:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
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

    # Конец наблюдения — последнее ФАКТИЧЕСКОЕ событие, а не плановая дата.
    snapshot = max(
        orders["order_purchase_timestamp"].max(),
        orders["order_delivered_customer_date"].max(),
    )
    snapshot_date = snapshot.date()

    report = ["# Шаг 04 — момент решения, таргет, момент узнавания метки", ""]
    report.append("Момент решения: `order_purchase_timestamp`")
    report.append(f"Конец наблюдения: {snapshot} (по фактическим событиям)")
    report.append("")

    in_period = orders.filter(
        (pl.col("order_purchase_timestamp") >= PERIOD_START)
        & (pl.col("order_purchase_timestamp") < PERIOD_STOP)
    )
    population = in_period.filter(~pl.col("order_status").is_in(NOT_INTENDED_FOR_DELIVERY))

    delivered_on = pl.col("order_delivered_customer_date").dt.date()
    promised_on = pl.col("order_estimated_delivery_date").dt.date()
    has_delivery = pl.col("order_delivered_customer_date").is_not_null()
    promise_expired = promised_on < pl.lit(snapshot_date)

    marked = population.with_columns(
        pl.when(has_delivery & (delivered_on <= promised_on))
        .then(pl.lit("on_time"))
        .when(has_delivery)
        .then(pl.lit("late_delivered"))
        .when(promise_expired)
        .then(pl.lit("late_not_delivered"))
        .otherwise(pl.lit("unobservable"))
        .alias("outcome"),
        # Исход известен в момент доставки, если она вовремя;
        # иначе — в конце обещанного дня, ждать фактической доставки не нужно.
        pl.when(has_delivery & (delivered_on <= promised_on))
        .then(pl.col("order_delivered_customer_date"))
        .otherwise(pl.col("order_estimated_delivery_date").dt.offset_by("1d"))
        .alias("label_known_at"),
    )

    report += ["## Исходы", "", "| исход | заказов | доля |", "|---|---|---|"]
    counts = marked.group_by("outcome").agg(pl.len().alias("n")).sort("n", descending=True)
    for row in counts.iter_rows(named=True):
        report.append(f"| {row['outcome']} | {row['n']:,} | {row['n'] / marked.height:.2%} |")
    report.append("")

    usable = marked.filter(pl.col("outcome") != "unobservable").with_columns(
        pl.col("outcome").str.starts_with("late").cast(pl.Int8).alias("is_late")
    )
    rate = usable["is_late"].mean()

    report += ["## Популяция", ""]
    report.append(f"- всего заказов в файле: {orders.height:,}")
    report.append(
        f"- в пригодном периоде {PERIOD_START:%Y-%m-%d} .. {PERIOD_STOP:%Y-%m-%d}: {in_period.height:,}"
    )
    report.append(
        f"- после исключения {NOT_INTENDED_FOR_DELIVERY}: {population.height:,} "
        f"(исключено {in_period.height - population.height:,})"
    )
    report.append(f"- обучающая популяция: {usable.height:,}")
    report.append(f"- **доля опозданий: {rate:.2%}**")
    report.append("")

    report += ["## Задержка метки", ""]
    lag = (
        usable.select(
            (pl.col("label_known_at") - pl.col("order_purchase_timestamp")).dt.total_days()
        )
        .to_series()
        .drop_nulls()
    )
    report.append(f"- медиана: {lag.median():.0f} дней")
    report.append(f"- p95: {lag.quantile(0.95):.0f} дней")
    report.append(f"- максимум: {lag.max():.0f} дней")
    report.append("")
    report.append(
        "Считается от момента узнавания исхода, а не от фактической доставки. "
        "Заказ, не приехавший к обещанной дате, известен как опоздавший сразу, "
        "даже если фактически приедет через полгода."
    )

    target = ARTIFACTS / "04_decision_moment.md"
    target.write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
