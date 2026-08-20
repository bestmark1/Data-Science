"""Шаг 04 — момент решения и определение таргета.

Задача: в момент оформления заказа предсказать, будет ли доставка позже
обещанной даты. Момент решения — order_purchase_timestamp. Всё, что после,
использовать нельзя.

Трудность не в формуле таргета, а в том, что делать с заказами, у которых
исход не наблюдаем.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

PROJECT = Path(__file__).resolve().parent.parent
PARQUET = PROJECT / "data" / "parquet"
ARTIFACTS = PROJECT / "artifacts"

DATA_END = None  # заполняется из данных


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

    data_end = orders["order_purchase_timestamp"].max()
    report = ["# Шаг 04 — момент решения и таргет", ""]
    report.append(f"Момент решения: `order_purchase_timestamp`")
    report.append(f"Конец наблюдения: {data_end}")
    report.append("")

    delivered = pl.col("order_delivered_customer_date").is_not_null()
    late_delivered = pl.col("order_delivered_customer_date") > pl.col(
        "order_estimated_delivery_date"
    )
    estimate_passed = pl.col("order_estimated_delivery_date") < data_end

    marked = orders.with_columns(
        pl.when(delivered)
        .then(pl.when(late_delivered).then(pl.lit("late")).otherwise(pl.lit("on_time")))
        .when(estimate_passed)
        .then(pl.lit("late_never_delivered"))
        .otherwise(pl.lit("unobservable"))
        .alias("outcome")
    )

    report += ["## Наблюдаемость исхода", "", "| исход | заказов | доля |", "|---|---|---|"]
    counts = marked.group_by("outcome").agg(pl.len().alias("n")).sort("n", descending=True)
    for row in counts.iter_rows(named=True):
        report.append(f"| {row['outcome']} | {row['n']:,} | {row['n'] / marked.height:.2%} |")
    report.append("")

    report += ["## Что даёт наивная разметка", ""]
    naive_positive = marked.filter(delivered & late_delivered).height
    honest_positive = marked.filter(pl.col("outcome").str.starts_with("late")).height
    report.append(f"- только доставленные, опоздание: {naive_positive:,}")
    report.append(f"- плюс недоставленные с истёкшим сроком: {honest_positive:,}")
    report.append(
        f"- пропущено положительных при наивной разметке: "
        f"{honest_positive - naive_positive:,} "
        f"({(honest_positive - naive_positive) / honest_positive:.1%} от всех опозданий)"
    )
    report.append("")

    usable = marked.filter(pl.col("outcome") != "unobservable")
    rate = usable.filter(pl.col("outcome").str.starts_with("late")).height / usable.height
    report += ["## Итоговая популяция", ""]
    report.append(f"- пригодных для обучения заказов: {usable.height:,}")
    report.append(f"- исключено как ненаблюдаемые: {marked.height - usable.height:,}")
    report.append(f"- доля положительного класса: {rate:.2%}")
    report.append("")

    report += ["## Задержка появления метки", ""]
    lag = (
        marked.filter(delivered)
        .select(
            (
                pl.col("order_delivered_customer_date") - pl.col("order_purchase_timestamp")
            ).dt.total_days()
        )
        .to_series()
    )
    report.append(f"- медиана дней от заказа до доставки: {lag.median():.0f}")
    report.append(f"- 95-й процентиль: {lag.quantile(0.95):.0f}")
    report.append(f"- максимум: {lag.max():.0f}")
    report.append("")
    report.append(
        "Метка становится известна в среднем через две недели после момента решения. "
        "Это определяет минимальный зазор между обучающим и тестовым периодами."
    )

    target = ARTIFACTS / "04_decision_moment.md"
    target.write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
