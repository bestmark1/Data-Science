"""Шаг 02 — временная ось заказа.

Пять меток на заказ. Задача: понять, какая из них event_time, какая приближает
available_at, и в каком порядке они обязаны идти.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

PROJECT = Path(__file__).resolve().parent.parent
PARQUET = PROJECT / "data" / "parquet"
ARTIFACTS = PROJECT / "artifacts"

STAMPS = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
]

# Порядок, в котором метки обязаны идти по смыслу процесса.
EXPECTED_ORDER = [
    ("order_purchase_timestamp", "order_approved_at"),
    ("order_approved_at", "order_delivered_carrier_date"),
    ("order_delivered_carrier_date", "order_delivered_customer_date"),
]


def main() -> int:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    orders = pl.read_parquet(PARQUET / "olist_orders_dataset.parquet")

    report = ["# Шаг 02 — временная ось", ""]
    report.append(f"Типы до разбора: {[str(orders[c].dtype) for c in STAMPS]}")
    report.append("")

    parsed = orders.with_columns([pl.col(c).str.to_datetime(strict=False).alias(c) for c in STAMPS])

    report += [
        "## Диапазоны",
        "",
        "| метка | минимум | максимум | пропусков |",
        "|---|---|---|---|",
    ]
    for stamp in STAMPS:
        column = parsed[stamp]
        report.append(
            f"| {stamp} | {column.min()} | {column.max()} | "
            f"{column.null_count():,} ({column.null_count() / parsed.height:.1%}) |"
        )
    report.append("")

    report += ["## Нарушения ожидаемого порядка", "", "| пара | нарушений |", "|---|---|"]
    for earlier, later in EXPECTED_ORDER:
        broken = parsed.filter(pl.col(later) < pl.col(earlier)).height
        report.append(f"| {earlier} -> {later} | {broken:,} |")
    report.append("")

    report += [
        "## Статусы заказов и доставка",
        "",
        "| статус | заказов | без даты доставки |",
        "|---|---|---|",
    ]
    by_status = (
        parsed.group_by("order_status")
        .agg(
            pl.len().alias("orders"),
            pl.col("order_delivered_customer_date").null_count().alias("no_delivery"),
        )
        .sort("orders", descending=True)
    )
    for row in by_status.iter_rows(named=True):
        report.append(f"| {row['order_status']} | {row['orders']:,} | {row['no_delivery']:,} |")
    report.append("")

    target = ARTIFACTS / "02_time_axis.md"
    target.write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
