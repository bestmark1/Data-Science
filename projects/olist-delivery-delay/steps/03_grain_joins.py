"""Шаг 03 — грануляция и fanout при join.

Заказ, позиции и платежи имеют разную грануляцию. Наивный join задваивает суммы.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

PROJECT = Path(__file__).resolve().parent.parent
PARQUET = PROJECT / "data" / "parquet"
ARTIFACTS = PROJECT / "artifacts"


def main() -> int:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    orders = pl.read_parquet(PARQUET / "olist_orders_dataset.parquet")
    items = pl.read_parquet(PARQUET / "olist_order_items_dataset.parquet")
    payments = pl.read_parquet(PARQUET / "olist_order_payments_dataset.parquet")
    reviews = pl.read_parquet(PARQUET / "olist_order_reviews_dataset.parquet")

    report = ["# Шаг 03 — грануляция и fanout", ""]

    report += ["## Грануляция по order_id", "", "| таблица | строк | уникальных order_id | строк на заказ |", "|---|---|---|---|"]
    for name, frame in [
        ("orders", orders),
        ("order_items", items),
        ("order_payments", payments),
        ("order_reviews", reviews),
    ]:
        unique = frame["order_id"].n_unique()
        report.append(f"| {name} | {frame.height:,} | {unique:,} | {frame.height / unique:.3f} |")
    report.append("")

    # Наивный join: суммируем цену позиций и стоимость платежей вместе.
    naive = orders.join(items, on="order_id", how="inner").join(
        payments, on="order_id", how="inner"
    )
    honest_items = items.group_by("order_id").agg(pl.col("price").sum().alias("items_total"))
    honest_payments = payments.group_by("order_id").agg(
        pl.col("payment_value").sum().alias("paid_total")
    )
    honest = orders.join(honest_items, on="order_id", how="inner").join(
        honest_payments, on="order_id", how="inner"
    )

    report += ["## Цена ошибки", ""]
    report.append(f"строк после наивного join: {naive.height:,}")
    report.append(f"строк после агрегации до join: {honest.height:,}")
    report.append(f"раздувание: x{naive.height / honest.height:.2f}")
    report.append("")
    report.append(f"сумма price при наивном join: {naive['price'].sum():,.2f}")
    report.append(f"сумма price честная: {honest['items_total'].sum():,.2f}")
    report.append(
        f"завышение выручки: x{naive['price'].sum() / honest['items_total'].sum():.2f}"
    )
    report.append("")

    report += ["## Заказы без связанных записей", ""]
    for name, frame in [("позиций", items), ("платежей", payments), ("отзывов", reviews)]:
        orphan = orders.height - orders.join(frame, on="order_id", how="semi").height
        report.append(f"- заказов без {name}: {orphan:,}")
    report.append("")

    # Отзывы: несколько отзывов на заказ означает, что review_id не ключ заказа.
    dup_reviews = reviews.height - reviews["order_id"].n_unique()
    report.append(f"- заказов с несколькими отзывами: {dup_reviews:,}")

    target = ARTIFACTS / "03_grain_joins.md"
    target.write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
