"""Шаг 05 — поиск лика относительно момента решения.

Два слоя. Первый — объявление: для каждой колонки вручную решается, известна ли
она в момент оформления заказа. Второй — эмпирический: сила связи с таргетом,
аномально высокая для признака, который знать заранее нельзя.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

PROJECT = Path(__file__).resolve().parent.parent
PARQUET = PROJECT / "data" / "parquet"
ARTIFACTS = PROJECT / "artifacts"

# Объявление доступности. Заполнено вручную: в данных этого нет нигде.
AVAILABILITY = {
    "order_purchase_timestamp": "at_decision",
    "order_estimated_delivery_date": "at_decision",
    "customer_zip_code_prefix": "at_decision",
    "customer_state": "at_decision",
    "customer_city": "at_decision",
    "items_count": "at_decision",
    "items_total": "at_decision",
    "freight_total": "at_decision",
    "sellers_count": "at_decision",
    "product_weight_g": "at_decision",
    "product_volume_cm3": "at_decision",
    "payment_installments": "at_decision",
    "payment_value": "at_decision",
    "payment_types": "at_decision",
    "order_approved_at": "after",
    "order_delivered_carrier_date": "after",
    "order_delivered_customer_date": "after",
    "order_status": "after",
    "review_score": "after",
    "shipping_limit_max": "ambiguous",
}


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

    delivered = pl.col("order_delivered_customer_date").is_not_null()
    late = pl.col("order_delivered_customer_date") > pl.col("order_estimated_delivery_date")
    estimate_passed = pl.col("order_estimated_delivery_date") < data_end

    labelled = (
        orders.with_columns(
            pl.when(delivered)
            .then(late.cast(pl.Int8))
            .when(estimate_passed)
            .then(pl.lit(1, dtype=pl.Int8))
            .otherwise(None)
            .alias("is_late")
        )
        .filter(pl.col("is_late").is_not_null())
        .select("order_id", "order_purchase_timestamp", "order_status", "is_late")
    )

    items = pl.read_parquet(PARQUET / "olist_order_items_dataset.parquet")
    items_agg = items.group_by("order_id").agg(
        pl.len().alias("items_count"),
        pl.col("price").sum().alias("items_total"),
        pl.col("freight_value").sum().alias("freight_total"),
        pl.col("seller_id").n_unique().alias("sellers_count"),
        pl.col("shipping_limit_date").max().alias("shipping_limit_max"),
    )
    reviews = pl.read_parquet(PARQUET / "olist_order_reviews_dataset.parquet")
    reviews_agg = reviews.group_by("order_id").agg(pl.col("review_score").mean())

    wide = labelled.join(items_agg, on="order_id", how="left").join(
        reviews_agg, on="order_id", how="left"
    )

    report = ["# Шаг 05 — поиск лика", ""]
    report.append(f"Популяция: {wide.height:,} заказов, доля опозданий {wide['is_late'].mean():.2%}")
    report.append("")

    report += [
        "## Объявленная доступность",
        "",
        "| признак | доступен | комментарий |",
        "|---|---|---|",
    ]
    for name, status in sorted(AVAILABILITY.items(), key=lambda kv: (kv[1], kv[0])):
        note = {
            "at_decision": "известен при оформлении",
            "after": "появляется после момента решения — использовать нельзя",
            "ambiguous": "требует доменного решения",
        }[status]
        report.append(f"| {name} | {status} | {note} |")
    report.append("")

    report += [
        "## Эмпирическая связь с таргетом",
        "",
        "| признак | доступен | доля опозданий при низких | при высоких | разрыв |",
        "|---|---|---|---|---|",
    ]
    numeric = [
        c
        for c, dtype in zip(wide.columns, wide.dtypes, strict=True)
        if dtype.is_numeric() and c != "is_late"
    ]
    rows = []
    for name in numeric:
        column = wide[name]
        if column.null_count() == wide.height:
            continue
        low_cut, high_cut = column.quantile(0.25), column.quantile(0.75)
        low = wide.filter(pl.col(name) <= low_cut)["is_late"].mean()
        high = wide.filter(pl.col(name) >= high_cut)["is_late"].mean()
        if low is None or high is None:
            continue
        rows.append((name, abs(high - low), low, high))

    for name, gap, low, high in sorted(rows, key=lambda r: r[1], reverse=True):
        status = AVAILABILITY.get(name, "не объявлен")
        flag = " **подозрительно**" if gap > 0.10 and status != "at_decision" else ""
        report.append(f"| {name} | {status} | {low:.2%} | {high:.2%} | {gap:.1%}{flag} |")
    report.append("")

    # Категориальный: статус заказа знать заранее нельзя, но он объясняет таргет почти полностью.
    report += ["## Статус заказа против таргета", "", "| статус | заказов | доля опозданий |", "|---|---|---|"]
    by_status = (
        wide.group_by("order_status")
        .agg(pl.len().alias("n"), pl.col("is_late").mean().alias("late_rate"))
        .sort("n", descending=True)
    )
    for row in by_status.iter_rows(named=True):
        report.append(f"| {row['order_status']} | {row['n']:,} | {row['late_rate']:.2%} |")
    report.append("")
    report.append(
        "Статус объясняет таргет почти полностью, но известен только после доставки. "
        "Признак такого рода в обучении даст отличную метрику и бесполезную модель."
    )

    target = ARTIFACTS / "05_leakage_hunt.md"
    target.write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
