"""Сборка таблицы решений четвёртого кейса.

Момент решения — подписка. Вопрос: отменит ли клиент подписку в течение
горизонта после её оформления.

Четыре таблицы разной грануляции, и соединения объявляются: клиент к подписке
один к одному, продукт к подписке справочником. Обращения в поддержку
НЕ присоединяются: они происходят после момента решения, и включать их значило
бы дать признаку знать будущее. Их место — во втором протоколе со скользящим
моментом решения, если он будет.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from dsx.join import Cardinality, guarded_join

PROJECT = Path(__file__).resolve().parent
RAW = PROJECT / "data" / "raw"

HORIZONS = (30, 90, 180, 365)
"""Кандидаты на срок в днях от оформления подписки. Строятся все; какой из них
является сроком, объявляется в форме."""

NULLS = ["NA"]
"""Отсутствие в этой выгрузке записано строкой «NA».

Без этого объявления Polars читает её значением, и колонка отмены выглядит
заполненной у всех: первый профиль дал «100% отменённых» вместо 22.1%.
"""


def build() -> pl.DataFrame:
    """Одна строка на подписку."""
    product = pl.read_csv(RAW / "customer_product.csv", null_values=NULLS).drop("")
    info = pl.read_csv(RAW / "customer_info.csv", null_values=NULLS).drop("")
    catalogue = pl.read_csv(RAW / "product_info.csv", null_values=NULLS)

    subscriptions = product.with_columns(
        pl.col("signup_date_time").str.to_datetime(strict=False).alias("signed_up_at"),
        pl.col("cancel_date_time").str.to_datetime(strict=False).alias("cancelled_at"),
    ).drop("signup_date_time", "cancel_date_time")

    # Клиент к подписке: ровно одна подписка на клиента, повторов нет — это
    # проверено значениями и записано поправкой A-2.
    with_customer = guarded_join(
        subscriptions, info, on=["customer_id"], expect=Cardinality.MANY_TO_ONE, how="left"
    )
    with_product = guarded_join(
        with_customer,
        catalogue.rename({"product_id": "product"}),
        on=["product"],
        expect=Cardinality.MANY_TO_ONE,
        how="left",
    )

    return with_product.with_columns(
        *[
            (pl.col("signed_up_at") + pl.duration(days=days)).alias(f"horizon_{days}d")
            for days in HORIZONS
        ],
    ).sort("signed_up_at")


def main() -> int:
    frame = build()
    print(f"строк: {frame.height:,}, колонок: {frame.width}")
    print(
        f"период: {frame['signed_up_at'].min():%Y-%m-%d} .. {frame['signed_up_at'].max():%Y-%m-%d}"
    )
    print(f"отменено: {frame['cancelled_at'].is_not_null().sum():,}")
    print()
    for name in frame.columns:
        print(f"  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
