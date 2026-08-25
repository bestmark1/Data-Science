"""Пересчёт ответов, которые вывел агент, а принял заполняющий.

Каждая проверка упала бы, будь утверждение ложным.
"""

from __future__ import annotations

import datetime as dt
import importlib.util
from pathlib import Path

import polars as pl

PROJECT = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("build", PROJECT / "build.py")
_build = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_build)

failures: list[str] = []


def claim(name: str, stated: float, actual: float, unit: str = "%", tol: float = 0.05) -> None:
    ok = abs(stated - actual) <= tol
    print(
        f"  [{'совпало' if ok else 'РАСХОЖДЕНИЕ'}] {name}: заявлено {stated:g}{unit}, "
        f"пересчитано {actual:g}{unit}"
    )
    if not ok:
        failures.append(name)


def exact(name: str, stated: object, actual: object) -> None:
    ok = stated == actual
    print(
        f"  [{'совпало' if ok else 'РАСХОЖДЕНИЕ'}] {name}: заявлено {stated!r}, "
        f"пересчитано {actual!r}"
    )
    if not ok:
        failures.append(name)


def main() -> int:
    f = _build.build()
    n = f.height
    print(f"строк: {n:,}\n")

    print("СТРУКТУРА")
    exact("подписок", 508_932, n)
    exact("клиентов", 508_932, f["customer_id"].n_unique())
    exact("одна подписка на клиента", True, f["customer_id"].n_unique() == n)
    exact("отменённых", 112_485, int(f["cancelled_at"].is_not_null().sum()))

    print("\nГРАНУЛЯЦИЯ")
    for column in ("signed_up_at", "cancelled_at"):
        series = f[column].drop_nulls()
        exact(f"{column} хранит момент", True, bool((series.dt.hour() != 0).any()))

    print("\nДОЛЯ ОТМЕН В ГОРИЗОНТЕ")
    for horizon, stated in ((30, 1.01), (90, 3.06), (180, 5.97), (365, 10.90)):
        hit = f.filter(
            pl.col("cancelled_at").is_not_null()
            & (pl.col("cancelled_at") <= pl.col(f"horizon_{horizon}d"))
        ).height
        claim(f"в {horizon} дн", stated, round(100 * hit / n, 2))

    print("\nТРЕНД")
    mature = f.filter(pl.col("horizon_180d") < dt.datetime(2022, 1, 1))
    by_year = (
        mature.with_columns(pl.col("signed_up_at").dt.year().alias("y"))
        .group_by("y")
        .agg(
            (
                pl.col("cancelled_at").is_not_null()
                & (pl.col("cancelled_at") <= pl.col("horizon_180d"))
            )
            .mean()
            .alias("rate")
        )
        .sort("y")
    )
    rates = by_year["rate"].to_list()
    exact(
        "доля растёт каждый год", True, all(b > a for a, b in zip(rates, rates[1:], strict=False))
    )
    # Заявленные 4.45% и 7.64% — ПОЛУГОДОВЫЕ величины, и это не было сказано.
    # Проверяются они; годовое отношение приводится рядом, потому что число,
    # зависящее от способа группировки, без него неполно.
    halves = (
        mature.with_columns(
            pl.col("signed_up_at").dt.year().alias("y"),
            ((pl.col("signed_up_at").dt.month() - 1) // 6 + 1).alias("h"),
        )
        .group_by("y", "h")
        .agg(
            (
                pl.col("cancelled_at").is_not_null()
                & (pl.col("cancelled_at") <= pl.col("horizon_180d"))
            )
            .mean()
            .alias("rate")
        )
        .sort("y", "h")
    )["rate"].to_list()
    claim("первое полугодие 2017", 4.45, round(100 * halves[0], 2))
    claim("первое полугодие 2021", 7.64, round(100 * halves[-2], 2))
    claim("рост по полугодиям, разы", 1.72, round(halves[-2] / halves[0], 2), unit="", tol=0.02)
    print(
        f"  [к сведению] по ГОДАМ отношение {rates[-1] / rates[0]:.2f} — величина зависит "
        "от способа агрегации"
    )

    print("\nПРИЗНАКИ")
    exact("различных возрастов", 58, f["age"].n_unique())
    for column, count in (("gender", 2), ("product", 2), ("price", 2), ("billing_cycle", 2)):
        exact(f"различных {column}", count, f[column].n_unique())

    print("\n" + "=" * 60)
    if failures:
        print(f"РАСХОЖДЕНИЙ: {len(failures)} — {', '.join(failures)}")
        return 1
    print("Расхождений нет.")
    print()
    print("ОШИБКИ, ДОПУЩЕННЫЕ АГЕНТОМ В ЭТОМ КЕЙСЕ:")
    print("  1. «отменено 100% подписок» — Polars прочёл строку «NA» значением.")
    print("     Верно 22.1%. Опровергнуто собственной проверкой в том же обмене,")
    print("     до того как на числе было что-либо построено.")
    print("  2. «рост на 72%» назван без указания способа агрегации. По полугодиям")
    print("     1.72, по годам 1.65. Число, зависящее от группировки, неполно без неё.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
