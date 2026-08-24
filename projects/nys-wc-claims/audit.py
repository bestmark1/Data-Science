"""P-2b: пересчёт ответов, которые вывел агент, а принял заполняющий.

Пре-регистрация, поправка A-1: «Из ответов, которые агент вывел, а заполняющий
принял, ни один не окажется неверным при последующей сверке». Здесь эта сверка
и проводится.

Каждое утверждение сопровождается расчётом, который УПАЛ БЫ, будь утверждение
ложным. Проверка, срабатывающая всегда, не отличается от её отсутствия.

Отдельно перечислены утверждения, которые расчётом не проверяются: прочтения
документа и интерпретации. По P-2c ответами они не являются.
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import json
from pathlib import Path

import polars as pl

PROJECT = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("build", PROJECT / "build.py")
_build = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_build)

TOLERANCE = 0.05
"""Допуск на округление в озвученных долях, в процентных пунктах."""

failures: list[str] = []


def claim(name: str, stated: float, actual: float, unit: str = "%") -> None:
    ok = abs(stated - actual) <= TOLERANCE
    mark = "совпало" if ok else "РАСХОЖДЕНИЕ"
    print(f"  [{mark}] {name}: заявлено {stated:g}{unit}, пересчитано {actual:g}{unit}")
    if not ok:
        failures.append(name)


def exact(name: str, stated: object, actual: object) -> None:
    ok = stated == actual
    mark = "совпало" if ok else "РАСХОЖДЕНИЕ"
    print(f"  [{mark}] {name}: заявлено {stated!r}, пересчитано {actual!r}")
    if not ok:
        failures.append(name)


def main() -> int:
    f = _build.build(since="2015-01-01")
    n = f.height
    print(f"строк в популяции: {n:,}\n")

    print("СТРУКТУРА")
    exact("ключ уникален", True, f["claim_id"].n_unique() == n)
    temporal = [c for c in f.columns if f[c].dtype == pl.Datetime]
    exact("временных колонок", 16, len(temporal))
    with_time = [
        c
        for c in temporal
        if (s := f[c].drop_nulls()).len() and (s.dt.hour().sum() or s.dt.minute().sum())
    ]
    exact("колонок с ненулевым временем суток", [], with_time)

    print("\nИСХОД")
    hearing = f["first_hearing_at"].is_not_null().sum()
    claim("доля со слушанием когда-либо", 19.3, round(100 * hearing / n, 1))
    lag = f.select(
        (pl.col("first_hearing_at") - pl.col("assembled_at")).dt.total_days().alias("d")
    ).drop_nulls()
    positive_lag = lag.filter(pl.col("d") >= 0)["d"]
    claim("медиана дней до слушания", 189, float(positive_lag.median()), unit=" дн")
    for horizon, stated in ((30, 0.1), (90, 3.9), (180, 9.2), (365, 14.9)):
        share = positive_lag.filter(positive_lag <= horizon).len() / n
        claim(f"доля со слушанием в {horizon} дн", stated, round(100 * share, 1))

    print("\nДОСТУПНОСТЬ")
    c3 = f.select(
        (pl.col("c3_received_at") - pl.col("assembled_at")).dt.total_days().alias("d")
    ).drop_nulls()["d"]
    claim("дат C-3 позже сборки", 60.5, round(100 * c3.filter(c3 > 0).len() / c3.len(), 1))
    claim("заполненность C-2", 1.1, round(100 * f["c2_received_at"].is_not_null().sum() / n, 1))
    claim(
        "заполненность даты травмы", 99.4, round(100 * f["accident_at"].is_not_null().sum() / n, 1)
    )

    print("\nСТАТЬЯ 32")
    s32 = f.filter(pl.col("section32_at").is_not_null())
    both = s32.filter(pl.col("first_hearing_at").is_not_null())
    claim("из них со слушанием", 78.8, round(100 * both.height / s32.height, 1))
    first = both.filter(pl.col("first_hearing_at") < pl.col("section32_at")).height
    claim("слушание раньше соглашения", 99.8, round(100 * first / both.height, 1))

    print("\nОТМЕНЁННЫЕ ДЕЛА")
    cancelled = f.filter(pl.col("current_status") == "CASE CANCELLED")
    exact("отменённых всего", 27_374, cancelled.height)
    exact(
        "из них без слушания",
        22_815,
        cancelled.filter(pl.col("first_hearing_at").is_null()).height,
    )

    print("\nКЛАСТЕРЫ И РИТМ")
    top10 = f.group_by("carrier_name").len().sort("len", descending=True).head(10)["len"].sum()
    claim("топ-10 страховщиков", 44.4, round(100 * top10 / n, 1))
    days = f.select(pl.col("assembled_at").dt.date().alias("d")).unique().sort("d")["d"]
    gaps = days.diff().drop_nulls().dt.total_days()
    exact("медиана промежутка, дн", 1.0, float(gaps.median()))
    exact("максимум промежутка, дн", 3.0, float(gaps.max()))

    print("\nГРАНИЦЫ")
    meta = json.loads((PROJECT / "data" / "raw" / "socrata_metadata.json").read_text())
    updated = dt.datetime.fromtimestamp(meta["rowsUpdatedAt"], dt.UTC)
    exact("дата обновления выгрузки", "2021-06-18", f"{updated:%Y-%m-%d}")
    lo = f["assembled_at"].min()
    mature_until = dt.datetime(2021, 6, 18) - dt.timedelta(days=180)
    exact("день созревания от начала", 2180, (mature_until - lo).days)
    exact("он же датой", "2020-12-20", f"{mature_until:%Y-%m-%d}")

    print("\n" + "=" * 60)
    if failures:
        print(f"РАСХОЖДЕНИЙ: {len(failures)} — {', '.join(failures)}")
        print("P-2b провалено: ответ агента, принятый заполняющим, оказался неверным.")
        return 1
    print("Расхождений нет: все пересчитанные ответы подтвердились.")
    print()
    print("НЕ ПРОВЕРЯЕТСЯ РАСЧЁТОМ (по P-2c ответами не является):")
    print("  - что колонки на момент выгрузки описаны словарём именно как текущее")
    print("    состояние — прочтение документа, вопрос владельцу;")
    print("  - что две претензии с дореволюционной травмой суть заглушки миграции —")
    print("    интерпретация двух строк, записана причиной обхода A13;")
    print("  - что свойства работника известны Совету к моменту сборки — прочтение")
    print("    словаря, записано допущением с основанием document.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
