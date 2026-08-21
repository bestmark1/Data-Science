"""Шаг 12 — скользящая оценка по нескольким окнам.

Вывод шага 11 — что порядок кандидатов переворачивается — получен на одной паре
окон. Одного переворота мало: тестовое окно могло оказаться аномальным.

Здесь тот же протокол прогоняется по пяти последовательным окнам. Для каждого
окна обучение видит только заказы, исход которых стал известен до его начала.
Никакой настройки по окнам не делается: конфигурации фиксированы заранее.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import lightgbm as lgb
import polars as pl
from sklearn.metrics import average_precision_score, roc_auc_score

PROJECT = Path(__file__).resolve().parent.parent
ARTIFACTS = PROJECT / "artifacts"

NUM = [
    "promised_lead_days",
    "items_count",
    "items_total",
    "freight_total",
    "sellers_count",
    "seller_states_count",
    "weight_total_g",
    "volume_total_cm3",
    "payment_value",
    "payment_installments",
    "payments_count",
    "purchase_month",
    "purchase_weekday",
    "purchase_hour",
    "freight_ratio",
    "customer_zip_code_prefix",
]
CAT = ["customer_state", "seller_state", "main_category", "payment_type", "cross_state"]
BASE = {
    "objective": "binary",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbose": -1,
    "seed": 42,
    "deterministic": True,
}
MONOTONE = [-1 if f == "promised_lead_days" else 0 for f in NUM] + [0] * len(CAT)

WINDOWS = [
    (dt.datetime(2017, 10, 1), dt.datetime(2017, 12, 1)),
    (dt.datetime(2017, 12, 1), dt.datetime(2018, 2, 1)),
    (dt.datetime(2018, 2, 1), dt.datetime(2018, 4, 1)),
    (dt.datetime(2018, 4, 1), dt.datetime(2018, 6, 1)),
    (dt.datetime(2018, 6, 1), dt.datetime(2018, 8, 21)),
]
CONFIGS = [
    ("модель, вся история", None, False),
    ("модель, 6 месяцев", 183, False),
    ("модель, 6 мес + монотонность", 183, True),
]


def prep(frame: pl.DataFrame, reference=None):
    pdf = (
        frame.with_columns(
            [pl.col(c).cast(pl.Utf8).fill_null("__m").cast(pl.Categorical) for c in CAT]
        )
        .select([*NUM, *CAT])
        .to_pandas()
    )
    if reference is not None:
        for column in CAT:
            pdf[column] = pdf[column].cat.set_categories(reference[column].cat.categories)
    return pdf


def main() -> int:
    table = pl.concat(
        [
            pl.read_parquet(ARTIFACTS / name)
            for name in ("train.parquet", "valid.parquet", "test.parquet")
        ]
    ).sort("order_purchase_timestamp")

    rows = []
    for start, stop in WINDOWS:
        window = table.filter(
            (pl.col("order_purchase_timestamp") >= start)
            & (pl.col("order_purchase_timestamp") < stop)
        )
        history = table.filter(pl.col("label_known_at") < start)
        if window.height < 1000 or history.height < 5000:
            continue

        y = window["is_late"].to_numpy()
        scores = {"правило 1/(1+срок)": 1.0 / (1.0 + window["promised_lead_days"].to_numpy())}

        for name, days, monotone in CONFIGS:
            subset = history
            if days is not None:
                subset = history.filter(
                    pl.col("order_purchase_timestamp") >= start - dt.timedelta(days=days)
                )
            if subset.height < 3000 or subset["is_late"].sum() < 100:
                continue
            params = dict(BASE)
            if monotone:
                params["monotone_constraints"] = MONOTONE
            x_ref = prep(subset)
            booster = lgb.train(
                params,
                lgb.Dataset(x_ref, label=subset["is_late"].to_numpy(), categorical_feature=CAT),
                400,
            )
            scores[name] = booster.predict(prep(window, x_ref))

        rows.append(
            {
                "window": f"{start:%Y-%m} .. {stop:%Y-%m}",
                "n": window.height,
                "rate": float(y.mean()),
                "history": history.height,
                "pr": {k: average_precision_score(y, v) for k, v in scores.items()},
                "auc": {k: roc_auc_score(y, v) for k, v in scores.items()},
            }
        )

    names = list(rows[0]["pr"].keys())

    report = ["# Шаг 12 — скользящая оценка", ""]
    report.append(
        f"Окон: {len(rows)}. Конфигурации фиксированы заранее, по окнам не настраивались."
    )
    report.append("")
    report += [
        "| окно | заказов | доля опозданий | история | " + " | ".join(names) + " |",
        "|---|---|---|---|" + "---|" * len(names),
    ]
    for row in rows:
        cells = " | ".join(f"{row['pr'][n]:.4f}" for n in names)
        report.append(
            f"| {row['window']} | {row['n']:,} | {row['rate']:.2%} | {row['history']:,} | {cells} |"
        )
    report.append("")
    report.append(
        "Значения — PR-AUC. Для сравнения: PR-AUC случайного предсказания равен доле опозданий."
    )
    report.append("")

    report += [
        "## Кто выигрывает в каждом окне",
        "",
        "| окно | победитель | отрыв от правила |",
        "|---|---|---|",
    ]
    wins = {}
    for row in rows:
        best = max(row["pr"], key=row["pr"].get)
        wins[best] = wins.get(best, 0) + 1
        gap = row["pr"][best] - row["pr"]["правило 1/(1+срок)"]
        report.append(f"| {row['window']} | {best} | {gap:+.4f} |")
    report.append("")

    report += ["## Устойчивость порядка", ""]
    for name in names:
        ranks = [sorted(row["pr"], key=row["pr"].get, reverse=True).index(name) + 1 for row in rows]
        report.append(f"- {name}: места по окнам {ranks}")
    report.append("")

    rule_wins = wins.get("правило 1/(1+срок)", 0)
    report.append(
        f"Правило выигрывает в {rule_wins} окнах из {len(rows)}, модели — в "
        f"{len(rows) - rule_wins}."
    )
    report.append("")
    if len({tuple(sorted(r["pr"], key=r["pr"].get, reverse=True)) for r in rows}) > 1:
        report.append(
            "**Порядок кандидатов различается между окнами.** Вывод шага 11 не был "
            "артефактом одного окна: на этих данных выбор лучшего кандидата "
            "зависит от того, какое окно назначено оценочным."
        )
    else:
        report.append(
            "**Порядок кандидатов одинаков во всех окнах.** Переворот в шаге 11 был "
            "особенностью конкретной пары окон, а не общим свойством данных."
        )

    (ARTIFACTS / "12_rolling.md").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
