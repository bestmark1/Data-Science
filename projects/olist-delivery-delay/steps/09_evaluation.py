"""Шаг 09 — калибровка и разбор ошибок.

Правило обыгрывает модель по ранжированию, но выдаёт не вероятности. Чтобы
сравнивать честно и выбирать порог, оба кандидата калибруются на обучении.
"""

from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

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
PARAMS = {
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
    "monotone_constraints": [-1 if f == "promised_lead_days" else 0 for f in NUM] + [0] * len(CAT),
}


def prep(frame: pl.DataFrame):
    return (
        frame.with_columns(
            [pl.col(c).cast(pl.Utf8).fill_null("__m").cast(pl.Categorical) for c in CAT]
        )
        .select([*NUM, *CAT])
        .to_pandas()
    )


def calibration_table(y: np.ndarray, p: np.ndarray, bins: int = 10) -> list[str]:
    order = np.argsort(p)
    chunks = np.array_split(order, bins)
    lines = ["| дециль | средний прогноз | фактическая доля | заказов |", "|---|---|---|---|"]
    for i, idx in enumerate(chunks, 1):
        lines.append(f"| {i} | {p[idx].mean():.4f} | {y[idx].mean():.4f} | {len(idx):,} |")
    return lines


def main() -> int:
    train = pl.read_parquet(ARTIFACTS / "train.parquet")
    test = pl.read_parquet(ARTIFACTS / "test.parquet")
    y_tr, y_te = train["is_late"].to_numpy(), test["is_late"].to_numpy()

    x_tr, x_te = prep(train), prep(test)
    for column in CAT:
        x_te[column] = x_te[column].cat.set_categories(x_tr[column].cat.categories)
    booster = lgb.train(PARAMS, lgb.Dataset(x_tr, label=y_tr, categorical_feature=CAT), 400)

    raw_model_tr = booster.predict(x_tr)
    raw_model_te = booster.predict(x_te)
    raw_rule_tr = 1.0 / (1.0 + train["promised_lead_days"].to_numpy())
    raw_rule_te = 1.0 / (1.0 + test["promised_lead_days"].to_numpy())

    candidates = {}
    for name, tr_raw, te_raw in (
        ("модель", raw_model_tr, raw_model_te),
        ("правило", raw_rule_tr, raw_rule_te),
    ):
        iso = IsotonicRegression(out_of_bounds="clip").fit(tr_raw, y_tr)
        candidates[name] = {"raw": te_raw, "cal": iso.predict(te_raw)}

    report = ["# Шаг 09 — калибровка и разбор ошибок", ""]
    report.append(f"Тест: {test.height:,} заказов, фактическая доля опозданий {y_te.mean():.2%}")
    report.append("")
    report += [
        "| кандидат | ROC-AUC | PR-AUC | Brier сырой | Brier калиброванный | средний прогноз |",
        "|---|---|---|---|---|---|",
    ]
    for name, values in candidates.items():
        report.append(
            f"| {name} | {roc_auc_score(y_te, values['cal']):.4f} | "
            f"{average_precision_score(y_te, values['cal']):.4f} | "
            f"{brier_score_loss(y_te, values['raw']):.4f} | "
            f"{brier_score_loss(y_te, values['cal']):.4f} | {values['cal'].mean():.4f} |"
        )
    report.append("")
    report.append(
        "Калибровка обучена на train, где доля опозданий 9.11%, а применяется к test, "
        "где 4.76%. Средний калиброванный прогноз выше факта — это смещение уровня, "
        "унаследованное от периода обучения, а не ошибка ранжирования."
    )
    report.append("")

    best = (
        "правило"
        if average_precision_score(y_te, candidates["правило"]["cal"])
        > average_precision_score(y_te, candidates["модель"]["cal"])
        else "модель"
    )
    p_best = candidates[best]["cal"]
    report.append(f"**Лучший кандидат по PR-AUC: {best}**")
    report.append("")

    report += ["## Калибровка лучшего кандидата по децилям", ""]
    report += calibration_table(y_te, p_best)
    report.append("")

    scored = test.with_columns(pl.Series("p", p_best))
    report += ["## Ошибки по сегментам", ""]

    for column, title in (("customer_state", "штат покупателя"), ("main_category", "категория")):
        group = (
            scored.group_by(column)
            .agg(
                pl.len().alias("n"),
                pl.col("is_late").mean().alias("actual"),
                pl.col("p").mean().alias("predicted"),
            )
            .filter(pl.col("n") >= 200)
            .with_columns((pl.col("predicted") - pl.col("actual")).alias("bias"))
            .sort("bias")
        )
        report += [
            f"### {title}: наибольшее смещение",
            "",
            "| сегмент | заказов | факт | прогноз | смещение |",
            "|---|---|---|---|---|",
        ]
        for row in [*group.head(3).iter_rows(named=True), *group.tail(3).iter_rows(named=True)]:
            report.append(
                f"| {row[column]} | {row['n']:,} | {row['actual']:.2%} | "
                f"{row['predicted']:.2%} | {row['bias']:+.2%} |"
            )
        report.append("")

    np.save(ARTIFACTS / "test_pred_calibrated.npy", p_best)
    (ARTIFACTS / "09_evaluation.md").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
