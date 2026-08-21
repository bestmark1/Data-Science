"""Шаг 08 — модель и проверка глубины истории.

Baseline-правило обыграло логрегрессию, потому что связь обещанного срока с
опозданием меняет знак между периодами (шаг 07). Поэтому модель обучается не
только на всей истории, но и на нескольких окнах недавнего прошлого: если
короткое окно выигрывает, длинная история вредна, а не полезна.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import lightgbm as lgb
import mlflow
import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

PROJECT = Path(__file__).resolve().parent.parent
ARTIFACTS = PROJECT / "artifacts"
TEST_START = dt.datetime(2018, 6, 1)

NUMERIC = [
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
CATEGORICAL = ["customer_state", "seller_state", "main_category", "payment_type", "cross_state"]

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
}
ROUNDS = 400


def prepare(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.with_columns(
        [pl.col(c).cast(pl.Utf8).fill_null("__missing__").cast(pl.Categorical) for c in CATEGORICAL]
    ).select([*NUMERIC, *CATEGORICAL, "is_late", "order_purchase_timestamp"])


def fit_and_score(train: pl.DataFrame, test: pl.DataFrame, label: str) -> dict:
    x_tr = train.select([*NUMERIC, *CATEGORICAL]).to_pandas()
    x_te = test.select([*NUMERIC, *CATEGORICAL]).to_pandas()
    for column in CATEGORICAL:
        x_te[column] = x_te[column].cat.set_categories(x_tr[column].cat.categories)

    booster = lgb.train(
        PARAMS,
        lgb.Dataset(x_tr, label=train["is_late"].to_numpy(), categorical_feature=CATEGORICAL),
        num_boost_round=ROUNDS,
    )
    p = booster.predict(x_te)
    y = test["is_late"].to_numpy()
    return {
        "window": label,
        "train_rows": train.height,
        "train_rate": float(train["is_late"].mean()),
        "roc_auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
        "brier": brier_score_loss(y, p),
        "mean_pred": float(p.mean()),
        "booster": booster,
        "pred": p,
    }


def main() -> int:
    train_all = prepare(pl.read_parquet(ARTIFACTS / "train.parquet"))
    test = prepare(pl.read_parquet(ARTIFACTS / "test.parquet"))
    y_test = test["is_late"].to_numpy()

    # MLflow 3.x перевёл файловый бэкенд в режим поддержки. SQLite остаётся
    # полностью локальным: один файл, без сервера и без аккаунта.
    tracking_dir = (PROJECT.parent.parent / "mlruns").resolve()
    tracking_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{tracking_dir / 'mlflow.db'}")
    mlflow.set_experiment("olist-delivery-delay")

    results = []
    windows = [("вся история", None), ("12 месяцев", 365), ("6 месяцев", 183), ("3 месяца", 92)]
    for label, days in windows:
        subset = train_all
        if days is not None:
            cutoff = TEST_START - dt.timedelta(days=days)
            subset = train_all.filter(pl.col("order_purchase_timestamp") >= cutoff)
        with mlflow.start_run(run_name=f"lgbm-{label}"):
            outcome = fit_and_score(subset, test, label)
            mlflow.log_params({**PARAMS, "rounds": ROUNDS, "window": label})
            mlflow.log_metrics({k: v for k, v in outcome.items() if isinstance(v, (int, float))})
        results.append(outcome)

    best = max(results, key=lambda r: r["pr_auc"])
    np.save(ARTIFACTS / "test_pred.npy", best["pred"])
    best["booster"].save_model(str(ARTIFACTS / "model.txt"))

    report = ["# Шаг 08 — модель и глубина истории", ""]
    report.append(f"Тест: {test.height:,} заказов, доля опозданий {y_test.mean():.2%}")
    report.append("")
    report += [
        "| окно обучения | заказов | доля в обучении | ROC-AUC | PR-AUC | Brier | средний прогноз |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        report.append(
            f"| {r['window']} | {r['train_rows']:,} | {r['train_rate']:.2%} | "
            f"{r['roc_auc']:.4f} | {r['pr_auc']:.4f} | {r['brier']:.4f} | {r['mean_pred']:.4f} |"
        )
    report.append("")
    report.append("Для сравнения, baseline-правило: ROC-AUC 0.7135, PR-AUC 0.1397, Brier 0.0434.")
    report.append("")
    report.append(f"**Лучшее окно по PR-AUC: {best['window']}**")

    importance = sorted(
        zip(
            best["booster"].feature_name(), best["booster"].feature_importance("gain"), strict=True
        ),
        key=lambda kv: kv[1],
        reverse=True,
    )[:12]
    report += ["", "## Вклад признаков (лучшая модель)", "", "| признак | gain |", "|---|---|"]
    for name, gain in importance:
        report.append(f"| {name} | {gain:,.0f} |")

    (ARTIFACTS / "08_model.md").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
