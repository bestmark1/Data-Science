"""Шаг 07 — baseline до всякой модели.

Сложная модель обязана доказать улучшение относительно этих трёх.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

PROJECT = Path(__file__).resolve().parent.parent
ARTIFACTS = PROJECT / "artifacts"

SIMPLE_FEATURES = ["promised_lead_days", "freight_total", "items_total", "items_count"]


def scores(name: str, y: np.ndarray, p: np.ndarray) -> dict:
    return {
        "baseline": name,
        "roc_auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
        "brier": brier_score_loss(y, p),
    }


def main() -> int:
    train = pl.read_parquet(ARTIFACTS / "train.parquet")
    test = pl.read_parquet(ARTIFACTS / "test.parquet")

    y_train = train["is_late"].to_numpy()
    y_test = test["is_late"].to_numpy()
    base_rate = y_train.mean()

    rows = []

    # 1. Константа: доля опозданий из обучения.
    rows.append(scores("константа (доля из train)", y_test, np.full(len(y_test), base_rate)))

    # 2. Правило: чем короче обещанный срок, тем выше риск не успеть.
    lead_test = test["promised_lead_days"].to_numpy().astype(float)
    rule = 1.0 / (1.0 + lead_test)
    rows.append(scores("правило: обратный обещанный срок", y_test, rule))

    # 3. Логистическая регрессия на четырёх очевидных признаках.
    x_train = train.select(SIMPLE_FEATURES).fill_null(0).to_numpy()
    x_test = test.select(SIMPLE_FEATURES).fill_null(0).to_numpy()
    scaler = StandardScaler().fit(x_train)
    model = LogisticRegression(max_iter=1000, class_weight="balanced").fit(
        scaler.transform(x_train), y_train
    )
    p_logit = model.predict_proba(scaler.transform(x_test))[:, 1]
    rows.append(scores("логрегрессия, 4 признака", y_test, p_logit))

    report = ["# Шаг 07 — baseline", ""]
    report.append(f"Обучение: {train.height:,} заказов, доля опозданий {base_rate:.2%}")
    report.append(f"Тест: {test.height:,} заказов, доля опозданий {y_test.mean():.2%}")
    report.append("")
    report += ["| baseline | ROC-AUC | PR-AUC | Brier |", "|---|---|---|---|"]
    for row in rows:
        report.append(
            f"| {row['baseline']} | {row['roc_auc']:.4f} | {row['pr_auc']:.4f} | {row['brier']:.4f} |"
        )
    report.append("")
    report.append(
        f"Доля положительных в тесте: {y_test.mean():.4f} — это PR-AUC случайного предсказания."
    )
    report.append("")
    report.append(
        "Расхождение долей между обучением и тестом означает, что константный "
        "baseline систематически завышает вероятность: Brier у него плох не "
        "из-за ранжирования, а из-за смещения уровня."
    )

    (ARTIFACTS / "07_baseline.md").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
