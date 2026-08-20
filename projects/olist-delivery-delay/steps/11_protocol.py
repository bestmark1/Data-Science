"""Шаг 11 — корректный протокол оценки.

Шаги 07-10 были первым проходом, в котором тест использовался для выбора окна
обучения, кандидата, калибровки и порога. Внешнее ревью указало, что после
этого итоговые цифры не являются несмещёнными. Здесь протокол исправлен:

- все решения принимаются на ВАЛИДАЦИИ;
- калибровка обучается на валидации, а не на обучающей выборке;
- тест оценивается ОДИН раз, готовой конфигурацией;
- для ключевых разниц считаются бутстрэп-интервалы.

Первый проход не удалён намеренно: он документирует, как выглядит вывод до
исправления протокола.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

PROJECT = Path(__file__).resolve().parent.parent
ARTIFACTS = PROJECT / "artifacts"
VALID_START = dt.datetime(2018, 4, 1)

NUM = [
    "promised_lead_days", "items_count", "items_total", "freight_total", "sellers_count",
    "seller_states_count", "weight_total_g", "volume_total_cm3", "payment_value",
    "payment_installments", "payments_count", "purchase_month", "purchase_weekday",
    "purchase_hour", "freight_ratio", "customer_zip_code_prefix",
]
CAT = ["customer_state", "seller_state", "main_category", "payment_type", "cross_state"]
BASE = {
    "objective": "binary", "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 100,
    "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "verbose": -1,
    "seed": 42, "deterministic": True,
}
COST_WARN, COST_MISS = 1.0, 10.0
RNG = np.random.default_rng(42)


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


def train_model(frame: pl.DataFrame, monotone: bool):
    params = dict(BASE)
    if monotone:
        params["monotone_constraints"] = [
            -1 if f == "promised_lead_days" else 0 for f in NUM
        ] + [0] * len(CAT)
    x = prep(frame)
    booster = lgb.train(
        params, lgb.Dataset(x, label=frame["is_late"].to_numpy(), categorical_feature=CAT), 400
    )
    return booster, x


def bootstrap_diff(y, p_a, p_b, metric, draws: int = 400) -> tuple[float, float]:
    """Интервал для разницы метрик двух кандидатов на одной выборке."""
    diffs = []
    n = len(y)
    for _ in range(draws):
        idx = RNG.integers(0, n, n)
        if y[idx].sum() == 0:
            continue
        diffs.append(metric(y[idx], p_a[idx]) - metric(y[idx], p_b[idx]))
    return float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))


def cost_of(y, p, threshold, cost_miss: float = COST_MISS) -> float:
    flagged = p >= threshold
    return float(flagged.sum() * COST_WARN + ((~flagged) & (y == 1)).sum() * cost_miss)


def main() -> int:
    train = pl.read_parquet(ARTIFACTS / "train.parquet")
    valid = pl.read_parquet(ARTIFACTS / "valid.parquet")
    test = pl.read_parquet(ARTIFACTS / "test.parquet")
    y_va, y_te = valid["is_late"].to_numpy(), test["is_late"].to_numpy()

    report = ["# Шаг 11 — корректный протокол", ""]
    report.append(
        f"Обучение {train.height:,} ({train['is_late'].mean():.2%}) | "
        f"валидация {valid.height:,} ({y_va.mean():.2%}) | "
        f"тест {test.height:,} ({y_te.mean():.2%})"
    )
    report.append("")

    # --- ВСЕ РЕШЕНИЯ НА ВАЛИДАЦИИ ---
    rule_va = 1.0 / (1.0 + valid["promised_lead_days"].to_numpy())
    candidates = {"правило 1/(1+срок)": rule_va}

    windows = [("вся история", None), ("12 месяцев", 365), ("6 месяцев", 183)]
    fitted = {}
    for label, days in windows:
        subset = train
        if days is not None:
            subset = train.filter(
                pl.col("order_purchase_timestamp") >= VALID_START - dt.timedelta(days=days)
            )
        for monotone in (False, True):
            name = f"lgbm {label}{' + монотонность' if monotone else ''}"
            booster, x_ref = train_model(subset, monotone)
            candidates[name] = booster.predict(prep(valid, x_ref))
            fitted[name] = (booster, x_ref, subset, monotone)

    report += ["## Выбор на валидации", "", "| кандидат | ROC-AUC | PR-AUC |", "|---|---|---|"]
    scored = {
        name: (roc_auc_score(y_va, p), average_precision_score(y_va, p))
        for name, p in candidates.items()
    }
    for name, (auc, ap) in sorted(scored.items(), key=lambda kv: -kv[1][1]):
        report.append(f"| {name} | {auc:.4f} | {ap:.4f} |")
    report.append("")

    winner = max(scored, key=lambda k: scored[k][1])
    report.append(f"**Выбран: {winner}**")
    report.append("")

    # --- КАЛИБРОВКА И ПОРОГ — ТОЖЕ НА ВАЛИДАЦИИ ---
    p_va = candidates[winner]
    calibrator = IsotonicRegression(out_of_bounds="clip").fit(p_va, y_va)
    cal_va = calibrator.predict(p_va)
    grid = np.unique(np.round(np.quantile(cal_va, np.linspace(0.50, 0.999, 80)), 6))
    threshold = min(grid, key=lambda t: cost_of(y_va, cal_va, t))
    report.append(f"Порог, выбранный на валидации: {threshold:.4f}")
    report.append("")

    # --- ТЕСТ: ОДИН РАЗ ---
    if winner.startswith("правило"):
        p_te_raw = 1.0 / (1.0 + test["promised_lead_days"].to_numpy())
    else:
        booster, x_ref, _, _ = fitted[winner]
        p_te_raw = booster.predict(prep(test, x_ref))
    p_te = calibrator.predict(p_te_raw)

    rule_te = 1.0 / (1.0 + test["promised_lead_days"].to_numpy())
    lo, hi = bootstrap_diff(y_te, p_te_raw, rule_te, average_precision_score)

    report += ["## Тест (единственная оценка)", "", "| метрика | значение |", "|---|---|"]
    report.append(f"| ROC-AUC (сырой скор) | {roc_auc_score(y_te, p_te_raw):.4f} |")
    report.append(f"| PR-AUC (сырой скор) | {average_precision_score(y_te, p_te_raw):.4f} |")
    report.append(f"| PR-AUC правила | {average_precision_score(y_te, rule_te):.4f} |")
    report.append(f"| Brier после калибровки | {brier_score_loss(y_te, p_te):.4f} |")
    report.append(f"| средний прогноз | {p_te.mean():.4f} при факте {y_te.mean():.4f} |")
    report.append("")
    report.append(
        f"Разница PR-AUC (выбранный минус правило), 95% бутстрэп: "
        f"[{lo:+.4f}, {hi:+.4f}]"
    )
    if lo <= 0 <= hi:
        report.append("Интервал накрывает ноль: превосходство не установлено.")
    report.append("")

    # --- ЭКОНОМИКА С ИНТЕРВАЛОМ И ЧУВСТВИТЕЛЬНОСТЬЮ ---
    cost_model = cost_of(y_te, p_te, threshold)
    cost_nothing = float((y_te == 1).sum() * COST_MISS)
    cost_all = float(len(y_te) * COST_WARN)

    gains = []
    n = len(y_te)
    for _ in range(400):
        idx = RNG.integers(0, n, n)
        gains.append(
            min(float((y_te[idx] == 1).sum() * COST_MISS), float(n * COST_WARN))
            - cost_of(y_te[idx], p_te[idx], threshold)
        )
    g_lo, g_hi = float(np.quantile(gains, 0.025)), float(np.quantile(gains, 0.975))

    report += ["## Экономика", "", "| стратегия | издержки |", "|---|---|"]
    report.append(f"| ничего не делать | {cost_nothing:,.0f} |")
    report.append(f"| предупреждать всех | {cost_all:,.0f} |")
    report.append(f"| порог {threshold:.4f} | {cost_model:,.0f} |")
    report.append("")
    report.append(
        f"Выигрыш относительно лучшей опорной: {min(cost_nothing, cost_all) - cost_model:,.0f}, "
        f"95% бутстрэп [{g_lo:,.0f}, {g_hi:,.0f}]"
    )
    if g_lo <= 0:
        report.append("Интервал включает ноль или отрицательные значения: выигрыш не установлен.")
    report.append("")

    report += [
        "## Чувствительность к цене ошибки",
        "",
        "| цена пропуска | порог с валидации | издержки | выигрыш |",
        "|---|---|---|---|",
    ]
    for miss in (3.0, 5.0, 10.0, 20.0, 50.0):
        t = min(grid, key=lambda x: cost_of(y_va, cal_va, x, miss))
        c = cost_of(y_te, p_te, t, miss)
        base = min(float((y_te == 1).sum() * miss), float(len(y_te) * COST_WARN))
        report.append(f"| {miss:.0f} | {t:.4f} | {c:,.0f} | {base - c:+,.0f} |")

    (ARTIFACTS / "11_protocol.md").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
