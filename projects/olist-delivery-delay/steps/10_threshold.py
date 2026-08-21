"""Шаг 10 — порог под цену ошибки.

Порог не максимизирует F1. Он выбирается из несимметричной цены: пропущенное
опоздание дороже лишнего предупреждения. Отдельно показано, во что обходится
выбор порога на обучающем распределении при дрейфе.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

PROJECT = Path(__file__).resolve().parent.parent
ARTIFACTS = PROJECT / "artifacts"

# Предупреждение стоит 1 условную единицу, пропущенное опоздание — 10.
# Соотношение принято допущением: у автора нет цифр от бизнеса.
COST_WARN = 1.0
COST_MISS = 10.0


def value_at(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    flagged = p >= threshold
    tp = int((flagged & (y == 1)).sum())
    fp = int((flagged & (y == 0)).sum())
    fn = int((~flagged & (y == 1)).sum())
    cost = (tp + fp) * COST_WARN + fn * COST_MISS
    return {
        "threshold": threshold,
        "flagged": int(flagged.sum()),
        "flagged_share": float(flagged.mean()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "recall": tp / max(tp + fn, 1),
        "precision": tp / max(tp + fp, 1),
        "cost": cost,
    }


def main() -> int:
    test = pl.read_parquet(ARTIFACTS / "test.parquet")
    y = test["is_late"].to_numpy()
    p = np.load(ARTIFACTS / "test_pred_calibrated.npy")

    grid = np.unique(np.round(np.quantile(p, np.linspace(0.50, 0.999, 60)), 6))
    rows = [value_at(y, p, t) for t in grid]
    best = min(rows, key=lambda r: r["cost"])

    do_nothing = len(y) * 0 + int((y == 1).sum()) * COST_MISS
    warn_all = len(y) * COST_WARN

    report = ["# Шаг 10 — порог под цену ошибки", ""]
    report.append(
        f"Допущение: предупреждение стоит {COST_WARN:.0f}, пропущенное опоздание {COST_MISS:.0f}."
    )
    report.append("Цифры от бизнеса отсутствуют, соотношение принято автором.")
    report.append("")
    report += ["## Опорные стратегии", "", "| стратегия | стоимость |", "|---|---|"]
    report.append(f"| никого не предупреждать | {do_nothing:,.0f} |")
    report.append(f"| предупреждать всех | {warn_all:,.0f} |")
    report.append(f"| **порог {best['threshold']:.4f}** | **{best['cost']:,.0f}** |")
    report.append("")
    report.append(
        f"Выигрыш относительно лучшей опорной стратегии: "
        f"{min(do_nothing, warn_all) - best['cost']:,.0f} единиц "
        f"({1 - best['cost'] / min(do_nothing, warn_all):.1%})."
    )
    report.append("")

    report += [
        "## Окрестность оптимума",
        "",
        "| порог | помечено | доля | полнота | точность | стоимость |",
        "|---|---|---|---|---|---|",
    ]
    near = sorted(rows, key=lambda r: abs(r["threshold"] - best["threshold"]))[:9]
    for row in sorted(near, key=lambda r: r["threshold"]):
        mark = " **<-**" if row["threshold"] == best["threshold"] else ""
        report.append(
            f"| {row['threshold']:.4f} | {row['flagged']:,} | {row['flagged_share']:.1%} | "
            f"{row['recall']:.1%} | {row['precision']:.1%} | {row['cost']:,.0f}{mark} |"
        )
    report.append("")

    # Цена дрейфа: порог, выбранный по обучающему уровню, применён к тесту.
    train_rate, test_rate = 0.0911, float(y.mean())
    naive_threshold = float(np.quantile(p, 1 - train_rate))
    naive = value_at(y, p, naive_threshold)
    report += ["## Во что обходится дрейф", ""]
    report.append(
        f"Если помечать долю заказов, равную доле опозданий В ОБУЧЕНИИ ({train_rate:.2%}), "
        f"порог получается {naive_threshold:.4f}, стоимость {naive['cost']:,.0f} — "
        f"на {naive['cost'] - best['cost']:,.0f} единиц хуже оптимума."
    )
    report.append("")
    report.append(
        f"Фактическая доля опозданий в тесте {test_rate:.2%}. Порог, откалиброванный "
        f"на устаревшем уровне, помечает лишних {naive['flagged'] - best['flagged']:,} заказов."
    )

    (ARTIFACTS / "10_threshold.md").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
