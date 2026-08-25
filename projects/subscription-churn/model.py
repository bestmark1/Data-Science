"""Обучение и измерение на четвёртом кейсе.

Обучение живёт здесь, ядро принимает готовые оценки. Сценарий тот же, что на
третьем кейсе, и намеренно тот же: если он потребует правок, это находка о
переносимости, а не о задаче.

Признаков пять, но работающих по существу два: возраст и пол. Продукт, цена и
цикл оплаты — одно и то же в трёх видах, два продукта на всю выгрузку. Модель
ожидается слабой, и инструмент обязан это сказать, а не выдать правдоподобное
число.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import lightgbm as lgb
import polars as pl

from dsx.label import LABEL
from dsx.measure import (
    BaselineRule,
    RuleKind,
    measure_against_baseline,
    segment_bias,
    segment_section,
    stability_across_windows,
)
from dsx.project import load
from dsx.runner import RESERVE, run
from dsx.samples import Purpose

PROJECT = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("build", PROJECT / "build.py")
_build = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_build)

SETTINGS = {
    "objective": "binary",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "n_estimators": 200,
    "min_child_samples": 200,
    "verbose": -1,
    "deterministic": True,
    "seed": 20260101,
}

SEGMENTS = ["gender", "product", "billing_cycle"]
RULE = BaselineRule(kind=RuleKind.CONSTANT, constant=0.06)
"""Базовое правило: постоянная оценка около доли класса в окнах."""


def features(form) -> list[str]:
    from dsx.roles import Availability, Role

    return [
        c.name
        for c in form.schema_spec().columns
        if c.role is Role.FEATURE and c.availability is Availability.AT_DECISION
    ]


def category_map(table: pl.DataFrame, names: list[str]) -> dict[str, list]:
    return {
        name: sorted(table[name].unique().drop_nulls().to_list())
        for name in names
        if not table[name].dtype.is_numeric()
    }


def prepare(frame: pl.DataFrame, names: list[str], categories: dict[str, list]):
    import pandas as pd

    data = {}
    for name in names:
        column = frame[name]
        if name in categories:
            data[name] = pd.Categorical(column.to_list(), categories=categories[name])
        else:
            data[name] = column.cast(pl.Float64).to_numpy()
    return pd.DataFrame(data)


def main() -> int:
    form = load(PROJECT / "project.yaml")
    names = features(form)
    table = _build.build()
    categories = category_map(table, names)

    result = run(form, table, PROJECT / "report")
    print(result.summary())
    print()

    last = result.split.parts[-1]
    train = last.train.filter(pl.col(LABEL).is_not_null())
    result.samples.fit(last.name, f"обучение на обучающей части окна {last.name}")
    print(f"обучение: {train.height:,} строк, признаков {len(names)}")

    model = lgb.LGBMClassifier(**SETTINGS)
    model.fit(prepare(train, names, categories), train[LABEL].to_numpy())

    def predict(frame: pl.DataFrame) -> pl.Series:
        return pl.Series("score", model.predict_proba(prepare(frame, names, categories))[:, 1])

    print("\n=== УСТОЙЧИВОСТЬ ПО ОКНАМ (N7) ===")
    by_window = {
        part.name: predict(result.samples.checkout(part.name, Purpose.AUDIT, "подготовка оценок"))
        for part in result.split.parts
    }
    print(stability_across_windows(result.samples, by_window, RULE).report_section())

    print("\n=== ИТОГОВОЕ ИЗМЕРЕНИЕ НА РЕЗЕРВЕ ===")
    reserve = result.samples.checkout(RESERVE, Purpose.AUDIT, "подготовка оценок")
    scores = predict(reserve)
    verdict = measure_against_baseline(result.samples, RESERVE, scores, RULE)
    print(verdict.report_section())

    print("\n=== СМЕЩЕНИЕ ПО СЕГМЕНТАМ (N10) ===")
    print(segment_section(segment_bias(reserve, scores, SEGMENTS, min_rows=500)))

    shifted = any(
        s.finding.value == "non_stationary_target" and "РЕЗЕРВЕ" in s.detail
        for s in result.checks.signals
    )
    caveat = (
        " Измерение проведено на популяции с более высокой долей класса, чем в "
        "обучающих окнах: доля отмен растёт весь период наблюдения."
        if shifted
        else ""
    )
    result.study.measurement = verdict
    result.study.conclude(verdict.statement() + caveat)
    (PROJECT / "report" / "measured.md").write_text(result.study.render(), encoding="utf-8")
    print(f"\nотчёт: {PROJECT / 'report' / 'measured.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
