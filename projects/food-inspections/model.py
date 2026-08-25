"""Обучение и измерение на пятом кейсе.

Признаки истории заведения неизбежно пересекаются между обучением и оценкой:
у долгоживущего объекта всякое «что было раньше» лежит в прошлом, а прошлое
лежит в обучении. Устранить это можно только разбиением по объектам, но тогда
рушится временной порядок, а он первичен.

Поэтому обход записывается, а его цена измеряется: две модели, с историей и
без, сравниваются НА ОКНАХ — они и есть выборки выбора. На резерве измерение
одно, как и положено.
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
from dsx.policy import OverrideLedger
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

HISTORY = ["failures_before", "inspections_before", "previous_failed", "days_since_previous"]
"""Признаки, из-за которых окна пересекаются. Их цена и измеряется."""

SEGMENTS = ["risk", "facility_type", "inspection_type"]
RULE = BaselineRule(kind=RuleKind.CONSTANT, constant=0.053)


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


def train(frame: pl.DataFrame, names: list[str], categories: dict[str, list]):
    model = lgb.LGBMClassifier(**SETTINGS)
    model.fit(prepare(frame, names, categories), frame[LABEL].to_numpy())
    return model


def main() -> int:
    form = load(PROJECT / "project.yaml")
    names = features(form)
    without_history = [n for n in names if n not in HISTORY]
    table = _build.build()
    categories = category_map(table, names)

    overrides = OverrideLedger()
    overrides.override(
        "N2i",
        reason="пересечение окон признаков у долгоживущего объекта неустранимо: всякий "
        "признак истории смотрит в прошлое, а прошлое лежит в обучении. Устранить "
        "можно только разбиением по заведениям, но тогда рушится временной порядок. "
        "Цена обхода измерена ниже: сравнение моделей с историей и без на окнах",
        author="автор",
    )
    overrides.override(
        "A14",
        reason="строка 'unknown' в названии заведения встречается однажды; это опечатка "
        "в реестре, а не способ записи отсутствия",
        author="автор",
    )
    result = run(form, table, PROJECT / "report", overrides=overrides)
    print(result.summary())

    last = result.split.parts[-1]
    train_rows = last.train.filter(pl.col(LABEL).is_not_null())
    result.samples.fit(last.name, "обучение обеих моделей на обучающей части последнего окна")
    print(f"\nобучение: {train_rows.height:,} строк")

    full = train(train_rows, names, categories)
    plain = train(train_rows, without_history, categories)

    def score(model, frame, cols):
        return pl.Series("score", model.predict_proba(prepare(frame, cols, categories))[:, 1])

    print("\n=== ЦЕНА ПРИЗНАКОВ ИСТОРИИ (сравнение на окнах) ===")
    with_hist = {
        p.name: score(full, result.samples.checkout(p.name, Purpose.AUDIT, "оценки"), names)
        for p in result.split.parts
    }
    no_hist = {
        p.name: score(
            plain, result.samples.checkout(p.name, Purpose.AUDIT, "оценки"), without_history
        )
        for p in result.split.parts
    }
    a = stability_across_windows(result.samples, with_hist, RULE, "сравнение с историей")
    print("С ИСТОРИЕЙ:")
    for name, comparison in sorted(a.per_window.items()):
        print(f"  {name}: {comparison}")
    b = stability_across_windows(result.samples, no_hist, RULE, "сравнение без истории")
    print("БЕЗ ИСТОРИИ:")
    for name, comparison in sorted(b.per_window.items()):
        print(f"  {name}: {comparison}")

    print("\n=== ИТОГОВОЕ ИЗМЕРЕНИЕ НА РЕЗЕРВЕ ===")
    reserve = result.samples.checkout(RESERVE, Purpose.AUDIT, "подготовка оценок")
    scores = score(full, reserve, names)
    verdict = measure_against_baseline(result.samples, RESERVE, scores, RULE)
    print(verdict.report_section())

    print("\n=== СМЕЩЕНИЕ ПО СЕГМЕНТАМ (N10) ===")
    print(segment_section(segment_bias(reserve, scores, SEGMENTS, min_rows=300)))

    result.study.measurement = verdict
    result.study.conclude(
        verdict.statement()
        + " Признаки истории пересекаются между обучением и оценкой неустранимо; "
        "обход записан, цена измерена сравнением моделей с историей и без."
    )
    (PROJECT / "report" / "measured.md").write_text(result.study.render(), encoding="utf-8")
    print(f"\nотчёт: {PROJECT / 'report' / 'measured.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
