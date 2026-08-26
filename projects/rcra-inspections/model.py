"""Измерение цены ПОВОДА на десятом кейсе.

Вопрос кейса: меняется ли смысл признаков от того, ПОЧЕМУ состоялось
наблюдение. Проверяется переносом.

Обучаются две модели на одном и том же окне, различающиеся только выборкой:
одна на проверках ПЛАНОВЫХ, другая на вызванных СИГНАЛОМ. Затем каждая строка
резерва получает две оценки:

  «своей моделью»   — обученной на проверках того же повода;
  «чужой моделью»   — обученной на проверках другого повода.

Разница разрешающей способности между этими двумя наборами оценок и есть цена
повода. Резерв расходуется ОДИН раз: обе оценки сравниваются парно.

Повод НЕ входит в признаки намеренно. Иначе измерялось бы, умеет ли модель
прочесть код повода, а вопрос в другом: меняется ли смысл ОСТАЛЬНЫХ признаков
в зависимости от него.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

from dsx.label import LABEL
from dsx.measure import measure_contrast
from dsx.policy import OverrideLedger
from dsx.project import load
from dsx.runner import RESERVE, run
from dsx.samples import Purpose

PROJECT = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("build", PROJECT / "build.py")
_build = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_build)

import sys  # noqa: E402

sys.path.insert(0, str(PROJECT))
from triggers import ROUTINE, TRIGGERED  # noqa: E402

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

FEATURES = ["EVALUATION_AGENCY", "ACTIVITY_LOCATION", "STATE_CODE", "sector", "prior_violations"]
"""Повода среди них нет намеренно: см. заголовок модуля."""


def ledger() -> OverrideLedger:
    overrides = OverrideLedger()
    overrides.override(
        "S10",
        reason="штат площадки и код отрасли взяты из справочника сегодняшнего состояния, "
        "истории он не хранит. Обход записан, потому что обе величины на практике не "
        "меняются: площадка не переезжает между штатами и не меняет вид производства. "
        "Изменчивые свойства из той же таблицы убраны из признаков, а не обойдены",
        author="автор",
    )
    # N3 НЕ обходится намеренно: доля нарушений в резерве 30.2% против 26.4% в
    # окнах. Находка обязана остаться видимой, иначе измеренное число прочтут
    # как относящееся к тем же годам, на которых модель училась.
    return overrides


def category_map(table: pl.DataFrame) -> dict[str, list]:
    return {
        name: sorted(table[name].unique().drop_nulls().to_list())
        for name in FEATURES
        if not table[name].dtype.is_numeric()
    }


def prepare(frame: pl.DataFrame, categories: dict[str, list]):
    import pandas as pd

    data = {}
    for name in FEATURES:
        column = frame[name]
        if name in categories:
            data[name] = pd.Categorical(column.to_list(), categories=categories[name])
        else:
            data[name] = column.cast(pl.Float64, strict=False).to_numpy()
    return pd.DataFrame(data)


def main() -> int:
    form = load(PROJECT / "project.yaml")
    table = _build.build()
    categories = category_map(table)

    result = run(form, table, PROJECT / "report", overrides=ledger())
    print(result.summary())

    last = result.split.parts[-1]
    train = last.train.filter(pl.col(LABEL).is_not_null())
    result.samples.fit(last.name, f"обучение двух моделей на окне {last.name}")

    models = {}
    for kind in (ROUTINE, TRIGGERED):
        part = train.filter(pl.col("trigger_kind") == kind)
        model = lgb.LGBMClassifier(**SETTINGS)
        model.fit(prepare(part, categories), part[LABEL].to_numpy())
        models[kind] = model
        print(
            f"  обучение на поводах «{kind}»: {part.height:,} строк, "
            f"доля нарушений {part[LABEL].mean():.1%}"
        )

    reserve = result.samples.checkout(RESERVE, Purpose.AUDIT, "подготовка оценок")
    prepared = prepare(reserve, categories)
    scored = {kind: model.predict_proba(prepared)[:, 1] for kind, model in models.items()}

    own = np.where(
        (reserve["trigger_kind"] == TRIGGERED).to_numpy(), scored[TRIGGERED], scored[ROUTINE]
    )
    other = np.where(
        (reserve["trigger_kind"] == TRIGGERED).to_numpy(), scored[ROUTINE], scored[TRIGGERED]
    )

    print("\n=== ЦЕНА ПОВОДА (P-2) ===")
    contrast = measure_contrast(
        result.samples,
        RESERVE,
        pl.Series("own", own),
        pl.Series("other", other),
        left_name="модель своего повода",
        right_name="модель чужого повода",
    )
    print(contrast)

    result.study.conclude(
        str(contrast) + " Измерение проведено на резерве 2024-2025 годов, где доля нарушений 30.2% "
        "против 26.4% в обучающих окнах (находка N3): число описывает поздний набор, "
        "а не годы обучения. Повод в признаки не входил: измерялось, меняется ли смысл "
        "ОСТАЛЬНЫХ признаков от того, почему состоялось наблюдение."
    )
    (PROJECT / "report" / "measured.md").write_text(result.study.render(), encoding="utf-8")
    print(f"\nотчёт: {PROJECT / 'report' / 'measured.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
