"""Обучение и измерение на восемнадцатом кейсе: наряды на работы с деревьями.

Обучение живёт ЗДЕСЬ: ядро предсказаний не производит, оно принимает готовую
колонку оценок и проводит протокол измерения. Подбора настроек нет — подбор
есть выбор, а выбор расходует выборку.

Кейс обычный, и измерение здесь не предмет находки, а проверка того, что
протокол доходит до конца на новой отрасли. Главные находки кейса лежат в
данных, а не в качестве модели.
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

FEATURES = ["wocategory", "wotype", "boroughcode", "communityboard", "zipcode", "citycouncil"]
"""Всё, что известно в момент заведения наряда. Приоритета среди них нет: он
объявлен причиной наблюдения, а не признаком."""

RULE = BaselineRule(kind=RuleKind.CONSTANT, constant=0.289)
"""Базовое правило: всем одна оценка, равная доле закрытых к сроку в окнах."""

SEGMENTS = ["wocategory", "wotype", "boroughcode"]
"""Оси разбора смещения объявлены заранее: перебор всех признаков ради самого
смещённого — тот же подбор, расходующий выборку."""


def ledger() -> OverrideLedger:
    """Обходы с причинами. Каждый — решение, а не умолчание."""
    overrides = OverrideLedger()
    overrides.override(
        "N14",
        reason="районов в городе 378, и все они присутствуют по обе стороны сплита: "
        "наряды заводятся всюду каждый год. Расхождение верное и принимается как "
        "условие задачи — предсказание делается по НОВОМУ наряду в известном районе, "
        "а не по новому району",
        author="автор",
    )
    # N3 НЕ обходится намеренно: доля прыгает между окнами в 2.5 раза, и это
    # след обрыва заполнения, а не шум. Обойти значило бы спрятать главную
    # находку кейса.
    #
    # S6 (observation_reason_matters) тоже не обходится: у приоритета '12' доля
    # 99%, у нерасставленного приоритета — 15%. Задача у этих поводов разная.
    #
    # event_before_decision не обходится: 32 строки ядро само оставляет без
    # метки, и обход ничего бы не изменил, кроме видимости чистоты.
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
    result.uses(FEATURES)
    print(result.summary())

    last = result.split.parts[-1]
    train = last.train.filter(pl.col(LABEL).is_not_null())
    result.samples.fit(last.name, f"обучение на обучающей части окна {last.name}")

    model = lgb.LGBMClassifier(**SETTINGS)
    model.fit(prepare(train, categories), train[LABEL].to_numpy())
    print(f"  обучение: {train.height:,} строк, признаков {len(FEATURES)}")

    reserve = result.samples.checkout(RESERVE, Purpose.AUDIT, "подготовка оценок")
    scores = pl.Series("score", model.predict_proba(prepare(reserve, categories))[:, 1])

    print("\n=== ИТОГОВОЕ ИЗМЕРЕНИЕ НА РЕЗЕРВЕ ===")
    verdict = measure_against_baseline(result.samples, RESERVE, scores, RULE)
    print(verdict.report_section())

    print("\n=== СМЕЩЕНИЕ ПО СЕГМЕНТАМ ===")
    print(segment_section(segment_bias(reserve, scores, SEGMENTS, min_rows=500)))

    result.study.measurement = verdict
    result.study.conclude(
        verdict.statement()
        + " Число описывает УЗКУЮ полосу: резерв взят с 2023-03-02 по 2023-05-31, потому "
        "что с июля 2023 дата закрытия не проставлена ни у одного наряда, хотя 34 550 из "
        "них числятся закрытыми. Доля закрытых к сроку прыгает между окнами в 2.5 раза "
        "(24.6%, 23.1%, 57.2%) — это след того же обрыва, ползущий назад по времени: чем "
        "ближе окно к обрыву, тем большая доля попавших в него закрытий быстрая. Мерить "
        "на таких данных можно, переносить измеренное на другой период нельзя."
    )
    (PROJECT / "report" / "measured.md").write_text(result.study.render(), encoding="utf-8")
    print(f"\nотчёт: {PROJECT / 'report' / 'measured.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
