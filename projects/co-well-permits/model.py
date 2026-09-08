"""Обучение и измерение на двадцатом кейсе: разрешения на скважины Колорадо.

Обучение живёт ЗДЕСЬ: ядро предсказаний не производит, оно принимает готовую
колонку оценок и проводит протокол измерения. Подбора настроек нет — подбор
есть выбор, а выбор расходует выборку.

Главная находка кейса лежит не в качестве модели: срок здесь взят из источника,
и он оказался щедрым — среди построенных скважин 99.8% уложились в два года.
Значит метрика меряет факт постройки, а не соблюдение срока.
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
    "seed": 20260908,
}

FEATURES = [
    "county",
    "div",
    "wd",
    "denver_basin_aquifer",
    "designated_basin",
    "management_district",
    "associated_uses",
    "elev",
]
"""Всё, что известно в момент выдачи разрешения. Вида разрешения среди них нет:
он объявлен причиной наблюдения, а не признаком."""

RULE = BaselineRule(kind=RuleKind.CONSTANT, constant=0.701)
"""Базовое правило: всем одна оценка, равная доле построенных в срок по окнам
(70.1%)."""

SEGMENTS = ["county", "denver_basin_aquifer", "associated_uses"]
"""Оси разбора смещения объявлены заранее: перебор всех признаков ради самого
смещённого — тот же подбор, расходующий выборку."""


def ledger() -> OverrideLedger:
    """Обходы с причинами. Каждый — решение, а не умолчание."""
    overrides = OverrideLedger()
    overrides.override(
        "N14",
        reason="группового уровня в доступных колонках нет. Ядро называет кандидатами "
        "округ (65 значений на 33 852 строки) и назначение воды (128) — это "
        "географические и правовые КАТЕГОРИИ, а не группы: делить по ним выборку "
        "значило бы объявить, что скважины одного округа принадлежат одному владельцу. "
        "Настоящая группа — участок или владелец — лежит в колонках, которые не "
        "скачивались: это персональные сведения",
        author="автор",
    )
    overrides.override(
        "P11",
        reason="незрелых исходов 1.1% в последнем окне и 2.6% в резерве. Они есть "
        "продлённые разрешения, у которых срок уходит за два года; отбросить их — "
        "сместить выборку в сторону коротких сроков, оставить — принять смещение "
        "известного размера. Второе меньше первого, и размер назван",
        author="автор",
    )
    # НЕ обходятся намеренно:
    #
    # event_before_decision — 1 454 строки, где скважина построена раньше выдачи
    # разрешения, худшая на 45 903 дня. Это старые скважины, узаконенные задним
    # числом; ядро зануляет им метку само, и обход ничего бы не изменил.
    #
    # informative_unobservability — наблюдаемость связана с бассейном и районом
    # управления. Это находка о данных, и прятать её обходом нельзя.
    #
    # observation_reason_matters — у жилых скважин доля 76.7%, у наблюдательных
    # скважин 40.0%. Задача у этих поводов разная.
    #
    # non_stationary_target — резерв 66.2% против окон 70.1%.
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
        + " Срок здесь взят ИЗ ИСТОЧНИКА, а не назначен автором, и это первый такой "
        "кейс из двадцати. Он же оказался щедрым: среди построенных скважин 99.8% "
        "уложились в два года, поэтому измеренное описывает факт постройки, а не "
        "соблюдение срока. Доля построенных в резерве 66.2% против 70.1% в окнах, и "
        "наблюдаемость исхода связана с назначенным бассейном — метрика описывает "
        "популяцию разрешений со ПРОСТАВЛЕННЫМ сроком, то есть 70% выданных."
    )
    (PROJECT / "report" / "measured.md").write_text(result.study.render(), encoding="utf-8")
    print(f"\nотчёт: {PROJECT / 'report' / 'measured.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
