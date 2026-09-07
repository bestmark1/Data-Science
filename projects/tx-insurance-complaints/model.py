"""Обучение и измерение на девятнадцатом кейсе: жалобы на страховщиков Техаса.

Обучение живёт ЗДЕСЬ: ядро предсказаний не производит, оно принимает готовую
колонку оценок и проводит протокол измерения. Подбора настроек нет — подбор
есть выбор, а выбор расходует выборку.

Главная находка кейса лежит не в качестве модели, а в том, что доля закрытых к
сроку меняется втрое внутри периода: департамент замедлился во второй половине
2024 (медиана с 45 суток до 137) и восстанавливался весь 2025 год. Измеренное
здесь описывает вторую половину 2025 года и никакой другой период.
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
    "seed": 20260907,
}

FEATURES = [
    "coverage_type",
    "coverage_level",
    "respondent_type",
    "respondent_role",
    "complainant_role",
    "complainant_type",
]
"""Всё, что известно в момент получения жалобы. Вида жалобы среди них нет: он
объявлен причиной наблюдения, а не признаком."""

RULE = BaselineRule(kind=RuleKind.CONSTANT, constant=0.229)
"""Базовое правило: всем одна оценка, равная доле закрытых к сроку в последнем
окне (w2, 22.9%)."""

SEGMENTS = ["coverage_type", "complainant_role", "respondent_type"]
"""Оси разбора смещения объявлены заранее: перебор всех признаков ради самого
смещённого — тот же подбор, расходующий выборку."""


def ledger() -> OverrideLedger:
    """Обходы с причинами. Каждый — решение, а не умолчание."""
    overrides = OverrideLedger()
    overrides.override(
        "N14",
        reason="страховых компаний 5 286, и крупные присутствуют по обе стороны сплита "
        "каждый год: жалобы на них поступают непрерывно. Расхождение верное и "
        "принимается как условие задачи — предсказание делается по НОВОЙ жалобе на "
        "известную компанию, а не по новой компании. 88–92% оценочного окна, и иначе "
        "быть не может: рынок страхования Техаса не обновляется за год",
        author="автор",
    )
    # N3 (non_stationary_target) НЕ обходится: доля прыгает между окнами в 2.8
    # раза, и это главная находка кейса — замедление департамента в 2024 году.
    #
    # S6 (observation_reason_matters) не обходится: у 'Portal' доля 54.8%, у
    # 'Property and Casualty' — 27.2%. Задача у этих поводов разная.
    #
    # outcome_known_at_decision не обходится: 556 строк закрыты в день подачи,
    # и это часть ответа, а не поломка.
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
        + " Число описывает ВТОРУЮ ПОЛОВИНУ 2025 ГОДА и никакой другой период. Доля "
        "закрытых к сроку внутри периода менялась втрое: 45.7% в первом окне, 16.3% "
        "во втором, 22.9% в третьем. За этим стоит замедление департамента — медиана "
        "срока закрытия выросла с 45 суток летом 2023 до 137 суток в ноябре 2024 и к "
        "декабрю 2025 вернулась к 50. Модель обучена на восстанавливающемся участке и "
        "измерена на нём же; переносить измеренное на 2023 или на середину 2024 нельзя."
    )
    (PROJECT / "report" / "measured.md").write_text(result.study.render(), encoding="utf-8")
    print(f"\nотчёт: {PROJECT / 'report' / 'measured.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
