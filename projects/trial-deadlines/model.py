"""Обучение и измерение на седьмом кейсе.

Обучение живёт ЗДЕСЬ: ядро предсказаний не производит, оно принимает готовую
колонку оценок и проводит протокол измерения. Подбора настроек нет — подбор
есть выбор, а выбор расходует выборку.

Три блокирующих сигнала обходятся с записанной причиной, один — НЕТ.
Различие не в удобстве: обход уместен там, где расхождение осознано и
принято как условие задачи, и неуместен там, где оно должно остаться в
заключении, иначе измеренное число будет прочитано не о той популяции.
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
from dsx.roles import Availability, Role
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

SEGMENTS = ["sponsor_class", "allocation", "masking", "primary_purpose", "intervention_model"]
"""Оси разбора смещения объявлены заранее: перебор всех категориальных
признаков ради самого смещённого — тот же подбор, расходующий выборку."""

RULE = BaselineRule(kind=RuleKind.CONSTANT, constant=0.365)
"""Базовое правило: всем одна оценка, равная доле класса в окнах (36.5%).

Ничего не различает и потому калибровано хорошо. Честный минимум: не
превзойдя его по разрешающей способности, строить модель незачем.
"""


def features(form) -> list[str]:
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
            data[name] = column.cast(pl.Float64, strict=False).to_numpy()
    return pd.DataFrame(data)


def ledger() -> OverrideLedger:
    """Обходы с причинами. Каждый — решение, а не умолчание."""
    overrides = OverrideLedger()
    overrides.override(
        "A14",
        reason="'UNKNOWN' у статуса — законный статус реестра: спонсор перестал "
        "обновлять запись, и это сведение, а не пропуск. 'NA' у схемы распределения — "
        "законное «неприменимо» у одногрупповых исследований. 'UNKNOWN' у класса "
        "спонсора — то же. Проверка требует либо исправить при сборке, либо записать "
        "обход с причиной; здесь верно второе, потому что значения содержательны",
        author="автор",
    )
    overrides.override(
        "N14",
        reason="спонсоры присутствуют по обе стороны сплита у 26-55% оценочных "
        "исследований. Расхождение верное и принимается как условие задачи: "
        "предсказание делается по НОВОМУ исследованию известного спонсора. Холодный "
        "старт спонсора этой постановкой не покрыт",
        author="автор",
    )
    return overrides


def main() -> int:
    form = load(PROJECT / "project.yaml")
    names = features(form)

    table = _build.build()
    categories = category_map(table, names)

    result = run(form, table, PROJECT / "report", overrides=ledger())
    print(result.summary())

    last = result.split.parts[-1]
    train = last.train.filter(pl.col(LABEL).is_not_null())
    result.samples.fit(last.name, f"обучение на обучающей части окна {last.name}")
    print(f"\nобучение: {train.height:,} строк, признаков {len(names)}")

    model = lgb.LGBMClassifier(**SETTINGS)
    model.fit(prepare(train, names, categories), train[LABEL].to_numpy())

    reserve = result.samples.checkout(RESERVE, Purpose.AUDIT, "подготовка оценок")
    scores = pl.Series("score", model.predict_proba(prepare(reserve, names, categories))[:, 1])

    print("\n=== ИТОГОВОЕ ИЗМЕРЕНИЕ НА РЕЗЕРВЕ ===")
    verdict = measure_against_baseline(result.samples, RESERVE, scores, RULE)
    print(verdict.report_section())

    print("\n=== СМЕЩЕНИЕ ПО СЕГМЕНТАМ ===")
    print(segment_section(segment_bias(reserve, scores, SEGMENTS, min_rows=500)))

    result.study.measurement = verdict
    result.study.conclude(
        verdict.statement()
        + " Из популяции исключены 10.0% исследований, зарегистрированных ПОСЛЕ "
        "первичного завершения: они обещания не давали. Исход наблюдается лишь у тех, "
        "чья дата завершения помечена как фактическая; у 1 529 исследований спонсор "
        "перестал обновлять запись, и их исход неизвестен, а не отрицателен."
    )
    (PROJECT / "report" / "measured.md").write_text(result.study.render(), encoding="utf-8")
    print(f"\nотчёт: {PROJECT / 'report' / 'measured.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
