"""Обучение и измерение на одиннадцатом кейсе: соблюдение расписания рейсов.

Обучение живёт ЗДЕСЬ: ядро предсказаний не производит, оно принимает готовую
колонку оценок и проводит протокол измерения. Подбора настроек нет — подбор
есть выбор, а выбор расходует выборку.

Модель обучается на последнем окне и меряется на резерве против постоянного
правила. Кейс обычный, и измерение здесь не предмет находки, а проверка того,
что протокол доходит до конца на новой отрасли.
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

FEATURES = [
    "call_type",
    "call_type_group",
    "battalion",
    "station_area",
    "zipcode_of_incident",
    "at_night",
    "at_rush_hour",
]
"""Всё, что известно в момент приёма вызова. Момента освобождения расчёта среди
них нет: он был контролем К-1 и объявлен ролью ignored."""

RULE = BaselineRule(kind=RuleKind.CONSTANT, constant=0.537)
"""Базовое правило: всем одна оценка, равная доле прибывших к сроку в окнах."""

SEGMENTS = ["call_type_group", "battalion", "at_night"]
"""Оси разбора смещения объявлены заранее: перебор всех признаков ради самого
смещённого — тот же подбор, расходующий выборку."""


def ledger() -> OverrideLedger:
    """Обходы с причинами. Каждый — решение, а не умолчание."""
    overrides = OverrideLedger()
    overrides.override(
        "N14",
        reason="районов в городе сорок один, и все они присутствуют по обе стороны "
        "сплита. Иначе быть не может: вызовы поступают отовсюду каждый день. "
        "Расхождение верное и принимается как условие задачи: предсказание делается "
        "по НОВОМУ вызову в известном районе, а не по новому району",
        author="автор",
    )
    # N17 НЕ обходится намеренно. Доля прибывших к сроку у приоритета 'E' 91.6%,
    # у '1' — 3.3%: это разные задачи, сведённые в одну выборку. Находка обязана
    # остаться видимой, иначе итоговое число прочтут как относящееся к вызовам
    # вообще, а оно относится к их смеси в наблюдённой пропорции.
    #
    # N3 тоже не обходится: доля растёт по окнам и продолжает расти в резерве.
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

    # Ядру объявляется, чем ДЕЙСТВИТЕЛЬНО питалась модель. Без этого

    # измерение отказывает (P11): утечку иначе достаточно объявить в

    # схеме ролью ignored, и ядро о ней не узнает.

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
        + " Измерение проведено на резерве последней четверти периода, где доля закрытых "
        "к сроку 42.1% против 48.5% в обучающих окнах (находка N3): число описывает "
        "поздний набор, а не весь период. Из популяции исключены 43 110 обращений, "
        "закрытых раньше приёма: их исход к их приёму не относится."
    )
    (PROJECT / "report" / "measured.md").write_text(result.study.render(), encoding="utf-8")
    print(f"\nотчёт: {PROJECT / 'report' / 'measured.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
