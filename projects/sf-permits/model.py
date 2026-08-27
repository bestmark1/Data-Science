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
    "permit_type",
    "estimated_cost",
    "number_of_existing_stories",
    "number_of_proposed_stories",
    "plansets",
    "site_permit",
    "fire_only_permit",
    "application_submission_method",
    "existing_use",
    "proposed_use",
    "neighborhoods_analysis_boundaries",
]
"""Всё, что известно при подаче заявки. Дня изменения состояния среди них нет:
он был контролем К-1 и объявлен ролью ignored."""

RULE = BaselineRule(kind=RuleKind.CONSTANT, constant=0.792)
"""Базовое правило: всем одна оценка, равная доле выданных к сроку в окнах."""

SEGMENTS = ["permit_type", "neighborhoods_analysis_boundaries", "application_submission_method"]
"""Оси разбора смещения объявлены заранее: перебор всех признаков ради самого
смещённого — тот же подбор, расходующий выборку."""


def ledger() -> OverrideLedger:
    """Обходы с причинами. Каждый — решение, а не умолчание."""
    overrides = OverrideLedger()
    overrides.override(
        "N14",
        reason="округов надзора в городе одиннадцать, и все одиннадцать присутствуют по "
        "обе стороны сплита. Иначе быть не может: держать округ целиком вне обучения "
        "значило бы выбросить около десятой части заявок. Расхождение верное и "
        "принимается как условие задачи: предсказание делается по НОВОЙ заявке в "
        "известном округе, а не по новому округу",
        author="автор",
    )
    # N3 НЕ обходится намеренно: доля устранённых в срок в резерве 25.0% против
    # 21.8% в окнах. Находка обязана остаться видимой, иначе измеренное число
    # прочтут как относящееся к тем же годам, на которых модель училась.
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
        + " Измерение проведено на резерве 2022-2023 годов. Доля выданных к сроку падает "
        "по окнам с 83.1% до 69.8%: число описывает поздний набор, а не весь период. "
        "Единица решения — заявка, а не пара «заявка-участок»: строки сведены к одной "
        "на заявку, иначе заявка на сто один участок считалась бы сто один раз."
    )
    (PROJECT / "report" / "measured.md").write_text(result.study.render(), encoding="utf-8")
    print(f"\nотчёт: {PROJECT / 'report' / 'measured.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
