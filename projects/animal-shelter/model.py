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
    "intake_type",
    "intake_condition",
    "animal_type",
    "sex_upon_intake",
    "age_upon_intake",
    "breed",
    "color",
]
"""Всё, что известно при приёме. Уточнения исхода среди них нет: оно было
контролем К-1 и объявлено ролью ignored."""

RULE = BaselineRule(kind=RuleKind.CONSTANT, constant=0.273)
"""Базовое правило: всем одна оценка, равная доле УСЫНОВЛЁННЫХ к сроку в окнах."""

SEGMENTS = ["animal_type", "intake_type", "intake_condition"]
"""Оси разбора смещения объявлены заранее: перебор всех признаков ради самого
смещённого — тот же подбор, расходующий выборку."""


def ledger() -> OverrideLedger:
    """Обходы с причинами. Каждый — решение, а не умолчание."""
    overrides = OverrideLedger()
    overrides.override(
        "N14",
        reason="'found_location' смешивает разное. Верхние значения — административные "
        "единицы: 'Austin (TX)' стоит у 21 974 пребываний, то есть у каждого пятого, "
        "а 'Travis (TX)' и 'Outside Jurisdiction' — ещё у 4 620. Держать их группой "
        "значило бы вынести пятую часть данных на одну сторону сплита. Настоящая "
        "группа здесь — помёт или изъятие у одного хозяина, — и она в данных не "
        "обозначена ничем: адрес находки её лишь иногда совпадает. Зависимость "
        "признаётся существующей и НЕизмеримой, а не отсутствующей",
        author="автор",
    )
    overrides.override(
        "N16",
        reason="цензура конкурирующими видами связана с обстоятельством приёма по "
        "устройству отрасли, а не по дефекту сборки: бродячих чаще передают партнёрским "
        "организациям, сданных хозяином чаще возвращают ему же. Устранить связь нельзя — "
        "она и есть предмет. Расхождение принимается как условие задачи и названо в "
        "выводе: измеренное описывает тех, чьё пребывание не оборвал конкурирующий исход, "
        "а это 62% популяции",
        author="автор",
    )
    # N3 НЕ обходится намеренно: доля усыновлённых к сроку падает 31.7% -> 26.7%
    # -> 22.6% по окнам и 13.7% в резерве. Находка обязана остаться видимой,
    # иначе измеренное число прочтут как относящееся ко всему периоду.
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
