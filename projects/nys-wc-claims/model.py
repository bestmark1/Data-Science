"""Обучение и измерение на третьем кейсе.

Обучение живёт ЗДЕСЬ, а не в ядре. Ядро предсказаний не производит: оно
принимает готовую колонку оценок и проводит протокол измерения.

Сценарий один и жёсткий, потому что каждое ветвление — это выбор, а выбор
расходует выборку:

  1. модель обучается на обучающей части последнего окна;
  2. предсказания по окнам идут на проверку устойчивости знака (N7);
  3. предсказания по резерву идут на итоговое измерение — один раз;
  4. разбор смещения по сегментам считается на резерве (N10).

Подбора гиперпараметров нет намеренно. Подбор есть выбор, выбор расходует
выборку, и на этом протокол сломается.
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

SINCE = "2015-01-01"

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
"""Настройки зафиксированы. Их подбор был бы выбором, расходующим выборку."""

SEGMENTS = ["district", "carrier_type", "industry", "gender", "medical_fee_region"]
"""Оси, по которым смотрится смещение. Объявлены заранее: перебор всех
категориальных признаков ради самого смещённого — тот же подбор."""

RULE = BaselineRule(kind=RuleKind.CONSTANT, constant=0.09)
"""Базовое правило: всем одна и та же оценка, близкая к доле класса.

Ничего не различает и потому калибровано хорошо. Это честный минимум: если
модель его не превосходит по разрешающей способности, строить её незачем.
"""


def features(form) -> list[str]:
    """Признаки, объявленные доступными в момент решения."""
    from dsx.roles import Availability, Role

    return [
        c.name
        for c in form.schema_spec().columns
        if c.role is Role.FEATURE and c.availability is Availability.AT_DECISION
    ]


def category_map(table: pl.DataFrame, names: list[str]) -> dict[str, list]:
    """Один общий словарь категорий на всю таблицу.

    Коды, построенные по каждой выборке отдельно, разошлись бы между обучением
    и оценкой: третье значение в обучении и третье в резерве — разные вещи.
    """
    return {
        name: sorted(table[name].unique().drop_nulls().to_list())
        for name in names
        if not table[name].dtype.is_numeric()
    }


def prepare(frame: pl.DataFrame, names: list[str], categories: dict[str, list]):
    """Кадр для LightGBM: категории по общему словарю, числа как есть.

    Строковые колонки остаются в таблице СТРОКАМИ и кодируются только здесь.
    Первая версия кодировала их прямо в таблице и требовала обратной сборки
    названий для разбора сегментов — там я и ошибся, взяв последние строки
    файла вместо строк резерва, потому что таблица по дате не отсортирована.
    """
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

    table = _build.build(since=SINCE)
    categories = category_map(table, names)

    overrides = OverrideLedger()
    overrides.override(
        "A13",
        reason="две строки из 1 804 676 — заглушки миграции старых дел: "
        "травмы 1941 года, слушания 1999 и 2001, дата сборки 1 января. "
        "Размечены пустой меткой и в анализ не входят",
        author="автор",
    )
    result = run(form, table, PROJECT / "report", overrides=overrides)
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
    steadiness = stability_across_windows(result.samples, by_window, RULE)
    print(steadiness.report_section())

    print("\n=== ИТОГОВОЕ ИЗМЕРЕНИЕ НА РЕЗЕРВЕ ===")
    reserve = result.samples.checkout(RESERVE, Purpose.AUDIT, "подготовка оценок")
    scores = predict(reserve)
    verdict = measure_against_baseline(result.samples, RESERVE, scores, RULE)
    print(verdict.report_section())

    print("\n=== СМЕЩЕНИЕ ПО СЕГМЕНТАМ (N10) ===")
    # Названия сегментов берутся из САМОГО резерва: строковые колонки никуда
    # не кодировались и путешествуют вместе со строками.
    segments = segment_bias(reserve, scores, SEGMENTS, min_rows=500)
    print(segment_section(segments))

    result.study.measurement = verdict
    # Проверка N3 нашла, что доля класса в резерве выше, чем в окнах. Обход НЕ
    # записывается намеренно: находка должна остаться видимой, а заключение —
    # назвать её, иначе измеренное число будет прочитано как относящееся к той
    # же популяции, на которой модель училась.
    shifted = any(
        s.finding.value == "non_stationary_target" and "РЕЗЕРВЕ" in s.detail
        for s in result.checks.signals
    )
    caveat = (
        " Измерение проведено на популяции с более высокой долей класса, "
        "чем в обучающих окнах (см. находку N3): число описывает резерв, а не окна."
        if shifted
        else ""
    )
    result.study.conclude(verdict.statement() + caveat)
    (PROJECT / "report" / "measured.md").write_text(result.study.render(), encoding="utf-8")
    print(f"\nотчёт: {PROJECT / 'report' / 'measured.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
