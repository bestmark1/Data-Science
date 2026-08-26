"""Обучение и измерение на шестом кейсе: сколько стоит изменчивость истории.

Две постановки, различающиеся ОДНИМ признаком:

  «момент решения»  — история участка посчитана по сводкам, ЗАРЕГИСТРИРОВАННЫМ
                      до решения. Так знает тот, кто решает;
  «момент выгрузки» — история посчитана по сводкам, ПРОИСШЕДШИМ в окне,
                      независимо от того, когда о них записали. Так считает
                      всякий, кто взял готовую выгрузку и не подумал о
                      запаздывании регистрации.

Остальные признаки у постановок общие. Разница между ними и есть цена
изменчивости — предсказание P-2, объявленное до контакта с данными.

Обучение живёт ЗДЕСЬ: ядро предсказаний не производит, оно принимает готовую
колонку оценок и проводит протокол измерения. Подбора настроек нет: подбор
есть выбор, а выбор расходует выборку.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import lightgbm as lgb
import polars as pl

from dsx.label import LABEL
from dsx.measure import measure_contrast
from dsx.policy import OverrideLedger
from dsx.project import load
from dsx.roles import Availability, Role
from dsx.runner import RESERVE, run
from dsx.samples import Purpose

PROJECT = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("build", PROJECT / "build.py")
_build = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_build)

KNOWN = "nearby_90d_known"
EXPORT = "nearby_90d_export"

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
"""Настройки зафиксированы и у обеих постановок ОДИНАКОВЫ: иначе сравнивались
бы не признаки, а настройки."""


def shared_features(form) -> list[str]:
    """Признаки, общие для обеих постановок: всё, кроме двух историй."""
    return [
        c.name
        for c in form.schema_spec().columns
        if c.role is Role.FEATURE
        and c.availability is Availability.AT_DECISION
        and c.name not in (KNOWN, EXPORT)
    ]


def category_map(table: pl.DataFrame, names: list[str]) -> dict[str, list]:
    """Один общий словарь категорий на всю таблицу.

    Коды, построенные по каждой выборке отдельно, разошлись бы между обучением
    и резервом: третье значение здесь и третье там — разные вещи.
    """
    return {
        name: sorted(table[name].unique().drop_nulls().to_list())
        for name in names
        if not table[name].dtype.is_numeric()
    }


def prepare(frame: pl.DataFrame, names: list[str], categories: dict[str, list]):
    """Кадр для LightGBM: категории по общему словарю, числа как есть."""
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
    shared = shared_features(form)

    table = _build.build()
    categories = category_map(table, shared)

    overrides = OverrideLedger()
    # Нестационарность исхода НЕ обходится намеренно: находка обязана остаться
    # видимой, а заключение — назвать её. Обход убрал бы её из отчёта, и
    # измеренное число прочиталось бы как относящееся к той же популяции, на
    # которой модель училась.
    result = run(form, table, PROJECT / "report", overrides=overrides)
    print(result.summary())

    last = result.split.parts[-1]
    train = last.train.filter(pl.col(LABEL).is_not_null())
    result.samples.fit(last.name, f"обучение обеих постановок на окне {last.name}")
    print(f"\nобучение: {train.height:,} строк, общих признаков {len(shared)}")

    def fit(history: str):
        names = [*shared, history]
        model = lgb.LGBMClassifier(**SETTINGS)
        model.fit(prepare(train, names, categories), train[LABEL].to_numpy())
        return model, names

    known_model, known_names = fit(KNOWN)
    export_model, export_names = fit(EXPORT)

    reserve = result.samples.checkout(RESERVE, Purpose.AUDIT, "подготовка оценок")

    def predict(model, names) -> pl.Series:
        return pl.Series("score", model.predict_proba(prepare(reserve, names, categories))[:, 1])

    print("\n=== ЦЕНА ИЗМЕНЧИВОСТИ ИСТОРИИ (P-2) ===")
    contrast = measure_contrast(
        result.samples,
        RESERVE,
        predict(export_model, export_names),
        predict(known_model, known_names),
        left_name="момент выгрузки",
        right_name="момент решения",
    )
    print(contrast)

    result.study.conclude(
        str(contrast) + " Измерение проведено на резерве, где доля класса выше, чем в "
        "обучающих окнах (см. находку N3): число описывает резерв, а не окна."
    )
    (PROJECT / "report" / "measured.md").write_text(result.study.render(), encoding="utf-8")
    print(f"\nотчёт: {PROJECT / 'report' / 'measured.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
