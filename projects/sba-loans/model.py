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

SEGMENTS = ["state", "bank_state", "sector", "urban_rural", "new_business"]
"""Оси разбора смещения объявлены заранее: перебор всех категориальных
признаков ради самого смещённого — тот же подбор, расходующий выборку."""

RULE = BaselineRule(kind=RuleKind.CONSTANT, constant=0.077)
"""Базовое правило: всем одна оценка, равная доле класса в окнах.

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
        "N14",
        reason="банки присутствуют по обе стороны сплита у 93-96% оценочных кредитов. "
        "Расхождение верное и принимается как условие задачи: предсказание делается "
        "по НОВОМУ кредиту в уже известном банке, а не по новому банку. Измеренное "
        "число описывает именно эту постановку, и холодный старт банка ею не покрыт",
        author="автор",
    )
    overrides.override(
        "A12",
        reason="незрелость 17-28% по окнам — прямое следствие ПЕРЕМЕННОГО срока: "
        "срок кредита от 1 до 569 месяцев, и метки зреют в разное время. Это "
        "объявлено осью до контакта с данными (порог >=2%) и принято. Смещение в "
        "сторону коротких сроков названо в заключении, а не спрятано",
        author="автор",
    )
    overrides.override(
        "N4",
        reason="связь срока и сумм с исходом меняет знак между окнами 1999-2001 и "
        "2003-2005. Это находка о предметной области, а не дефект: режим кредитования "
        "изменился. Модель обучается на ПОСЛЕДНЕМ окне, где связь та же, что в резерве",
        author="автор",
    )
    # N3 НЕ обходится намеренно: доля дефолтов в резерве 17.6% против 7.7% в
    # окнах — это предкризисный набор 2005-2006 годов. Находка обязана остаться
    # видимой, иначе измеренное число прочтут как относящееся к тем же годам,
    # на которых модель училась.
    return overrides


def main() -> int:
    form = load(PROJECT / "project.yaml")
    names = features(form)

    table = _build.build()
    categories = category_map(table, names)

    result = run(form, table, PROJECT / "report", overrides=ledger())
    # Ядру объявляется, чем ДЕЙСТВИТЕЛЬНО питалась модель (P11). Здесь
    # список берётся из самого формуляра, и утечка невозможна по
    # построению, — но объявление обязательно всё равно: механизм,
    # который можно не позвать, в этом проекте ломался трижды.
    result.uses(names)

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
        + " Измерение проведено на резерве 2005-2006 годов, где доля дефолтов 17.6% "
        "против 7.7% в обучающих окнах (находка N3): число описывает предкризисный "
        "набор, а не годы обучения. Из измерения исключены 14.7% кредитов, чей срок "
        "не истёк к концу наблюдения, поэтому оно смещено в сторону коротких сроков."
    )
    (PROJECT / "report" / "measured.md").write_text(result.study.render(), encoding="utf-8")
    print(f"\nотчёт: {PROJECT / 'report' / 'measured.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
