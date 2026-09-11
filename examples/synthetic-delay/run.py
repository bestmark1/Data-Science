"""Прикладной пример: форма → проверки → отчёт → измерение против базового правила.

    .venv/bin/python examples/synthetic-delay/run.py

Ни сети, ни внешних выгрузок: данные порождает синтетический мир стенда
детерминированно. Всё остальное — настоящие публичные пути ядра, те же, что
у кейсов: `dsx.project.load`, `dsx.runner.run`, `dsx.measure`.

ЧТО ЭТОТ ПРИМЕР ПОКАЗЫВАЕТ. Процедуру целиком и один её оборот: объявления
проверяются, возражения печатаются, измерение доводится до разницы с базовым
правилом и доверительных границ.

ЧЕГО ОН НЕ ПОКАЗЫВАЕТ — и это важнее. Ограничения перечислены в README и
печатаются рядом с результатом. Коротко: инструмент не подтверждает, что с
данными всё хорошо; сигнал не есть доказанный дефект; числа на синтетике
ничего не говорят о качестве на настоящей задаче.

ПОЧЕМУ ЭТОТ МИР. `feature-falsely-declared-available` — единственный, чья
ожидаемая находка есть ровно та проверка (N6), про которую предложение прямо
говорит: она НЕ отличает утечку от сильного честного признака. Пример на нём
показывает не только отчёт, но и границу его толкования. Мир выбран по этому
основанию, а не по величине прироста метрики: прирост здесь не улучшался и не
подбирался.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import lightgbm as lgb
import polars as pl

from dsx.evals.registry import BY_ID
from dsx.label import LABEL
from dsx.measure import BaselineRule, RuleKind, measure_against_baseline
from dsx.project import load
from dsx.runner import RESERVE, run
from dsx.samples import Purpose

HERE = Path(__file__).resolve().parent
WORLD = "feature-falsely-declared-available"

FEATURES = ["lead_days", "size", "region"]
"""Что подаётся модели.

`overrun_days` НЕ подаётся, хотя форма объявляет её доступной в момент решения
и ядро её объявление принимает. Это решение ЧЕЛОВЕКА, принятое после того, как
ядро возразило: сигнал N6 — повод спросить, когда величина становится
известной, а не приговор признаку. Здесь ответ известен по построению мира,
и он таков, что колонка вычислена из исхода.

Инструмент такого решения не принимает и принять не может.
"""

SETTINGS = {
    "objective": "binary",
    "num_leaves": 15,
    "learning_rate": 0.05,
    "n_estimators": 150,
    "min_child_samples": 50,
    "verbose": -1,
    "deterministic": True,
    "seed": 20260910,
}
"""Настройки фиксированы и не подбирались: подбор есть выбор, а выбор
расходует выборку. Seed задан ради воспроизводимости, а не ради числа."""

ОГРАНИЧЕНИЯ = """
ЧЕГО ЭТОТ ОТЧЁТ НЕ ГОВОРИТ

1. Инструмент НЕ подтверждает, что с данными всё хорошо. Он возражает по тому,
   что умеет проверить, и молчит обо всём остальном. Молчание проверки не есть
   свидетельство исправности.
2. Ядро принимает `binary`/`delayed` и `competing`/`delayed`. Это не поддержка
   всех задач машинного обучения: regression, ranking, survival, uplift и любой
   `immediate` оно отвергает явно. Пример показывает первую комбинацию.
3. Сигнал — не доказанный дефект. N6 сравнивает силу связи признака с исходом и
   по устройству НЕ отличает утечку от сильного добросовестного признака; их
   различает только знание о том, когда величина становится известной.
4. Пре-регистрация, слепой контроль и исследовательские учёты в этот сценарий
   не входят: это протокол проверки самого инструмента, а не работа
   пользователя.
5. Измерение на синтетическом мире показывает ПРОЦЕДУРУ, а не качество. Мир
   построен так, что ответ в нём известен заранее; о настоящей задаче эти
   числа не говорят ничего.
"""


def prepare(frame: pl.DataFrame, categories: list) -> object:
    import pandas as pd

    return pd.DataFrame(
        {
            "lead_days": frame["lead_days"].cast(pl.Float64).to_numpy(),
            "size": frame["size"].cast(pl.Float64).to_numpy(),
            "region": pd.Categorical(frame["region"].to_list(), categories=categories),
        }
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=HERE / "report",
        help="каталог результатов (по умолчанию examples/synthetic-delay/report)",
    )
    out = parser.parse_args(argv).out
    out.mkdir(parents=True, exist_ok=True)

    # 1. ФОРМА. Объявления читаются ядром, а не собираются здесь руками.
    form = load(HERE / "project.yaml")

    # 2. ДАННЫЕ. Синтетический мир, порождаемый детерминированно.
    frame = BY_ID[WORLD].build().main

    # 3. ПРОВЕРКИ И ОТЧЁТ. Сплит, учёт выборок и report.md делает ядро.
    result = run(form, frame, out)
    print(result.summary())

    # 4. МОДЕЛЬ. Ядро предсказаний не производит: оно принимает готовые оценки.
    result.uses(FEATURES)
    last = result.split.parts[-1]
    train = last.train.filter(pl.col(LABEL).is_not_null())
    result.samples.fit(last.name, f"обучение на обучающей части окна {last.name}")

    доля = float(train[LABEL].mean())
    categories = sorted(train["region"].unique().drop_nulls().to_list())
    model = lgb.LGBMClassifier(**SETTINGS)
    model.fit(prepare(train, categories), train[LABEL].to_numpy())
    print(f"  обучение: {train.height:,} строк, признаков {len(FEATURES)}")

    # 5. ИЗМЕРЕНИЕ. Базовое правило — постоянная оценка, равная доле в
    # ОБУЧАЮЩЕЙ части. Взять её из резерва значило бы подсмотреть в выборку,
    # которой измеряют.
    reserve = result.samples.checkout(RESERVE, Purpose.AUDIT, "подготовка оценок")
    scores = pl.Series("score", model.predict_proba(prepare(reserve, categories))[:, 1])
    rule = BaselineRule(kind=RuleKind.CONSTANT, constant=round(доля, 4))
    verdict = measure_against_baseline(result.samples, RESERVE, scores, rule)
    print("\n=== ИЗМЕРЕНИЕ НА РЕЗЕРВЕ ===")
    print(verdict.report_section())

    # МАШИНОЧИТАЕМЫЙ ИТОГ ПРОВЕРОК — рядом с человеческим, не вместо него.
    #
    # `report.md` печатает `signal.detail` — фразу для человека, а не
    # `finding.value`. Сверять по нему множество находок нельзя: ровно на этом
    # уже обжёгся протокол, где доказательством служила копия отчёта, и сверка
    # не работала ни разу. Здесь берутся идентификаторы, порождённые ядром.
    (out / "findings.txt").write_text(
        "\n".join(sorted(f.value for f in result.checks.findings)) + "\n", encoding="utf-8"
    )

    result.study.measurement = verdict
    result.study.conclude(
        verdict.statement()
        + " Мир синтетический: разница с базовым правилом описывает процедуру, а не "
        "качество на настоящей задаче. Превосходства над базовым правилом пример не "
        "обязан показывать — требовать его значило бы подбирать мир под красивое число."
        + ОГРАНИЧЕНИЯ
    )
    (out / "measured.md").write_text(result.study.render(), encoding="utf-8")

    print(ОГРАНИЧЕНИЯ)
    print(f"отчёт проверок: {out / 'report.md'}")
    print(f"находки списком: {out / 'findings.txt'}")
    print(f"отчёт измерения: {out / 'measured.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
