"""Шаг 05 — полный прогон протокола B текущим ядром.

Проверяются три предсказания пре-регистрации, введённые поправкой A-1:
P-9 (момент узнавания исхода), P-10 (журнал выборок и перекрытие по времени),
P-11 (недостаточность ролей). Каждое проверяется наблюдением, а не мнением.
"""

from __future__ import annotations

import datetime as dt
import importlib.util
from pathlib import Path

import polars as pl

from dsx.assumptions import AssumptionRegistry, Basis
from dsx.checks import ALL_CHECKS, Context, run_checks
from dsx.outcome import ComparisonMode, OutcomeDefinition
from dsx.policy import OverrideLedger
from dsx.report import Study
from dsx.roles import Role
from dsx.samples import SampleLedger
from dsx.split import Window, entity_overlap, positive_rates, split_by_windows
from dsx.task import ObjectLifetime, OutcomeTiming, TargetKind, TaskSpec

PROJECT = Path(__file__).resolve().parent.parent


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, PROJECT / "steps" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


protocol_a = _load("protocol_a", "02_protocol_a.py")
protocol_b = _load("protocol_b", "04_protocol_b.py")

HORIZON_DAYS = protocol_b.HORIZON_DAYS
LOOKBACK_DAYS = protocol_b.LOOKBACK_DAYS


def main() -> int:
    world = protocol_b.build_case()
    frame = world.main

    definition = OutcomeDefinition(
        event_column="failed_at",
        deadline_column="horizon_on",
        comparison=ComparisonMode.DIRECT,
        # Причины те же: они свойство предметной области, а не постановки.
        missing_causes=protocol_a.CAUSES,
        estimand=f"отказ любого компонента в течение {HORIZON_DAYS} дней после дня решения",
    )

    task = TaskSpec(
        target_kind=TargetKind.BINARY,
        outcome_timing=OutcomeTiming.DELAYED,
        has_process=False,
        is_stream=True,  # решения назначает календарь, это поток
        object_lifetime=ObjectLifetime.RECURRING,
    )

    lo = frame["decided_at"].min()
    hi = frame["decided_at"].max()
    snapshot = hi
    windows = [
        Window(f"w{i}", lo + dt.timedelta(days=150 + i * 60), lo + dt.timedelta(days=210 + i * 60))
        for i in range(3)
    ]

    print(f"решений: {frame.height:,}, машин: {frame['machine_key'].n_unique()}")
    print(f"период: {lo:%Y-%m-%d} .. {hi:%Y-%m-%d}")
    print(f"окно признаков: {LOOKBACK_DAYS} дн, горизонт исхода: {HORIZON_DAYS} дн")
    print()

    split = split_by_windows(world, definition, windows, snapshot)
    print(f"выпало между выборками: {split.dropped_not_yet_known:,}")
    for part in split.parts:
        print(f"  {part.name}: обучение {part.train.height:6,}  оценка {part.evaluate.height:5,}")
    print("незрелых:", split.immature or "нет")
    print("доли положительных:", {k: f"{v:.1%}" for k, v in positive_rates(split.parts).items()})
    print("машин по обе стороны:", entity_overlap(split.parts, world) or "нет", "(ожидаемо)")
    print("повторов решения:", entity_overlap(split.parts, world, role=Role.ENTITY_ID) or "нет")
    print()

    report = run_checks(list(ALL_CHECKS), Context(world, definition, task, split))
    print("=== СИГНАЛЫ ===")
    for signal in report.signals:
        print(f"  {signal}")
    if not report.signals:
        print("  нет")
    print()
    print("=== ПРОПУЩЕННЫЕ ПРОВЕРКИ ===")
    for item in report.skipped:
        print(f"  {item}")
    print()
    print(f"блокирующих: {len(report.blocking)}, пропущено проверок: {len(report.skipped)}")
    print()

    print("=== P-10: журнал выборок против перекрытия по времени ===")
    samples = SampleLedger(OverrideLedger())
    samples.register(*[p.name for p in split.parts])
    samples.select("w0", "выбор окна признаков и горизонта")
    samples.measure("w2")
    print("журнал принял: выбор на w0, измерение на w2 — разные имена, возражений нет")

    by_name = {p.name: p for p in split.parts}
    shared_rows = set(by_name["w0"].train["decision_key"].to_list()) & set(
        by_name["w2"].train["decision_key"].to_list()
    )
    print(
        f"а на деле обучающие выборки w0 и w2 делят {len(shared_rows):,} решений "
        f"из {by_name['w0'].train.height:,} — это {len(shared_rows) / by_name['w0'].train.height:.0%}"
    )

    w0_eval_days = set(by_name["w0"].evaluate["decided_at"].dt.date().to_list())
    w2_train_days = set(by_name["w2"].train["decided_at"].dt.date().to_list())
    print(
        f"дней оценки w0, попавших в обучение w2: {len(w0_eval_days & w2_train_days):,} "
        f"из {len(w0_eval_days):,}"
    )
    print()

    print("=== F-5: пересечение окон признаков ===")
    boundary = by_name["w0"].train["decided_at"].max()
    reach = boundary + dt.timedelta(days=LOOKBACK_DAYS)
    crossing = by_name["w0"].evaluate.filter(pl.col("decided_at") < reach).height
    print(
        f"последнее обучающее решение: {boundary:%Y-%m-%d}, окно признаков достаёт "
        f"до {reach:%Y-%m-%d}"
    )
    print(
        f"оценочных решений, чьё окно признаков перекрывается с обучающими: "
        f"{crossing:,} из {by_name['w0'].evaluate.height:,}"
    )
    print("ядро об этом не сообщило: окно признаков нигде не объявлено")
    print()

    print("=== P-9: момент узнавания исхода ===")
    labelled = frame.filter(pl.col("failed_at").is_not_null())
    positives = labelled.filter(pl.col("failed_at") <= pl.col("horizon_on"))
    gap = (pl.col("horizon_on") - pl.col("failed_at")).dt.total_days()
    lag = positives.select(gap.alias("d"))["d"]
    print(
        f"положительных решений: {positives.height:,}; исход у них известен в среднем "
        f"за {lag.mean():.1f} дн до конца горизонта (медиана {lag.median():.0f})"
    )
    print(
        "сплит откладывает их до конца горизонта: отрицательные становятся известны "
        "только там, положительные — раньше, но ядро хранит одно время узнавания"
    )
    print()

    print("=== P-11: достаточность ролей ===")
    raw_failures = pl.read_csv(protocol_b.RAW / "PdM_failures.csv")
    print(
        f"типов отказа в данных: {raw_failures['failure'].n_unique()} "
        f"({', '.join(sorted(raw_failures['failure'].unique().to_list()))})"
    )
    print("роли для компонента отдельно от актива нет: отказы склеены в один исход")
    print(
        "роли для времени измерения отдельно от времени события нет: телеметрия "
        "свёрнута в суточные агрегаты, и час измерения потерян"
    )
    print()

    assumptions = AssumptionRegistry()
    assumptions.record(
        f"признаки строятся по окну {LOOKBACK_DAYS} дней назад и замыкаются днём решения",
        basis=Basis.DOMAIN_KNOWLEDGE,
        author="автор",
        consequence="окно захватывает измерения, недоступные к моменту решения, "
        "либо теряет сигнал последних суток",
    )
    assumptions.record(
        "ежедневный скоринг соответствует реальному темпу принятия решений",
        basis=Basis.DOMAIN_KNOWLEDGE,
        author="автор",
        consequence="моменты решения назначены календарём там, где вмешательство "
        "невозможно, и оценка описывает несуществующий процесс",
    )
    assumptions.record(
        "отказы разных компонентов считаются одним исходом",
        basis=Basis.DOMAIN_KNOWLEDGE,
        author="автор",
        consequence="конкурирующие отказы склеены: модель предсказывает отказ вообще, "
        "а вмешательство требуется адресное",
    )

    study = Study(title="Azure PdM, протокол B: скользящая постановка")
    study.checks = report
    study.samples = samples
    study.assumptions = assumptions
    study.conclude(
        "протокол B проходится по форме, но три предсказанных излома подтвердились "
        "наблюдением: P-10 и F-5 не видны ядру, P-11 обходится склейкой"
    )

    (PROJECT / "report").mkdir(exist_ok=True)
    (PROJECT / "report" / "protocol_b.md").write_text(study.render(), encoding="utf-8")
    print(f"отчёт: {PROJECT / 'report' / 'protocol_b.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
