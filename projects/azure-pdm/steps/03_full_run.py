"""Шаг 03 — полный прогон протокола A текущим ядром.

Цель не получить модель, а увидеть, где ядро ломается.

После правок F-1, F-2 и F-3 прогон повторён: исход строится на перечне причин
без выдуманного статуса, а машина объявлена долгоживущим объектом.
"""

from __future__ import annotations

import datetime as dt
import importlib.util
from pathlib import Path

from dsx.assumptions import AssumptionRegistry, Basis
from dsx.checks import ALL_CHECKS, Context, run_checks
from dsx.outcome import ComparisonMode, OutcomeDefinition, PositiveClass
from dsx.policy import OverrideLedger
from dsx.report import Study
from dsx.roles import Role
from dsx.samples import SampleLedger
from dsx.split import (
    Window,
    entity_overlap,
    extent_of,
    positive_rates,
    reserved_extent,
    split_by_windows,
)
from dsx.task import ObjectLifetime, OutcomeTiming, TargetKind, TaskSpec

PROJECT = Path(__file__).resolve().parent.parent
HORIZON_DAYS = 30

spec = importlib.util.spec_from_file_location("protocol_a", PROJECT / "steps" / "02_protocol_a.py")
protocol_a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(protocol_a)


def main() -> int:
    world = protocol_a.build_case()
    definition = OutcomeDefinition(
        event_column="failed_at",
        deadline_column="horizon_on",
        comparison=ComparisonMode.DIRECT,
        # Положителен отказ В ПРЕДЕЛАХ горизонта. До правки F-10 ядро
        # считало обратное, и объявленный estimand это не ловил.
        positive_class=PositiveClass.EVENT_WITHIN_DEADLINE,
        missing_causes=protocol_a.CAUSES,
        estimand=f"отказ любого компонента в течение {HORIZON_DAYS} дней после визита",
    )

    task = TaskSpec(
        target_kind=TargetKind.BINARY,
        outcome_timing=OutcomeTiming.DELAYED,
        has_process=False,  # статуса процесса нет
        is_stream=False,  # решения событийные, не поток
        # Машина обслуживается многократно: единица решения — визит,
        # объект — машина. Дробление намеренное (F-2, F-3).
        object_lifetime=ObjectLifetime.RECURRING,
    )

    lo = world.main["decided_at"].min()
    hi = world.main["decided_at"].max()
    snapshot = hi
    span = (hi - lo).days
    windows = [
        Window(f"w{i}", lo + dt.timedelta(days=180 + i * 60), lo + dt.timedelta(days=240 + i * 60))
        for i in range(3)
    ]
    # Резерв отрезается до построения окон и не входит ни в одно из них (F-9).
    reserve_from = lo + dt.timedelta(days=400)

    print(f"период визитов: {lo:%Y-%m-%d} .. {hi:%Y-%m-%d} ({span} дней)")
    split = split_by_windows(world, definition, windows, snapshot, reserve_from)
    print(f"выпало между выборками: {split.dropped_not_yet_known:,}")
    for part in split.parts:
        print(f"  {part.name}: обучение {part.train.height:5,}  оценка {part.evaluate.height:4,}")
    print("незрелых:", split.immature or "нет")
    print("доли положительных:", {k: f"{v:.1%}" for k, v in positive_rates(split.parts).items()})
    print("машин по обе стороны:", entity_overlap(split.parts, world) or "нет", "(ожидаемо)")
    print(
        "повторов решения:",
        entity_overlap(split.parts, world, role=Role.ENTITY_ID) or "нет",
    )
    print()

    report = run_checks(list(ALL_CHECKS), Context(world, definition, task, split))
    print("=== СИГНАЛЫ ===")
    for signal in report.signals:
        print(f"  {signal}")
    print()
    print("=== ПРОПУЩЕННЫЕ ПРОВЕРКИ ===")
    for item in report.skipped:
        print(f"  {item}")
    print()
    print(f"блокирующих: {len(report.blocking)}, пропущено проверок: {len(report.skipped)}")

    samples = SampleLedger(OverrideLedger())
    for part in split.parts:
        samples.register(part.name, extent_of(part, world))
    samples.register("резерв", reserved_extent(split, world))
    samples.select("w0", "выбор горизонта и окна")
    samples.measure("резерв")
    print(f"измерение проведено на резерве: {samples.extent('резерв')}")

    assumptions = AssumptionRegistry()
    assumptions.record(
        "визит на обслуживание является моментом, когда возможно вмешательство",
        basis=Basis.DOMAIN_KNOWLEDGE,
        author="автор",
        consequence="момент решения выбран там, где действие невозможно, и прогноз бесполезен",
    )
    assumptions.record(
        "отсутствие записи об отказе означает, что отказа не было",
        basis=Basis.DOMAIN_KNOWLEDGE,
        author="автор",
        consequence="цензура, вывод из эксплуатации и незарегистрированные отказы "
        "склеены с настоящим отсутствием отказа",
    )
    assumptions.record(
        "горизонт 30 дней соответствует времени реакции обслуживания",
        basis=Basis.DOMAIN_KNOWLEDGE,
        author="автор",
        consequence="прогноз не успевает изменить исход либо теряет сигнал",
    )

    study = Study(title="Azure PdM, протокол A: аудит")
    study.checks = report
    study.samples = samples
    study.assumptions = assumptions
    study.conclude(
        "протокол A проходится; F-1 и F-3 исправлены, но пересечение окон "
        "признаков не проверяется (F-5)"
    )

    (PROJECT / "report").mkdir(exist_ok=True)
    (PROJECT / "report" / "protocol_a.md").write_text(study.render(), encoding="utf-8")
    print(f"\nотчёт: {PROJECT / 'report' / 'protocol_a.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
