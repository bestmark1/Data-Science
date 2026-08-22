"""Шаг 02 — протокол A: вырожденный событийный подкейс.

Момент решения выбран там, где он почти естественен: визит на обслуживание.
Вопрос: случится ли отказ любого компонента в течение горизонта после визита.

Один класс исхода, один горизонт, событийный якорь. Если ядро не справляется
даже с этим, проблема глубже скользящих окон.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from dsx.evals.world import World
from dsx.label import compute
from dsx.outcome import (
    ComparisonMode,
    MissingEventCause,
    MissingEventMeaning,
    OutcomeDefinition,
    PositiveClass,
)
from dsx.roles import Availability, ColumnSpec, FeatureWindow, Role, Schema, TemporalKind

PROJECT = Path(__file__).resolve().parent.parent
RAW = PROJECT / "data" / "raw"
HORIZON_DAYS = 30
SOURCE = "владелец данных (гипотетический)"

# Все признаки протокола A мгновенные: они читаются из карточки машины или из
# самой записи о визите, а не считаются по интервалу назад.
INSTANT = FeatureWindow(lookback_days=0, source_of_claim="код сборки кейса, шаг 02")

# Статуса в данных нет. Причины отсутствия отказа объявляются перечнем, и все
# они неразличимы: ни одна не выводится из имеющихся таблиц. Каждая обязана
# назвать принимаемое допущение.
CAUSES = [
    MissingEventCause(
        name="отказа действительно не было",
        meaning=MissingEventMeaning.NOT_OCCURRED,
        assumption="отсутствие записи об отказе означает исправную работу, "
        "а не пробел в регистрации",
    ),
    MissingEventCause(
        name="отказ произошёл, но не зарегистрирован",
        meaning=MissingEventMeaning.NOT_OCCURRED,
        assumption="доля незарегистрированных отказов пренебрежимо мала",
    ),
    MissingEventCause(
        name="профилактический ремонт предотвратил отказ",
        meaning=MissingEventMeaning.NOT_OCCURRED,
        assumption="предотвращённый отказ считается отсутствием отказа, "
        "хотя это вмешательство, а не наблюдение",
    ),
    MissingEventCause(
        name="объект выведен из эксплуатации",
        meaning=MissingEventMeaning.NOT_OCCURRED,
        assumption="вывода из эксплуатации в наблюдаемом периоде не было",
    ),
]


def build_case() -> World:
    """Собрать таблицу решений: одна строка на визит обслуживания."""
    maint = pl.read_csv(RAW / "PdM_maint.csv").with_columns(pl.col("datetime").str.to_datetime())
    failures = pl.read_csv(RAW / "PdM_failures.csv").with_columns(
        pl.col("datetime").str.to_datetime().alias("failed_at")
    )
    machines = pl.read_csv(RAW / "PdM_machines.csv")

    visits = (
        maint.group_by("machineID", "datetime")
        .agg(pl.col("comp").n_unique().alias("components_serviced"))
        .rename({"datetime": "decided_at"})
        .sort("machineID", "decided_at")
        .with_row_index("visit_id")
    )

    # Ближайший отказ после визита.
    next_failure = (
        visits.join(failures.select("machineID", "failed_at"), on="machineID", how="left")
        .filter(pl.col("failed_at") > pl.col("decided_at"))
        .group_by("visit_id")
        .agg(pl.col("failed_at").min())
    )

    table = (
        visits.join(next_failure, on="visit_id", how="left")
        .join(machines, on="machineID", how="left")
        .with_columns(
            (pl.col("decided_at") + pl.duration(days=HORIZON_DAYS)).alias("horizon_on"),
            pl.col("visit_id").cast(pl.Utf8).alias("visit_key"),
            pl.col("machineID").cast(pl.Utf8).alias("machine_key"),
        )
    )

    schema = Schema(
        columns=[
            ColumnSpec(name="visit_key", role=Role.ENTITY_ID),
            ColumnSpec(name="machine_key", role=Role.NATURAL_KEY),
            ColumnSpec(name="decided_at", role=Role.DECISION_TIME, temporal=TemporalKind.INSTANT),
            ColumnSpec(
                name="failed_at", role=Role.OUTCOME_COMPONENT, temporal=TemporalKind.INSTANT
            ),
            ColumnSpec(name="horizon_on", role=Role.DEADLINE, temporal=TemporalKind.INSTANT),
            ColumnSpec(
                name="components_serviced",
                role=Role.FEATURE,
                availability=Availability.AT_DECISION,
                source_of_claim=SOURCE,
                window=INSTANT,
            ),
            ColumnSpec(
                name="age",
                role=Role.FEATURE,
                availability=Availability.AT_DECISION,
                source_of_claim=SOURCE,
                window=INSTANT,
            ),
            ColumnSpec(
                name="model",
                role=Role.FEATURE,
                availability=Availability.AT_DECISION,
                source_of_claim=SOURCE,
                window=INSTANT,
            ),
        ]
    )
    return World(frames={"main": table}, schema=schema)


def main() -> int:
    world = build_case()
    print(f"визитов: {world.main.height:,}, машин: {world.main['machine_key'].n_unique()}")
    print(
        f"с отказом в горизонте {HORIZON_DAYS} дн: "
        f"{world.main.filter(pl.col('failed_at') <= pl.col('horizon_on')).height:,}"
    )
    print()

    definition = OutcomeDefinition(
        event_column="failed_at",
        deadline_column="horizon_on",
        comparison=ComparisonMode.DIRECT,
        # Положителен отказ В ПРЕДЕЛАХ горизонта. До правки F-10 ядро
        # считало обратное, и объявленный estimand это не ловил.
        positive_class=PositiveClass.EVENT_WITHIN_DEADLINE,
        missing_causes=CAUSES,
        estimand=f"отказ любого компонента в течение {HORIZON_DAYS} дней после визита",
    )
    frame = compute(world, definition)
    observed = frame.filter(pl.col("__outcome").is_not_null())
    print("контракт собран БЕЗ выдуманного статуса")
    print(f"  строк размечено: {observed.height:,}")
    print(f"  доля положительных: {observed['__outcome'].mean():.2%}")
    print()
    print(
        f"причин объявлено: {len(definition.missing_causes)}, "
        f"из них неразличимых: {len(definition.indistinguishable)}"
    )
    print(f"смыслы неразличимых смешаны: {definition.conflated}")
    print()
    print("допущения, принятые из-за неразличимости:")
    for text in definition.assumptions():
        print(f"  - {text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
