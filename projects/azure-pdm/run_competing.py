"""Цена склейки видов отказа на втором кейсе (остаток F-8).

Протокол A объявлял исход как «отказ любого компонента». В данных компонентов
четыре, и ремонт адресный: склейка означала, что модель предсказывает не то,
чем управляют.

Скрипт считает исход отдельно по каждому компоненту и показывает, сколько
строк при этом оказываются не отрицательными, а НЕИЗВЕСТНЫМИ: отказ одного
компонента останавливает машину, и что случилось бы со вторым, не наблюдалось.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import polars as pl

from dsx.competing import compute_by_kind
from dsx.evals.world import World
from dsx.join import Cardinality, guarded_join
from dsx.label import LABEL, compute
from dsx.project import load
from dsx.roles import ColumnSpec, Role

PROJECT = Path(__file__).resolve().parent
RAW = PROJECT / "data" / "raw"

_spec = importlib.util.spec_from_file_location("run_a", PROJECT / "run_a.py")
_run_a = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_run_a)


SIMULTANEOUS = "одновременный отказ нескольких компонентов"


def build_with_kinds() -> tuple[pl.DataFrame, int]:
    """Та же таблица визитов, но с видом ближайшего отказа.

    В 42 моментах из 719 отказывают два компонента разом. Конкурирующие риски
    предполагают, что побеждает ровно один, и здесь это допущение нарушено:
    кто отказал бы первым, не наблюдалось.

    Такие визиты помечаются как НЕНАБЛЮДАЕМЫЕ для всех видов, а не относятся
    к одному из них произвольно. Соединение оставляет одну строку на визит:
    иначе визит с двумя отказами удвоился бы и вошёл в оценку дважды.
    """
    failures = (
        pl.read_csv(RAW / "PdM_failures.csv")
        .with_columns(
            pl.col("datetime").str.to_datetime().alias("failed_at"),
            pl.col("machineID").cast(pl.Utf8).alias("machine_key"),
        )
        .group_by("machine_key", "failed_at")
        .agg(
            pl.col("failure").first().alias("failure"),
            pl.len().alias("components_failed"),
        )
    )
    joined = guarded_join(
        _run_a.build(),
        failures,
        on=["machine_key", "failed_at"],
        expect=Cardinality.MANY_TO_ONE,
        how="left",
    )
    ambiguous = int(joined.filter(pl.col("components_failed") > 1).height)
    frame = joined.with_columns(
        pl.when(pl.col("failed_at") > pl.col("horizon_on"))
        .then(None)
        .when(pl.col("components_failed") > 1)
        .then(pl.lit(SIMULTANEOUS))
        .otherwise(pl.col("failure"))
        .alias("failure_kind")
    )
    return frame, ambiguous


def main() -> int:
    form = load(PROJECT / "protocol-a.yaml")
    frame, ambiguous = build_with_kinds()
    print(f"визитов, чей ближайший отказ затронул несколько компонентов: {ambiguous:,}")
    print("допущение конкурирующих рисков «побеждает ровно один» на них не выполнено")
    print()
    schema = form.schema_spec()
    world = World(
        frames={"main": frame},
        schema=type(schema)(
            columns=[*schema.columns, ColumnSpec(name="failure_kind", role=Role.EVENT_KIND)]
        ),
    )
    definition = form.outcome.to_definition()

    collapsed = compute(world, definition)
    observable = collapsed.filter(pl.col(LABEL).is_not_null())
    print("СКЛЕЙКА (как было объявлено в протоколе A)")
    print(f"  размечено {observable.height:,}, положительных {int(observable[LABEL].sum()):,}")
    print()

    print("ПО КОМПОНЕНТАМ")
    outcomes = compute_by_kind(world, definition)
    for outcome in outcomes:
        print(
            f"  {outcome.kind}: положительных {outcome.positives:,}, "
            f"наблюдаемых {outcome.observable:,}, "
            f"оборвано конкурентами {outcome.censored_by_others:,}"
        )
    print()

    censored = [o.censored_by_others for o in outcomes]
    print("ЧТО ТЕРЯЕТСЯ")
    print(
        f"  склейка: один исход на {int(observable[LABEL].sum()):,} положительных, "
        "но предсказывает «откажет что-нибудь». Ремонт адресный, и управлять этим нельзя"
    )
    print(
        f"  наивная поразрядная разметка: объявила бы исправностью от {min(censored):,} "
        f"до {max(censored):,} строк на каждый вид — те, где раньше отказал конкурент. "
        "Машину остановили и починили, и что случилось бы с этим компонентом, "
        "не наблюдалось"
    )
    print(
        f"  честный счёт: у каждого вида от {min(o.positives for o in outcomes):,} до "
        f"{max(o.positives for o in outcomes):,} положительных при "
        f"{min(o.observable for o in outcomes):,}–{max(o.observable for o in outcomes):,} "
        "наблюдаемых строках"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
