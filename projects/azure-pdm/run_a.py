"""Протокол A через форму проекта.

Скрипт делает единственное, что нельзя объявить: собирает таблицу решений из
пяти таблиц разного уровня. Всё остальное объявлено в protocol-a.yaml.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from dsx.project import load
from dsx.runner import objects_across_splits, run

PROJECT = Path(__file__).resolve().parent
RAW = PROJECT / "data" / "raw"
HORIZON_DAYS = 30


def build() -> pl.DataFrame:
    """Одна строка на визит обслуживания."""
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
    next_failure = (
        visits.join(failures.select("machineID", "failed_at"), on="machineID", how="left")
        .filter(pl.col("failed_at") > pl.col("decided_at"))
        .group_by("visit_id")
        .agg(pl.col("failed_at").min())
    )
    return (
        visits.join(next_failure, on="visit_id", how="left")
        .join(machines, on="machineID", how="left")
        .with_columns(
            (pl.col("decided_at") + pl.duration(days=HORIZON_DAYS)).alias("horizon_on"),
            pl.col("visit_id").cast(pl.Utf8).alias("visit_key"),
            pl.col("machineID").cast(pl.Utf8).alias("machine_key"),
        )
    )


def main() -> int:
    form = load(PROJECT / "protocol-a.yaml")
    result = run(form, build(), PROJECT / "report" / "a")

    print(result.summary())
    print("машин по обе стороны:", objects_across_splits(result) or "нет", "(ожидаемо)")

    result.samples.select("w0", "выбор горизонта и окна")
    result.samples.measure("резерв")
    result.study.conclude("протокол A проходится; измерение на зарезервированной выборке")
    print("измерение на резерве:", result.samples.extent("резерв"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
