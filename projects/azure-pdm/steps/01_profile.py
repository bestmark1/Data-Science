"""Шаг 01 — профилирование пяти таблиц.

Первый контакт с содержимым. Манифест зафиксирован ранее.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

PROJECT = Path(__file__).resolve().parent.parent
RAW = PROJECT / "data" / "raw"
ARTIFACTS = PROJECT / "artifacts"


def main() -> int:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    report = ["# Шаг 01 — профиль таблиц", ""]

    for path in sorted(RAW.glob("*.csv")):
        frame = pl.read_csv(path, infer_schema_length=20_000)
        report += [f"## {path.stem}", "", f"строк: {frame.height:,}  колонок: {frame.width}", ""]

        duplicates = frame.height - frame.unique().height
        if duplicates:
            report.append(f"**полных дубликатов строк: {duplicates:,}**")
            report.append("")

        report += ["| колонка | тип | пропусков | уникальных | пример |", "|---|---|---|---|---|"]
        for name in frame.columns:
            column = frame[name]
            nulls = column.null_count()
            share = f"{nulls / frame.height:.1%}" if nulls else "—"
            sample = str(column.drop_nulls().head(1).to_list()[:1])[1:-1][:28]
            report.append(
                f"| {name} | {column.dtype} | {share} | {column.n_unique():,} | `{sample}` |"
            )
        report.append("")

    target = ARTIFACTS / "01_profile.md"
    target.write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
