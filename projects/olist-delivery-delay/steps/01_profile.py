"""Шаг 01 — профилирование девяти таблиц Olist.

Ручной проход: код намеренно не обобщается. Всё, что раздражает при написании,
идёт в журнал трений, а не в модуль.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

PROJECT = Path(__file__).resolve().parent.parent
PARQUET = PROJECT / "data" / "parquet"
ARTIFACTS = PROJECT / "artifacts"


def profile_table(path: Path) -> list[str]:
    frame = pl.read_parquet(path)
    lines = [f"### {path.stem}", "", f"строк: {frame.height:,}  колонок: {frame.width}", ""]

    duplicates = frame.height - frame.unique().height
    if duplicates:
        lines.append(f"**полных дубликатов строк: {duplicates:,}**")
        lines.append("")

    lines.append("| колонка | тип | пропусков | уникальных |")
    lines.append("|---|---|---|---|")
    for name in frame.columns:
        column = frame[name]
        nulls = column.null_count()
        share = f"{nulls / frame.height:.1%}" if nulls else "—"
        lines.append(f"| {name} | {column.dtype} | {share} | {column.n_unique():,} |")
    lines.append("")
    return lines


def main() -> int:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    tables = sorted(PARQUET.glob("*.parquet"))

    report = ["# Шаг 01 — профиль таблиц", "", f"Таблиц: {len(tables)}", ""]
    for path in tables:
        report += profile_table(path)

    target = ARTIFACTS / "01_profile.md"
    target.write_text("\n".join(report), encoding="utf-8")
    print(f"Профиль записан: {target}")
    print("\n".join(report[:4]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
