"""Конвертация в Parquet обязана быть идемпотентной, иначе R10 недостижим."""

from __future__ import annotations

from pathlib import Path

import pytest

from dsx.io.acquire import csv_to_parquet
from dsx.io.manifest import sha256_of


@pytest.fixture
def csv_file(tmp_path: Path) -> Path:
    path = tmp_path / "orders.csv"
    path.write_text(
        "order_id,purchased_at,price\n1,2017-03-01 10:00:00,10.5\n2,2017-03-02 11:30:00,22.0\n",
        encoding="utf-8",
    )
    return path


def test_conversion_produces_readable_parquet(csv_file: Path, tmp_path: Path) -> None:
    import polars as pl

    target = csv_to_parquet(csv_file, tmp_path / "orders.parquet")
    frame = pl.read_parquet(target)

    assert frame.height == 2
    assert frame.columns == ["order_id", "purchased_at", "price"]


def test_conversion_is_idempotent(csv_file: Path, tmp_path: Path) -> None:
    first = csv_to_parquet(csv_file, tmp_path / "a.parquet")
    first_hash = sha256_of(first)

    second = csv_to_parquet(csv_file, tmp_path / "b.parquet")

    assert sha256_of(second) == first_hash
