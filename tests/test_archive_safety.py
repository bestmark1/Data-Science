"""Распаковка архива не должна писать вне каталога назначения."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from dsx.io.acquire import UnsafeArchive, extract_archive


def _archive(path: Path, entries: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w") as bundle:
        for name, payload in entries.items():
            bundle.writestr(name, payload)
    return path


def test_normal_archive_extracts_files_only(tmp_path: Path) -> None:
    archive = _archive(tmp_path / "a.zip", {"orders.csv": b"id\n1\n", "sub/items.csv": b"x\n"})

    extracted = extract_archive(archive, tmp_path / "out")

    # Порядок — по полному пути: файл в корне идёт раньше вложенного каталога.
    assert [p.name for p in extracted] == ["orders.csv", "items.csv"]
    assert all(p.is_file() for p in extracted)


def test_path_traversal_is_refused(tmp_path: Path) -> None:
    archive = _archive(tmp_path / "evil.zip", {"../escaped.csv": b"pwned\n"})

    with pytest.raises(UnsafeArchive, match="выходит за каталог"):
        extract_archive(archive, tmp_path / "out")

    assert not (tmp_path / "escaped.csv").exists()


def test_absolute_path_is_refused(tmp_path: Path) -> None:
    archive = _archive(tmp_path / "evil.zip", {"/tmp/escaped.csv": b"pwned\n"})

    with pytest.raises(UnsafeArchive, match="выходит за каталог"):
        extract_archive(archive, tmp_path / "out")
