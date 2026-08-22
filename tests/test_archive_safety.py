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


def test_oversized_archive_is_refused_by_default(tmp_path: Path, monkeypatch) -> None:
    """Архив, распакованный в тысячи раз больше себя, не должен пройти молча."""
    archive = tmp_path / "big.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr("payload.csv", "a" * 2_000_000)

    with pytest.raises(UnsafeArchive, match="превышает лимит"):
        extract_archive(archive, tmp_path / "out", max_uncompressed_bytes=1_000_000)


def test_limit_can_be_raised_by_the_caller(tmp_path: Path) -> None:
    """Поднять лимит можно, но только осознанно и на стороне вызова."""
    archive = tmp_path / "big.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr("payload.csv", "a" * 2_000_000)

    extracted = extract_archive(archive, tmp_path / "out", max_uncompressed_bytes=4_000_000)

    assert [p.name for p in extracted] == ["payload.csv"]


def test_refusal_names_both_sizes(tmp_path: Path) -> None:
    """Сообщение обязано сказать, сколько получилось и сколько разрешено."""
    archive = tmp_path / "big.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr("payload.csv", "a" * 2_000_000)

    with pytest.raises(UnsafeArchive) as excinfo:
        extract_archive(archive, tmp_path / "out", max_uncompressed_bytes=1_000_000)

    assert "ГиБ" in str(excinfo.value) and "явно" in str(excinfo.value)
