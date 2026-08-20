"""Манифест источника — носитель воспроизводимости (R4, R9, R10)."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from dsx.io.acquire import KaggleCredentialsMissing, read_credentials
from dsx.io.manifest import (
    ChangedFiles,
    ExtraFiles,
    FileEntry,
    ManifestError,
    MissingFiles,
    SourceManifest,
    build_manifest,
    matches,
    read_manifest,
    verify,
    write_manifest,
)

SOURCE_KW = {
    "source": "kaggle:olistbr/brazilian-ecommerce",
    "license": "CC BY-NC-SA 4.0",
    "url": "https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce",
}


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    root = tmp_path / "data"
    root.mkdir()
    (root / "orders.csv").write_text("id,status\n1,delivered\n", encoding="utf-8")
    (root / "items.csv").write_text("order_id,price\n1,10.0\n", encoding="utf-8")
    return root


def test_manifest_has_an_entry_per_file(data_dir: Path) -> None:
    manifest = build_manifest(data_dir, **SOURCE_KW)

    assert [f.name for f in manifest.files] == ["items.csv", "orders.csv"]
    assert all(len(f.sha256) == 64 for f in manifest.files)
    assert all(f.bytes > 0 for f in manifest.files)


def test_manifest_is_stable_across_rebuilds(data_dir: Path) -> None:
    first = build_manifest(data_dir, retrieved_at=dt.date(2026, 8, 12), **SOURCE_KW)
    second = build_manifest(data_dir, retrieved_at=dt.date(2026, 8, 12), **SOURCE_KW)

    assert first == second


def test_verify_passes_on_untouched_directory(data_dir: Path) -> None:
    verify(build_manifest(data_dir, **SOURCE_KW), data_dir)


def test_changed_byte_names_the_file(data_dir: Path) -> None:
    manifest = build_manifest(data_dir, **SOURCE_KW)
    (data_dir / "orders.csv").write_text("id,status\n1,shipped\n", encoding="utf-8")

    with pytest.raises(ChangedFiles) as excinfo:
        verify(manifest, data_dir)

    assert excinfo.value.names == ["orders.csv"]


def test_missing_file_is_a_distinct_error(data_dir: Path) -> None:
    manifest = build_manifest(data_dir, **SOURCE_KW)
    (data_dir / "orders.csv").unlink()

    with pytest.raises(MissingFiles) as excinfo:
        verify(manifest, data_dir)

    assert excinfo.value.names == ["orders.csv"]


def test_matches_reports_state_without_raising(data_dir: Path) -> None:
    manifest = build_manifest(data_dir, **SOURCE_KW)

    assert matches(manifest, data_dir) is True

    (data_dir / "items.csv").unlink()
    assert matches(manifest, data_dir) is False


def test_manifest_round_trips_through_yaml(data_dir: Path, tmp_path: Path) -> None:
    manifest = build_manifest(data_dir, **SOURCE_KW)
    path = tmp_path / "manifest.yaml"

    write_manifest(manifest, path)

    assert read_manifest(path) == manifest


def test_missing_credentials_explain_how_to_get_them(tmp_path: Path) -> None:
    with pytest.raises(KaggleCredentialsMissing, match="Create New Token"):
        read_credentials(tmp_path / "kaggle.json")


def test_malformed_credentials_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "kaggle.json"
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(KaggleCredentialsMissing, match="JSON"):
        read_credentials(path)


def test_credentials_without_key_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "kaggle.json"
    path.write_text(json.dumps({"username": "someone"}), encoding="utf-8")

    with pytest.raises(KaggleCredentialsMissing, match="username и key"):
        read_credentials(path)


def test_derived_files_are_recorded(data_dir: Path, tmp_path: Path) -> None:
    derived = tmp_path / "parquet"
    derived.mkdir()
    (derived / "orders.parquet").write_bytes(b"PAR1fake")

    manifest = build_manifest(data_dir, derived_root=derived, **SOURCE_KW)

    assert [f.name for f in manifest.derived] == ["orders.parquet"]
    assert manifest.derived[0].bytes == len(b"PAR1fake")


def test_changed_derived_file_is_caught(data_dir: Path, tmp_path: Path) -> None:
    derived = tmp_path / "parquet"
    derived.mkdir()
    target = derived / "orders.parquet"
    target.write_bytes(b"PAR1fake")

    manifest = build_manifest(data_dir, derived_root=derived, **SOURCE_KW)
    target.write_bytes(b"PAR1other")

    with pytest.raises(ChangedFiles) as excinfo:
        verify(manifest, data_dir, derived)

    assert excinfo.value.names == ["orders.parquet"]


def test_derived_files_are_skipped_when_root_not_given(data_dir: Path, tmp_path: Path) -> None:
    derived = tmp_path / "parquet"
    derived.mkdir()
    (derived / "orders.parquet").write_bytes(b"PAR1fake")
    manifest = build_manifest(data_dir, derived_root=derived, **SOURCE_KW)

    (derived / "orders.parquet").unlink()

    verify(manifest, data_dir)

    with pytest.raises(MissingFiles):
        verify(manifest, data_dir, derived)


def test_manifest_without_derived_defaults_to_empty(data_dir: Path) -> None:
    manifest = build_manifest(data_dir, **SOURCE_KW)

    assert manifest.derived == []


def test_empty_manifest_is_rejected() -> None:
    with pytest.raises(ValidationError):
        SourceManifest(
            source="s", license="l", url="u", retrieved_at=dt.date(2026, 8, 12), files=[]
        )


def test_duplicate_names_are_rejected() -> None:
    entry = FileEntry(name="a.csv", sha256="0" * 64, bytes=1)
    with pytest.raises(ValidationError, match="дубликаты"):
        SourceManifest(
            source="s",
            license="l",
            url="u",
            retrieved_at=dt.date(2026, 8, 12),
            files=[entry, entry],
        )


def test_extra_file_is_caught_in_strict_mode(data_dir: Path) -> None:
    manifest = build_manifest(data_dir, **SOURCE_KW)
    (data_dir / "leftover.csv").write_text("stale\n", encoding="utf-8")

    verify(manifest, data_dir)

    with pytest.raises(ExtraFiles) as excinfo:
        verify(manifest, data_dir, strict=True)

    assert excinfo.value.names == ["leftover.csv"]


def test_nested_names_use_posix_separators(tmp_path: Path) -> None:
    root = tmp_path / "data"
    (root / "sub").mkdir(parents=True)
    (root / "sub" / "orders.csv").write_text("id\n1\n", encoding="utf-8")

    manifest = build_manifest(root, **SOURCE_KW)

    assert [f.name for f in manifest.files] == ["sub/orders.csv"]


def test_corrupt_manifest_raises_manifest_error(tmp_path: Path) -> None:
    path = tmp_path / "manifest.yaml"
    path.write_text("files: [oops\n", encoding="utf-8")

    with pytest.raises(ManifestError):
        read_manifest(path)


def test_manifest_of_wrong_shape_raises_manifest_error(tmp_path: Path) -> None:
    path = tmp_path / "manifest.yaml"
    path.write_text("just a string\n", encoding="utf-8")

    with pytest.raises(ManifestError):
        read_manifest(path)
