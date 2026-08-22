"""Чтение токена Kaggle: любая некорректность даёт понятную ошибку, а не AttributeError."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dsx.io.acquire import KaggleCredentialsMissing, read_credentials


def _write(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_valid_credentials_are_read(tmp_path: Path) -> None:
    path = _write(tmp_path / "k.json", {"username": "someone", "key": "abc"})

    assert read_credentials(path) == ("someone", "abc")


@pytest.mark.parametrize(
    "payload",
    [
        "just a string",
        ["username", "key"],
        42,
        {"username": "someone", "key": 12345},
        {"username": "", "key": "abc"},
        {"username": "someone", "key": ""},
    ],
)
def test_wrong_shape_is_rejected(tmp_path: Path, payload: object) -> None:
    path = _write(tmp_path / "k.json", payload)

    with pytest.raises(KaggleCredentialsMissing):
        read_credentials(path)


def test_directory_instead_of_file_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(KaggleCredentialsMissing, match="не найден"):
        read_credentials(tmp_path)


# --- идентификатор датасета ------------------------------------------------


def test_apostrophe_is_allowed_in_a_slug() -> None:
    """Настоящие идентификаторы Kaggle его содержат."""
    from dsx.io.acquire import _validate_slug

    assert _validate_slug("new-york-state/nys-assembled-workers'-compensation-claims")


@pytest.mark.parametrize(
    "slug",
    ["owner/../etc", "owner/name.zip", "a/b/c", "/name", "owner/", "owner/na me"],
)
def test_slug_that_could_escape_the_data_directory_is_refused(slug: str) -> None:
    from dsx.io.acquire import _validate_slug

    with pytest.raises(ValueError, match="недопустимый идентификатор"):
        _validate_slug(slug)


def test_archive_name_keeps_only_predictable_characters() -> None:
    """Идентификатор в имя файла напрямую не переносится."""
    from dsx.io.acquire import _archive_name

    name = _archive_name("new-york-state/nys-assembled-workers'-compensation-claims")

    assert all(c.isalnum() or c in "-_." for c in name)
    assert name.endswith(".zip")
