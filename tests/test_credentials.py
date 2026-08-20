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
