"""Получение датасета с Kaggle и конвертация в Parquet.

Единственное место, где на этапе 0 пишется аккуратный код: получение данных не
является предметом наблюдения, переписывать его потом незачем.
"""

from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import polars as pl

DEFAULT_CREDENTIALS_PATH = Path.home() / ".kaggle" / "kaggle.json"
_API_ROOT = "https://www.kaggle.com/api/v1"
_CHUNK = 1 << 20


class KaggleCredentialsMissing(Exception):
    """Токен Kaggle не найден или испорчен."""

    HOWTO = (
        "Получить токен: kaggle.com -> Settings -> API -> Create New Token. "
        "Положить скачанный kaggle.json в ~/.kaggle/ и выполнить chmod 600."
    )

    def __init__(self, reason: str) -> None:
        super().__init__(f"{reason}. {self.HOWTO}")


def read_credentials(path: Path = DEFAULT_CREDENTIALS_PATH) -> tuple[str, str]:
    if not path.is_file():
        raise KaggleCredentialsMissing(f"файл {path} не найден")

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise KaggleCredentialsMissing(f"файл {path} не является корректным JSON") from exc

    username = payload.get("username")
    key = payload.get("key")
    if not username or not key:
        raise KaggleCredentialsMissing(f"в {path} нет полей username и key")

    return username, key


def download_dataset(slug: str, destination: Path, *, credentials: tuple[str, str]) -> Path:
    """Скачать архив датасета. Возвращает путь к архиву."""
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / f"{slug.replace('/', '_')}.zip"

    username, key = credentials
    token = base64.b64encode(f"{username}:{key}".encode()).decode()
    request = urllib.request.Request(
        f"{_API_ROOT}/datasets/download/{slug}",
        headers={"Authorization": f"Basic {token}"},
    )

    try:
        with urllib.request.urlopen(request) as response, archive.open("wb") as handle:
            while chunk := response.read(_CHUNK):
                handle.write(chunk)
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            raise KaggleCredentialsMissing("Kaggle отклонил токен") from exc
        raise

    return archive


def extract_archive(archive: Path, destination: Path) -> list[Path]:
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(destination)
        names = bundle.namelist()
    return [destination / name for name in sorted(names)]


def csv_to_parquet(source: Path, destination: Path) -> Path:
    """Сконвертировать CSV в Parquet детерминированно.

    Идемпотентность важна: повторный прогон обязан дать файл с тем же хэшем,
    иначе воспроизводимость прогона (R10) недостижима.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame = pl.read_csv(source, infer_schema_length=10_000)
    frame.write_parquet(destination, compression="uncompressed", statistics=False)
    return destination
