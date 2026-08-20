"""Получение датасета с Kaggle и конвертация в Parquet.

Единственное место, где на этапе 0 пишется аккуратный код: получение данных не
является предметом наблюдения, переписывать его потом незачем.
"""

from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

import polars as pl

DEFAULT_CREDENTIALS_PATH = Path.home() / ".kaggle" / "kaggle.json"
_API_ROOT = "https://www.kaggle.com/api/v1"
_CHUNK = 1 << 20
_TIMEOUT_SECONDS = 120
_MAX_UNCOMPRESSED_BYTES = 4 << 30
_ZIP_MAGIC = b"PK\x03\x04"


class KaggleCredentialsMissing(Exception):
    """Токен Kaggle не найден или испорчен."""

    HOWTO = (
        "Получить токен: kaggle.com -> Settings -> API -> Create New Token. "
        "Положить скачанный kaggle.json в ~/.kaggle/ и выполнить chmod 600."
    )

    def __init__(self, reason: str) -> None:
        super().__init__(f"{reason}. {self.HOWTO}")


class DownloadFailed(Exception):
    """Загрузка не удалась по причине, не связанной с токеном."""


class UnsafeArchive(Exception):
    """Архив пытается писать за пределы целевого каталога или слишком велик."""


def read_credentials(path: Path = DEFAULT_CREDENTIALS_PATH) -> tuple[str, str]:
    if not path.is_file():
        raise KaggleCredentialsMissing(f"файл {path} не найден")

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise KaggleCredentialsMissing(f"файл {path} не является корректным JSON") from exc

    if not isinstance(payload, dict):
        raise KaggleCredentialsMissing(f"в {path} ожидался объект JSON")

    username, key = payload.get("username"), payload.get("key")
    if not isinstance(username, str) or not isinstance(key, str) or not username or not key:
        raise KaggleCredentialsMissing(f"в {path} нет непустых строковых полей username и key")

    return username, key


class _StripAuthOnCrossHost(urllib.request.HTTPRedirectHandler):
    """Kaggle перенаправляет на хранилище. Заголовок с токеном туда уходить не должен."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001, ANN201
        new_request = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new_request is None:
            return None

        same_host = (
            urllib.parse.urlsplit(newurl).netloc == urllib.parse.urlsplit(req.full_url).netloc
        )
        if not same_host:
            new_request.remove_header("Authorization")
        return new_request


def _validate_slug(slug: str) -> str:
    owner, _, name = slug.partition("/")
    parts_ok = owner and name and "/" not in name
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")
    if not parts_ok or not set(owner + name) <= allowed:
        raise ValueError(f"недопустимый идентификатор датасета: {slug!r}")
    return slug


def download_dataset(slug: str, destination: Path, *, credentials: tuple[str, str]) -> Path:
    """Скачать архив датасета. Возвращает путь к архиву.

    Загрузка идёт во временный файл и переименовывается только целиком: оборванная
    попытка не должна оставить файл, который выглядит готовым.
    """
    _validate_slug(slug)
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / f"{slug.replace('/', '_')}.zip"
    partial = archive.with_suffix(".zip.part")

    username, key = credentials
    token = base64.b64encode(f"{username}:{key}".encode()).decode()
    request = urllib.request.Request(
        f"{_API_ROOT}/datasets/download/{slug}",
        headers={"Authorization": f"Basic {token}"},
    )
    opener = urllib.request.build_opener(_StripAuthOnCrossHost)

    try:
        with opener.open(request, timeout=_TIMEOUT_SECONDS) as response:
            with partial.open("wb") as handle:
                while chunk := response.read(_CHUNK):
                    handle.write(chunk)
    except urllib.error.HTTPError as exc:
        partial.unlink(missing_ok=True)
        if exc.code in (401, 403):
            raise KaggleCredentialsMissing(
                f"Kaggle отклонил запрос (HTTP {exc.code}): токен неверен, "
                "либо нужно принять условия датасета на его странице"
            ) from exc
        raise DownloadFailed(f"Kaggle ответил HTTP {exc.code}") from exc
    except OSError as exc:
        partial.unlink(missing_ok=True)
        raise DownloadFailed(f"сеть недоступна или чтение прервано: {exc}") from exc

    if partial.read_bytes()[:4] != _ZIP_MAGIC:
        partial.unlink(missing_ok=True)
        raise DownloadFailed("полученный файл не является ZIP-архивом")

    os.replace(partial, archive)
    return archive


def extract_archive(archive: Path, destination: Path) -> list[Path]:
    """Распаковать архив, возвращая пути извлечённых файлов.

    Пути проверяются до записи: архив не должен писать вне целевого каталога.
    """
    destination.mkdir(parents=True, exist_ok=True)
    resolved_root = destination.resolve()
    extracted: list[Path] = []

    with zipfile.ZipFile(archive) as bundle:
        total = sum(info.file_size for info in bundle.infolist())
        if total > _MAX_UNCOMPRESSED_BYTES:
            raise UnsafeArchive(f"распакованный размер {total} байт превышает лимит")

        for info in bundle.infolist():
            target = (resolved_root / info.filename).resolve()
            if not target.is_relative_to(resolved_root):
                raise UnsafeArchive(f"путь выходит за каталог назначения: {info.filename}")

            bundle.extract(info, destination)
            if not info.is_dir():
                extracted.append(destination / info.filename)

    return sorted(extracted)


def csv_to_parquet(source: Path, destination: Path) -> Path:
    """Сконвертировать CSV в Parquet детерминированно и атомарно.

    Идемпотентность важна: повторный прогон обязан дать файл с тем же хэшем,
    иначе воспроизводимость прогона (R10) недостижима. Запись через временный
    файл гарантирует, что прерванный прогон не оставит битый артефакт.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")

    frame = pl.read_csv(source, infer_schema_length=10_000)
    frame.write_parquet(partial, compression="uncompressed", statistics=False)
    os.replace(partial, destination)
    return destination
