"""Манифест источника: происхождение и контрольные суммы данных.

Данные в git не попадают (лицензии источников, правило рабочего места), поэтому
единственное, что остаётся в репозитории, — запись о том, что именно было
получено. Несовпадение контрольной суммы считается ошибкой, а не
предупреждением: это первый носитель принципа воспроизводимости.
"""

from __future__ import annotations

import datetime as dt
import hashlib
from pathlib import Path
from typing import Annotated

import yaml
from pydantic import BaseModel, ConfigDict, Field

_CHUNK = 1 << 20


class ManifestError(Exception):
    """Базовая ошибка манифеста."""


class MissingFiles(ManifestError):
    """Файлы, записанные в манифест, отсутствуют на диске."""

    def __init__(self, names: list[str]) -> None:
        self.names = names
        super().__init__("отсутствуют файлы: " + ", ".join(names))


class ChangedFiles(ManifestError):
    """Файлы на диске не совпадают с записанными контрольными суммами."""

    def __init__(self, names: list[str]) -> None:
        self.names = names
        super().__init__("контрольная сумма не совпадает: " + ", ".join(names))


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


class FileEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: Annotated[str, Field(min_length=1)]
    sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    bytes: Annotated[int, Field(ge=0)]


class SourceManifest(BaseModel):
    """Происхождение набора файлов."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source: Annotated[str, Field(min_length=1)]
    """Откуда получено, например 'kaggle:olistbr/brazilian-ecommerce'."""

    license: Annotated[str, Field(min_length=1)]
    url: Annotated[str, Field(min_length=1)]
    retrieved_at: dt.date

    files: list[FileEntry]
    """Файлы, полученные из источника."""

    derived: list[FileEntry] = Field(default_factory=list)
    """Файлы, порождённые из источника локально (например, Parquet).

    Учитываются наравне с исходными: производный артефакт, не покрытый
    контрольной суммой, ломает воспроизводимость так же, как исходный.
    """

    def entry(self, name: str) -> FileEntry | None:
        for item in (*self.files, *self.derived):
            if item.name == name:
                return item
        return None


def entries_for(root: Path, patterns: tuple[str, ...] = ("*",)) -> list[FileEntry]:
    """Записи по каталогу, упорядоченные по имени для стабильности манифеста."""
    paths: set[Path] = set()
    for pattern in patterns:
        paths.update(p for p in root.rglob(pattern) if p.is_file())

    return [
        FileEntry(
            name=str(path.relative_to(root)),
            sha256=sha256_of(path),
            bytes=path.stat().st_size,
        )
        for path in sorted(paths)
    ]


def build_manifest(
    root: Path,
    *,
    source: str,
    license: str,
    url: str,
    patterns: tuple[str, ...] = ("*",),
    retrieved_at: dt.date | None = None,
    derived_root: Path | None = None,
    derived_patterns: tuple[str, ...] = ("*",),
) -> SourceManifest:
    """Собрать манифест по каталогу источника и, если задан, по каталогу производных."""
    return SourceManifest(
        source=source,
        license=license,
        url=url,
        retrieved_at=retrieved_at or dt.date.today(),
        files=entries_for(root, patterns),
        derived=entries_for(derived_root, derived_patterns) if derived_root else [],
    )


def verify(manifest: SourceManifest, root: Path, derived_root: Path | None = None) -> None:
    """Проверить каталоги против манифеста.

    Отсутствие файла и изменение файла — разные ошибки: первое означает
    неполную загрузку, второе — что данные под нами поменялись.

    Производные файлы проверяются, только если указан derived_root: без него
    их отсутствие означает «ещё не собраны», а не «потеряны».
    """
    missing: list[str] = []
    changed: list[str] = []

    checks = [(item, root) for item in manifest.files]
    if derived_root is not None:
        checks += [(item, derived_root) for item in manifest.derived]

    for item, base in checks:
        path = base / item.name
        if not path.is_file():
            missing.append(item.name)
        elif sha256_of(path) != item.sha256:
            changed.append(item.name)

    if missing:
        raise MissingFiles(missing)
    if changed:
        raise ChangedFiles(changed)


def matches(manifest: SourceManifest, root: Path, derived_root: Path | None = None) -> bool:
    """Совпадают ли каталоги с манифестом.

    Без исключений — для решения о пропуске загрузки.
    """
    try:
        verify(manifest, root, derived_root)
    except ManifestError:
        return False
    return True


def write_manifest(manifest: SourceManifest, path: Path) -> None:
    payload = manifest.model_dump(mode="json")
    path.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def read_manifest(path: Path) -> SourceManifest:
    return SourceManifest.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
