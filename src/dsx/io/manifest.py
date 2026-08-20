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
from pydantic import BaseModel, ConfigDict, Field, model_validator

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


class ExtraFiles(ManifestError):
    """В каталоге есть файлы, которых нет в манифесте.

    Обычно означает остатки прошлой загрузки, смешавшиеся с новыми данными.
    """

    def __init__(self, names: list[str]) -> None:
        self.names = names
        super().__init__("файлы вне манифеста: " + ", ".join(names))


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

    files: Annotated[list[FileEntry], Field(min_length=1)]
    """Файлы, полученные из источника.

    Пустой список запрещён: манифест, не описывающий ничего, совпадёт с любым
    пустым каталогом и заставит пропустить загрузку.
    """

    derived: list[FileEntry] = Field(default_factory=list)
    """Файлы, порождённые из источника локально (например, Parquet).

    Учитываются наравне с исходными: производный артефакт, не покрытый
    контрольной суммой, ломает воспроизводимость так же, как исходный.
    """

    @model_validator(mode="after")
    def _names_are_unique(self) -> SourceManifest:
        for label, group in (("files", self.files), ("derived", self.derived)):
            names = [item.name for item in group]
            if len(names) != len(set(names)):
                raise ValueError(f"дубликаты имён в {label}")
        return self


def entries_for(root: Path, patterns: tuple[str, ...] = ("*",)) -> list[FileEntry]:
    """Записи по каталогу, упорядоченные по имени для стабильности манифеста."""
    paths: set[Path] = set()
    for pattern in patterns:
        paths.update(p for p in root.rglob(pattern) if p.is_file())

    entries = [
        FileEntry(
            # POSIX-вид: манифест, собранный на одной ОС, читается на другой.
            name=path.relative_to(root).as_posix(),
            sha256=sha256_of(path),
            bytes=path.stat().st_size,
        )
        for path in paths
    ]
    return sorted(entries, key=lambda item: item.name)


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


def verify(
    manifest: SourceManifest,
    root: Path,
    derived_root: Path | None = None,
    *,
    strict: bool = False,
) -> None:
    """Проверить каталоги против манифеста.

    Отсутствие файла и изменение файла — разные ошибки: первое означает
    неполную загрузку, второе — что данные под нами поменялись.

    Производные файлы проверяются, только если указан derived_root: без него
    их отсутствие означает «ещё не собраны», а не «потеряны».

    strict дополнительно требует, чтобы в каталогах не было файлов вне
    манифеста. Для каталога загрузки это обязательно: лишний файл означает,
    что данные смешались с остатками прошлой попытки.
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

    if strict:
        extra = _extra_names(manifest.files, root)
        if derived_root is not None:
            extra += _extra_names(manifest.derived, derived_root)
        if extra:
            raise ExtraFiles(sorted(extra))


def _extra_names(expected: list[FileEntry], root: Path) -> list[str]:
    if not root.is_dir():
        return []
    known = {item.name for item in expected}
    present = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    return sorted(present - known)


def matches(
    manifest: SourceManifest,
    root: Path,
    derived_root: Path | None = None,
    *,
    strict: bool = False,
) -> bool:
    """Совпадают ли каталоги с манифестом.

    Без исключений — для решения о пропуске загрузки.
    """
    try:
        verify(manifest, root, derived_root, strict=strict)
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
    """Прочитать манифест. Повреждённый манифест — ManifestError, а не что попало."""
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        return SourceManifest.model_validate(payload)
    except ManifestError:
        raise
    except Exception as exc:
        raise ManifestError(f"манифест {path} не читается: {exc}") from exc
