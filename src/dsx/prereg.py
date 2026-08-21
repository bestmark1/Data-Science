"""Предрегистрация: критерии, зафиксированные до контакта с данными.

Критерии, написанные после просмотра данных, превращаются в рационализацию:
любой исход объявляется подтверждением. Здесь они получают отпечаток, и правка
после фиксации перестаёт быть незаметной.

Отпечаток покрывает не только текст критериев, но и версию ядра и манифест
датасета. Хэшировать один файл недостаточно: подменить можно данные или код, а
критерии оставить нетронутыми.

Честная граница: соло-разработчик способен переписать историю и пересчитать
отпечаток. Механизм не делает обман невозможным — он делает явную поправку
дешевле тайной подмены.
"""

from __future__ import annotations

import datetime as dt
import hashlib
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field


class PreregViolation(Exception):
    """Состояние разошлось с зафиксированным."""


class Severity(BaseModel):
    """Уровни серьёзности находки. Заданы заранее, чтобы «критично» против
    «косметично» не решалось после результата."""

    model_config = ConfigDict(frozen=True)

    BLOCKER: str = "blocker"
    PROTOCOL_CHANGE: str = "protocol-change"
    ASSUMPTION_GAP: str = "assumption-gap"
    REPORT_ONLY: str = "report-only"
    COSMETIC: str = "cosmetic"


COUNTS_FOR_ACCEPTANCE = frozenset({"blocker", "protocol-change", "assumption-gap"})
"""Уровни, засчитываемые при приёмке. Остальные не закрывают требование."""


class Amendment(BaseModel):
    """Изменение зафиксированных критериев после контакта с данными."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    what: Annotated[str, Field(min_length=1)]
    why: Annotated[str, Field(min_length=10)]
    affected_predictions: Annotated[list[str], Field(min_length=1)]
    invalidates_conclusions: bool
    author: Annotated[str, Field(min_length=1)]
    at: dt.datetime = Field(default_factory=lambda: dt.datetime.now(dt.UTC))

    def __str__(self) -> str:
        mark = " (прежние заключения устарели)" if self.invalidates_conclusions else ""
        return f"{self.at:%Y-%m-%d} {self.what}: {self.why}{mark}"


class Preregistration(BaseModel):
    """Зафиксированное состояние: критерии, версия ядра, манифест данных."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    criteria_path: str
    criteria_digest: Annotated[str, Field(pattern=r"^[0-9a-f]{12}$")]
    core_version: Annotated[str, Field(min_length=1)]
    dataset_digest: str = ""
    """Пусто до получения данных. Заполняется сразу после, до анализа."""

    registered_at: dt.datetime = Field(default_factory=lambda: dt.datetime.now(dt.UTC))
    amendments: list[Amendment] = Field(default_factory=list)

    @property
    def data_contacted(self) -> bool:
        return bool(self.dataset_digest)


def digest_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def digest_file(path: Path) -> str:
    return digest_text(path.read_text(encoding="utf-8"))


def digest_dataset(paths: list[Path]) -> str:
    """Отпечаток набора файлов данных.

    Замена файла меняет отпечаток и требует поправки: перейти на другую версию
    датасета незаметно — один из способов обойти предрегистрацию.
    """
    parts = []
    for path in sorted(paths):
        payload = hashlib.sha256(path.read_bytes()).hexdigest()
        parts.append(f"{path.name}:{payload}")
    return digest_text("\n".join(parts))


def register(criteria: Path, core_version: str) -> Preregistration:
    """Зафиксировать критерии до контакта с данными."""
    return Preregistration(
        criteria_path=str(criteria.name),
        criteria_digest=digest_file(criteria),
        core_version=core_version,
    )


def verify(prereg: Preregistration, criteria: Path) -> None:
    """Проверить, что критерии не изменились после фиксации."""
    current = digest_file(criteria)
    if current != prereg.criteria_digest:
        raise PreregViolation(
            f"критерии изменились после фиксации: было {prereg.criteria_digest}, "
            f"стало {current}. Если изменение осознанное, оформите поправку "
            "с перечнем затронутых предсказаний"
        )


def accepts(findings: dict[str, str], areas: frozenset[str]) -> bool:
    """Достаточно ли находок для закрытия требования.

    Засчитывается расхождение засчитываемого уровня в заранее заданной области.
    «Хоть что-то сломалось» не годится: оно оптимизирует под наличие поломки, а
    не под переносимость.
    """
    return any(
        severity in COUNTS_FOR_ACCEPTANCE and area in areas for area, severity in findings.items()
    )
