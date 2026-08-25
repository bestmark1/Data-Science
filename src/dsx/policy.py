"""Блокировки и механизм явного обхода.

Блокирующее требование без механизма обхода — фикция. Под давлением срока его
обойдёт сам автор, и обойдёт незаметно: выборку переименуют, дрейф не объявят,
доступность заполнят словами «скорее всего доступно».

Поэтому обход существует, но он дороже по видимости и дешевле по усилию, чем
обман: одна запись с обязательной причиной, которая попадает в итоговый отчёт
отдельным разделом. Блокировка, которую проще обойти тайно, чем явно, защищает
хуже предупреждения — она даёт ложное чувство защиты.
"""

from __future__ import annotations

import datetime as dt
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field


class Blocked(Exception):
    """Требование нарушено и не обойдено явно."""

    def __init__(self, requirement: str, detail: str) -> None:
        self.requirement = requirement
        self.detail = detail
        super().__init__(
            f"[{requirement}] {detail}\n"
            f"Если это осознанное решение, зафиксируйте обход с причиной: "
            f"ledger.override({requirement!r}, reason=..., author=...)"
        )


class Override(BaseModel):
    """Явный, обоснованный и видимый обход одной блокировки."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    requirement: Annotated[str, Field(pattern=r"^[A-Z]\d{1,2}[a-z]?$")]
    """Идентификатор требования: буква, одна-две цифры и необязательный
    строчный суффикс.

    Суффикс добавлен по итогам пятого кейса. Проверка N2i объявляла себя этим
    именем и была блокирующей, а журнал обходов такого имени не принимал — то
    есть обойти её было нельзя в принципе. Механизм существовал, дотянуться до
    него было невозможно; тот же класс, что F-13 и недостижимый журнал обходов
    в runner.
    """
    reason: Annotated[str, Field(min_length=10)]
    """Причина обязательна и не может быть отпиской: обход без объяснения
    неотличим от обмана."""

    author: Annotated[str, Field(min_length=1)]
    at: dt.datetime = Field(default_factory=lambda: dt.datetime.now(dt.UTC))

    def __str__(self) -> str:
        return f"{self.requirement} обойдено ({self.author}, {self.at:%Y-%m-%d}): {self.reason}"


class OverrideLedger:
    """Журнал обходов. Попадает в отчёт целиком."""

    def __init__(self) -> None:
        self._entries: list[Override] = []

    def override(self, requirement: str, *, reason: str, author: str) -> Override:
        entry = Override(requirement=requirement, reason=reason, author=author)
        self._entries.append(entry)
        return entry

    def is_overridden(self, requirement: str) -> bool:
        return any(e.requirement == requirement for e in self._entries)

    @property
    def entries(self) -> tuple[Override, ...]:
        return tuple(self._entries)

    def enforce(self, requirement: str, detail: str) -> Override | None:
        """Заблокировать, если обход не зафиксирован.

        Возвращает запись обхода, когда он есть, — чтобы вызывающий код мог
        протащить её в отчёт, а не просто продолжить молча.
        """
        for entry in self._entries:
            if entry.requirement == requirement:
                return entry
        raise Blocked(requirement, detail)

    def report_section(self) -> str:
        """Раздел отчёта. Пустой журнал тоже отражается явно."""
        if not self._entries:
            return "## Обходы блокировок\n\nНе зафиксировано."
        lines = ["## Обходы блокировок", ""]
        lines += [f"- {entry}" for entry in self._entries]
        return "\n".join(lines)
