"""Реестр допущений.

На этапе 0 доменное знание применялось минимум в девяти решениях и ни разу не
было записано как доменное: порядок событий процесса, трактовка отмены,
доступность признаков при оформлении, направление монотонной связи, сама
постановка задачи. Каждое записывалось словами «заполнил вручную».

Причина ошибки называется здесь прямо: **скорость собственного ответа была
принята за отсутствие вопроса**. Человек, знающий отрасль, отвечает мгновенно
и не замечает, что ответил, — а новичок на том же месте встанет.

Поэтому реестр различает не «важное и неважное», а **источник ответа**. Всё,
что решено без обращения к данным, автоматически попадает в список вопросов
отрасли.
"""

from __future__ import annotations

import datetime as dt
from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field


class Basis(StrEnum):
    """Откуда взят ответ."""

    DATA = "data"
    """Выведено из данных. Проверяемо повторным вычислением."""

    OWNER = "owner"
    """Сказал владелец данных или системы. Проверяемо переспросом."""

    DOCUMENT = "document"
    """Записано в документации, схеме, регламенте."""

    DOMAIN_KNOWLEDGE = "domain_knowledge"
    """Известно из опыта в отрасли. Не проверяемо ничем внутри проекта и
    потому обязано стать вопросом для того, кто отрасли не знает."""


NEEDS_ASKING = frozenset({Basis.DOMAIN_KNOWLEDGE, Basis.OWNER})


class Assumption(BaseModel):
    """Одно допущение, на котором держится анализ."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    statement: Annotated[str, Field(min_length=1)]
    """Что именно принято. Формулируется утверждением, а не вопросом."""

    basis: Basis
    author: Annotated[str, Field(min_length=1)]

    evidence: str | None = None
    """Ссылка на подтверждение: имя документа, кто сказал, какой расчёт."""

    consequence: Annotated[str, Field(min_length=1)]
    """Что сломается, если допущение неверно. Допущение без последствия
    невозможно приоритизировать при перепроверке."""

    at: dt.datetime = Field(default_factory=lambda: dt.datetime.now(dt.UTC))

    @property
    def needs_asking(self) -> bool:
        """Требует ли подтверждения у человека на новом месте."""
        return self.basis in NEEDS_ASKING

    def as_question(self) -> str:
        """Переформулировать допущение вопросом к владельцу данных."""
        return f"Верно ли, что {self.statement.rstrip('.')}? (иначе {self.consequence})"

    def __str__(self) -> str:
        return f"[{self.basis.value}] {self.statement}"


class AssumptionRegistry:
    """Реестр допущений проекта (S2) и источник вопросов отрасли (S3)."""

    def __init__(self) -> None:
        self._items: list[Assumption] = []

    def record(
        self,
        statement: str,
        *,
        basis: Basis,
        author: str,
        consequence: str,
        evidence: str | None = None,
    ) -> Assumption:
        item = Assumption(
            statement=statement,
            basis=basis,
            author=author,
            consequence=consequence,
            evidence=evidence,
        )
        self._items.append(item)
        return item

    @property
    def items(self) -> tuple[Assumption, ...]:
        return tuple(self._items)

    def by_basis(self, basis: Basis) -> tuple[Assumption, ...]:
        return tuple(a for a in self._items if a.basis is basis)

    def open_questions(self) -> tuple[str, ...]:
        """Вопросы, которые нужно задать людям на новом месте (S3).

        Порождаются из допущений, а не пишутся отдельно: список, ведомый
        вручную, расходится с тем, на чём анализ держится на самом деле.
        """
        return tuple(a.as_question() for a in self._items if a.needs_asking)

    def report_section(self) -> str:
        if not self._items:
            return "## Допущения\n\nНе зафиксировано."

        lines = ["## Допущения", "", "| основание | утверждение | если неверно |", "|---|---|---|"]
        for item in self._items:
            lines.append(f"| {item.basis.value} | {item.statement} | {item.consequence} |")

        questions = self.open_questions()
        if questions:
            lines += ["", "### Требуют подтверждения на новом месте", ""]
            lines += [f"- {q}" for q in questions]
        return "\n".join(lines)
