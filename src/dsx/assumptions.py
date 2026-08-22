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

from pydantic import BaseModel, ConfigDict, Field, model_validator


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


NEEDS_ASKING = frozenset({Basis.DOMAIN_KNOWLEDGE, Basis.OWNER, Basis.DOCUMENT})
"""Основания, требующие подтверждения у людей на новом месте.

Документ добавлен по итогам третьего кейса. Я прочитал словарь данных, вывел
из него, когда приходят формы работника, и ошибся: они приходят позже сборки
претензии в шестидесяти процентах случаев. Прочтение документа — интерпретация,
а не факт, и проверяется оно перечитыванием вместе с тем, кто документ писал.
"""

NEEDS_EVIDENCE = frozenset({Basis.DATA, Basis.DOCUMENT})
"""Основания, обязанные предъявить ссылку.

«Из данных» без расчёта и «из документа» без страницы неотличимы от «мне так
кажется», но выглядят обоснованными.
"""


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

    @model_validator(mode="after")
    def _evidence_is_required_where_it_exists(self) -> Assumption:
        if self.basis in NEEDS_EVIDENCE and not (self.evidence or "").strip():
            raise ValueError(
                f"основание {self.basis.value!r} обязано предъявить ссылку: расчёт для "
                "данных, страницу или раздел для документа. Без неё утверждение "
                "неотличимо от догадки, но выглядит обоснованным"
            )
        return self

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
