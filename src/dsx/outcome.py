"""Определение исхода.

Исход вычисляется по объявленному правилу, а не произвольным выражением.
Причина в том, что самая дорогая ошибка этапа 0 сидела именно в способе
сравнения: момент времени сравнивался с датой, и 16.5% положительных меток
оказались ложными. Пока сравнение скрыто внутри кода анализа, проверить его
нечем.

Второе следствие: колонки, участвующие в вычислении исхода, известны явно и
потому не могут попасть в признаки (C6).
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dsx.roles import Schema, TemporalKind


class ComparisonMode(StrEnum):
    """Как сравниваются две временные колонки."""

    DIRECT = "direct"
    """Как есть. Допустимо только при совпадающей грануляции."""

    BY_DATE = "by_date"
    """Обе стороны приводятся к календарной дате явно."""


class MissingEventMeaning(StrEnum):
    """Что означает отсутствие события (C4)."""

    NOT_OCCURRED = "not_occurred"
    """Событие не произошло — это наблюдение, а не пропуск."""

    UNOBSERVED = "unobserved"
    """Исход ещё не наблюдаем; строка исключается."""

    EXCLUDED = "excluded"
    """Объект не предполагался к обработке; исключается из популяции."""


class OutcomeDefinition(BaseModel):
    """Исход как сравнение момента события с назначенным сроком."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    event_column: Annotated[str, Field(min_length=1)]
    deadline_column: Annotated[str, Field(min_length=1)]
    comparison: ComparisonMode

    missing_event: dict[str, MissingEventMeaning]
    """Трактовка отсутствия события по значению статуса. Пустой словарь
    запрещён: склейка разных причин в один класс — ошибка, которая на этапе 0
    добавила 11.3% ложных положительных."""

    estimand: Annotated[str, Field(min_length=1)]
    """Что именно оценивается, словами. Заполняется до вычисления таргета."""

    @model_validator(mode="after")
    def _missing_event_is_explicit(self) -> OutcomeDefinition:
        if not self.missing_event:
            raise ValueError(
                "трактовка отсутствия события обязана быть объявлена явно "
                "для каждого значения статуса"
            )
        return self

    def components(self) -> frozenset[str]:
        """Колонки, участвующие в вычислении исхода."""
        return frozenset({self.event_column, self.deadline_column})


class OutcomeContractError(Exception):
    """Определение исхода несовместимо со схемой."""


def validate_outcome(definition: OutcomeDefinition, schema: Schema) -> None:
    """Проверить определение исхода против объявленной схемы."""
    left = schema.get(definition.event_column)
    right = schema.get(definition.deadline_column)

    for name, column in (
        (definition.event_column, left),
        (definition.deadline_column, right),
    ):
        if column is None:
            raise OutcomeContractError(f"колонка {name!r} отсутствует в схеме")
        if column.temporal is None:
            raise OutcomeContractError(
                f"колонка {name!r} участвует в вычислении исхода "
                "и обязана объявить временную грануляцию"
            )

    assert left is not None and right is not None
    if definition.comparison is ComparisonMode.DIRECT and left.temporal is not right.temporal:
        date_side = left if left.temporal is TemporalKind.DATE else right
        instant_side = right if date_side is left else left
        raise OutcomeContractError(
            f"{date_side.name!r} хранит дату, {instant_side.name!r} — момент времени. "
            "Прямое сравнение пометит событие в тот же день как произошедшее позже. "
            "Используйте comparison=by_date."
        )
