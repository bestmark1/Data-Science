"""Запись трения — единица наблюдения этапа 0.

Журнал не побочный продукт прохода, а его основной выход. Схема существует,
чтобы запись нельзя было заполнить бессодержательно: поле tool_should_have
превращает жалобу в требование к ядру, и без него запись отклоняется.
"""

from __future__ import annotations

import datetime as dt
from typing import Annotated, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, model_validator


def _stripped(value: object) -> object:
    """Обрезать пробелы до проверки длины.

    Без этого строка из пробелов проходит min_length=1: формально поле
    заполнено, содержательно пусто — ровно то, что схема должна отсекать.
    """
    return value.strip() if isinstance(value, str) else value


NonEmpty = Annotated[str, BeforeValidator(_stripped), Field(min_length=1)]
OptionalText = Annotated[str | None, BeforeValidator(_stripped), Field(min_length=1)]

Scope = Literal["portable", "local"]
"""portable — обобщённый урок, едет между местами работы.
local — конкретика места: имена таблиц, продуктов, сегментов. В git не попадает."""


class FrictionRecord(BaseModel):
    """Одно трение, зафиксированное в момент возникновения."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    step: NonEmpty
    """Шаг прохода, на котором возникло. Например '04_decision_moment'."""

    did_manually: NonEmpty
    """Что пришлось делать руками."""

    minutes: Annotated[int, Field(gt=0)]
    """Сколько это заняло. Ноль означает, что трения не было."""

    tool_should_have: NonEmpty
    """Что вместо этого должен был бы сделать инструмент.

    Ключевое поле схемы: оно превращает наблюдение в требование. Запись без
    него описывает раздражение, а не спецификацию, и пользы для этапа 1 не несёт.
    """

    outcome: NonEmpty
    """Чем закончилось: получилось, обошёл, бросил."""

    looked_up: OptionalText = None
    """Что искалось в документации или интернете, если искалось."""

    contains_workplace_specifics: bool = False
    """Содержит ли запись имена таблиц, колонок, продуктов или сегментов
    конкретного места работы."""

    scope: Scope = "portable"
    created_at: dt.date = Field(default_factory=dt.date.today)

    @model_validator(mode="after")
    def _specifics_must_stay_local(self) -> FrictionRecord:
        """Отнесение — правило, а не аккуратность (R7)."""
        if self.contains_workplace_specifics and self.scope != "local":
            raise ValueError(
                "запись содержит конкретику места работы и обязана иметь scope='local'"
            )
        return self
