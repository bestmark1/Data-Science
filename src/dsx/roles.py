"""Роли колонок и временная семантика.

Ядро не знает имён колонок. Оно знает роли: что здесь идентификатор сущности,
что момент решения, что компонент исхода. Имена вроде order_id живут в
конфигурации проекта-экземпляра и в ядро не попадают — иначе оно прирастает к
схеме первого датасета.

Временная семантика объявляется, а не угадывается. Эвристика «одна уникальная
отметка времени суток означает дату» переобучена на конкретный формат выгрузки:
она даёт ложные срабатывания, которые пользователь научится затыкать, и защита
перестаёт работать.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Role(StrEnum):
    """Роль колонки в задаче."""

    ENTITY_ID = "entity_id"
    """Идентификатор сущности, о которой принимается решение."""

    NATURAL_KEY = "natural_key"
    """Идентификатор объекта реального мира (человек, устройство), если он
    отличается от идентификатора строки."""

    DECISION_TIME = "decision_time"
    """Момент принятия решения. Всё, что появляется позже, использовать нельзя."""

    EVENT_TIME = "event_time"
    """Момент наступления события процесса."""

    DEADLINE = "deadline"
    """Обещанный или назначенный срок, известный заранее."""

    AVAILABLE_AT = "available_at"
    """Момент, когда значение стало доступно системе."""

    STATUS = "status"
    """Категориальное состояние процесса."""

    OUTCOME_COMPONENT = "outcome_component"
    """Колонка, участвующая в вычислении исхода. Признаком быть не может."""

    FEATURE = "feature"
    IGNORED = "ignored"


class TemporalKind(StrEnum):
    """Фактическая грануляция временной колонки.

    Сравнение колонок разной грануляции — источник ошибки, которую не видно
    глазами: доставка в обещанный день оказывается опозданием.
    """

    DATE = "date"
    INSTANT = "instant"


class Availability(StrEnum):
    """Доступность значения в момент решения."""

    AT_DECISION = "at_decision"
    AFTER = "after"
    UNKNOWN = "unknown"
    """Полноправное состояние. Умолчание здесь опаснее отсутствия ответа."""


TEMPORAL_ROLES = frozenset({Role.DECISION_TIME, Role.EVENT_TIME, Role.DEADLINE, Role.AVAILABLE_AT})


class ColumnSpec(BaseModel):
    """Объявление одной колонки."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: Annotated[str, Field(min_length=1)]
    role: Role
    temporal: TemporalKind | None = None
    availability: Availability = Availability.UNKNOWN

    source_of_claim: str | None = None
    """Чем подтверждена объявленная доступность: владелец, схема, документ, лог.

    Без источника поле заполняется словами «скорее всего доступно», и требование
    порождает ложные ответы вместо защиты.
    """

    @model_validator(mode="after")
    def _temporal_roles_declare_granularity(self) -> ColumnSpec:
        if self.role in TEMPORAL_ROLES and self.temporal is None:
            raise ValueError(
                f"колонка {self.name!r} играет временную роль {self.role.value!r} "
                "и обязана объявить грануляцию: date или instant"
            )
        if self.role not in TEMPORAL_ROLES and self.temporal is not None:
            raise ValueError(
                f"колонка {self.name!r} не играет временной роли, грануляция неприменима"
            )
        return self

    @model_validator(mode="after")
    def _known_availability_needs_a_source(self) -> ColumnSpec:
        decided = self.availability in (Availability.AT_DECISION, Availability.AFTER)
        if decided and not self.source_of_claim:
            raise ValueError(
                f"колонка {self.name!r} объявляет доступность "
                f"{self.availability.value!r} без источника утверждения"
            )
        return self


class Schema(BaseModel):
    """Объявленная схема проекта в терминах ролей."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    columns: Annotated[list[ColumnSpec], Field(min_length=1)]

    @model_validator(mode="after")
    def _names_unique(self) -> Schema:
        names = [c.name for c in self.columns]
        if len(names) != len(set(names)):
            raise ValueError("дубликаты имён колонок")
        return self

    @model_validator(mode="after")
    def _exactly_one_decision_time(self) -> Schema:
        moments = [c for c in self.columns if c.role is Role.DECISION_TIME]
        if len(moments) != 1:
            raise ValueError(f"момент решения должен быть ровно один, объявлено {len(moments)}")
        return self

    def by_role(self, role: Role) -> list[ColumnSpec]:
        return [c for c in self.columns if c.role is role]

    def get(self, name: str) -> ColumnSpec | None:
        for column in self.columns:
            if column.name == name:
                return column
        return None

    @property
    def decision_time(self) -> ColumnSpec:
        return self.by_role(Role.DECISION_TIME)[0]

    def usable_features(self) -> list[ColumnSpec]:
        """Признаки, которые допустимо подавать в модель.

        Исключены компоненты исхода и всё, чья доступность не объявлена или
        объявлена как появляющаяся после момента решения.
        """
        return [
            c
            for c in self.columns
            if c.role is Role.FEATURE and c.availability is Availability.AT_DECISION
        ]
