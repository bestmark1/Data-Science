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

    MEASURED_AT = "measured_at"
    """Момент, когда значение признака было получено.

    Введена по F-11. Окно признака объявлялось и ни с чем не сверялось: ядро
    верило, что среднее посчитано за объявленные семь дней, и что мгновенное
    показание получено ровно в момент решения. Ни то, ни другое проверить было
    нечем, и объявление оставалось необеспеченным.
    """

    STATUS = "status"
    """Категориальное состояние процесса."""

    OUTCOME_COMPONENT = "outcome_component"
    """Колонка, участвующая в вычислении исхода. Признаком быть не может."""

    FEATURE = "feature"
    IGNORED = "ignored"


class FeatureWindow(BaseModel):
    """Интервал, по которому посчитан признак-агрегат.

    Второй кейс показал, что ядро не знает, откуда взялось значение признака.
    Среднее за неделю и мгновенное показание выглядят одинаково — колонка
    с числом. Но у первого есть протяжённость назад, и она может захватить
    измерения, попавшие в обучающие строки того же объекта.

    Окно решения в момент t покрывает [t - lag - lookback, t - lag].
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    lookback_days: Annotated[float, Field(ge=0.0)]
    """Протяжённость окна назад. Ноль означает мгновенное показание."""

    lag_days: Annotated[float, Field(ge=0.0)] = 0.0
    """Отступ: окно кончается за столько дней до момента решения.

    Ненулевой отступ — обычный способ учесть задержку поставки данных.
    """

    source_of_claim: Annotated[str, Field(min_length=1)]
    """Чем подтверждено окно: код построения признака, владелец, документ.

    Без источника поле заполняется по памяти, и объявление перестаёт быть
    свидетельством.
    """

    @property
    def instantaneous(self) -> bool:
        return self.lookback_days == 0.0

    def bounds_days(self) -> tuple[float, float]:
        """Границы окна в днях относительно момента решения, слева направо."""
        return (-(self.lag_days + self.lookback_days), -self.lag_days)


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


TEMPORAL_ROLES = frozenset(
    {
        Role.DECISION_TIME,
        Role.EVENT_TIME,
        Role.DEADLINE,
        Role.AVAILABLE_AT,
        Role.MEASURED_AT,
    }
)


class ColumnSpec(BaseModel):
    """Объявление одной колонки."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: Annotated[str, Field(min_length=1)]
    role: Role
    temporal: TemporalKind | None = None
    availability: Availability = Availability.UNKNOWN

    window: FeatureWindow | None = None
    """Окно, по которому посчитан признак. None означает, что ответа нет.

    Мгновенное показание объявляется окном нулевой длины, а не отсутствием
    окна: умолчание, совпадающее с честным ответом, неотличимо от пропуска.
    """

    measured_at: str | None = None
    """Колонка с моментом получения значения. None означает, что сверить
    объявленное окно с данными нечем, и оно остаётся объявлением."""

    source_of_claim: str | None = None
    """Чем подтверждена объявленная доступность: владелец, схема, документ, лог.

    Без источника поле заполняется словами «скорее всего доступно», и требование
    порождает ложные ответы вместо защиты.
    """

    @model_validator(mode="after")
    def _temporal_roles_declare_granularity(self) -> ColumnSpec:
        """Временные роли обязаны объявить грануляцию; остальные — могут.

        Компонент исхода часто является меткой времени, и именно сравнение его
        с датой дало 16.5% ложных положительных меток на этапе 0. Запрещать ему
        объявлять грануляцию значило бы закрыть глаза на самый дорогой случай.
        """
        if self.role in TEMPORAL_ROLES and self.temporal is None:
            raise ValueError(
                f"колонка {self.name!r} играет временную роль {self.role.value!r} "
                "и обязана объявить грануляцию: date или instant"
            )
        return self

    @model_validator(mode="after")
    def _only_features_have_windows(self) -> ColumnSpec:
        if self.window is not None and self.role is not Role.FEATURE:
            raise ValueError(
                f"колонка {self.name!r} играет роль {self.role.value!r} и окна "
                "признака иметь не может"
            )
        return self

    @model_validator(mode="after")
    def _measured_at_needs_a_window(self) -> ColumnSpec:
        """Время измерения без окна сверять не с чем."""
        if self.measured_at is not None and self.window is None:
            raise ValueError(
                f"колонка {self.name!r} называет время измерения, но не объявила "
                "окно признака: сверять нечего"
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
