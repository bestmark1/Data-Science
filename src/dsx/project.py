"""Объявления проекта в виде заполняемой формы.

Два кейса добавляли поверхность объявлений по одному полю к живому коду:
роли и грануляция, причины отсутствия события, направление исхода, жизненный
цикл объекта, окна признаков, время измерения, состав выборок, граница
резерва. Каждое поле обосновано находкой, но заполнить их разом как форму
никто ни разу не пробовал.

Здесь они сведены в одно место. Граница проходит честно: форма описывает
ОБЪЯВЛЕНИЯ, код проекта строит таблицу решений. Соединить пять таблиц разного
уровня декларацией нельзя, и делать вид, что можно, значило бы получить
конфигурацию, в которой прячется программа.

Форма читается из YAML и превращается в те же объекты, что раньше писались
руками. Ошибка в форме — отказ при загрузке, а не неверный результат позже.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Annotated

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from dsx.assumptions import AssumptionRegistry, Basis
from dsx.outcome import (
    ComparisonMode,
    MissingEventCause,
    MissingEventMeaning,
    OutcomeDefinition,
    PositiveClass,
)
from dsx.roles import (
    Availability,
    ColumnSpec,
    Direction,
    Evidence,
    FeatureWindow,
    Role,
    Schema,
    TemporalKind,
)
from dsx.split import Window
from dsx.task import (
    ObjectLifetime,
    OutcomeTiming,
    Simultaneity,
    TargetKind,
    TaskSpec,
)


class ColumnForm(BaseModel):
    """Одна колонка таблицы решений."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: Evidence
    role: Role
    temporal: TemporalKind | None = None
    availability: Availability = Availability.UNKNOWN
    source_of_claim: str | None = None

    window_lookback_days: float | None = None
    window_lag_days: float = 0.0
    window_source: str | None = None
    measured_at: str | None = None
    direction: Direction | None = None
    """Куда признак двигает риск по доменному знанию, до просмотра данных."""

    def to_spec(self) -> ColumnSpec:
        window = None
        if self.window_lookback_days is not None:
            if not self.window_source:
                raise ValueError(
                    f"колонка {self.name!r} объявила окно признака без источника "
                    "утверждения: чем подтверждено, что окно именно такое"
                )
            window = FeatureWindow(
                lookback_days=self.window_lookback_days,
                lag_days=self.window_lag_days,
                source_of_claim=self.window_source,
            )
        return ColumnSpec(
            name=self.name,
            role=self.role,
            temporal=self.temporal,
            availability=self.availability,
            source_of_claim=self.source_of_claim,
            window=window,
            measured_at=self.measured_at,
            direction=self.direction,
        )


class CauseForm(BaseModel):
    """Одна причина, по которой события могло не быть."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: Annotated[str, Field(min_length=1)]
    meaning: MissingEventMeaning
    status_value: str | None = None
    assumption: str | None = None

    def to_cause(self) -> MissingEventCause:
        return MissingEventCause(
            name=self.name,
            meaning=self.meaning,
            status_value=self.status_value,
            assumption=self.assumption,
        )


class OutcomeForm(BaseModel):
    """Контракт исхода."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    event_column: Annotated[str, Field(min_length=1)]
    deadline_column: Annotated[str, Field(min_length=1)]
    comparison: ComparisonMode
    positive_class: PositiveClass
    estimand: Annotated[str, Field(min_length=1)]
    missing_causes: Annotated[list[CauseForm], Field(min_length=1)]

    expected_positive_rate: Annotated[float, Field(gt=0.0, lt=1.0)] | None = None
    """Ожидаемая доля положительного класса по доменному знанию, до просмотра.

    Необязательно: ожидания может не быть, и выдумывать его хуже, чем не иметь.
    Но если оно есть, расхождение с наблюдаемым означает либо ошибку фильтрации
    и разметки, либо неверное понимание процесса — и то и другое дороже ошибки
    в модели. Тот же приём, что с доменным направлением признака.
    """

    def to_definition(self) -> OutcomeDefinition:
        return OutcomeDefinition(
            event_column=self.event_column,
            deadline_column=self.deadline_column,
            comparison=self.comparison,
            positive_class=self.positive_class,
            estimand=self.estimand,
            expected_positive_rate=self.expected_positive_rate,
            missing_causes=[c.to_cause() for c in self.missing_causes],
        )


class TaskForm(BaseModel):
    """Тип задачи и предпосылки."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    target_kind: TargetKind
    outcome_timing: OutcomeTiming
    has_process: bool
    is_stream: bool
    object_lifetime: ObjectLifetime
    kinds_collapsed: bool | None = None
    simultaneous_kinds: Simultaneity | None = None

    def to_spec(self) -> TaskSpec:
        return TaskSpec(
            target_kind=self.target_kind,
            outcome_timing=self.outcome_timing,
            has_process=self.has_process,
            is_stream=self.is_stream,
            object_lifetime=self.object_lifetime,
            kinds_collapsed=self.kinds_collapsed,
            simultaneous_kinds=self.simultaneous_kinds,
        )


class WindowForm(BaseModel):
    """Одно оценочное окно."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: Evidence
    start_day: int
    """Смещение от первого решения, в днях."""

    stop_day: int

    @model_validator(mode="after")
    def _ordered(self) -> WindowForm:
        if self.stop_day <= self.start_day:
            raise ValueError(f"окно {self.name!r}: конец не позже начала")
        return self


class SplitForm(BaseModel):
    """Как строится временной сплит."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    windows: Annotated[list[WindowForm], Field(min_length=1)]
    reserve_from_day: int
    """Смещение границы резерва от первого решения, в днях.

    Обязательно: измерительная выборка отрезается до начала работы, иначе она
    выбирается по уже увиденным метрикам (F-9).
    """

    @model_validator(mode="after")
    def _window_names_are_unique(self) -> SplitForm:
        """Одноимённые окна затирают друг друга в учёте выборок и в долях.

        Загрязнённое окно исчезало бы из проверки независимости только из-за
        повторённого имени.
        """
        names = [w.name for w in self.windows]
        if len(names) != len(set(names)):
            raise ValueError(f"имена окон должны быть различны, объявлено: {names}")
        return self

    @model_validator(mode="after")
    def _reserve_is_beyond_every_window(self) -> SplitForm:
        late = [w.name for w in self.windows if w.stop_day > self.reserve_from_day]
        if late:
            raise ValueError(
                f"окна {late!r} заходят за границу резерва: измерительная выборка "
                "перестала бы быть независимой"
            )
        return self

    def to_windows(self, origin: dt.datetime) -> list[Window]:
        return [
            Window(
                w.name,
                origin + dt.timedelta(days=w.start_day),
                origin + dt.timedelta(days=w.stop_day),
            )
            for w in self.windows
        ]


class AssumptionForm(BaseModel):
    """Одно допущение проекта."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    statement: Evidence
    basis: Basis
    author: Evidence
    consequence: Evidence
    evidence: str | None = None


class ProjectForm(BaseModel):
    """Проект целиком: всё, что объявляется, и ничего, что вычисляется."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    title: Evidence
    columns: Annotated[list[ColumnForm], Field(min_length=1)]
    outcome: OutcomeForm
    task: TaskForm
    split: SplitForm
    assumptions: Annotated[list[AssumptionForm], Field(min_length=1)]
    """Хотя бы одно. Проект без единого записанного допущения означает, что
    допущения принимались молча, а не что их не было."""

    def schema_spec(self) -> Schema:
        return Schema(columns=[c.to_spec() for c in self.columns])

    def registry(self) -> AssumptionRegistry:
        registry = AssumptionRegistry()

        # Допущения, принятые из-за неразличимости причин отсутствия события,
        # — такие же допущения проекта. Прежде они жили только в контракте
        # исхода и в реестр не попадали, то есть в отчёте их не было.
        for cause in self.outcome.missing_causes:
            if cause.assumption and cause.assumption.strip():
                registry.record(
                    cause.assumption,
                    basis=Basis.DOMAIN_KNOWLEDGE,
                    author="контракт исхода",
                    consequence=(
                        f"причина {cause.name!r} неотличима по данным, и её строки "
                        f"размечаются как {cause.meaning.value}"
                    ),
                )

        for item in self.assumptions:
            registry.record(
                item.statement,
                basis=item.basis,
                author=item.author,
                consequence=item.consequence,
                evidence=item.evidence,
            )
        return registry


def load(path: Path) -> ProjectForm:
    """Прочитать форму. Ошибка в объявлениях — отказ здесь, а не позже."""
    return ProjectForm(**yaml.safe_load(path.read_text(encoding="utf-8")))
