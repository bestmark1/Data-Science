"""Проверки объявленного контракта.

Все универсальны: они смотрят на объявления, а не на данные, и потому не
зависят ни от типа задачи, ни от наличия процесса.
"""

from __future__ import annotations

from dataclasses import dataclass

from dsx.checks.base import Context, Signal
from dsx.evals.case import Finding
from dsx.outcome import ComparisonMode
from dsx.roles import Availability, Role, TemporalKind
from dsx.task import Premise

UNIVERSAL = frozenset({Premise.UNIVERSAL})


@dataclass(frozen=True)
class MixedTemporalComparison:
    """A10. Сравнение даты с моментом времени в определении исхода.

    На этапе 0 это пометило поздними 1292 события, случившиеся в назначенный
    день — 16.5% положительного класса. Ошибка невидима глазами: обе колонки
    выглядят как даты.
    """

    requirement: str = "A10"
    premises: frozenset[Premise] = UNIVERSAL
    detects: frozenset[Finding] = frozenset({Finding.MIXED_TEMPORAL_COMPARISON})

    def run(self, context: Context) -> list[Signal]:
        if context.outcome.comparison is not ComparisonMode.DIRECT:
            return []

        schema = context.world.schema
        left = schema.get(context.outcome.event_column)
        right = schema.get(context.outcome.deadline_column)
        if left is None or right is None:
            return []
        if left.temporal is None or right.temporal is None or left.temporal is right.temporal:
            return []

        date_side = left if left.temporal is TemporalKind.DATE else right
        instant_side = right if date_side is left else left
        return [
            Signal(
                Finding.MIXED_TEMPORAL_COMPARISON,
                f"{date_side.name!r} хранит дату, {instant_side.name!r} — момент времени. "
                "Прямое сравнение пометит событие в тот же день как произошедшее позже.",
                blocking=True,
            )
        ]


@dataclass(frozen=True)
class UndeclaredTemporalKind:
    """A10. Колонка исхода без объявленной грануляции.

    Угадывать грануляцию эвристикой нельзя: эвристика даёт ложные срабатывания,
    которые пользователь научится затыкать.
    """

    requirement: str = "A10"
    premises: frozenset[Premise] = UNIVERSAL
    detects: frozenset[Finding] = frozenset({Finding.UNDECLARED_TEMPORAL_KIND})

    def run(self, context: Context) -> list[Signal]:
        schema = context.world.schema
        signals = []
        for name in sorted(context.outcome.components()):
            column = schema.get(name)
            if column is None or column.temporal is not None:
                continue
            signals.append(
                Signal(
                    Finding.UNDECLARED_TEMPORAL_KIND,
                    f"колонка {name!r} участвует в вычислении исхода, "
                    "но не объявила временную грануляцию",
                    blocking=True,
                )
            )
        return signals


@dataclass(frozen=True)
class OutcomeComponentAsFeature:
    """C6. Колонка, участвующая в вычислении метки, объявлена признаком.

    Единственный вид лика, который проверки данных увидеть не могут в принципе:
    связь возникает в формуле исхода, а не в данных.
    """

    requirement: str = "C6"
    premises: frozenset[Premise] = UNIVERSAL
    detects: frozenset[Finding] = frozenset({Finding.OUTCOME_COMPONENT_AS_FEATURE})

    def run(self, context: Context) -> list[Signal]:
        components = context.outcome.components()
        return [
            Signal(
                Finding.OUTCOME_COMPONENT_AS_FEATURE,
                f"колонка {column.name!r} участвует в вычислении исхода и не может быть признаком",
                blocking=True,
            )
            for column in context.world.schema.by_role(Role.FEATURE)
            if column.name in components
        ]


@dataclass(frozen=True)
class FeatureAfterDecision:
    """C1. Признак объявлен доступным после момента решения."""

    requirement: str = "C1"
    premises: frozenset[Premise] = UNIVERSAL
    detects: frozenset[Finding] = frozenset({Finding.FEATURE_AFTER_DECISION})

    def run(self, context: Context) -> list[Signal]:
        return [
            Signal(
                Finding.FEATURE_AFTER_DECISION,
                f"признак {column.name!r} появляется после момента решения",
                blocking=True,
            )
            for column in context.world.schema.by_role(Role.FEATURE)
            if column.availability is Availability.AFTER
        ]


@dataclass(frozen=True)
class UndeclaredAvailability:
    """C1, C2. Признак без объявленной доступности.

    Не блокирует: такие признаки просто не попадают в выборку, а вопрос о них
    уходит владельцу данных.
    """

    requirement: str = "C2"
    premises: frozenset[Premise] = UNIVERSAL
    detects: frozenset[Finding] = frozenset({Finding.UNDECLARED_AVAILABILITY})

    def run(self, context: Context) -> list[Signal]:
        return [
            Signal(
                Finding.UNDECLARED_AVAILABILITY,
                f"доступность признака {column.name!r} не объявлена — вопрос владельцу данных",
            )
            for column in context.world.schema.by_role(Role.FEATURE)
            if column.availability is Availability.UNKNOWN
        ]


CONTRACT_CHECKS = [
    MixedTemporalComparison(),
    UndeclaredTemporalKind(),
    OutcomeComponentAsFeature(),
    FeatureAfterDecision(),
    UndeclaredAvailability(),
]
