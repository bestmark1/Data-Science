"""Проверки объявленного контракта.

Все универсальны: они смотрят на объявления, а не на данные, и потому не
зависят ни от типа задачи, ни от наличия процесса.
"""

from __future__ import annotations

from dataclasses import dataclass

from dsx.checks.base import Context, Signal
from dsx.evals.case import Finding
from dsx.outcome import ComparisonMode
from dsx.roles import Availability, Role, TemporalKind, coarser
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
        schema = context.world.schema
        left = schema.get(context.outcome.event_column)
        right = schema.get(context.outcome.deadline_column)
        if left is None or right is None:
            return []
        if left.temporal is None or right.temporal is None:
            return []

        if left.temporal is right.temporal:
            # Одинаково грубые стороны не смешиваются, но месяц остаётся месяцем:
            # неопределённость никуда не делась, она лишь одинакова с обеих
            # сторон. Сигнал не блокирующий — запрещать тут нечего, — но
            # читатель обязан знать, что у числа есть месяц люфта.
            if left.temporal is TemporalKind.MONTH:
                return [
                    Signal(
                        Finding.MIXED_TEMPORAL_COMPARISON,
                        f"{left.name!r} и {right.name!r} известны с точностью до месяца. "
                        "Сравнение честное, но исход у части строк зависит от того, каким "
                        "днём развёрнут месяц: у полученного числа есть люфт до месяца",
                        blocking=False,
                    )
                ]
            return []

        rough = coarser(left.temporal, right.temporal)
        fine = right if rough is left.temporal else left
        rough_side = left if rough is left.temporal else right

        # Приведение к дате лечит расхождение дата/момент и только его. Месяц
        # оно не лечит: развернуть «2015-10» в дату можно четырьмя способами, и
        # на девятом кейсе выбор менял долю уложившихся с 26.3% до 36.6%.
        if context.outcome.comparison is ComparisonMode.BY_DATE and rough is not TemporalKind.MONTH:
            return []

        if rough is TemporalKind.MONTH:
            detail = (
                f"{rough_side.name!r} известна с точностью до месяца, а {fine.name!r} — "
                f"до {fine.temporal.value}. Развернуть месяц в дату можно по-разному, и "
                "выбор соглашения меняет исход у всех строк с месячной точностью. "
                "Приведение к дате этого не лечит"
            )
        else:
            detail = (
                f"{rough_side.name!r} хранит дату, {fine.name!r} — момент времени. "
                "Прямое сравнение пометит событие в тот же день как произошедшее позже."
            )
        return [Signal(Finding.MIXED_TEMPORAL_COMPARISON, detail, blocking=True)]


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
