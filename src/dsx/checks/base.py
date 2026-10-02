"""Каркас проверок.

Проверка объявляет три вещи: какой дефект ищет, при каких предпосылках
осмысленна и блокирует ли работу. Первое даёт механический вердикт на
проверочных кейсах, второе выключает её на задаче другого типа, третье
связывает её с журналом обходов.

Проверка, не объявившая предпосылок, считается универсальной — и это самое
опасное умолчание, поэтому оно требует явного `Premise.UNIVERSAL`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from dsx.evals.case import Finding
from dsx.evals.world import World
from dsx.outcome import OutcomeDefinition
from dsx.policy import OverrideLedger
from dsx.task import Premise, TaskSpec


@dataclass(frozen=True)
class Signal:
    """Одно срабатывание проверки."""

    finding: Finding
    detail: str
    blocking: bool = False

    def __str__(self) -> str:
        mark = "БЛОК" if self.blocking else "внимание"
        return f"[{mark}] {self.finding.value}: {self.detail}"


@dataclass(frozen=True)
class Context:
    """Всё, что проверке разрешено видеть."""

    world: World
    outcome: OutcomeDefinition
    task: TaskSpec
    split: object | None = None
    """Результат временного сплита, если он построен. Проверки, которым он
    нужен, отказываются выполняться без него по названной причине."""

    observed_until: object | None = None
    """Конец наблюдения, объявленный формой.

    Нужен там, где отсутствие события надо отличить от НЕ НАСТУПИВШЕГО СРОКА.
    Строка, чей срок ещё не истёк, молчит законно, и складывать её с той, у
    которой событие перестали записывать, значит смешивать созревание с
    порчей данных.

    Проверка, которой он нужен, отказывается работать без него по названной
    причине: вывести его из максимума даты события нельзя — это последнее
    СЛУЧИВШЕЕСЯ событие, а не конец сбора.
    """


@runtime_checkable
class Check(Protocol):
    """Контракт проверки."""

    requirement: str
    """Идентификатор требования спецификации: A8, C6, P1."""

    premises: frozenset[Premise]
    detects: frozenset[Finding]

    def run(self, context: Context) -> list[Signal]: ...


class NotApplicable(Exception):
    """Проверка не может быть выполнена по названной причине.

    Отличается от невыполненной предпосылки: предпосылка известна заранее, а
    это выясняется в ходе работы — например, когда контракт исхода нарушен и
    метку вычислить нечем.
    """


@dataclass(frozen=True)
class Skipped:
    """Проверка не применялась, и известно почему."""

    requirement: str
    reason: str
    unmet: frozenset[Premise] = frozenset()

    def __str__(self) -> str:
        return f"{self.requirement}: пропущена — {self.reason}"

    @classmethod
    def for_premises(cls, requirement: str, unmet: frozenset[Premise]) -> Skipped:
        names = ", ".join(sorted(p.value for p in unmet))
        return cls(requirement, f"не выполнены предпосылки: {names}", unmet)


@dataclass
class Report:
    """Результат прогона набора проверок."""

    signals: list[Signal] = field(default_factory=list)
    skipped: list[Skipped] = field(default_factory=list)
    limits: list[str] = field(default_factory=list)
    """Пределы молчания: чего промолчавшая проверка не заметила бы на этих данных.

    DS-007: N4 видит смену знака связи лишь сильнее порога шума окна, и её
    молчание на окнах в 300 строк не доказывало устойчивости (класс 18 журнала
    повторов). Предел — не находка: с ожиданием кейса он не сверяется."""

    overrides: tuple = ()

    @property
    def findings(self) -> frozenset[Finding]:
        """То, что сравнивается с ожиданием кейса."""
        return frozenset(s.finding for s in self.signals)

    @property
    def blocking(self) -> list[Signal]:
        return [s for s in self.signals if s.blocking]


def run_checks(
    checks: list[Check],
    context: Context,
    ledger: OverrideLedger | None = None,
) -> Report:
    """Прогнать проверки, выключая неприменимые и уважая журнал обходов.

    Пропущенная проверка не исчезает: она попадает в отчёт с указанием
    невыполненных предпосылок. Молча выключенная проверка неотличима от
    проверки, которая ничего не нашла.
    """
    ledger = ledger or OverrideLedger()
    report = Report()

    for check in checks:
        unmet = context.task.unmet(check.premises)
        if unmet:
            report.skipped.append(Skipped.for_premises(check.requirement, unmet))
            continue

        try:
            produced = check.run(context)
        except NotApplicable as exc:
            report.skipped.append(Skipped(check.requirement, str(exc)))
            continue

        for signal in produced:
            if signal.blocking and ledger.is_overridden(check.requirement):
                signal = Signal(signal.finding, signal.detail, blocking=False)
            report.signals.append(signal)

        # Проверка, умеющая назвать предел своего молчания, называет его, только
        # когда промолчала: сработавшей проверке он не нужен.
        silence = getattr(check, "silence", None)
        if not produced and silence is not None:
            report.limits.append(f"{check.requirement}: {silence(context)}")

    report.overrides = ledger.entries
    return report
