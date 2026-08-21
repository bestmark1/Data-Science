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


@runtime_checkable
class Check(Protocol):
    """Контракт проверки."""

    requirement: str
    """Идентификатор требования спецификации: A8, C6, P1."""

    premises: frozenset[Premise]
    detects: frozenset[Finding]

    def run(self, context: Context) -> list[Signal]: ...


@dataclass(frozen=True)
class Skipped:
    """Проверка не применялась, и известно почему."""

    requirement: str
    unmet: frozenset[Premise]

    def __str__(self) -> str:
        names = ", ".join(sorted(p.value for p in self.unmet))
        return f"{self.requirement}: пропущена, не выполнены предпосылки — {names}"


@dataclass
class Report:
    """Результат прогона набора проверок."""

    signals: list[Signal] = field(default_factory=list)
    skipped: list[Skipped] = field(default_factory=list)
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
            report.skipped.append(Skipped(check.requirement, unmet))
            continue

        for signal in check.run(context):
            if signal.blocking and ledger.is_overridden(check.requirement):
                signal = Signal(signal.finding, signal.detail, blocking=False)
            report.signals.append(signal)

    report.overrides = ledger.entries
    return report
