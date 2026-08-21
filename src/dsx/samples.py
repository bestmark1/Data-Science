"""Журнал расхода выборок.

Самая дорогая ошибка этапа 0 сидела здесь. Тест использовался для выбора окна
обучения, кандидата, калибровки и порога, после чего на нём же считались
итоговые метрики. Заключение перевернулось трижды, и каждый раз из-за
протокола, а не новых данных.

Выборка расходуется. Обращение к ней ради выбора решения тратит её как
измерительный инструмент: после этого измерение на ней смещено. Журнал делает
расход видимым, потому что незаписанный расход неотличим от его отсутствия.
"""

from __future__ import annotations

import datetime as dt
from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from dsx.policy import Blocked, OverrideLedger


class Purpose(StrEnum):
    """Зачем к выборке обращались."""

    FITTING = "fitting"
    """Обучение. Расходует выборку как обучающую, но не как измерительную."""

    SELECTION = "selection"
    """Выбор решения: окно, кандидат, гиперпараметры, калибровка, порог.
    Расходует выборку как измерительную."""

    MEASUREMENT = "measurement"
    """Итоговое измерение. Допустимо один раз и только на нерасходованной
    выборке."""


class Access(BaseModel):
    """Одно обращение к выборке."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sample: Annotated[str, Field(min_length=1)]
    purpose: Purpose
    decision: Annotated[str, Field(min_length=1)]
    """Какое решение принималось. Обращение без названного решения не
    отличается от необъяснимого расхода."""

    at: dt.datetime = Field(default_factory=lambda: dt.datetime.now(dt.UTC))

    def __str__(self) -> str:
        return f"{self.sample} / {self.purpose.value}: {self.decision}"


class SampleLedger:
    """Учёт обращений к выборкам (P1, P2, P7)."""

    def __init__(self, ledger: OverrideLedger | None = None) -> None:
        self._accesses: list[Access] = []
        self._known: set[str] = set()
        self._overrides = ledger or OverrideLedger()

    # --- регистрация ------------------------------------------------------

    def register(self, *names: str) -> None:
        """Объявить существующие выборки. Обращение к незарегистрированной
        выборке — ошибка: имя, придуманное на ходу, обходит учёт."""
        self._known.update(names)

    def _require_known(self, sample: str) -> None:
        if sample not in self._known:
            raise Blocked(
                "P2",
                f"выборка {sample!r} не зарегистрирована; "
                f"известны: {', '.join(sorted(self._known)) or 'ни одной'}",
            )

    # --- обращения --------------------------------------------------------

    def fit(self, sample: str, decision: str) -> None:
        self._require_known(sample)
        self._accesses.append(Access(sample=sample, purpose=Purpose.FITTING, decision=decision))

    def select(self, sample: str, decision: str) -> None:
        """Зафиксировать выбор решения по выборке.

        После итогового измерения любой новый выбор требует свежей выборки:
        протокол, изменённый после просмотра метрик, обесценивает измерение (P7).
        """
        self._require_known(sample)
        if self.was_measured(sample):
            self._overrides.enforce(
                "P7",
                f"на выборке {sample!r} уже проведено итоговое измерение; "
                "решение, принятое после просмотра метрик, требует свежей выборки",
            )
        self._accesses.append(Access(sample=sample, purpose=Purpose.SELECTION, decision=decision))

    def measure(self, sample: str, decision: str = "итоговая оценка") -> None:
        """Зафиксировать итоговое измерение.

        Блокируется, если выборка уже расходовалась на выбор решений (P1) либо
        измерялась ранее.
        """
        self._require_known(sample)

        spent = self.selections(sample)
        if spent:
            self._overrides.enforce(
                "P1",
                f"выборка {sample!r} уже участвовала в выборе решений "
                f"({len(spent)}): {'; '.join(a.decision for a in spent)}. "
                "Измерение на ней смещено",
            )
        if self.was_measured(sample):
            self._overrides.enforce(
                "P1", f"на выборке {sample!r} итоговое измерение уже проводилось"
            )

        self._accesses.append(Access(sample=sample, purpose=Purpose.MEASUREMENT, decision=decision))

    # --- состояние --------------------------------------------------------

    def selections(self, sample: str) -> tuple[Access, ...]:
        return tuple(
            a for a in self._accesses if a.sample == sample and a.purpose is Purpose.SELECTION
        )

    def was_measured(self, sample: str) -> bool:
        return any(a.sample == sample and a.purpose is Purpose.MEASUREMENT for a in self._accesses)

    def is_spent(self, sample: str) -> bool:
        """Израсходована ли выборка как измерительный инструмент."""
        return bool(self.selections(sample)) or self.was_measured(sample)

    def unspent(self) -> tuple[str, ...]:
        return tuple(sorted(n for n in self._known if not self.is_spent(n)))

    @property
    def accesses(self) -> tuple[Access, ...]:
        return tuple(self._accesses)

    # --- отчёт ------------------------------------------------------------

    def report_section(self) -> str:
        """Раздел отчёта: на чём принято каждое решение (P2)."""
        lines = ["## Расход выборок", ""]
        if not self._accesses:
            lines.append("Обращений не зафиксировано.")
            return "\n".join(lines)

        lines += ["| выборка | назначение | решение |", "|---|---|---|"]
        for access in self._accesses:
            lines.append(f"| {access.sample} | {access.purpose.value} | {access.decision} |")

        remaining = self.unspent()
        lines.append("")
        lines.append("Нерасходованные выборки: " + (", ".join(remaining) if remaining else "нет"))
        return "\n".join(lines)
