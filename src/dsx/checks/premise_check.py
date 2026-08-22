"""Проверка объявленных предпосылок (F-4)."""

from __future__ import annotations

from dataclasses import dataclass

from dsx.checks.base import Context, Signal
from dsx.evals.case import Finding
from dsx.premises import verify
from dsx.task import Premise


@dataclass(frozen=True)
class PremisesMatchData:
    """Объявленные предпосылки расходятся с данными.

    Предпосылка выключает проверки. Объявление, не сверенное с данными,
    позволяет выключить их ради тишины — и доказать обратное невозможно.
    """

    requirement: str = "S6"
    premises: frozenset[Premise] = frozenset({Premise.UNIVERSAL})
    detects: frozenset[Finding] = frozenset({Finding.PREMISE_MISMATCH})

    def run(self, context: Context) -> list[Signal]:
        return [
            Signal(
                Finding.PREMISE_MISMATCH,
                f"{discrepancy}. Выключенные этой предпосылкой проверки "
                "не проводились без основания",
                blocking=True,
            )
            for discrepancy in verify(context.world, context.task)
        ]


PREMISE_CHECKS = [PremisesMatchData()]
