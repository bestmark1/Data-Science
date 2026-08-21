"""Проверки, требующие построенного временного сплита."""

from __future__ import annotations

from dataclasses import dataclass

from dsx.checks.base import Context, NotApplicable, Signal
from dsx.evals.case import Finding
from dsx.split import SplitResult, entity_overlap
from dsx.task import Premise


def _require_split(context: Context) -> SplitResult:
    if not isinstance(context.split, SplitResult):
        raise NotApplicable("временной сплит не построен")
    return context.split


@dataclass(frozen=True)
class EntityOverlapAcrossSplits:
    """N2. Один объект и в обучении, и в оценке.

    При временном сплите это означает, что модель видела объект и оценивается
    на нём же: оценка завышена, и завышена незаметно.
    """

    requirement: str = "N2"
    premises: frozenset[Premise] = frozenset({Premise.UNIVERSAL})
    detects: frozenset[Finding] = frozenset({Finding.ENTITY_OVERLAP_ACROSS_SPLITS})

    def run(self, context: Context) -> list[Signal]:
        split = _require_split(context)
        overlaps = entity_overlap(split.parts, context.world)
        return [
            Signal(
                Finding.ENTITY_OVERLAP_ACROSS_SPLITS,
                f"в окне {name!r} {count:,} объектов присутствуют и в обучении, "
                "и в оценке: модель оценивается на том, что видела",
                blocking=True,
            )
            for name, count in sorted(overlaps.items())
        ]


@dataclass(frozen=True)
class LabelImmaturity:
    """A12. Исход части объектов оценочного окна ещё не наблюдаем.

    Если такие объекты просто отбросить, окно сместится в сторону тех, чей срок
    короче, и оценка перестанет описывать популяцию.
    """

    requirement: str = "A12"
    premises: frozenset[Premise] = frozenset({Premise.DELAYED_OUTCOME})
    detects: frozenset[Finding] = frozenset({Finding.LABEL_IMMATURITY})
    tolerance: float = 0.01

    def run(self, context: Context) -> list[Signal]:
        split = _require_split(context)
        signals = []
        for part in split.parts:
            immature = split.immature.get(part.name, 0)
            if not immature or not part.evaluate.height:
                continue
            share = immature / part.evaluate.height
            if share < self.tolerance:
                continue
            signals.append(
                Signal(
                    Finding.LABEL_IMMATURITY,
                    f"в окне {part.name!r} исход {immature:,} объектов "
                    f"({share:.1%}) не наблюдаем к концу наблюдения: отбросив их, "
                    "получим окно из объектов с короткими сроками",
                    blocking=True,
                )
            )
        return signals


SPLIT_CHECKS = [EntityOverlapAcrossSplits(), LabelImmaturity()]
