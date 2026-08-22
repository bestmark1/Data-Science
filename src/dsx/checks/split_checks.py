"""Проверки, требующие построенного временного сплита."""

from __future__ import annotations

from dataclasses import dataclass

from dsx.checks.base import Context, NotApplicable, Signal
from dsx.evals.case import Finding
from dsx.roles import Role
from dsx.split import SplitResult, entity_overlap
from dsx.task import ObjectLifetime, Premise


def _require_split(context: Context) -> SplitResult:
    if not isinstance(context.split, SplitResult):
        raise NotApplicable("временной сплит не построен")
    return context.split


@dataclass(frozen=True)
class EntityOverlapAcrossSplits:
    """N2. Один и тот же объект и в обучении, и в оценке.

    Проверка выведена на заказах, где объект одноразов: его появление по обе
    стороны сплита означало утечку. Второй кейс показал скрытое допущение —
    сотня машин, наблюдаемых полтора года, даёт полное пересечение при любом
    сплите, и это повторные измерения, а не утечка.

    Поэтому предмет проверки зависит от жизненного цикла. У одноразового
    объекта утечка — присутствие объекта. У долгоживущего — повтор самой
    единицы решения; присутствие объекта нормально.
    """

    requirement: str = "N2"
    premises: frozenset[Premise] = frozenset({Premise.UNIVERSAL})
    detects: frozenset[Finding] = frozenset({Finding.ENTITY_OVERLAP_ACROSS_SPLITS})

    def run(self, context: Context) -> list[Signal]:
        split = _require_split(context)
        recurring = context.task.object_lifetime is ObjectLifetime.RECURRING

        if not recurring:
            # Жизненный цикл не объявлен или объект одноразов. Неизвестность
            # трактуется строго: объявить её обязывает A2.
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

        overlaps = entity_overlap(split.parts, context.world, role=Role.ENTITY_ID)
        return [
            Signal(
                Finding.ENTITY_OVERLAP_ACROSS_SPLITS,
                f"в окне {name!r} {count:,} ЕДИНИЦ РЕШЕНИЯ повторяются в обучении "
                "и в оценке. Объект долгоживущий, и его присутствие по обе стороны "
                "нормально, но одно и то же решение оценивается на себе же",
                blocking=True,
            )
            for name, count in sorted(overlaps.items())
        ]


@dataclass(frozen=True)
class ObservationIntervalOverlap:
    """N2i. Пересечение интервалов наблюдения у долгоживущего объекта.

    Для машины, клиента или пациента присутствие по обе стороны сплита
    неизбежно, и настоящей утечкой является пересечение интервалов, из которых
    построены признаки. Проверить это нечем: окна признаков нигде не
    объявляются.

    Проверка существует, чтобы разрыв был виден. Ноль блокирующих сигналов на
    задаче с долгоживущими объектами иначе читался бы как отсутствие утечки,
    хотя означает лишь, что её не искали.
    """

    requirement: str = "N2i"
    premises: frozenset[Premise] = frozenset({Premise.UNIVERSAL})
    detects: frozenset[Finding] = frozenset({Finding.ENTITY_OVERLAP_ACROSS_SPLITS})

    def run(self, context: Context) -> list[Signal]:
        _require_split(context)
        if context.task.object_lifetime is not ObjectLifetime.RECURRING:
            raise NotApplicable("объект одноразов: пересечение объектов уже проверено N2")
        raise NotApplicable(
            "объект долгоживущий, и утечкой является пересечение интервалов "
            "наблюдения, а не присутствие объекта. Окна признаков нигде не "
            "объявляются, поэтому проверить это нечем — утечка не исключена"
        )


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


SPLIT_CHECKS = [EntityOverlapAcrossSplits(), ObservationIntervalOverlap(), LabelImmaturity()]
