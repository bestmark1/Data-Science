"""Временной сплит по моменту узнавания исхода.

Обучение видит не всё, что случилось до отсечки, а только то, чей исход был
ИЗВЕСТЕН к ней. Разница существенна: на этапе 0 между выборками выпало 3342
объекта, купленных до отсечки, но узнанных после. Использовать их в обучении
значило бы знать будущее.

Момент узнавания выводится из определения исхода, а не объявляется отдельно.
Для исхода вида «событие не произошло к сроку» ответ известен в конце срока —
ждать фактического наступления не нужно. На этапе 0 это сократило зазор с 29
дней до 26, а максимум — с 209 до 145.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import polars as pl

from dsx.evals.world import World
from dsx.label import LABEL, compute, observable
from dsx.outcome import OutcomeDefinition
from dsx.roles import Role

KNOWN_AT = "__label_known_at"


def with_label_known_at(
    world: World, definition: OutcomeDefinition, snapshot: dt.datetime | None = None
) -> pl.DataFrame:
    """Добавить момент узнавания исхода к размеченной таблице.

    Исход известен в момент события, если оно произошло в срок; иначе — в конце
    срока, когда стало ясно, что событие не успело.
    """
    frame = compute(world, definition, snapshot)
    event = pl.col(definition.event_column)
    deadline = pl.col(definition.deadline_column)

    return frame.with_columns(
        pl.when(event.is_not_null() & (event.dt.date() <= deadline.dt.date()))
        .then(event)
        .otherwise(deadline.dt.offset_by("1d"))
        .alias(KNOWN_AT)
    )


@dataclass(frozen=True)
class Window:
    """Оценочное окно."""

    name: str
    start: dt.datetime
    stop: dt.datetime


@dataclass
class Part:
    """Одна часть сплита: чему учимся и на чём оцениваемся."""

    window: Window
    train: pl.DataFrame
    evaluate: pl.DataFrame

    @property
    def name(self) -> str:
        return self.window.name


@dataclass
class SplitResult:
    parts: list[Part] = field(default_factory=list)
    dropped_not_yet_known: int = 0
    """Объекты, решение по которым принято до отсечки, а исход стал известен
    после. В обучение попасть не могут."""

    immature: dict[str, int] = field(default_factory=dict)
    """Объекты оценочного окна, чей исход не наблюдаем к концу наблюдения."""

    def part(self, name: str) -> Part:
        return next(p for p in self.parts if p.name == name)


def split_by_windows(
    world: World,
    definition: OutcomeDefinition,
    windows: list[Window],
    snapshot: dt.datetime,
) -> SplitResult:
    """Построить обучающие и оценочные части по временным окнам.

    Для каждого окна обучение — объекты, чей исход был известен до его начала.
    Оценка — объекты, решение по которым принято внутри окна.
    """
    labelled = with_label_known_at(world, definition, snapshot)
    decision = world.schema.decision_time.name
    result = SplitResult()

    for window in windows:
        train = labelled.filter(pl.col(LABEL).is_not_null() & (pl.col(KNOWN_AT) < window.start))
        evaluate = labelled.filter(
            (pl.col(decision) >= window.start) & (pl.col(decision) < window.stop)
        )
        result.parts.append(Part(window=window, train=train, evaluate=evaluate))

        # Незрелость — это пустая метка в окне: исход ещё не наблюдаем.
        immature = evaluate.filter(pl.col(LABEL).is_null()).height
        if immature:
            result.immature[window.name] = immature

    if windows:
        first = windows[0]
        before_cutoff = labelled.filter(pl.col(decision) < first.start)
        result.dropped_not_yet_known = before_cutoff.filter(pl.col(KNOWN_AT) >= first.start).height

    return result


def entity_overlap(parts: list[Part], world: World) -> dict[str, int]:
    """Объекты, встречающиеся и в обучении, и в оценке одного окна.

    Пересечение сущностей при временном сплите означает, что модель видела тот
    же объект и оценивается на нём же.
    """
    keys = world.schema.by_role(Role.NATURAL_KEY) or world.schema.by_role(Role.ENTITY_ID)
    if not keys:
        return {}
    key = keys[0].name

    overlaps: dict[str, int] = {}
    for part in parts:
        if key not in part.train.columns or key not in part.evaluate.columns:
            continue
        shared = set(part.train[key].unique().to_list()) & set(
            part.evaluate[key].unique().to_list()
        )
        if shared:
            overlaps[part.name] = len(shared)
    return overlaps


def positive_rates(parts: list[Part]) -> dict[str, float]:
    """Доля положительного класса по окнам.

    На этапе 0 она менялась от 1.95% до 20.8%, и это ломало и калибровку, и
    выбор порога независимо от качества модели.
    """
    return {
        part.name: float(observable(part.evaluate)[LABEL].mean())
        for part in parts
        if observable(part.evaluate).height
    }
