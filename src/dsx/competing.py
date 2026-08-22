"""Конкурирующие исходы: несколько видов события, наступает не более одного.

Остаток F-8. Во втором кейсе отказывал один из четырёх компонентов, и я свёл
их в исход «отказало хоть что-то». Ядро не возразило, потому что о видах
событий не знало.

Главное здесь не многоклассовость, а ЦЕНЗУРА. Отказ первого компонента не
является отрицательным исходом для второго: машину остановили и починили, и
что случилось бы со вторым, неизвестно. Разметить такую строку нулём значит
объявить наблюдением то, чего не наблюдали.

Поэтому исход каждого вида считается отдельно, а строка, где раньше наступил
конкурирующий вид, получает не ноль, а пустоту — тот же смысл, что
`MissingEventMeaning.UNOBSERVED`.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from dsx.evals.world import World
from dsx.label import LABEL, compute
from dsx.outcome import OutcomeDefinition
from dsx.roles import Role


class CompetingError(Exception):
    """Конкурирующие исходы объявлены несогласованно с данными."""


@dataclass(frozen=True)
class KindOutcome:
    """Исход одного вида вместе с ценой, которую он взял с остальных."""

    kind: str
    frame: pl.DataFrame
    censored_by_others: int
    """Строки, где раньше наступил конкурирующий вид: исход неизвестен."""

    @property
    def positives(self) -> int:
        return int(self.frame[LABEL].fill_null(0).sum())

    @property
    def observable(self) -> int:
        return int(self.frame[LABEL].is_not_null().sum())


def kinds_in(world: World) -> tuple[str, ...]:
    """Виды событий, встречающиеся в данных."""
    columns = world.schema.by_role(Role.EVENT_KIND)
    if not columns:
        return ()
    name = columns[0].name
    if name not in world.main.columns:
        return ()
    return tuple(sorted(world.main[name].drop_nulls().unique().to_list()))


def compute_by_kind(
    world: World, definition: OutcomeDefinition, exclude_kind: str | None = None
) -> list[KindOutcome]:
    """Разметить исход отдельно по каждому виду события.

    Строка, где наступил другой вид, для данного вида не отрицательна, а
    ненаблюдаема: конкурирующее событие оборвало наблюдение.
    """
    columns = world.schema.by_role(Role.EVENT_KIND)
    if not columns:
        raise CompetingError(
            "конкурирующие исходы объявлены, но колонка вида события не названа: различать нечего"
        )
    kind_column = columns[0].name
    kinds = kinds_in(world)
    if len(kinds) < 2:
        raise CompetingError(
            f"вид события {kind_column!r} принимает {len(kinds)} значений: "
            "конкуренции нет, и задача бинарная"
        )

    base = compute(world, definition)
    if exclude_kind is not None:
        # Одновременный случай объявлен выводимым из популяции: строки уходят
        # целиком, а не размечаются произвольным видом.
        base = base.filter(pl.col(kind_column).is_null() | (pl.col(kind_column) != exclude_kind))
        kinds = tuple(k for k in kinds if k != exclude_kind)

    results = []
    for kind in kinds:
        # Событие другого вида обрывает наблюдение, только если случилось ДО
        # конца срока. Конкурирующий отказ после горизонта наблюдению не мешает:
        # к концу срока уже видно, что целевого отказа не было.
        censored = (
            pl.col(kind_column).is_not_null()
            & (pl.col(kind_column) != kind)
            & (pl.col(definition.event_column) <= pl.col(definition.deadline_column))
        )
        marked = base.with_columns(
            pl.when(censored).then(None).otherwise(pl.col(LABEL)).alias(LABEL)
        )
        results.append(
            KindOutcome(
                kind=kind,
                frame=marked,
                censored_by_others=int(base.select(censored.sum()).item()),
            )
        )
    return results


def collapse_cost(world: World, definition: OutcomeDefinition) -> str:
    """Что теряется при сведении видов в один исход, словами и числами."""
    outcomes = compute_by_kind(world, definition)
    lines = [
        f"  {o.kind}: положительных {o.positives:,}, наблюдаемых {o.observable:,}, "
        f"оборвано конкурентами {o.censored_by_others:,}"
        for o in outcomes
    ]
    return "\n".join(lines)
