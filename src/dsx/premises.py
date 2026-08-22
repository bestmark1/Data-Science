"""Сверка объявленных предпосылок с данными.

Второй кейс показал дыру: предпосылки объявляет сам аналитик, и никто их не
проверяет. Объявив отсутствие процесса, я выключил две проверки и не смог бы
доказать, что сделал это по существу, а не ради тишины.

Тот же дефект, что с выдуманной колонкой статуса. Механизм, удовлетворяемый
декларацией, которую никто не сверяет, защищает хуже своего отсутствия: он
создаёт видимость проверки.

Сверяется не всё: предпосылка о процессе и о потоке выводима из данных,
предпосылка о типе таргета — нет. Невыводимое остаётся объявлением, и это
сказано прямо, а не замаскировано.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from dsx.evals.world import World
from dsx.roles import Role
from dsx.task import Premise, TaskSpec


@dataclass(frozen=True)
class Discrepancy:
    """Объявление разошлось с данными."""

    premise: Premise
    declared: bool
    observed: bool
    detail: str

    def __str__(self) -> str:
        return (
            f"{self.premise.value}: объявлено {self.declared}, "
            f"по данным {self.observed} — {self.detail}"
        )


VERIFIABLE = frozenset({Premise.PROCESS, Premise.STREAM})
"""Предпосылки, выводимые из данных. Остальные остаются объявлениями."""


def _process_observed(world: World) -> tuple[bool, str]:
    """Есть ли в данных событийный процесс со статусами.

    Критерий — присутствие объявленной колонки статуса, а не число её значений.
    Первая версия требовала минимум двух значений и перестала видеть конфликт
    статуса с событием там, где статус один: процесс существует, просто
    состояние не меняется, и проверка согласованности остаётся осмысленной.
    """
    statuses = world.schema.by_role(Role.STATUS)
    if not statuses:
        return False, "колонка статуса не объявлена"

    column = statuses[0].name
    if column not in world.main.columns:
        return False, f"колонка {column!r} объявлена статусом, но отсутствует в данных"

    distinct = world.main[column].n_unique()
    return True, f"статус {column!r} присутствует, значений: {distinct}"


def _stream_observed(world: World, gap_share: float = 0.1) -> tuple[bool, str]:
    """Собираются ли наблюдения регулярным потоком.

    Признак потока — покрытие календаря: наблюдения есть в большинстве дней
    периода. Событийные решения оставляют календарь дырявым.
    """
    column = world.schema.decision_time.name
    if column not in world.main.columns:
        return False, f"колонка решения {column!r} отсутствует в данных"

    daily = world.main.select(pl.col(column).dt.date().alias("day")).group_by("day").agg(pl.len())
    lo, hi = world.main[column].min(), world.main[column].max()
    span_days = max((hi - lo).days + 1, 1)
    coverage = daily.height / span_days

    if coverage >= 1 - gap_share:
        return True, f"наблюдения есть в {coverage:.0%} дней периода"
    return False, f"наблюдения есть лишь в {coverage:.0%} дней периода"


def verify(world: World, task: TaskSpec) -> list[Discrepancy]:
    """Сверить выводимые предпосылки с данными.

    Возвращает расхождения. Пустой список означает, что объявленное совпало с
    наблюдаемым — но только по выводимым предпосылкам.
    """
    checks = {
        Premise.PROCESS: (_process_observed(world), task.has_process),
        Premise.STREAM: (_stream_observed(world), task.is_stream),
    }

    return [
        Discrepancy(premise=premise, declared=declared, observed=observed, detail=detail)
        for premise, ((observed, detail), declared) in checks.items()
        if observed != declared
    ]


def unverifiable() -> frozenset[Premise]:
    """Предпосылки, которые из данных не выводятся и остаются объявлениями."""
    return frozenset(Premise) - VERIFIABLE
