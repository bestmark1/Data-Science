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


def _stream_observed(world: World, irregularity: float = 3.0) -> tuple[bool, str]:
    """Собираются ли наблюдения регулярным потоком.

    Признак потока — РЕГУЛЯРНОСТЬ, а не ежедневность. Первая версия требовала
    наблюдений в девяноста процентах календарных дней и объявляла не-потоком
    сбор по рабочим дням (около 71%), еженедельный и любой иной ритм. После
    согласованного `is_stream=false` проверки полноты периода выключались, и
    настоящий пропуск недели в таком потоке оставался незамеченным.

    Регулярность меряется разбросом промежутков между днями наблюдений:
    у потока они одинаковы, у событийного сбора — нет.
    """
    column = world.schema.decision_time.name
    if column not in world.main.columns:
        return False, f"колонка решения {column!r} отсутствует в данных"

    days = (
        world.main.select(pl.col(column).dt.date().alias("day"))
        .unique()
        .sort("day")
        .get_column("day")
    )
    if days.len() < 4:
        return False, f"дней с наблюдениями всего {days.len()}: о ритме говорить рано"

    gaps = days.diff().drop_nulls().dt.total_days()
    typical = float(gaps.median())
    if typical <= 0:
        return False, "промежутки между днями наблюдений вырождены"

    worst = float(gaps.quantile(0.9))
    longest = float(gaps.max())
    # Худший промежуток называется всегда: ритмичный в среднем сбор может иметь
    # разреженное начало, и медиана этого не показывает. Сам разрыв — предмет
    # отдельной проверки полноты периода, но знать о нём нужно уже здесь.
    tail = f", самый длинный {longest:g} дн" if longest > typical * irregularity else ""
    if worst <= typical * irregularity:
        return True, (
            f"наблюдения идут ритмично: типичный промежуток {typical:g} дн, "
            f"девяностый процентиль {worst:g} дн{tail}"
        )
    return False, (
        f"промежутки между наблюдениями неровные: типичный {typical:g} дн, "
        f"девяностый процентиль {worst:g} дн{tail}"
    )


def verify(world: World, task: TaskSpec, event_column: str | None = None) -> list[Discrepancy]:
    """Сверить выводимые предпосылки с данными.

    Возвращает расхождения. Пустой список означает, что объявленное совпало с
    наблюдаемым — но только по выводимым предпосылкам.

    `DELAYED_OUTCOME` сюда НЕ входит, и это исправление, а не упущение.
    Семнадцатый кейс показал, что сравнивать там нечего: `delayed` —
    единственное значение, допустимое в работающем прогоне, `immediate`
    отвергается `require_supported` целиком. Наблюдатель сравнивал объявление,
    у которого нет альтернативы, и называл свойство данных расхождением.

    Обоснование, которым он вводился, тоже было получено в обход рабочего пути:
    объявление `immediate` подменялось прямо в `TaskSpec`, минуя отказ прогона.
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
