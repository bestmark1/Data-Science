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
from dsx.task import OutcomeTiming, Premise, TaskSpec


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


VERIFIABLE = frozenset({Premise.PROCESS, Premise.STREAM, Premise.DELAYED_OUTCOME})
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


def _delay_observed(world: World, event_column: str | None) -> tuple[bool | None, str]:
    """Наступает ли исход ПОЗЖЕ решения.

    Пятнадцатый кейс назвал класс: предпосылка, выводимая из данных, но не
    выводимая ядром, выключает проверку по одному слову автора. `DELAYED_OUTCOME`
    была последней такой. Показано счётом на кейсе стенда `label-immaturity`:
    объявление `immediate` при 20% незрелых исходов убирало ЧЕТЫРЕ блокирующие
    находки — по каждому окну и по резерву, — и ни одна проверка не возражала.

    Критерий без порога: медиана задержки среди строк С СОБЫТИЕМ. Больше нуля —
    исход отложен. Порог здесь был бы лишним: вопрос не «насколько поздно», а
    «позже ли вообще», и половина строк отвечает на него без произвольных чисел.

    None означает, что судить не по чему: событий в данных нет. Тогда нет и
    незрелости, и выключенная проверка ничего не теряет.
    """
    # Имя колонки берётся из КОНТРАКТА ИСХОДА, а не из роли. Роль события в
    # схеме может быть иной — на стенде она `outcome_component`, — а
    # авторитетом здесь является то, по чему считается метка.
    column = event_column
    if column is None:
        return None, "колонка события не названа в контракте исхода"
    if column not in world.main.columns:
        return None, f"колонка {column!r} объявлена событием, но отсутствует в данных"

    decided = world.schema.decision_time.name
    gaps = (
        world.main.filter(pl.col(column).is_not_null())
        .select((pl.col(column) - pl.col(decided)).dt.total_seconds().alias("gap"))
        .get_column("gap")
    )
    if gaps.is_empty():
        return None, f"в колонке {column!r} нет ни одного события"

    median = float(gaps.median())
    if median > 0:
        return True, f"медианная задержка исхода {median / 86400:.1f} сут"
    return False, "медианная задержка исхода не больше нуля: исход известен сразу"


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
    """
    checks = {
        Premise.PROCESS: (_process_observed(world), task.has_process),
        Premise.STREAM: (_stream_observed(world), task.is_stream),
        Premise.DELAYED_OUTCOME: (
            _delay_observed(world, event_column),
            task.outcome_timing is OutcomeTiming.DELAYED,
        ),
    }

    # Наблюдение None означает, что судить не по чему. Молчание здесь не
    # согласие с автором, а отсутствие предмета спора, и оно отличается от
    # совпадения объявленного с наблюдаемым только тем, что сказать нечего.
    return [
        Discrepancy(premise=premise, declared=declared, observed=observed, detail=detail)
        for premise, ((observed, detail), declared) in checks.items()
        if observed is not None and observed != declared
    ]


def unverifiable() -> frozenset[Premise]:
    """Предпосылки, которые из данных не выводятся и остаются объявлениями."""
    return frozenset(Premise) - VERIFIABLE
