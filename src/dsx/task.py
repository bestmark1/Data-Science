"""Описание задачи и предпосылки, при которых проверки осмысленны.

Половина требований, выведенных на одном кейсе, несла скрытые предпосылки:
событийный процесс со статусами, отложенный исход, бинарный вероятностный
таргет, регулярный поток сбора. Поданы они были как универсальные.

Здесь предпосылки становятся машинными. Проверка объявляет, при каких условиях
она осмысленна, и выключается вместе с ними — а не срабатывает вхолостую на
задаче другого типа.

Реализация пока одна: бинарный таргет с отложенным исходом. Интерфейс обязан
допускать остальные, иначе второй кейс в другой отрасли упрётся в него.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class TargetKind(StrEnum):
    """Тип целевой величины."""

    BINARY = "binary"
    REGRESSION = "regression"
    RANKING = "ranking"
    SURVIVAL = "survival"
    UPLIFT = "uplift"


class OutcomeTiming(StrEnum):
    """Когда исход становится известен относительно момента решения."""

    IMMEDIATE = "immediate"
    """Известен сразу. Зазор в сплите и зрелость метки неприменимы."""

    DELAYED = "delayed"
    """Созревает позже. Момент узнавания выводится из определения исхода."""


class Premise(StrEnum):
    """Предпосылка, при которой требование осмысленно.

    Соответствует пометкам области действия в наблюдённой спецификации.
    """

    UNIVERSAL = "universal"
    PROCESS = "process"
    """Есть событийный процесс со статусами и упорядоченными метками."""

    DELAYED_OUTCOME = "delayed_outcome"
    BINARY_TARGET = "binary_target"
    STREAM = "stream"
    """Данные собираются регулярным временным потоком."""


class TaskSpec(BaseModel):
    """Что за задача решается и какие предпосылки выполнены."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    target_kind: TargetKind
    outcome_timing: OutcomeTiming

    has_process: bool = False
    """Есть ли событийный процесс со статусами."""

    is_stream: bool = False
    """Собираются ли данные регулярным временным потоком."""

    def satisfies(self, premise: Premise) -> bool:
        """Выполнена ли предпосылка."""
        match premise:
            case Premise.UNIVERSAL:
                return True
            case Premise.PROCESS:
                return self.has_process
            case Premise.DELAYED_OUTCOME:
                return self.outcome_timing is OutcomeTiming.DELAYED
            case Premise.BINARY_TARGET:
                return self.target_kind is TargetKind.BINARY
            case Premise.STREAM:
                return self.is_stream

    def satisfies_all(self, premises: frozenset[Premise]) -> bool:
        return all(self.satisfies(p) for p in premises)

    def unmet(self, premises: frozenset[Premise]) -> frozenset[Premise]:
        """Предпосылки, которые не выполнены. Пусто — проверка применима."""
        return frozenset(p for p in premises if not self.satisfies(p))


SUPPORTED = frozenset({(TargetKind.BINARY, OutcomeTiming.DELAYED)})
"""Комбинации, для которых есть реализация. Интерфейс шире реализации намеренно."""


class UnsupportedTask(Exception):
    """Тип задачи описан корректно, но реализации для него пока нет."""


def require_supported(task: TaskSpec) -> None:
    """Отказать явно, а не выдать неверный результат молча."""
    if (task.target_kind, task.outcome_timing) not in SUPPORTED:
        raise UnsupportedTask(
            f"задача {task.target_kind.value} / {task.outcome_timing.value} описана "
            "корректно, но реализации для неё нет. Поддерживается: "
            + ", ".join(f"{k.value}/{t.value}" for k, t in sorted(SUPPORTED))
        )
