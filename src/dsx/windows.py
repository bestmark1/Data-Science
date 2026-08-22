"""Сверка объявленных окон признаков с данными.

F-11: окно признака объявлялось и ни с чем не сверялось. Мгновенный признак
объявляется окном нулевой длины, но такого признака не существует — у любого
значения есть возраст. «Последнее показание датчика» может быть трёхдневной
давности, и ядро об этом не узнавало.

Сверяется то, что можно: признак, назвавший колонку времени измерения. Прочие
остаются объявлениями, и это говорится прямо. Тот же приём, что с
предпосылками: невыводимое не маскируется под проверенное.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import polars as pl

from dsx.evals.world import World
from dsx.roles import Role


class WindowFault(StrEnum):
    """Как измерение разошлось с объявленным окном."""

    TOO_NEW = "too_new"
    """Значение получено позже, чем кончается окно: знание из будущего."""

    TOO_OLD = "too_old"
    """Значение старше начала окна: признак устарел сильнее объявленного."""


@dataclass(frozen=True)
class WindowDiscrepancy:
    """Объявленное окно разошлось с временем измерения."""

    feature: str
    fault: WindowFault
    rows: int
    worst_days: float
    """Насколько сильно нарушена граница в худшей строке, в днях."""

    def __str__(self) -> str:
        side = "позже конца окна" if self.fault is WindowFault.TOO_NEW else "раньше начала окна"
        return (
            f"{self.feature}: у {self.rows:,} строк значение получено {side}, "
            f"худшее расхождение {self.worst_days:g} дн"
        )


def verify_windows(world: World) -> list[WindowDiscrepancy]:
    """Сверить объявленные окна с временем измерения там, где оно названо."""
    frame = world.main
    moment = world.schema.decision_time.name
    discrepancies: list[WindowDiscrepancy] = []

    for column in world.schema.columns:
        if column.role is not Role.FEATURE or column.measured_at is None:
            continue
        if column.measured_at not in frame.columns or moment not in frame.columns:
            continue

        window = column.window
        assert window is not None  # обеспечено проверкой схемы
        age = (pl.col(moment) - pl.col(column.measured_at)).dt.total_seconds() / 86400.0
        start, stop = -window.bounds_days()[0], -window.bounds_days()[1]

        too_new = frame.select((stop - age).alias("d")).filter(pl.col("d") > 0)["d"]
        if too_new.len():
            discrepancies.append(
                WindowDiscrepancy(column.name, WindowFault.TOO_NEW, too_new.len(), too_new.max())
            )

        too_old = frame.select((age - start).alias("d")).filter(pl.col("d") > 0)["d"]
        if too_old.len():
            discrepancies.append(
                WindowDiscrepancy(column.name, WindowFault.TOO_OLD, too_old.len(), too_old.max())
            )

    return discrepancies


def unverified(world: World) -> tuple[str, ...]:
    """Признаки, чьё окно объявлено, но сверить его не с чем."""
    return tuple(
        sorted(
            c.name
            for c in world.schema.columns
            if c.role is Role.FEATURE and c.window is not None and c.measured_at is None
        )
    )
