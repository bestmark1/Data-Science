"""Сверка объявлений с данными: предпосылки (F-4) и окна признаков (F-11)."""

from __future__ import annotations

from dataclasses import dataclass

from dsx.checks.base import Context, Signal
from dsx.evals.case import Finding
from dsx.premises import verify
from dsx.task import Premise
from dsx.windows import verify_windows


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


@dataclass(frozen=True)
class FeatureWindowsMatchData:
    """S7. Объявленное окно признака расходится со временем измерения.

    Окно объявлялось и ни с чем не сверялось. Мгновенный признак объявляется
    окном нулевой длины, но такого признака не существует: у любого значения
    есть возраст. Признак, назвавший колонку времени измерения, сверяется;
    остальные остаются объявлениями, и это видно в отчёте.
    """

    requirement: str = "S7"
    premises: frozenset[Premise] = frozenset({Premise.UNIVERSAL})
    detects: frozenset[Finding] = frozenset({Finding.FEATURE_WINDOW_MISMATCH})

    def run(self, context: Context) -> list[Signal]:
        return [
            Signal(
                Finding.FEATURE_WINDOW_MISMATCH,
                f"{discrepancy}. Окно объявлено, но данные его не подтверждают",
                blocking=True,
            )
            for discrepancy in verify_windows(context.world)
        ]


@dataclass(frozen=True)
class EveryColumnIsDeclared:
    """S8. В таблице решений есть колонка, которой нет в схеме.

    Ядро видит только объявленное. Незаявленная колонка не попадает ни в одну
    проверку: можно не объявить настоящую колонку статуса, выставить
    `has_process: false`, и сверка предпосылок согласится с объявлением, хотя
    данные говорят обратное. Тем же способом остаётся невидимым признак,
    вычисленный из исхода.

    Роль `ignored` существует ровно для этого: сказать «колонка есть, и она не
    нужна» — ответ, а промолчать — нет.
    """

    requirement: str = "S8"
    premises: frozenset[Premise] = frozenset({Premise.UNIVERSAL})
    detects: frozenset[Finding] = frozenset({Finding.UNDECLARED_COLUMN})

    def run(self, context: Context) -> list[Signal]:
        declared = {c.name for c in context.world.schema.columns}
        present = set(context.world.main.columns)
        undeclared = sorted(present - declared)
        if not undeclared:
            return []
        return [
            Signal(
                Finding.UNDECLARED_COLUMN,
                f"колонки {undeclared!r} есть в таблице решений, но не объявлены в "
                "схеме. Ядро их не видит: ни одна проверка к ним не применится. "
                "Объявите роль, в том числе 'ignored', если колонка не нужна",
                blocking=True,
            )
        ]


PREMISE_CHECKS = [PremisesMatchData(), FeatureWindowsMatchData(), EveryColumnIsDeclared()]
