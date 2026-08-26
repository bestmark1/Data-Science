"""Сверка объявлений с данными: предпосылки (F-4) и окна признаков (F-11)."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from dsx.checks.base import Context, Signal
from dsx.evals.case import Finding
from dsx.premises import verify
from dsx.roles import Role
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


@dataclass(frozen=True)
class WindowClockIsKnowable:
    """S9. Окно признака отсчитано по времени, которого на момент решения нет.

    У записи бывает два времени: когда описанное ею случилось и когда о ней
    стало известно. Окно, отсчитанное по первому, захватывает записи, о которых
    на момент решения ещё не сообщили. В таблице обе колонки — просто числа, и
    отличить одно окно от другого без объявления нельзя: шестой кейс собрал два
    признака, различающиеся ровно этим, и ядро не возразило ни разу.

    Объявление проверяется данными. Если объявленные часы систематически
    отстают от момента, когда запись становится известной, окно захватывает
    неизвестное — и это находка, а не оговорка.

    Там, где часы лежат за пределами таблицы решений (показания приборов,
    отдельный журнал), сверить нечем, и проверка честно молчит: объявление
    остаётся объявлением и видно в отчёте как непроверяемое.
    """

    requirement: str = "S9"
    premises: frozenset[Premise] = frozenset({Premise.UNIVERSAL})
    detects: frozenset[Finding] = frozenset({Finding.WINDOW_CLOCK_UNKNOWABLE})

    share: float = 0.05
    """Какая доля отставших строк считается систематической.

    Единичное отставание бывает опечаткой в источнике; порог отделяет её от
    устройства данных. На шестом кейсе отставали 25.3% строк.
    """

    def _knowable_at(self, context: Context) -> str | None:
        """Колонка, по которой запись становится известной.

        Порядок не произволен. Объявленный момент поступления сведений главнее
        всего; за ним момент измерения — показание прибора становится известным
        тогда, когда снято, и отдельного «сообщили» у него нет. Если не
        объявлено ни того ни другого, запись становится известной в собственный
        момент решения.

        Первая версия всегда брала момент решения и кричала на законном: у
        показаний за неделю до визита время замера, разумеется, раньше визита.
        Проверка, срабатывающая на обычном устройстве данных, запрещена
        правилами этого проекта.
        """
        for role in (Role.AVAILABLE_AT, Role.MEASURED_AT):
            declared = context.world.schema.by_role(role)
            if declared:
                return declared[0].name
        decision = context.world.schema.decision_time
        return decision.name if decision else None

    def run(self, context: Context) -> list[Signal]:
        features = [
            c
            for c in context.world.schema.usable_features()
            if c.window is not None and c.window.lookback_days > 0.0
        ]
        if not features:
            return []

        undeclared = sorted(c.name for c in features if not c.window.clock)
        signals = [
            Signal(
                Finding.WINDOW_CLOCK_UNKNOWABLE,
                f"признаки {undeclared!r} объявили окно, но не сказали, ПО КАКОМУ "
                "времени строка в него отбирается. Время события и время, когда о "
                "нём стало известно, — разные колонки, и окно по первому захватывает "
                "то, чего на момент решения ещё нет",
                blocking=True,
            )
        ]
        if not undeclared:
            signals = []

        known = self._knowable_at(context)
        frame = context.world.main
        for column in features:
            clock = column.window.clock
            if not clock or not known or clock == known:
                continue
            if clock not in frame.columns or known not in frame.columns:
                continue  # часы вне таблицы решений: сверить нечем, и это видно в отчёте
            behind = frame.filter(pl.col(clock) < pl.col(known)).height
            if behind < frame.height * self.share:
                continue
            signals.append(
                Signal(
                    Finding.WINDOW_CLOCK_UNKNOWABLE,
                    f"окно признака {column.name!r} отсчитывается по {clock!r}, а запись "
                    f"становится известной в {known!r}: у {behind:,} строк "
                    f"({behind / frame.height:.1%}) первое раньше второго. Окно "
                    "захватывает записи, о которых на момент решения ещё не сообщили",
                    blocking=True,
                )
            )
        return signals


@dataclass(frozen=True)
class ValueFixedBeforeDecision:
    """S10. Значение поля зафиксировано ПОЗЖЕ момента, когда решение принималось.

    Поле, которое источник переписывает, читается сегодня не тем, каким было
    тогда. Девятый кейс: срок клинического исследования пересматривался у 73.9%
    записей с медианой 184 дня, а у завершившихся переписывался в дату самого
    завершения. Взятый из текущей выгрузки, он давал 100% выполненных обещаний
    вместо 36.5% — разницу в 63.5 процентных пункта.

    Проверяется прежде всего СРОК. Именно через него пересмотр смертелен:
    переписанный срок перестаёт быть обещанием и становится записью о
    случившемся, после чего исход выполняется тождественно. Признаки
    проверяются тоже, если объявили момент фиксации.

    Отсутствие объявления у срока блокирует. Вопрос «на какой момент
    зафиксировано это значение» девять кейсов не задавался ни разу, и умолчание
    здесь неотличимо от невнимательности.

    Предел назван прямо: объявление сверяется данными только тогда, когда
    названная колонка есть в таблице решений. Объявить момент фиксации равным
    моменту решения и солгать — можно; проверка увидит это лишь если в данных
    есть чем возразить. Та же граница, что у S9.
    """

    requirement: str = "S10"
    premises: frozenset[Premise] = frozenset({Premise.UNIVERSAL})
    detects: frozenset[Finding] = frozenset({Finding.VALUE_REVISED_AFTER_DECISION})

    share: float = 0.05
    """Какая доля позже зафиксированных значений считается систематической."""

    def run(self, context: Context) -> list[Signal]:
        schema = context.world.schema
        decision = schema.decision_time
        if decision is None:
            return []

        deadlines = schema.by_role(Role.DEADLINE)
        signals: list[Signal] = []

        undeclared = [c.name for c in deadlines if not c.value_as_of]
        if undeclared:
            signals.append(
                Signal(
                    Finding.VALUE_REVISED_AFTER_DECISION,
                    f"срок {sorted(undeclared)!r} не объявил, НА КАКОЙ МОМЕНТ "
                    "зафиксировано его значение. Поле, переписанное после исхода, "
                    "перестаёт быть обещанием и становится записью о случившемся, "
                    "и отличить одно от другого без объявления нельзя",
                    blocking=True,
                )
            )

        frame = context.world.main
        checked = [c for c in (*deadlines, *schema.usable_features()) if c.value_as_of]
        for column in checked:
            fixed = column.value_as_of
            if fixed not in frame.columns or decision.name not in frame.columns:
                continue  # сверить нечем, и это видно в отчёте как непроверяемое
            if fixed == decision.name:
                continue  # значение зафиксировано в момент решения: пересмотра нет
            later = frame.filter(pl.col(fixed) > pl.col(decision.name)).height
            if later < frame.height * self.share:
                continue
            signals.append(
                Signal(
                    Finding.VALUE_REVISED_AFTER_DECISION,
                    f"значение {column.name!r} зафиксировано на момент {fixed!r}, "
                    f"а решение принималось в {decision.name!r}: у {later:,} строк "
                    f"({later / frame.height:.1%}) первое позже второго. Это значение "
                    "решавшему известно не было",
                    blocking=True,
                )
            )
        return signals


PREMISE_CHECKS = [
    PremisesMatchData(),
    FeatureWindowsMatchData(),
    EveryColumnIsDeclared(),
    WindowClockIsKnowable(),
    ValueFixedBeforeDecision(),
]
