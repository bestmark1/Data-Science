"""Журнал расхода выборок.

Самая дорогая ошибка этапа 0 сидела здесь. Тест использовался для выбора окна
обучения, кандидата, калибровки и порога, после чего на нём же считались
итоговые метрики. Заключение перевернулось трижды, и каждый раз из-за
протокола, а не новых данных.

Выборка расходуется. Обращение к ней ради выбора решения тратит её как
измерительный инструмент: после этого измерение на ней смещено. Журнал делает
расход видимым, потому что незаписанный расход неотличим от его отсутствия.

Второй кейс показал, что учёта по именам недостаточно. При скользящих окнах
выборки w0 и w2 назывались по-разному, а делили сто процентов строк: выбор
делался на том же, на чём потом измеряли. Имя выборки не описывает её
содержимое, поэтому выборка объявляет состав.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from dsx.policy import Blocked, OverrideLedger


class Purpose(StrEnum):
    """Зачем к выборке обращались."""

    AUDIT = "audit"
    """Чтение ради проверок. Выборку не расходует: аудит не принимает решений
    и потому не смещает измерение."""

    FITTING = "fitting"
    """Обучение. Расходует выборку как обучающую, но не как измерительную."""

    SELECTION = "selection"
    """Выбор решения: окно, кандидат, гиперпараметры, калибровка, порог.
    Расходует выборку как измерительную."""

    MEASUREMENT = "measurement"
    """Итоговое измерение. Допустимо один раз и только на нерасходованной
    выборке."""


class Extent(BaseModel):
    """Состав выборки: какие единицы решения в неё входят и какой период.

    Существует потому, что имя ничего не гарантирует. Две выборки с разными
    именами могут быть одним и тем же множеством строк.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    units: Annotated[frozenset[str], Field(min_length=1)]
    """Идентификаторы единиц решения, входящих в выборку."""

    since: dt.datetime
    until: dt.datetime

    def shared_with(self, other: Extent) -> frozenset[str]:
        return self.units & other.units

    def __str__(self) -> str:
        return f"{len(self.units):,} единиц, {self.since:%Y-%m-%d}..{self.until:%Y-%m-%d}"


class Access(BaseModel):
    """Одно обращение к выборке."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sample: Annotated[str, Field(min_length=1)]
    purpose: Purpose
    decision: Annotated[str, Field(min_length=1)]
    """Какое решение принималось. Обращение без названного решения не
    отличается от необъяснимого расхода."""

    at: dt.datetime = Field(default_factory=lambda: dt.datetime.now(dt.UTC))

    def __str__(self) -> str:
        return f"{self.sample} / {self.purpose.value}: {self.decision}"


def _strictly_before(earlier: dt.datetime, later: dt.datetime) -> bool:
    """`earlier < later` как моменты, а не как показания часов.

    Ревью `da79e10`: при ОДНОМ объекте пояса Python сравнивает местное время и
    не учитывает `fold`. В час перевода часов 02:15+01:00 (второй проход) позже
    02:30+02:00 на 45 минут, а сравнение говорило «раньше» — обучение на будущих
    метках принималось. polars сравнивает в UTC, и так же — здесь. Наивное время
    неоднозначности не имеет и сравнивается как есть; смешение наивного с
    поясным отвергается при регистрации окна.
    """
    if earlier.tzinfo is None:
        return earlier < later
    return earlier.astimezone(dt.UTC) < later.astimezone(dt.UTC)


def _moment(value: dt.datetime) -> str:
    """Момент для сообщения: со смещением пояса, если оно есть."""
    return f"{value:%Y-%m-%d %H:%M%z}"


@dataclass(frozen=True)
class Training:
    """Обучающая часть окна, как её передаёт журналу прогон.

    Журнал не знает схемы данных, поэтому единицы решения и момент знания меток
    считает прогон, у которого схема есть.
    """

    frame: object
    rows: int
    units: frozenset[str]
    labels_known_until: dt.datetime
    """Момент, к которому известны ВСЕ метки обучающих строк."""


@dataclass(frozen=True, eq=False)
class Fit:
    """Запись обучения: обучающая часть окна, выданная журналом.

    Класс 19 журнала повторов: N7 принимал прогнозы по окнам как объявление их
    происхождения, и в кейсах 3–5 мерил ранние окна моделью, обученной на их же
    строках. Запись выдаёт только `SampleLedger.training`, тем же действием,
    что и кадр для обучения; измерение по окнам сверяет её с каждым окном.
    Сравнение — по тождеству: запись, которую журнал не выдавал, не принимается.
    """

    window: str
    frame: object
    rows: int
    units: frozenset[str]
    labels_known_until: dt.datetime

    def __str__(self) -> str:
        return (
            f"обучающая часть окна {self.window}: {self.rows:,} строк, "
            f"метки известны до {_moment(self.labels_known_until)}"
        )


@dataclass(frozen=True)
class _Window:
    start: dt.datetime
    evaluation: frozenset[str]
    fit: Fit | None


def _require_window_metadata(
    name: str, start: object, evaluation: object, training: object
) -> None:
    """Проверить то, что журнал принимает об окне, когда он это принимает.

    Ревью `ab5d402`: `Training(rows=[])` проходил регистрацию, и текст
    происхождения падал ПОСЛЕ расходования всех окон. Типы — ровно свои: время
    — `datetime` с той же привязкой к поясу, что начало окна, иначе сравнение
    упало бы при проверке происхождения.
    """
    if type(start) is not dt.datetime:
        raise ValueError(f"окно {name!r}: начало — не datetime")
    if type(evaluation) is not frozenset or any(type(u) is not str for u in evaluation):
        raise ValueError(f"окно {name!r}: оценочный состав — не frozenset строк")
    if training is None:
        return
    where = f"обучающая часть окна {name!r}"
    if type(training) is not Training:
        raise ValueError(f"{where}: ожидается Training")
    if type(training.rows) is not int or training.rows < 1:
        raise ValueError(f"{where}: число строк — не целое больше нуля; пустая часть — None")
    units = training.units
    if type(units) is not frozenset or not units or any(type(u) is not str for u in units):
        raise ValueError(f"{where}: единицы решения — не непустой frozenset строк")
    known = training.labels_known_until
    if type(known) is not dt.datetime:
        raise ValueError(f"{where}: момент знания меток — не datetime")
    if (known.tzinfo is None) != (start.tzinfo is None):
        raise ValueError(f"{where}: момент знания меток и начало окна по-разному привязаны к поясу")


class SampleLedger:
    """Учёт обращений к выборкам (P1, P2, P7) и того, чем питалась модель (P11)."""

    def __init__(self, ledger: OverrideLedger | None = None) -> None:
        self._accesses: list[Access] = []
        self._known: set[str] = set()
        self._extents: dict[str, Extent] = {}
        self._frames: dict[str, object] = {}
        self._features: tuple[str, ...] | None = None
        self._windows: dict[str, _Window] = {}
        self._issued: set[str] = set()
        self._overrides = ledger or OverrideLedger()

    # --- регистрация ------------------------------------------------------

    def register(self, name: str, extent: Extent | None = None, frame: object = None) -> None:
        """Объявить выборку, её состав и, если журнал ею владеет, сами данные.

        Обращение к незарегистрированной выборке — ошибка: имя, придуманное на
        ходу, обходит учёт. Состав необязателен, но без него журнал не сможет
        утверждать независимость измерения и скажет об этом прямо.

        `frame` передаётся тогда, когда данные должны быть доступны ТОЛЬКО
        через журнал. Тогда чтение и запись расхода становятся одной операцией,
        и обойти учёт нельзя не по невнимательности, а вовсе.
        """
        self._known.add(name)
        if extent is not None:
            self._extents[name] = extent
        if frame is not None:
            self._frames[name] = frame

    def extent(self, sample: str) -> Extent | None:
        return self._extents.get(sample)

    def register_window(
        self,
        name: str,
        *,
        start: dt.datetime,
        extent: Extent | None,
        evaluation: frozenset[str],
        frame: object,
        training: Training | None,
    ) -> None:
        """Объявить оценочное окно: общий состав, оценочную часть и обучающую.

        Общий состав (обучение и оценка вместе) по-прежнему служит независимости
        итогового измерения. Оценочная часть и начало окна нужны измерению по
        окнам: прогнозы окна принимаются, только если модель училась не на его
        строках и не на метках, известных после его начала (DS-010).

        Обучающая часть переходит к журналу, как резерв: получить её можно только
        через `training`, который записывает обращение и выдаёт запись обучения.
        `training=None` — обучающая часть пуста.
        """
        _require_window_metadata(name, start, evaluation, training)
        self.register(name, extent, frame)
        fit = (
            None
            if training is None
            else Fit(
                window=name,
                frame=training.frame,
                rows=training.rows,
                units=training.units,
                labels_known_until=training.labels_known_until,
            )
        )
        self._windows[name] = _Window(start=start, evaluation=evaluation, fit=fit)

    def training(self, window: str, decision: str) -> Fit:
        """Выдать обучающую часть окна, записав обучение тем же действием.

        Единственный способ получить кадр для обучения и запись, без которой
        измерение по окнам прогнозы не примет. Прежде обучающая часть лежала
        публичным полем `Part.train`, а `fit(...)` было объявлением, которое
        ничего не связывало: кейсы 3–5 учились на последнем окне и мерили все.
        """
        self._require_known(window)
        if window not in self._windows:
            raise Blocked("P2", f"{window!r} не зарегистрирована как окно: обучающей части нет")
        fit = self._windows[window].fit
        if fit is None:
            raise ValueError(
                f"обучающая часть окна {window!r} пуста: учить нечему, и момент, к "
                "которому известны метки, не определён"
            )
        self.fit(window, decision)
        self._issued.add(window)
        return fit

    def provenance_defects(self, window: str, fit: object) -> list[str]:
        """Почему прогнозы окна, полученные по записи `fit`, не принимаются.

        Проверяется состав и время, а не число моделей. Пустой список — запись
        выдана этим журналом, строки обучения не пересекают оценочную часть окна
        и все метки обучения известны строго до его начала — так же строго, как
        `split_by_windows` отбирает обучение окна.
        """
        if window not in self._windows:
            return [f"{window!r} не зарегистрирована как окно: начало и состав оценки неизвестны"]
        if type(fit) is not Fit:
            return ["прогнозы без записи обучения, выданной журналом: происхождение неизвестно"]
        # Сначала тождество с ВЫДАННЫМИ записями, не трогая полей чужой: её
        # `window` служил ключом поиска, и непригодное значение роняло
        # `TypeError` (ревью `648a185`). Поля читаются только у своей записи.
        if not any(fit is self._windows[name].fit for name in self._issued):
            return ["запись обучения выдана не этим журналом: происхождение не проверяется"]

        target = self._windows[window]
        defects = []
        shared = fit.units & target.evaluation
        if shared:
            defects.append(
                f"модель училась на {len(shared):,} из {len(target.evaluation):,} оценочных "
                f"единиц решения окна ({fit})"
            )
        if not _strictly_before(fit.labels_known_until, target.start):
            defects.append(
                f"модель училась на метках, известных до {_moment(fit.labels_known_until)}, "
                f"а окно начинается {_moment(target.start)}: метки из будущего окна"
            )
        return defects

    # --- чем питалась модель (P11) ---------------------------------------

    def declare_features(self, names: Sequence[str]) -> None:
        """Записать колонки, которые модель ДЕЙСТВИТЕЛЬНО получила на вход.

        Пятнадцатый кейс: ядро проверяло объявленную СХЕМУ и не знало, чем
        питается модель. Признаки жили отдельным списком в коде проекта, и
        связи между ним и формуляром не было никакой.

        Цена показана опытом на закрытом кейсе. Колонка `outcome_subtype`,
        объявленная признаком, мгновенно даёт `value_revised_after_decision` на
        113 790 строках из 116 671 — контроль К-1 её ловил. Объявленная
        `role: ignored` и скормленная модели в обход, она подняла разрешающую
        способность с 0.7178 до 0.8009, и ядро не сказало НИ СЛОВА: сигналы
        совпали дословно.

        Дыра обесценивала три механизма разом — N6, S10 и подложенные контроли:
        любую утечку достаточно было объявить `ignored`.

        Список проверяется на согласие со схемой вызывающей стороной, у которой
        схема есть; журнал хранит факт объявления и сами имена.
        """
        self._features = tuple(names)

    @property
    def features(self) -> tuple[str, ...] | None:
        """Объявленные входы модели. None — объявления не было."""
        return self._features

    def require_features(self) -> None:
        """Отказать в измерении, пока входы модели не объявлены.

        Необязательное объявление никто не делает, и механизм, которого можно
        не позвать, в этом проекте ломался трижды. Поэтому отказывает само
        измерение: обойти его, ничего не заметив, нельзя.
        """
        if self._features is None:
            self._overrides.enforce(
                "P11",
                "входы модели не объявлены: измерять нечего, пока неизвестно, "
                "чем модель питалась. Объявите их через result.uses(...) — "
                "иначе утечку достаточно объявить в схеме ролью ignored",
            )

    def _require_known(self, sample: str) -> None:
        if sample not in self._known:
            raise Blocked(
                "P2",
                f"выборка {sample!r} не зарегистрирована; "
                f"известны: {', '.join(sorted(self._known)) or 'ни одной'}",
            )

    # --- обращения --------------------------------------------------------

    def checkout(self, sample: str, purpose: Purpose, decision: str) -> object:
        """Выдать данные выборки, записав расход тем же действием.

        Единственный способ получить кадр, которым владеет журнал. Прежде
        данные лежали публичным полем результата сплита, а журнал был отдельным
        объектом, который надо не забыть позвать. Механизм, зависящий от
        памяти зовущего, в этом проекте ломался дважды.
        """
        self._require_known(sample)
        if sample not in self._frames:
            raise Blocked(
                "P2",
                f"журнал не владеет данными выборки {sample!r}: выдать нечего. "
                "Либо выборка зарегистрирована без данных, либо имя ошибочно",
            )

        match purpose:
            case Purpose.AUDIT:
                pass  # аудит решений не принимает и выборку не расходует
            case Purpose.FITTING:
                self.fit(sample, decision)
            case Purpose.SELECTION:
                self.select(sample, decision)
            case Purpose.MEASUREMENT:
                self.measure(sample, decision)

        if purpose is Purpose.AUDIT:
            self._accesses.append(Access(sample=sample, purpose=Purpose.AUDIT, decision=decision))
        return self._frames[sample]

    def owns(self, sample: str) -> bool:
        """Владеет ли журнал данными выборки."""
        return sample in self._frames

    def fit(self, sample: str, decision: str) -> None:
        self._require_known(sample)
        self._accesses.append(Access(sample=sample, purpose=Purpose.FITTING, decision=decision))

    def select(self, sample: str, decision: str) -> None:
        """Зафиксировать выбор решения по выборке.

        После итогового измерения любой новый выбор требует свежей выборки:
        протокол, изменённый после просмотра метрик, обесценивает измерение (P7).
        """
        self.require_selectable(sample)
        self._accesses.append(Access(sample=sample, purpose=Purpose.SELECTION, decision=decision))

    def require_selectable(self, sample: str) -> None:
        """Отказать, если выбор по выборке запрещён, — ничего не записывая.

        Нужен тому, кто расходует несколько выборок разом: ревью `ab5d402`
        показало, что отказ P7 по второму окну наступал после расходования
        первого, хотя был известен заранее. Обход P7 действует так же, как в
        `select`: `enforce` ничего не записывает.
        """
        self._require_known(sample)
        if self.was_measured(sample):
            self._overrides.enforce(
                "P7",
                f"на выборке {sample!r} уже проведено итоговое измерение; "
                "решение, принятое после просмотра метрик, требует свежей выборки",
            )

    def measure(self, sample: str, decision: str = "итоговая оценка") -> None:
        """Зафиксировать итоговое измерение.

        Блокируется, если выборка уже расходовалась на выбор решений (P1) либо
        измерялась ранее.
        """
        self._require_known(sample)

        self._require_independent(sample)

        spent = self.selections(sample)
        if spent:
            self._overrides.enforce(
                "P1",
                f"выборка {sample!r} уже участвовала в выборе решений "
                f"({len(spent)}): {'; '.join(a.decision for a in spent)}. "
                "Измерение на ней смещено",
            )
        if self.was_measured(sample):
            self._overrides.enforce(
                "P1", f"на выборке {sample!r} итоговое измерение уже проводилось"
            )

        self._accesses.append(Access(sample=sample, purpose=Purpose.MEASUREMENT, decision=decision))

    def _require_independent(self, sample: str) -> None:
        """Убедиться, что измерительная выборка не пересекается с теми, на
        которых учились и выбирали.

        Проверяется состав, а не имя. Имя не описывает содержимое: на втором
        кейсе выборки w0 и w2 делили все строки до единой.
        """
        # Прежние ИЗМЕРЕНИЯ тоже расходуют состав: измерить одни и те же строки
        # дважды под разными именами — ровно тот обход, ради которого учёт по
        # содержимому и вводился.
        touched = [
            a for a in self._accesses if a.sample != sample and a.purpose is not Purpose.AUDIT
        ]
        if not touched:
            return

        mine = self._extents.get(sample)
        unknown = sorted({a.sample for a in touched if a.sample not in self._extents})
        if mine is None or unknown:
            missing = sorted(set(unknown) | ({sample} if mine is None else set()))
            self._overrides.enforce(
                "P2",
                f"состав выборок {missing!r} не объявлен, поэтому независимость "
                f"измерения на {sample!r} не проверена. Имя выборки её содержимого "
                "не описывает",
            )
            return

        for other in sorted({a.sample for a in touched}):
            shared = mine.shared_with(self._extents[other])
            if not shared:
                continue
            purposes = sorted({a.purpose.value for a in touched if a.sample == other})
            self._overrides.enforce(
                "P1",
                f"выборки {sample!r} и {other!r} делят {len(shared):,} единиц решения "
                f"из {len(mine.units):,} ({len(shared) / len(mine.units):.0%}), "
                f"а {other!r} уже использована для: {', '.join(purposes)}. "
                "Измерение проводится на том же, на чём принимались решения",
            )

    # --- состояние --------------------------------------------------------

    def selections(self, sample: str) -> tuple[Access, ...]:
        return tuple(
            a for a in self._accesses if a.sample == sample and a.purpose is Purpose.SELECTION
        )

    def was_measured(self, sample: str) -> bool:
        return any(a.sample == sample and a.purpose is Purpose.MEASUREMENT for a in self._accesses)

    def is_spent(self, sample: str) -> bool:
        """Израсходована ли выборка как измерительный инструмент."""
        return bool(self.selections(sample)) or self.was_measured(sample)

    def unspent(self) -> tuple[str, ...]:
        return tuple(sorted(n for n in self._known if not self.is_spent(n)))

    @property
    def accesses(self) -> tuple[Access, ...]:
        return tuple(self._accesses)

    # --- отчёт ------------------------------------------------------------

    def report_section(self) -> str:
        """Раздел отчёта: на чём принято каждое решение (P2)."""
        lines = ["## Расход выборок", ""]
        if not self._accesses:
            lines.append("Обращений не зафиксировано.")
            return "\n".join(lines)

        lines += ["| выборка | назначение | решение |", "|---|---|---|"]
        for access in self._accesses:
            lines.append(f"| {access.sample} | {access.purpose.value} | {access.decision} |")

        undeclared = sorted(n for n in self._known if n not in self._extents)
        if undeclared:
            lines.append("")
            lines.append(
                "Состав не объявлен у выборок: "
                + ", ".join(undeclared)
                + ". Для них независимость измерения проверена только по именам."
            )

        remaining = self.unspent()
        lines.append("")
        lines.append("Нерасходованные выборки: " + (", ".join(remaining) if remaining else "нет"))
        return "\n".join(lines)
