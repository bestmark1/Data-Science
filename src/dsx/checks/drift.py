"""Проверки изменчивости во времени (N3, N4, N5, N8).

Заключение этапа 0 переворачивалось трижды, и дважды причиной была связь,
державшаяся не весь период. Проверки здесь ищут ровно это: долю класса,
меняющуюся между окнами, и признак, чья связь с исходом меняет знак или силу.

Порядок между ними не случаен. Прежде чем утверждать, что связь изменилась,
нужно убедиться, что окна вообще сравнимы: признак, принимающий в разных
окнах непересекающиеся значения, даёт видимость смены знака там, где сравнение
неприменимо. Поэтому N5 не дополнение к N4, а условие его осмысленности.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import polars as pl

from dsx.checks.base import Context, NotApplicable, Signal
from dsx.checks.empirical import association
from dsx.evals.case import Finding
from dsx.label import LABEL, REASON, OutcomeReason
from dsx.roles import Availability, Direction, Role
from dsx.split import SplitResult
from dsx.task import Premise

MIN_ROWS = 100
"""Меньше этого в окне — говорить об изменении связи не о чем."""

SIGMA = 3.0
"""Во сколько стандартных ошибок должно укладываться расхождение, чтобы
считаться наблюдением, а не колебанием выборки.

Первая версия порогов этого не учитывала и дала три ложные тревоги на чистых
мирах, а на кейсе дрейфа — ложную УДАЧУ: доли 0.7% и 1.1% при трёхстах строках
в окне отличаются на два события, и проверка объявила это дрейфом. Порог,
который срабатывает по шуму, не отличается от отсутствия порога.
"""


def _rate_error(rate: float, rows: int) -> float:
    """Стандартная ошибка доли."""
    return math.sqrt(max(rate * (1 - rate), 0.0) / rows) if rows else float("inf")


def _association_error(labels: pl.Series) -> float:
    """Стандартная ошибка связи вблизи нуля.

    Приближение достаточное для порога: связь считается наблюдаемой, если
    превосходит несколько таких ошибок.
    """
    positives = int(labels.sum())
    negatives = labels.len() - positives
    smaller = min(positives, negatives)
    return 0.29 / math.sqrt(smaller) if smaller else float("inf")


def _require_split(context: Context) -> SplitResult:
    if not isinstance(context.split, SplitResult):
        raise NotApplicable("временной сплит не построен")
    return context.split


def _observable(frame: pl.DataFrame) -> pl.DataFrame:
    if LABEL not in frame.columns:
        return frame.head(0)
    return frame.filter(pl.col(LABEL).is_not_null())


def _windows_with_labels(context: Context) -> list[tuple[str, pl.DataFrame]]:
    split = _require_split(context)
    parts = [(p.name, _observable(p.evaluate)) for p in split.parts]
    usable = [(name, frame) for name, frame in parts if frame.height >= MIN_ROWS]
    if len(usable) < 2:
        raise NotApplicable(
            f"окон с достаточным числом размеченных строк меньше двух (нужно {MIN_ROWS} в каждом)"
        )
    return usable


def _numeric_features(context: Context) -> list[str]:
    return [
        c.name
        for c in context.world.schema.by_role(Role.FEATURE)
        if c.availability is Availability.AT_DECISION
    ]


@dataclass(frozen=True)
class TargetRateStationarity:
    """N3. Доля положительного класса меняется между окнами.

    Модель, обученная на окне с одной долей и оценённая на окне с другой,
    сравнивается не сама с собой. На этапе 0 это дало вывод, продержавшийся
    до первой же проверки.
    """

    requirement: str = "N3"
    premises: frozenset[Premise] = frozenset({Premise.BINARY_TARGET})
    detects: frozenset[Finding] = frozenset({Finding.NON_STATIONARY_TARGET})

    ratio: float = 1.5
    """Во сколько раз доли МЕЖДУ ОКНАМИ могут различаться. Небольшие колебания
    во времени нормальны, и порог по кратности здесь уместен."""

    def _reserve_signals(
        self, context: Context, windows: list[tuple[str, pl.DataFrame]]
    ) -> list[Signal]:
        """Отличается ли доля класса в резерве от окон.

        Здесь порог по кратности НЕ применяется, и это не небрежность. Резерв —
        измерительный инструмент: инструмент, откалиброванный на другой
        популяции, смещает итоговое число независимо от величины сдвига.
        Достаточно того, что разница различима на фоне шума.

        На третьем кейсе резерв дал 11.9% против 8.7–9.1% в окнах. Кратность
        1.37 порога 1.5 не достигала, а разница составляла сорок стандартных
        ошибок, и модель систематически занижала риск.
        """
        split = _require_split(context)
        if split.reserved is None:
            return []
        reserved = _observable(split.reserved)
        if reserved.height < MIN_ROWS or not windows:
            return []

        pooled = pl.concat([f for _, f in windows], how="vertical_relaxed")
        in_windows = float(pooled[LABEL].mean())
        in_reserve = float(reserved[LABEL].mean())
        spread = math.sqrt(
            _rate_error(in_windows, pooled.height) ** 2
            + _rate_error(in_reserve, reserved.height) ** 2
        )
        # Сравнение НЕСТРОГОЕ намеренно. При доле ровно 100% или ровно 0%
        # разброс равен нулю, и строгое `0 < 0` ложно: проверка сообщала о
        # расхождении между 100% и 100%. Расхождение, равное шумовому порогу,
        # свидетельством не является — тем более нулевое.
        if abs(in_reserve - in_windows) <= SIGMA * spread:
            return []

        return [
            Signal(
                Finding.NON_STATIONARY_TARGET,
                f"доля положительного класса в РЕЗЕРВЕ {in_reserve:.1%}, а в окнах "
                f"{in_windows:.1%}. Резерв — измерительный инструмент: измеренное на нём "
                "описывает другую популяцию, и модель, обученная на окнах, будет на нём "
                "систематически смещена",
                blocking=True,
            )
        ]

    def _trend_signals(
        self, context: Context, windows: list[tuple[str, pl.DataFrame]]
    ) -> list[Signal]:
        """Медленный однонаправленный дрейф, невидимый сравнению по кратности.

        Соседние окна различаются мало, а накопленное за период различие велико.
        На четвёртом кейсе доля отмен росла с 4.45% до 7.64% — на 72% за пять
        лет, — при отношении соседних окон 1.14 и пороге 1.5. Проверка молчала,
        и дрейф поймал только резерв, потому что лежал в конце.

        Требуется одновременно направленность и различимость: три окна,
        упорядоченные случайно, дают монотонность с вероятностью около трети,
        поэтому одной её мало.

        Направление НЕ переносится за последнее окно само собой. На
        четырнадцатом кейсе доля росла 45.9% → 47.6% → 52.1%, а в резерве,
        лежащем сразу следом, падала до 42.1%. Прежняя формулировка обещала,
        что смещение тем сильнее, чем дальше от обучения, — обещание о
        будущем, которого проверка не измеряла, хотя резерв был у неё под
        рукой. Разворот — известие ХУДШЕЕ продолжения: направление нельзя
        продлить даже на шаг вперёд.
        """
        if len(windows) < 3:
            return []

        rates = [float(frame[LABEL].mean()) for _, frame in windows]
        sizes = [frame.height for _, frame in windows]
        rising = all(b > a for a, b in zip(rates, rates[1:], strict=False))
        falling = all(b < a for a, b in zip(rates, rates[1:], strict=False))
        if not (rising or falling):
            return []

        spread = math.sqrt(
            _rate_error(rates[0], sizes[0]) ** 2 + _rate_error(rates[-1], sizes[-1]) ** 2
        )
        if abs(rates[-1] - rates[0]) <= SIGMA * spread:
            return []

        shown = ", ".join(
            f"{name}: {rate:.1%}" for (name, _), rate in zip(windows, rates, strict=False)
        )
        direction = "растёт" if rising else "падает"
        return [
            Signal(
                Finding.NON_STATIONARY_TARGET,
                f"доля положительного класса однонаправленно {direction} по окнам "
                f"({shown}). Соседние окна различаются мало, и сравнение по кратности "
                f"этого не видит, но за период накопилось "
                f"{abs(rates[-1] - rates[0]):.1%}. "
                + self._beyond_last_window(context, rates, rising),
                blocking=True,
            )
        ]

    def _beyond_last_window(self, context: Context, rates: list[float], rising: bool) -> str:
        """Что делает доля СРАЗУ ЗА последним окном — по резерву, а не по вере.

        Возвращает готовую фразу. Когда сказать нечего, так и говорит: молчание
        здесь означало бы, что направление продлевается, а это и есть
        непроверенное обещание.
        """
        split = _require_split(context)
        if split.reserved is None:
            return (
                "Продолжается ли направление за последним окном, сказать нечем: резерв не объявлен"
            )
        reserved = _observable(split.reserved)
        if reserved.height < MIN_ROWS:
            return (
                "Продолжается ли направление за последним окном, сказать нечем: "
                f"в резерве наблюдаемых исходов {reserved.height}, меньше {MIN_ROWS}"
            )

        after = float(reserved[LABEL].mean())
        last = rates[-1]
        spread = _rate_error(after, reserved.height)
        if abs(after - last) <= SIGMA * spread:
            return (
                f"В резерве, лежащем сразу следом, доля {after:.1%} — от последнего "
                "окна неотличима: направление за окнами не продолжается и не "
                "разворачивается, оно там просто кончается"
            )
        if (after > last) == rising:
            return (
                f"В резерве, лежащем сразу следом, доля {after:.1%} — направление "
                "продолжается, и чем дальше от обучения делается предсказание, тем "
                "сильнее смещение"
            )
        return (
            f"В резерве, лежащем сразу следом, доля {after:.1%} — направление "
            "РАЗВЕРНУЛОСЬ. Это известие хуже продолжения: доля меняется, но её "
            "направление не продлевается даже на шаг вперёд, и поправка «по тренду» "
            "уводила бы в сторону, противоположную нужной"
        )

    def run(self, context: Context) -> list[Signal]:
        windows = _windows_with_labels(context)
        signals = self._reserve_signals(context, windows) + self._trend_signals(context, windows)
        rates = {name: float(frame[LABEL].mean()) for name, frame in windows}
        sizes = {name: frame.height for name, frame in windows}

        low_name = min(rates, key=lambda k: rates[k])
        high_name = max(rates, key=lambda k: rates[k])
        low, high = rates[low_name], rates[high_name]

        # Нулевая доля в одном окне при ненулевой в другом — расхождение
        # бесконечной кратности, а не повод к раннему возврату. Первая версия
        # трактовала самый крайний случай как отсутствие дефекта.
        if (low > 0 and high / low < self.ratio) or high <= 0:
            return signals

        # Кратность сама по себе ничего не значит: при редком исходе два
        # события против трёх дают полуторную разницу, будучи шумом.
        spread = math.sqrt(
            _rate_error(low, sizes[low_name]) ** 2 + _rate_error(high, sizes[high_name]) ** 2
        )
        if high - low <= SIGMA * spread:
            return signals

        shown = ", ".join(f"{name}: {value:.1%}" for name, value in sorted(rates.items()))
        times = (
            f"{high / low:.1f} раза"
            if low > 0
            else "бесконечное число раз (в окне нет ни одного положительного)"
        )
        return [
            *signals,
            Signal(
                Finding.NON_STATIONARY_TARGET,
                f"доля положительного класса различается между окнами в "
                f"{times} ({shown}). Модель, обученная на одной доле "
                "и оценённая на другой, сравнивается не сама с собой",
                blocking=True,
            ),
        ]


@dataclass(frozen=True)
class ComparableSupport:
    """N5. Значения признака в разных окнах почти не пересекаются.

    Пока поддержка не пересекается, вывод об изменении зависимости делать
    нельзя: сравниваются разные участки шкалы, а не разные времена.
    """

    requirement: str = "N5"
    premises: frozenset[Premise] = frozenset({Premise.UNIVERSAL})
    detects: frozenset[Finding] = frozenset({Finding.INCOMPARABLE_SUPPORT})

    floor: float = 0.5
    """Какую долю диапазона окна обязано покрывать пересечение."""

    def run(self, context: Context) -> list[Signal]:
        windows = _windows_with_labels(context)
        return [
            Signal(
                Finding.INCOMPARABLE_SUPPORT,
                f"признак {name!r} принимает в окнах почти непересекающиеся значения "
                f"(пересечение покрывает {share:.0%} диапазона). Вывод об изменении "
                "зависимости по нему неприменим: сравниваются разные участки шкалы",
                blocking=True,
            )
            for name, share in sorted(incomparable(context, windows, self.floor).items())
        ]


def support_overlap(windows: list[tuple[str, pl.DataFrame]], feature: str) -> float | None:
    """Какую долю типичного диапазона покрывает пересечение окон.

    Границы берутся по 5-му и 95-му процентилю: одиночный выброс не должен
    создавать видимость пересечения.
    """
    spans = []
    for _, frame in windows:
        if feature not in frame.columns:
            return None
        series = frame[feature].drop_nulls()
        if series.len() < MIN_ROWS or not series.dtype.is_numeric():
            return None
        spans.append((float(series.quantile(0.05)), float(series.quantile(0.95))))

    lo = max(s[0] for s in spans)
    hi = min(s[1] for s in spans)
    widths = [s[1] - s[0] for s in spans]
    typical = min(widths)
    if typical <= 0:
        # Признак постоянен хотя бы в одном окне. Совпадающие константы
        # сопоставимы полностью, разошедшиеся — не сопоставимы вовсе. Возврат
        # None прятал второй случай как невычислимость.
        return 1.0 if hi >= lo else 0.0
    return max(0.0, (hi - lo) / typical)


def incomparable(
    context: Context, windows: list[tuple[str, pl.DataFrame]], floor: float
) -> dict[str, float]:
    """Признаки, чью поддержку сравнивать между окнами нельзя."""
    result = {}
    for name in _numeric_features(context):
        share = support_overlap(windows, name)
        if share is not None and share < floor:
            result[name] = share
    return result


@dataclass(frozen=True)
class FeatureRelationStability:
    """N4. Связь признака с исходом меняет знак или силу между окнами.

    Смена знака и ослабление связи — разные болезни. Первая означает, что
    процесс изменился, вторая — что признак перестал работать. Сообщения
    разные, потому что действия разные.
    """

    requirement: str = "N4"
    premises: frozenset[Premise] = frozenset({Premise.BINARY_TARGET})
    detects: frozenset[Finding] = frozenset({Finding.UNSTABLE_FEATURE_RELATION})

    floor: float = 0.05
    """Ниже этой силы связь считается шумом, и её знак ничего не значит."""

    ratio: float = 3.0
    """Во сколько раз сила может меняться, прежде чем это станет вопросом."""

    strong: float = 0.15
    """Связь, ниже которой ослабление неотличимо от шума.

    Без этого порога проверка кричала на чистом мире: связь 0.00 в одном окне
    и 0.07 в другом даёт кратность в семнадцать раз, будучи шумом в обоих.
    """

    def run(self, context: Context) -> list[Signal]:
        windows = _windows_with_labels(context)
        skip = set(incomparable(context, windows, ComparableSupport.floor))

        signals = []
        for name in _numeric_features(context):
            if name in skip:
                continue  # об этом сказала N5, и вывод здесь был бы ложным
            values: dict[str, float] = {}
            for window, frame in windows:
                if name not in frame.columns or not frame[name].dtype.is_numeric():
                    break
                value = association(frame[name], frame[LABEL])
                if value is None:
                    break
                values[window] = value
            if len(values) != len(windows):
                continue

            # Значимой считается связь, превосходящая шум выборки этого окна.
            noise = {w: SIGMA * _association_error(f[LABEL]) for w, f in windows}
            strong = {w: v for w, v in values.items() if abs(v) >= max(self.floor, noise[w])}
            shown = ", ".join(f"{w}: {v:+.2f}" for w, v in sorted(values.items()))

            signs = {v > 0 for v in strong.values()}
            if len(signs) > 1:
                signals.append(
                    Signal(
                        Finding.UNSTABLE_FEATURE_RELATION,
                        f"признак {name!r} меняет ЗНАК связи с исходом между окнами "
                        f"({shown}). Процесс изменился, и модель, обученная на раннем "
                        "окне, на позднем будет ошибаться в обратную сторону",
                        blocking=True,
                    )
                )
                continue

            # Ослабление сильной связи до шума — это и есть «признак перестал
            # работать», и требовать двух сильных окон значило бы молчать
            # именно в этом случае.
            #
            # Но кратности мало: 0.20 против 0.05 при шуме 0.15 — одно и то же
            # значение, измеренное дважды. Поэтому требуется, чтобы РАЗРЫВ
            # между окнами превосходил совместный шум, а не только максимум.
            magnitudes = {w: abs(v) for w, v in values.items()}
            high_window = max(magnitudes, key=lambda w: magnitudes[w])
            low_window = min(magnitudes, key=lambda w: magnitudes[w])
            high, low = magnitudes[high_window], magnitudes[low_window]
            spread = math.sqrt((noise[high_window] / SIGMA) ** 2 + (noise[low_window] / SIGMA) ** 2)
            ratio_broken = low <= 0 or high / low >= self.ratio
            if (
                high >= max(self.strong, noise[high_window])
                and ratio_broken
                and high - low >= SIGMA * spread
            ):
                signals.append(
                    Signal(
                        Finding.UNSTABLE_FEATURE_RELATION,
                        f"сила связи признака {name!r} с исходом меняется между окнами "
                        f"более чем в {self.ratio:g} раза ({shown}). Признак работает "
                        "не весь период",
                        blocking=True,
                    )
                )
        return signals


@dataclass(frozen=True)
class DeclaredDirectionHolds:
    """N8. Наблюдаемое направление связи противоречит объявленному.

    Направление объявляется из доменного знания до просмотра данных.
    Расхождение означает либо неверное понимание процесса, либо испорченный
    признак — оба случая дороже, чем ошибка в модели.
    """

    requirement: str = "N8"
    premises: frozenset[Premise] = frozenset({Premise.BINARY_TARGET})
    detects: frozenset[Finding] = frozenset({Finding.DIRECTION_CONTRADICTS_DOMAIN})

    floor: float = 0.05
    """Слабая связь знака не имеет: спорить с доменным знанием нечем."""

    def run(self, context: Context) -> list[Signal]:
        declared = [
            c
            for c in context.world.schema.by_role(Role.FEATURE)
            if c.direction is not None and c.availability is Availability.AT_DECISION
        ]
        if not declared:
            raise NotApplicable("доменное направление не объявлено ни у одного признака")

        windows = _windows_with_labels(context)
        frame = pl.concat([f for _, f in windows], how="vertical_relaxed")
        # Порог считается с шумом выборки: на сотне строк случайный признак
        # даёт связь около 0.06 и противоречил бы объявленному направлению.
        threshold = max(self.floor, SIGMA * _association_error(frame[LABEL]))

        signals = []
        for column in declared:
            if column.name not in frame.columns or not frame[column.name].dtype.is_numeric():
                continue
            value = association(frame[column.name], frame[LABEL])
            if value is None or abs(value) < threshold:
                continue
            observed = Direction.INCREASES if value > 0 else Direction.DECREASES
            if observed is column.direction:
                continue
            signals.append(
                Signal(
                    Finding.DIRECTION_CONTRADICTS_DOMAIN,
                    f"признак {column.name!r} объявлен как {column.direction.value}, "
                    f"а по данным {observed.value} (связь {value:+.2f}). Либо процесс "
                    "понят неверно, либо признак построен неправильно",
                    blocking=True,
                )
            )
        return signals


@dataclass(frozen=True)
class CompetingKindsDeclared:
    """N12. Виды событий сведены в один исход, и не объявлено, намеренно ли.

    Свести четыре вида отказа в исход «отказало хоть что-то» бывает верно:
    если вмешательство одно на все виды, различать их незачем. Бывает и
    ошибкой: если ремонт адресный, модель предсказывает не то, чем управляют.

    По данным эти случаи неотличимы, поэтому блокируется отсутствие ответа,
    а не сама склейка. Ровно та же развилка, что с жизненным циклом объекта.
    """

    requirement: str = "N12"
    premises: frozenset[Premise] = frozenset({Premise.UNIVERSAL})
    detects: frozenset[Finding] = frozenset(
        {Finding.COMPETING_KINDS_COLLAPSED, Finding.SIMULTANEITY_UNDECLARED}
    )

    def run(self, context: Context) -> list[Signal]:
        from dsx.competing import kinds_in
        from dsx.task import TargetKind

        kinds = kinds_in(context.world)
        if len(kinds) < 2:
            raise NotApplicable("видов события меньше двух: конкуренции нет")

        if context.task.target_kind is TargetKind.COMPETING:
            declared = context.task.simultaneous_kinds
            if declared is not None:
                return _simultaneity_signals(context, declared, kinds)
            return [
                Signal(
                    Finding.SIMULTANEITY_UNDECLARED,
                    "конкурирующие исходы предполагают, что побеждает ровно один, "
                    "но что делать при одновременном наступлении, не объявлено. "
                    "По таблице решений это не проверить: там один вид на строку, "
                    "а нарушение происходит раньше, при её сборке",
                    blocking=True,
                )
            ]

        if context.task.kinds_collapsed is True:
            return []  # склейка объявлена намеренной

        # kinds_collapsed=False — это признание, что склейка НЕ задумана. Считать
        # такое признание ответом значит закрывать находку её же описанием.
        answered = context.task.kinds_collapsed is False
        preface = (
            "склейка объявлена НЕнамеренной, но исход всё равно один"
            if answered
            else f"в данных {len(kinds)} видов события ({', '.join(kinds)}), "
            "а исход объявлен одним. Объявите, намеренно ли"
        )
        return [
            Signal(
                Finding.COMPETING_KINDS_COLLAPSED,
                f"{preface}. Если вмешательство адресное, модель предсказывает не то, "
                "чем управляют. Наступление одного вида к тому же обрывает наблюдение "
                "за остальными, и такие строки не отрицательны, а неизвестны",
                blocking=True,
            )
        ]


def _simultaneity_signals(context: Context, declared, kinds: tuple[str, ...]) -> list[Signal]:
    """Сверить объявление об одновременности с данными, если оно проверяемо.

    Проверяемым его делает названное значение вида. Без него остаётся только
    записать допущение — что и делает запуск проекта.
    """
    from dsx.task import Simultaneity

    value = context.task.simultaneous_kind_value
    if value is None:
        return []

    present = value in kinds
    if declared is Simultaneity.NOT_POSSIBLE and present:
        return [
            Signal(
                Finding.SIMULTANEITY_UNDECLARED,
                f"объявлено, что одновременное наступление невозможно, но вид "
                f"{value!r} присутствует в данных: объявление противоречит данным",
                blocking=True,
            )
        ]
    if declared is not Simultaneity.NOT_POSSIBLE and not present:
        return [
            Signal(
                Finding.SIMULTANEITY_UNDECLARED,
                f"вид {value!r} назван обозначением одновременного случая, но в данных "
                "его нет: объявление описывает то, чего не происходит",
                blocking=True,
            )
        ]
    return []


@dataclass(frozen=True)
class ExpectedRateHolds:
    """N13. Наблюдаемая доля класса расходится с объявленным ожиданием.

    Ожидание объявляется из доменного знания до просмотра данных. Без него
    отличить честный дисбаланс от ошибки нечем: доля в один процент выглядит
    одинаково и там, и там.

    Расхождение меряется В ДОЛЯХ, а не в разах, и допуск объявляет автор.
    Прежняя версия сравнивала по кратности с порогом 2.0 и пропустила на
    шестнадцатом кейсе ошибку автора в 31 процентный пункт: 0.85 против 54.2%
    дают 1.57, что меньше двух. Кратность у долей около половины сжимает крупные
    расхождения, а у редких исходов раздувает мелкие.

    Шум тоже не подошёл: при 168 тысячах строк три стандартные ошибки — 0.36
    процентного пункта. Здесь сравнивается измерение с ДОГАДКОЙ, а у догадки
    выборочной ошибки нет; есть неуверенность автора, и объявляет её он.

    Проверка эта — единственное, чем ядро видит ошибку ПОСТРОИТЕЛЯ: свёртки,
    фильтра и разметки оно не видит вовсе, а их след в распределении исхода —
    видит. На том же кейсе подмена `min` на `max` в свёртке была поймана только
    ею.
    """

    requirement: str = "N13"
    premises: frozenset[Premise] = frozenset({Premise.BINARY_TARGET})
    detects: frozenset[Finding] = frozenset({Finding.RATE_CONTRADICTS_EXPECTATION})

    def run(self, context: Context) -> list[Signal]:
        expected = context.outcome.expected_positive_rate
        if expected is None:
            raise NotApplicable("ожидаемая доля класса не объявлена")
        tolerance = context.outcome.expected_within
        assert tolerance is not None, "контракт исхода требует допуск вместе с ожиданием"

        windows = _windows_with_labels(context)
        frame = pl.concat([f for _, f in windows], how="vertical_relaxed")
        observed = float(frame[LABEL].mean())

        # Наблюдаемый ноль при ненулевом ожидании — крайнее расхождение, но
        # только если строк достаточно, чтобы ноль что-то значил.
        if observed <= 0:
            if frame.height * expected < SIGMA:
                raise NotApplicable(
                    f"при ожидаемой доле {expected:.1%} и {frame.height:,} строках ноль "
                    "положительных ещё ни о чём не говорит"
                )
        elif abs(observed - expected) <= tolerance:
            return []
        return [
            Signal(
                Finding.RATE_CONTRADICTS_EXPECTATION,
                f"ожидалась доля положительного класса {expected:.1%} с допуском "
                f"{tolerance:.1%}, наблюдается {observed:.1%} — расхождение "
                f"{abs(observed - expected):.1%}. Либо разметка или фильтрация теряют "
                "строки, либо процесс понят неверно",
                blocking=True,
            )
        ]


@dataclass(frozen=True)
class NonDegenerateOutcome:
    """N15. Доля класса такова, что предсказывать нечего.

    Шестой кейс дал в первой постановке 95.3% одного класса, и НИ ОДНА
    проверка не возразила. N13 сверяет долю только с объявленным ожиданием и
    без него пропускается — то есть выключается ровно у того, кто о
    вырожденности не подумал, а значит и ожидания не объявил.

    Порог объявляет автор. Общего числа здесь нет: доля в один процент
    вырождена для просрочки доставки и совершенно законна для мошенничества.
    Порог, назначенный ядром, кричал бы на редких событиях — а проверка,
    кричащая на законном, запрещена правилами этого проекта.

    Отсутствие порога тоже блокирует, и это главное в проверке. Иначе она
    повторила бы дефект N13: молчание там, где нужнее всего.
    """

    requirement: str = "N15"
    premises: frozenset[Premise] = frozenset({Premise.BINARY_TARGET})
    detects: frozenset[Finding] = frozenset({Finding.DEGENERATE_OUTCOME})

    def run(self, context: Context) -> list[Signal]:
        floor = context.outcome.degenerate_beyond
        if floor is None:
            return [
                Signal(
                    Finding.DEGENERATE_OUTCOME,
                    "порог невырожденности не объявлен: не сказано, при какой доле "
                    "класса задача перестаёт иметь смысл. Без него вырожденный исход "
                    "проходит молча, и заключение делается о цели, в которой нечего "
                    "предсказывать",
                    blocking=True,
                )
            ]

        windows = _windows_with_labels(context)
        frame = pl.concat([f for _, f in windows], how="vertical_relaxed")
        observed = float(frame[LABEL].mean())
        if floor <= observed <= 1.0 - floor:
            return []

        constant = max(observed, 1.0 - observed)
        return [
            Signal(
                Finding.DEGENERATE_OUTCOME,
                f"доля положительного класса {observed:.1%} при объявленном пороге "
                f"невырожденности {floor:.1%}. Постоянное правило, не смотрящее ни "
                f"на что, верно в {constant:.1%} случаев: измеренное качество будет "
                "свойством популяции, а не модели",
                blocking=True,
            )
        ]


@dataclass(frozen=True)
class UnobservedCostIsNamed:
    """N16. Цена ненаблюдаемых исходов не названа, и она может быть велика.

    Ядро девять кейсов знало, что исход бывает ненаблюдаем: `OutcomeReason`
    различает незрелость, цензурирование и исключение. Но знание это
    заканчивалось разметкой — строки просто выпадали из счёта, и отчёт называл
    долю среди ОСТАВШИХСЯ, не говоря, скольких он не считал и чем они могли
    оказаться.

    Девятый кейс: 20.1% исследований исход не показали. Отчёт называл 36.6%
    уложившихся в срок; границы, в которых лежит правда, — [29.2%, 49.3%].
    Двадцать процентных пунктов, о которых читатель не узнавал.

    Проверка делает две разные вещи, и различие существенно.

    **Границы называются всегда**, когда ненаблюдаемых заметная доля. Это не
    возражение: ненаблюдаемость нормальна. Это отказ выдавать число без его
    цены.

    **Блокирует** же не ненаблюдаемость, а её СВЯЗЬ с объявленным признаком.
    Если наблюдаемость зависит от того, что известно в момент решения, значит
    выпавшие — не случайная часть популяции, а её отличимый кусок, и метрика
    описывает не ту популяцию, о которой сделан вывод. На девятом кейсе среди
    наблюдаемых 57.2% исследований спонсированы индустрией, среди молчащих —
    18.6%.
    """

    requirement: str = "N16"
    premises: frozenset[Premise] = frozenset({Premise.BINARY_TARGET})
    detects: frozenset[Finding] = frozenset({Finding.INFORMATIVE_UNOBSERVABILITY})

    share: float = 0.05
    """С какой доли ненаблюдаемых их цена перестаёт быть мелочью."""

    def run(self, context: Context) -> list[Signal]:
        split = _require_split(context)
        parts = [p.evaluate for p in split.parts if LABEL in p.evaluate.columns]
        if not parts:
            raise NotApplicable("оценочных частей с меткой нет")
        frame = pl.concat(parts, how="vertical_relaxed")
        if frame.height < MIN_ROWS:
            raise NotApplicable("строк слишком мало, чтобы говорить о границах")

        # Исключённый из популяции — не ненаблюдаемый. Различие существенно:
        # исключение объявлено автором и означает «объект не предполагался к
        # обработке», а ненаблюдаемость означает «исход есть, но его не видно».
        # Смешивать их значило бы предъявлять счёт за честное объявление.
        if REASON in frame.columns:
            frame = frame.filter(pl.col(REASON) != OutcomeReason.EXCLUDED.value)
        if frame.height < MIN_ROWS:
            raise NotApplicable("после исключённых строк слишком мало")

        observed = frame.filter(pl.col(LABEL).is_not_null())
        missing = frame.height - observed.height
        if observed.height == 0 or missing < frame.height * self.share:
            return []

        rate = float(observed[LABEL].mean())
        low = rate * observed.height / frame.height
        high = (rate * observed.height + missing) / frame.height
        signals = [
            Signal(
                Finding.INFORMATIVE_UNOBSERVABILITY,
                f"исход не наблюдается у {missing:,} из {frame.height:,} строк "
                f"({missing / frame.height:.1%}). Доля положительного класса среди "
                f"наблюдаемых {rate:.1%}, но правда лежит в границах "
                f"[{low:.1%}; {high:.1%}] — их ширина и есть цена ненаблюдаемости",
                blocking=False,
            )
        ]

        # Связь наблюдаемости с объявленным признаком: выпавшие — отличимый
        # кусок популяции, а не случайная её часть.
        seen = frame.with_columns(pl.col(LABEL).is_not_null().cast(pl.Int64).alias("__seen"))
        threshold = max(0.10, SIGMA * _association_error(seen["__seen"]))
        for column in context.world.schema.usable_features():
            if column.name not in seen.columns:
                continue
            value = association(seen[column.name], seen["__seen"])
            if abs(value) <= threshold:
                continue
            signals.append(
                Signal(
                    Finding.INFORMATIVE_UNOBSERVABILITY,
                    f"наблюдаемость исхода связана с признаком {column.name!r} "
                    f"(связь {value:+.2f} при пороге {threshold:.2f}). Выпавшие строки — "
                    "не случайная часть популяции, и метрика описывает не ту популяцию, "
                    "о которой будет сделан вывод",
                    blocking=True,
                )
            )
        return signals


@dataclass(frozen=True)
class ReasonForObservationTransfers:
    """N17. Смысл задачи меняется вместе с ПРИЧИНОЙ наблюдения.

    Строка существует не сама по себе. Проверка приходит на площадку по жалобе
    или по расписанию, анализ назначают заболевшему, инспекция идёт туда, где
    подозревают. Метрика, измеренная на смеси поводов, верна для смеси и неверна
    для каждой части — а применяют её всегда к части.

    Десятый кейс измерил цену: доля нарушений 35.5% у плановых проверок против
    15.0% у вызванных сигналом, и модель, обученная на одном поводе и
    применённая к другому, теряет **0.215** разрешающей способности.

    Проверяется доля класса, а не качество модели: ядро предсказаний не
    производит. Расхождение долей — необходимый признак того, что задача у
    поводов разная, и его достаточно, чтобы возразить.

    Пропускается, если колонка повода не объявлена. Требовать её всюду значило
    бы требовать ответа, которого неоткуда взять: в большинстве выгрузок
    причины наблюдения нет. Но пропуск виден в отчёте, и молчание перестаёт
    быть незаметным.
    """

    requirement: str = "N17"
    premises: frozenset[Premise] = frozenset({Premise.BINARY_TARGET})
    detects: frozenset[Finding] = frozenset({Finding.OBSERVATION_REASON_MATTERS})

    ratio: float = 1.5
    """Во сколько раз доли у разных поводов могут различаться."""

    min_rows: int = MIN_ROWS
    """Повод, встречающийся реже, в сравнение не входит: доля по горстке строк
    гуляет сама по себе."""

    def run(self, context: Context) -> list[Signal]:
        declared = context.world.schema.by_role(Role.OBSERVATION_REASON)
        if not declared:
            raise NotApplicable(
                "причина наблюдения не объявлена: нечем сказать, почему строка существует"
            )

        column = declared[0].name
        parts = [p.evaluate for p in _require_split(context).parts if LABEL in p.evaluate.columns]
        if not parts:
            raise NotApplicable("оценочных частей с меткой нет")
        frame = _observable(pl.concat(parts, how="vertical_relaxed"))
        if column not in frame.columns or frame.height < self.min_rows:
            raise NotApplicable("строк с наблюдаемым исходом слишком мало")

        rates = (
            frame.group_by(column)
            .agg(pl.len().alias("rows"), pl.col(LABEL).mean().alias("rate"))
            .filter(pl.col("rows") >= self.min_rows)
            .sort(column, nulls_last=True)
        )
        if rates.height < 2:
            raise NotApplicable("поводов с достаточным числом строк меньше двух")

        low = rates.filter(pl.col("rate") == pl.col("rate").min()).row(0, named=True)
        high = rates.filter(pl.col("rate") == pl.col("rate").max()).row(0, named=True)
        if high["rate"] <= 0:
            return []  # исхода нет ни у одного повода: расходиться нечему

        # Нулевая доля у ОДНОГО повода при ненулевой у другого — расхождение
        # бесконечной кратности, а не повод к раннему возврату. Ровно эту ошибку
        # уже проходила N3, и я повторил её здесь дословно: первая версия
        # молчала на самом крайнем случае из всех.
        spread = math.sqrt(
            _rate_error(low["rate"], low["rows"]) ** 2
            + _rate_error(high["rate"], high["rows"]) ** 2
        )
        if low["rate"] > 0 and high["rate"] / low["rate"] < self.ratio:
            return []
        if high["rate"] - low["rate"] <= SIGMA * spread:
            return []

        return [
            Signal(
                Finding.OBSERVATION_REASON_MATTERS,
                f"доля положительного класса зависит от причины наблюдения: у "
                f"{high[column]!r} она {high['rate']:.1%} ({high['rows']:,} строк), у "
                f"{low[column]!r} — {low['rate']:.1%} ({low['rows']:,}). Задача у этих "
                "поводов разная, и метрика, измеренная на их смеси, не переносится ни на "
                "один из них по отдельности",
                blocking=True,
            )
        ]


DRIFT_CHECKS = [
    NonDegenerateOutcome(),
    UnobservedCostIsNamed(),
    ReasonForObservationTransfers(),
    TargetRateStationarity(),
    ComparableSupport(),
    FeatureRelationStability(),
    DeclaredDirectionHolds(),
    CompetingKindsDeclared(),
    ExpectedRateHolds(),
]
