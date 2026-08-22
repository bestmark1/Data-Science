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
from dsx.label import LABEL
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
    """Во сколько раз доли могут различаться, прежде чем это станет вопросом."""

    def run(self, context: Context) -> list[Signal]:
        windows = _windows_with_labels(context)
        rates = {name: float(frame[LABEL].mean()) for name, frame in windows}
        sizes = {name: frame.height for name, frame in windows}

        low_name = min(rates, key=lambda k: rates[k])
        high_name = max(rates, key=lambda k: rates[k])
        low, high = rates[low_name], rates[high_name]
        if low <= 0 or high / low < self.ratio:
            return []

        # Кратность сама по себе ничего не значит: при редком исходе два
        # события против трёх дают полуторную разницу, будучи шумом.
        spread = math.sqrt(
            _rate_error(low, sizes[low_name]) ** 2 + _rate_error(high, sizes[high_name]) ** 2
        )
        if high - low < SIGMA * spread:
            return []

        shown = ", ".join(f"{name}: {value:.1%}" for name, value in sorted(rates.items()))
        return [
            Signal(
                Finding.NON_STATIONARY_TARGET,
                f"доля положительного класса различается между окнами в "
                f"{high / low:.1f} раза ({shown}). Модель, обученная на одной доле "
                "и оценённая на другой, сравнивается не сама с собой",
                blocking=True,
            )
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
        return None
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

            # Ослабление можно утверждать, только когда обе стороны сравнения
            # различимы на фоне шума: иначе сравнивается связь с пустотой.
            if len(strong) < 2:
                continue
            magnitudes = [abs(v) for v in strong.values()]
            low, high = min(magnitudes), max(magnitudes)
            if high >= self.strong and high / low >= self.ratio:
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

        signals = []
        for column in declared:
            if column.name not in frame.columns or not frame[column.name].dtype.is_numeric():
                continue
            value = association(frame[column.name], frame[LABEL])
            if value is None or abs(value) < self.floor:
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
            if context.task.simultaneous_kinds is not None:
                return []  # виды различаются, склейки нет, ответ об одновременности дан
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

        if context.task.kinds_collapsed is not None:
            return []  # ответ дан

        return [
            Signal(
                Finding.COMPETING_KINDS_COLLAPSED,
                f"в данных {len(kinds)} видов события ({', '.join(kinds)}), а исход "
                "объявлен одним. Объявите, намеренно ли: если вмешательство адресное, "
                "модель предсказывает не то, чем управляют. Наступление одного вида "
                "к тому же обрывает наблюдение за остальными, и такие строки не "
                "отрицательны, а неизвестны",
                blocking=True,
            )
        ]


@dataclass(frozen=True)
class ExpectedRateHolds:
    """N13. Наблюдаемая доля класса расходится с объявленным ожиданием.

    Ожидание объявляется из доменного знания до просмотра данных. Расхождение
    в разы означает либо ошибку фильтрации и разметки, либо неверное понимание
    процесса. Без объявления отличить честный дисбаланс от ошибки нечем: доля
    в один процент выглядит одинаково и там, и там.
    """

    requirement: str = "N13"
    premises: frozenset[Premise] = frozenset({Premise.BINARY_TARGET})
    detects: frozenset[Finding] = frozenset({Finding.RATE_CONTRADICTS_EXPECTATION})

    ratio: float = 2.0
    """Во сколько раз наблюдаемое может отличаться от ожидаемого."""

    def run(self, context: Context) -> list[Signal]:
        expected = context.outcome.expected_positive_rate
        if expected is None:
            raise NotApplicable("ожидаемая доля класса не объявлена")

        windows = _windows_with_labels(context)
        frame = pl.concat([f for _, f in windows], how="vertical_relaxed")
        observed = float(frame[LABEL].mean())
        if observed <= 0:
            return []

        high, low = max(observed, expected), min(observed, expected)
        if high / low < self.ratio:
            return []
        return [
            Signal(
                Finding.RATE_CONTRADICTS_EXPECTATION,
                f"ожидалась доля положительного класса {expected:.1%}, наблюдается "
                f"{observed:.1%} — расхождение в {high / low:.1f} раза. Либо разметка "
                "или фильтрация теряют строки, либо процесс понят неверно",
                blocking=True,
            )
        ]


DRIFT_CHECKS = [
    TargetRateStationarity(),
    ComparableSupport(),
    FeatureRelationStability(),
    DeclaredDirectionHolds(),
    CompetingKindsDeclared(),
    ExpectedRateHolds(),
]
