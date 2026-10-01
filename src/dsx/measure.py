"""Измерение: приём предсказаний и сравнение с объявленным правилом.

Ядро предсказаний НЕ ПРОИЗВОДИТ. Обучение живёт в коде проекта — там же, где
сборка таблицы решений. Сюда подаётся готовая колонка оценок, и здесь
проводится протокол измерения. `.fit()` в ядре не появляется: как только
инструмент владеет моделью, он становится очередным ML-фреймворком и
проигрывает зрелым, теряя единственное, чем ценен.

Измеряется не метрика модели, а РАЗНИЦА с базовым правилом, интервалом.
История проекта начинается с того, что вывод «модель лучше правила»
переворачивался трижды из-за протокола. Одинокое число 0.71 не говорит ничего:
сравнивать не с чем, а точность оценки неизвестна.

Разрешающая способность и калибровка считаются ПОРОЗНЬ. Модель бывает лучше
ранжирует и хуже калибруется; склейка этих двух и позволяет перевороту.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from fractions import Fraction
from typing import Annotated

import numpy as np
import polars as pl
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sklearn.metrics import average_precision_score

from dsx.label import LABEL
from dsx.samples import Fit, Purpose, SampleLedger

BOOTSTRAP = 400
"""Число пересборок для интервала. Фиксировано: подбор числа пересборок под
желаемый результат — тот же выбор, что подбор порога."""

SEED = 20260101
"""Зерно пересборки. Отчёт обязан воспроизводиться побайтово (R10)."""

BINS = 10
"""Групп для оценки калибровки."""


class RuleKind(StrEnum):
    """Что делают без модели."""

    CONSTANT = "constant"
    """Всем одна и та же оценка — доля класса в обучении. Ничего не различает,
    но калибровано идеально: полезный контраст."""

    THRESHOLD = "threshold"
    """Порог по одному объявленному признаку. То, что напишет аналитик за час."""


class BaselineRule(BaseModel):
    """Базовое правило, с которым сравнивается модель.

    Обязательно. Без него измерять не с чем: одинокая метрика не отвечает на
    вопрос «стоило ли строить модель».
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: RuleKind
    feature: str | None = None
    """Признак для порогового правила."""

    threshold: float | None = None
    above_is_positive: bool = True
    constant: Annotated[float, Field(ge=0.0, le=1.0)] | None = None

    @model_validator(mode="after")
    def _fields_match_the_kind(self) -> BaselineRule:
        if self.kind is RuleKind.THRESHOLD and (self.feature is None or self.threshold is None):
            raise ValueError("пороговое правило обязано назвать признак и порог")
        if self.kind is RuleKind.CONSTANT and self.constant is None:
            raise ValueError("постоянное правило обязано назвать значение")
        return self

    def score(self, frame: pl.DataFrame) -> pl.Series:
        """Оценки правила для строк выборки, от 0 до 1."""
        if self.kind is RuleKind.CONSTANT:
            assert self.constant is not None
            return pl.Series("rule", np.full(frame.height, self.constant))

        assert self.feature is not None and self.threshold is not None
        if self.feature not in frame.columns:
            raise ValueError(f"признак правила {self.feature!r} отсутствует в выборке")
        side = (
            pl.col(self.feature) > self.threshold
            if self.above_is_positive
            else pl.col(self.feature) < self.threshold
        )
        return frame.select(side.cast(pl.Float64).alias("rule"))["rule"]

    def __str__(self) -> str:
        if self.kind is RuleKind.CONSTANT:
            return f"постоянная оценка {self.constant:.3f}"
        sign = ">" if self.above_is_positive else "<"
        return f"{self.feature} {sign} {self.threshold:g}"


def discrimination(scores: np.ndarray, labels: np.ndarray) -> float:
    """Разрешающая способность: доля верно упорядоченных пар, от 0 до 1.

    Половина означает, что оценки не различают классы.
    """
    positives = int(labels.sum())
    negatives = labels.size - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, scores.size + 1)
    # Связки получают средний ранг: без этого постоянное правило выглядело бы
    # различающим.
    unique, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    sums = np.zeros(unique.size)
    np.add.at(sums, inverse, ranks)
    ranks = (sums / counts)[inverse]
    return float(
        (ranks[labels == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives)
    )


def calibration_error(scores: np.ndarray, labels: np.ndarray, bins: int = BINS) -> float:
    """Средняя по группам разница между обещанной и наблюдённой долей.

    Ноль означает, что оценка 0.3 действительно означает три случая из десяти.
    """
    if scores.size == 0:
        return float("nan")
    edges = np.quantile(scores, np.linspace(0, 1, bins + 1))
    edges[-1] = np.nextafter(edges[-1], np.inf)
    index = np.clip(np.searchsorted(edges, scores, side="right") - 1, 0, bins - 1)

    total = 0.0
    for group in range(bins):
        mask = index == group
        if not mask.any():
            continue
        total += mask.mean() * abs(scores[mask].mean() - labels[mask].mean())
    return float(total)


NUMPY_FORMS = frozenset(
    {
        ("b", 1),
        ("i", 1), ("i", 2), ("i", 4), ("i", 8),
        ("u", 1), ("u", 2), ("u", 4), ("u", 8),
        ("f", 2), ("f", 4), ("f", 8),
    }
)  # fmt: skip
"""Представления numpy во входах метрик — ПЕРЕЧНЕМ пар (вид, размер в байтах).

Седьмое ревью DS-008: категория `f` пропускала longdouble; там, где он точнее
float64, 1 + 2**-63 при переводе становилось 1, и оценка выше 1 проходила как
вероятность. Восьмое ревью (`6b808ae`): перечень скалярных ТИПОВ сверялся по
тождеству и отвергал `np.longlong` и `np.ulonglong` — те же восемь байт, что
int64 и uint64, но другой класс. Значение определяет представление, а не имя
класса: сверяется пара `(dtype.kind, dtype.itemsize)`. Порядок байтов в неё не
входит: `>f8` — тот же float64. longdouble больше восьми байт отвергается;
восьмибайтовый (macOS arm64) совпадает с float64 и переводится без потерь.
Пара описывает только СКАЛЯР: структурный и подмассивный dtype отвергаются до
неё — вид и размер у них берутся от основы (девятое ревью, `e955c8b`)."""


EXACT_INTEGER = 2**53
"""Целые по модулю до 2**53 переходят в float64 без потерь."""


def _type_name(kind: type) -> str:
    """Имя класса без обращения к его метаклассу.

    `kind.__name__` вызывает `__getattribute__` метакласса, и в ревью `c53a247`
    он обнулял оценки, пока ядро печатало сообщение отказа. Дескриптор берётся у
    самого `type`: он читает имя из структуры класса. Имя можно подменить
    подклассом `str` со своим `__format__` — `str.__str__` делает из него
    обычную строку, не вызывая его методов (ревью `0c71a26`).
    """
    return str.__str__(type.__dict__["__name__"].__get__(kind, type))


def _require_series(what: str, series: object) -> pl.Series:
    """Ровно `pl.Series` — до первого вызова его методов, включая `len` и `filter`.

    Ревью `c53a247`: подкласс переопределял `filter` или `cast`, оценки
    [-0.1, 1.1] становились [0, 1] и уходили в numpy обычным массивом.
    """
    if type(series) is not pl.Series:
        raise ValueError(
            f"{what}: ожидается ровно pl.Series, получено {_type_name(type(series))} — "
            "методы подкласса могли бы изменить значения до проверки"
        )
    return series


def _require_frame(frame: object) -> pl.DataFrame:
    """Ровно `pl.DataFrame` — сразу после выдачи журналом, до `height` и правила.

    Ревью `0c71a26`: после `checkout` шли `isinstance` и `frame.height`, точная
    проверка — только в `_observed`, после `rule.score`. Getter `height`
    подкласса переписывал метки и сам класс: «правило лучше» становилось
    «модель лучше», AUC в N7 — 0 → 1.
    """
    if type(frame) is not pl.DataFrame:
        raise ValueError(
            f"выборка: ожидается ровно pl.DataFrame, получено {_type_name(type(frame))}"
        )
    return frame


def _exact_real(what: str, value: object) -> np.ndarray:
    """Массив `float64`, полученный из исходного БЕЗ ПОТЕРЬ, — или отказ.

    Проверяется исходное значение, а не приведённое. Четыре круга ревью DS-008
    находили одно и то же — значение молча менялось до проверки, как строка в
    `preflight` (класс 20 журнала повторов): `np.asarray` снимал маску
    (`cdf9638`), приведение к float64 склеивало 2**53 и 2**53 + 1 (`cdf9638`),
    отбрасывало мнимую часть (`e3c4788`), Decimal выше 1 округлялся до 1
    (`cdf9638`). Приведение здесь только такое, которое ничего не меняет.
    """
    kind = type(value)
    if kind is not np.ndarray:
        # Тип берётся `type()` ПЕРВЫМ действием: `isinstance` читает `__class__`
        # объекта, и подменённый getter обнулял оценки до отказа (ревью
        # `c53a247`). `issubclass` сверяет классы, не трогая объект.
        if issubclass(kind, np.ma.MaskedArray):
            raise ValueError(f"{what}: маскированный массив — неизвестные значения скрыты маской")
        if issubclass(kind, np.ndarray):
            # Ревью `7dc1568`: приведение исполняло `astype` и `__array_finalize__`
            # подкласса — оценки [-0.1, 1.1] обрезались до [0, 1]. Подкласс можно
            # было бы снять `np.asarray` без вызова его кода; отказ — выбранная
            # политика: ядро принимает ровно `np.ndarray`, снимает вызывающий.
            raise ValueError(
                f"{what}: подкласс np.ndarray {_type_name(kind)} — передайте обычный "
                "np.ndarray, например np.asarray(...)"
            )
        # Последовательность Python приводится к ОБЩЕМУ типу до проверки:
        # [2**53, 2**53 + 1, 0.5] становился float64 целиком, и первые две оценки
        # совпадали (ревью `3bfbb95`). Принимается только готовый массив.
        raise ValueError(
            f"{what}: ожидается np.ndarray, получено {_type_name(kind)} — "
            "последовательность приводилась бы к общему типу с потерей"
        )
    array = value
    if array.dtype.fields is not None or array.dtype.subdtype is not None:
        # Девятое ревью (`e955c8b`): у структурного dtype с основой float64 вид `f`
        # и размер 8, а приведение снимает поля и считает по основе. Перечень
        # описывает скаляр; запись под него не подходит, каким бы ни был её вид.
        raise ValueError(
            f"{what}: структурный dtype {array.dtype} — выберите поле явно, "
            "приведение сняло бы структуру молча"
        )
    if (array.dtype.kind, array.dtype.itemsize) not in NUMPY_FORMS:
        raise ValueError(
            f"{what} не вещественные в поддерживаемом виде: dtype {array.dtype} не поддерживается"
        )
    converted = array.astype(np.float64)
    if array.dtype.kind in "iu" and array.size:
        beyond = (array > EXACT_INTEGER) | (array < -EXACT_INTEGER)
        if beyond.any() and any(
            int(f) != int(v) for f, v in zip(converted[beyond], array[beyond], strict=True)
        ):
            raise ValueError(
                f"{what}: целые вне ±2**53 теряют точность в float64 — порядок оценок изменился бы"
            )
    return converted


def _require_aligned(scores: object, labels: object) -> tuple[np.ndarray, np.ndarray]:
    """Входы метрики, приведённые к счёту без потерь: оценки и метки `float64`→`int64`.

    Отказ, если массивы маскированы, не одномерные, разной длины, не
    вещественного вида, теряют точность при приведении или метка не 0 и не 1.
    Все метрики и `_compare` берут входы ТОЛЬКО отсюда.

    Ревью `75a6f1c`: NaN-метка становилась отрицательным исходом; одна метка
    растягивалась на две оценки (broadcasting). Ревью `e3c4788`: комплексные
    оценки проходили с отброшенной мнимой частью; булевы 0/1 роняли вычитание.
    """
    scores, labels = _exact_real("оценки", scores), _exact_real("метки", labels)
    if scores.ndim != 1 or labels.ndim != 1:
        raise ValueError("оценки и метки обязаны быть одномерными")
    if scores.shape != labels.shape:
        raise ValueError(f"оценок {scores.size}, меток {labels.size}: не выровнены")
    if not np.all(np.isin(labels, (0, 1))):
        raise ValueError("метка не 0 и не 1: неизвестный исход не бывает отрицательным")
    return scores, labels.astype(np.int64)


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    """PR-AUC как average precision — определение `sklearn`, не трапеция.

    Точность, усреднённая по порогам с весом прироста полноты; равные оценки —
    один порог. Трапеция по кривой точность–полнота даёт другое число и молча с
    этим не приравнивается. Опорное значение — доля класса: это точный AP
    постоянной оценки и ориентир для случайного ранжирования (не его точное
    значение: у двух строк с одним событием случайный порядок даёт в среднем
    0.75 при доле 0.5). `nan`, если нет положительных или оценки не конечны.
    """
    scores, labels = _require_aligned(scores, labels)
    if labels.size == 0 or int(labels.sum()) == 0 or not np.all(np.isfinite(scores)):
        return float("nan")
    return float(average_precision_score(labels, scores))


def _probabilities(scores: np.ndarray) -> bool:
    """Оценки — вероятности: конечные и в [0, 1]. Иначе Brier и log loss не определены."""
    if scores.size == 0:
        return False
    return bool(np.all(np.isfinite(scores)) and scores.min() >= 0.0 and scores.max() <= 1.0)


def brier_score(scores: np.ndarray, labels: np.ndarray) -> float:
    """Средний квадрат разницы вероятности и исхода: разрешающая способность и
    калибровка сразу. `nan`, если оценки не вероятности."""
    scores, labels = _require_aligned(scores, labels)
    if scores.size == 0 or not _probabilities(scores):
        return float("nan")
    return float(np.mean((scores - labels) ** 2))


def log_loss(scores: np.ndarray, labels: np.ndarray) -> float:
    """Средний минус логарифм вероятности, отданной случившемуся исходу.

    БЕЗ ПОДРЕЗКИ. Оценка ровно 0 или 1 при противоположном исходе даёт
    бесконечность: модель поставила всё и проиграла. Подрезка до 1e-15 выдала бы
    за число то, что числом не является, — это молчаливая починка. `nan`, если
    оценки не вероятности.
    """
    scores, labels = _require_aligned(scores, labels)
    if scores.size == 0 or not _probabilities(scores):
        return float("nan")
    # log1p(−p), а не log(1 − p): 1 − 1e-20 округлялось до 1, и потеря 1e-20
    # становилась −0.0 (ревью `cdf9638`). При p = 1 log1p(−1) = −∞ — уверенная
    # ошибка остаётся бесконечностью.
    with np.errstate(divide="ignore"):
        losses = np.where(labels == 1, -np.log(scores), -np.log1p(-scores))
    return float(np.mean(losses))


class Costs(BaseModel):
    """Стоимость ошибок, объявленная для решения по оценке.

    Умолчания нет: равные стоимости — такое же объявление, как любые другие, и
    приниматься молча не может.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    false_positive: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    """Цена действия там, где события не будет."""
    false_negative: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    """Цена бездействия там, где событие будет."""

    @model_validator(mode="after")
    def _threshold_is_representable(self) -> Costs:
        threshold = self.threshold
        if not 0.0 < threshold < 1.0:
            raise ValueError(
                f"порог {threshold!r} непредставим строго внутри (0, 1): стоимости "
                "несоизмеримы в двойной точности. Подрезки нет — объявите соизмеримые"
            )
        return self

    @property
    def threshold(self) -> float:
        """Порог вероятности, выше которого действовать дешевле. Верен для
        КАЛИБРОВАННЫХ вероятностей: при смещённой калибровке порог смещается тоже.

        Точной дробью из двух `float`, округлённой один раз. Три круга ревью
        находили край арифметики с плавающей точкой: сумма переполнялась
        (`0dad089`), отношение переполнялось (`75a6f1c`), `1 + 1e-16` округлялось
        до 1 и порог 0.9999999999999999 терялся (`e3c4788`). Точный счёт краёв не
        имеет: непредставим только порог, который округляется в 0 или 1."""
        fp, fn = Fraction(self.false_positive), Fraction(self.false_negative)
        return float(fp / (fp + fn))


def net_benefit(scores: np.ndarray, labels: np.ndarray, threshold: float) -> float:
    """Чистая польза решения «действовать при оценке ≥ порога» (decision curve).

    В единицах верных срабатываний на строку: верные срабатывания минус ложные,
    взвешенные ценой, заложенной в порог, — p/(1−p). «Не действовать никогда»
    даёт ноль. `nan`, если оценки не вероятности: порог стоит на шкале вероятности.
    """
    if not 0.0 < threshold < 1.0:
        raise ValueError(f"порог {threshold!r} вне (0, 1): вес ложного срабатывания не определён")
    scores, labels = _require_aligned(scores, labels)
    if labels.size == 0 or not _probabilities(scores):
        return float("nan")
    act = scores >= threshold
    true = int(np.sum(act & (labels == 1)))
    false = int(np.sum(act & (labels == 0)))
    return (true - false * threshold / (1.0 - threshold)) / labels.size


@dataclass(frozen=True)
class Unmeasured:
    """Величина, которую на этой выборке честно посчитать нельзя, и почему."""

    name: str
    reason: str

    decisive = False
    """Неопределённое ничего не показывает: ни победы, ни поражения."""
    model_wins = False

    def __str__(self) -> str:
        return f"{self.name}: не определено — {self.reason}"


@dataclass(frozen=True)
class Comparison:
    """Сравнение модели с правилом по одной величине."""

    name: str
    model: float
    baseline: float
    low: float
    high: float
    lower_is_better: bool = False

    @property
    def difference(self) -> float:
        return self.model - self.baseline

    @property
    def decisive(self) -> bool:
        """Исключает ли интервал ноль."""
        return self.low > 0 or self.high < 0

    @property
    def model_wins(self) -> bool:
        if not self.decisive:
            return False
        return self.difference < 0 if self.lower_is_better else self.difference > 0

    def __str__(self) -> str:
        verdict = (
            ("модель лучше" if self.model_wins else "правило лучше")
            if self.decisive
            else "превосходство не показано"
        )
        return (
            f"{self.name}: модель {self.model:.4f}, правило {self.baseline:.4f}, "
            f"разница {self.difference:+.4f} [{self.low:+.4f}; {self.high:+.4f}] — {verdict}"
        )


def _interval(differences: np.ndarray) -> tuple[float, float]:
    return float(np.quantile(differences, 0.025)), float(np.quantile(differences, 0.975))


@dataclass(frozen=True)
class Verdict:
    """Итог измерения: два сравнения и вывод словами."""

    sample: str
    rows: int
    rule: BaselineRule
    ranking: Comparison | Unmeasured
    calibration: Comparison | Unmeasured
    prevalence: float = float("nan")
    precision: Comparison | Unmeasured | None = None
    brier: Comparison | Unmeasured | None = None
    log_loss: Comparison | Unmeasured | None = None
    costs: Costs | None = None
    benefit: Comparison | Unmeasured | None = None
    treat_all: float = float("nan")

    def statement(self) -> str:
        """Вывод словами. Осторожный там, где интервал ноль не исключает."""
        if isinstance(self.ranking, Unmeasured):
            return (
                f"превосходство модели над правилом «{self.rule}» не показано: "
                f"разрешающая способность не определена — {self.ranking.reason}"
            )
        if not self.ranking.decisive:
            return (
                f"превосходство модели над правилом «{self.rule}» не показано: "
                f"интервал на разницу разрешающей способности включает ноль"
            )
        if (
            self.ranking.model_wins
            and self.calibration.decisive
            and not self.calibration.model_wins
        ):
            return (
                f"модель ранжирует лучше правила «{self.rule}», но калибрована хуже: "
                "порядок верен, обещанные вероятности — нет"
            )
        if self.ranking.model_wins:
            return f"модель превосходит правило «{self.rule}» по разрешающей способности"
        return f"правило «{self.rule}» превосходит модель по разрешающей способности"

    def report_section(self) -> str:
        lines = [
            "## Измерение",
            "",
            f"Выборка: {self.sample}, строк {self.rows:,}. Базовое правило: {self.rule}.",
            "",
            f"- {self.ranking}",
            f"- {self.calibration}",
        ]
        # Доля класса и польза «всегда» не определены только при n = 0: метки уже
        # проверены на 0 и 1. Ревью `22eebc0`: отчёт печатал их как `nan`, а
        # «никогда 0» — как измеренный ноль.
        undefined = "не определено — нет строк с наблюдаемым исходом"
        if self.precision is not None:
            lines += [
                f"- доля класса {self.prevalence:.4f} — PR-AUC постоянной оценки и ориентир "
                "для случайного ранжирования"
                if np.isfinite(self.prevalence)
                else f"- доля класса: {undefined}",
                f"- {self.precision}",
            ]
        lines += [f"- {c}" for c in (self.brier, self.log_loss) if c is not None]
        if self.costs is None:
            lines.append(
                "- стоимость ошибок не объявлена: порог решения и чистая польза не считались"
            )
        else:
            threshold = self.costs.threshold
            lines.append(
                f"- стоимость ошибок: ложное срабатывание {self.costs.false_positive:g}, "
                f"пропуск {self.costs.false_negative:g} — действовать при оценке "
                f"≥ {threshold:.4f}; порог верен для калиброванных вероятностей"
            )
            lines.append(f"- {self.benefit}")
            lines.append(
                f"- чистая польза при том же пороге: действовать всегда {self.treat_all:+.4f}, "
                "никогда 0"
                if np.isfinite(self.treat_all)
                else f"- чистая польза стратегий «всегда» и «никогда»: {undefined}"
            )
        lines += ["", self.statement()]
        return "\n".join(lines)


SERIES_TYPES = frozenset(
    {
        pl.Boolean,
        pl.Int8, pl.Int16, pl.Int32, pl.Int64,
        pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
        pl.Float32, pl.Float64,
        pl.Null,
    }
)  # fmt: skip
"""Типы столбцов, которые переходят в numpy без потерь, — ПЕРЕЧНЕМ, а не признаком.

Шестое ревью DS-008: проверка «целое или вещественное» пропускала Int128, а его
`to_numpy()` в polars не реализован и падает паникой, которую не ловит даже
`except Exception`. Перечень разрешённого не расширяется молча с новыми типами
polars. `pl.Null` — столбец, где все значения пусты: законно, исход не наблюдался.
"""


def _series_exact(what: str, series: pl.Series) -> np.ndarray:
    """Столбец polars в `float64` без потерь; null — NaN, то есть неизвестное.

    Проверка идёт на ГРАНИЦЕ polars → numpy, по dtype polars: `to_numpy()` сам
    меняет значения — nullable Int64 переводит в float64, склеивая 2**53 и
    2**53 + 1, nullable Boolean превращает в object, Duration отдаёт как
    timedelta, который проходил `isin((0, 1))` (ревью `3bfbb95`). Значения без
    null уходят в numpy в своём dtype и проверяются на точность; затем перевод в
    float64 уже ничего не меняет.
    """
    dtype = _require_series(what, series).dtype
    if dtype not in SERIES_TYPES:
        raise ValueError(
            f"{what} не вещественные в поддерживаемом виде: тип {dtype} не поддерживается — "
            "допустимы Boolean, целые до 64 бит, Float32, Float64"
        )
    _exact_real(what, series.drop_nulls().to_numpy())
    return series.cast(pl.Float64).to_numpy()


def _observed(frame: pl.DataFrame, *scores: pl.Series) -> tuple[np.ndarray, ...]:
    """Метки и оценки строк с наблюдаемым исходом — проверенные ДО приведения.

    Единственный путь входов для измерения против правила, сравнения постановок
    и устойчивости по окнам. Прежде каждая функция приводила сама:
    `fill_null(0).astype(int64)` делал из NaN-метки число, `.astype(float64)`
    округлял Decimal и принимал строки (класс 20 журнала повторов). Строки без
    наблюдаемого исхода отбрасываются ещё в polars — до перевода, который мог бы
    испортить их оценки (nullable Boolean → object).
    """
    _require_frame(frame)
    for series in scores:
        _require_series("оценки", series)
    observable = frame[LABEL].is_not_null()
    labels = _series_exact("метки", frame[LABEL].filter(observable))
    if not np.all(np.isin(labels, (0, 1))):
        raise ValueError(
            "в колонке исхода метка не 0 и не 1 (NaN — не пусто): неизвестный исход "
            "не бывает отрицательным"
        )
    return (
        labels.astype(np.int64),
        *(_series_exact("оценки", s.filter(observable)) for s in scores),
    )


def measure_against_baseline(
    ledger: SampleLedger,
    sample: str,
    scores: pl.Series,
    rule: BaselineRule,
    decision: str = "итоговая оценка",
    costs: Costs | None = None,
) -> Verdict:
    """Измерить на выборке, израсходовав её тем же действием.

    Данные берутся ТОЛЬКО через журнал: получить их иначе нельзя, и потому
    измерение не может обойти учёт. Повторный заход блокируется — посмотреть
    метрику, подкрутить порог и посмотреть снова здесь невозможно.
    """
    _require_series("оценки", scores)
    frame = _require_frame(ledger.checkout(sample, Purpose.MEASUREMENT, decision))
    ledger.require_features()

    if scores.len() != frame.height:
        raise ValueError(
            f"предсказаний {scores.len():,}, а строк в выборке {frame.height:,}: "
            "оценки не выровнены по единице решения"
        )

    labels, model, baseline = _observed(frame, scores, rule.score(frame))

    return Verdict(
        sample=sample,
        rows=int(labels.size),
        rule=rule,
        ranking=_compare("разрешающая способность", discrimination, model, baseline, labels),
        calibration=_compare(
            "ошибка калибровки", calibration_error, model, baseline, labels, lower_is_better=True
        ),
        prevalence=float(labels.mean()) if labels.size else float("nan"),
        precision=_compare(
            "PR-AUC (average precision)", average_precision, model, baseline, labels
        ),
        brier=_compare("Brier", brier_score, model, baseline, labels, lower_is_better=True),
        log_loss=_compare("log loss", log_loss, model, baseline, labels, lower_is_better=True),
        costs=costs,
        benefit=None
        if costs is None
        else _compare(
            f"чистая польза при пороге {costs.threshold:.4f}",
            lambda s, y: net_benefit(s, y, costs.threshold),
            model,
            baseline,
            labels,
        ),
        treat_all=float("nan")
        if costs is None
        else net_benefit(np.ones(labels.size), labels, costs.threshold),
    )


def _compare(
    name: str,
    metric,
    model: np.ndarray,
    baseline: np.ndarray,
    labels: np.ndarray,
    *,
    lower_is_better: bool = False,
    who: tuple[str, str] = ("модели", "правила"),
) -> Comparison | Unmeasured:
    """Сравнение по одной величине с парным bootstrap, либо честное «не определено».

    Одна пересборка для всех величин: то же зерно, те же строки на каждом шаге у
    обеих сторон. Неопределённое объявляется с причиной: нет строк, неконечные
    оценки, бесконечная или неопределённая величина, негодные пересборки.
    """
    model, _ = _require_aligned(model, labels)
    baseline, labels = _require_aligned(baseline, labels)
    if labels.size == 0:
        return Unmeasured(name, "нет строк с наблюдаемым исходом")
    for side, scores in zip(who, (model, baseline), strict=True):
        if not np.all(np.isfinite(scores)):
            return Unmeasured(name, f"оценки {side} не конечны: NaN или бесконечность")
    left, right = metric(model, labels), metric(baseline, labels)
    for side, value, scores in zip(who, (left, right), (model, baseline), strict=True):
        if np.isfinite(value):
            continue
        if np.isinf(value):
            return Unmeasured(
                name, f"у {side} бесконечно: оценка ровно 0 или 1 при противоположном исходе"
            )
        if np.unique(labels).size < 2 and metric in (discrimination, average_precision):
            return Unmeasured(name, "на выборке нет обоих классов")
        if not _probabilities(scores):
            return Unmeasured(name, f"оценки {side} не вероятности: вне [0, 1]")
        return Unmeasured(name, "на выборке нет обоих классов")
    rng = np.random.default_rng(SEED)
    differences = np.empty(BOOTSTRAP)
    for i in range(BOOTSTRAP):
        pick = rng.integers(0, labels.size, labels.size)
        differences[i] = metric(model[pick], labels[pick]) - metric(baseline[pick], labels[pick])
    undefined = int(np.sum(~np.isfinite(differences)))
    if undefined:
        # Отбросить негодные пересборки молча значило бы построить интервал по
        # другой, отобранной выборке (ревью `0dad089`: y=[0,1] — 100 из 400).
        return Unmeasured(
            name,
            f"интервал не построен: в {undefined} из {BOOTSTRAP} пересборок величина "
            "не определена (нет обоих классов)",
        )
    low, high = _interval(differences)
    return Comparison(name, left, right, low, high, lower_is_better=lower_is_better)


@dataclass(frozen=True)
class Contrast:
    """Сравнение двух ПОСТАНОВОК на одной выборке.

    Отличается от `Verdict` предметом: там модель меряется против правила и
    отвечает на вопрос «стоило ли строить», здесь две постановки меряются друг
    против друга и отвечают на вопрос «сколько стоит различие между ними».

    Шестой кейс потребовал этого впервые: цена изменчивости истории есть
    разница между моделью на признаках момента решения и моделью на признаках
    момента выгрузки. Сравнить каждую с правилом по отдельности недостаточно —
    два интервала, каждый из которых ноль не включает, ничего не говорят о том,
    исключает ли ноль их РАЗНИЦА.
    """

    sample: str
    rows: int
    left_name: str
    right_name: str
    left: float
    right: float
    low: float
    high: float
    reason: str | None = None
    """Почему сравнение не определено; `None` — определено."""

    @property
    def difference(self) -> float:
        return self.left - self.right

    @property
    def decisive(self) -> bool:
        return self.reason is None and (self.low > 0 or self.high < 0)

    def __str__(self) -> str:
        if self.reason is not None:
            return f"разрешающая способность: не определено — {self.reason}"
        verdict = (
            f"{self.left_name if self.difference > 0 else self.right_name} различает лучше"
            if self.decisive
            else "различие не показано"
        )
        return (
            f"разрешающая способность: {self.left_name} {self.left:.4f}, "
            f"{self.right_name} {self.right:.4f}, разница {self.difference:+.4f} "
            f"[{self.low:+.4f}; {self.high:+.4f}] — {verdict}"
        )

    def report_section(self) -> str:
        return "\n".join(
            [
                "## Сравнение постановок",
                "",
                f"Выборка: {self.sample}, строк {self.rows:,}.",
                "",
                f"- {self}",
            ]
        )


def measure_contrast(
    ledger: SampleLedger,
    sample: str,
    left_scores: pl.Series,
    right_scores: pl.Series,
    left_name: str,
    right_name: str,
    decision: str = "сравнение двух постановок",
) -> Contrast:
    """Сравнить две постановки на одной выборке, израсходовав её один раз.

    Выборка берётся ОДНИМ checkout: две отдельные меры израсходовали бы её
    дважды, а второй заход журнал не пропустит — и правильно, потому что
    посмотреть, подкрутить и посмотреть снова здесь так же недопустимо.

    Пересчёт парный: обе постановки на каждом шаге считаются по ОДНОМУ И ТОМУ
    ЖЕ набору строк. Непарный дал бы интервал шире истинного, сложив в него
    разброс выборки, который у обеих постановок общий и потому сокращается.
    """
    _require_series(f"оценки {left_name!r}", left_scores)
    _require_series(f"оценки {right_name!r}", right_scores)
    frame = _require_frame(ledger.checkout(sample, Purpose.MEASUREMENT, decision))
    ledger.require_features()

    for name, scores in ((left_name, left_scores), (right_name, right_scores)):
        if scores.len() != frame.height:
            raise ValueError(
                f"у постановки {name!r} предсказаний {scores.len():,}, а строк "
                f"в выборке {frame.height:,}: оценки не выровнены по единице решения"
            )

    labels, left, right = _observed(frame, left_scores, right_scores)
    result = _compare(
        "разрешающая способность",
        discrimination,
        left,
        right,
        labels,
        who=(f"постановки «{left_name}»", f"постановки «{right_name}»"),
    )
    nan = float("nan")
    if isinstance(result, Unmeasured):
        return Contrast(
            sample,
            int(labels.size),
            left_name,
            right_name,
            nan,
            nan,
            nan,
            nan,
            reason=result.reason,
        )
    return Contrast(
        sample=sample,
        rows=int(labels.size),
        left_name=left_name,
        right_name=right_name,
        left=result.model,
        right=result.baseline,
        low=result.low,
        high=result.high,
    )


# --- разбор смещения по сегментам (N10) ------------------------------------


@dataclass(frozen=True)
class Segment:
    """Один сегмент: сколько строк, что обещано, что наблюдалось."""

    feature: str
    value: str
    rows: int
    predicted: float
    observed: float

    @property
    def bias(self) -> float:
        """Насколько обещанное расходится с наблюдённым. Знак важен."""
        return self.predicted - self.observed

    def __str__(self) -> str:
        side = "завышает" if self.bias > 0 else "занижает"
        return (
            f"{self.feature}={self.value!r}: {self.rows:,} строк, обещано "
            f"{self.predicted:.3f}, наблюдалось {self.observed:.3f} — "
            f"{side} на {abs(self.bias):.3f}"
        )


def segment_bias(
    frame: pl.DataFrame,
    scores: pl.Series,
    features: list[str],
    min_rows: int = 200,
) -> list[Segment]:
    """Смещение по сегментам, упорядоченное по ВЕЛИЧИНЕ, а не по объёму.

    Требование N10. Порядок по объёму прячет самое дорогое: крупный сегмент
    со смещением 0.01 неинтересен, мелкий со смещением 0.4 определяет, кому
    модель систематически вредит.

    Сегменты меньше `min_rows` не рассматриваются: на них смещение неотличимо
    от случайности, и включать их значило бы наполнить список шумом.
    """
    labelled = frame.with_columns(scores.alias("__score")).filter(pl.col(LABEL).is_not_null())
    found: list[Segment] = []

    for feature in features:
        if feature not in labelled.columns:
            continue
        grouped = (
            labelled.group_by(feature)
            .agg(
                pl.len().alias("rows"),
                pl.col("__score").mean().alias("predicted"),
                pl.col(LABEL).mean().alias("observed"),
            )
            .filter(pl.col("rows") >= min_rows)
        )
        found += [
            Segment(
                feature=feature,
                value=str(row[feature]),
                rows=int(row["rows"]),
                predicted=float(row["predicted"]),
                observed=float(row["observed"]),
            )
            for row in grouped.iter_rows(named=True)
        ]

    # Второй и третий ключи — ради воспроизводимости: при равном смещении
    # порядок иначе решает group_by, чей порядок в polars не определён.
    return sorted(found, key=lambda s: (-abs(s.bias), s.feature, s.value))


def segment_section(segments: list[Segment], show: int = 10) -> str:
    """Раздел отчёта: самые смещённые сегменты."""
    lines = ["## Смещение по сегментам", ""]
    if not segments:
        lines.append("Сегментов достаточного объёма не нашлось.")
        return "\n".join(lines)

    lines.append(
        f"Сегментов рассмотрено: {len(segments):,}. Упорядочены по величине смещения, "
        "а не по объёму: крупный сегмент с малым смещением дешевле мелкого с большим."
    )
    lines.append("")
    lines += [f"- {segment}" for segment in segments[:show]]
    return "\n".join(lines)


# --- устойчивость знака по окнам (N7, P5) ----------------------------------


@dataclass(frozen=True)
class Prediction:
    """Прогнозы окна вместе с записью обучения модели, которая их дала.

    Запись выдаёт журнал выборок (`SampleLedger.training`); без неё измерение
    по окнам прогнозы не принимает (DS-010). Связь прогнозов с самой моделью
    ядро не видит — это названный предел: построитель, обучивший модель не на
    выданном кадре, не обнаруживается.
    """

    fit: Fit
    scores: pl.Series


@dataclass(frozen=True)
class Stability:
    """Знак превосходства по окнам: держится ли вывод во времени."""

    per_window: dict[str, Comparison | Unmeasured]
    provenance: dict[str, str] = field(default_factory=dict)
    """Окно → на чём обучена модель, давшая его прогнозы."""

    @property
    def signs(self) -> set[bool]:
        return {c.difference > 0 for c in self.per_window.values() if c.decisive}

    @property
    def steady(self) -> bool:
        """Один и тот же знак во всех окнах, где он вообще различим."""
        return len(self.signs) <= 1

    def statement(self) -> str:
        decisive = [name for name, c in self.per_window.items() if c.decisive]
        undefined = sorted(n for n, c in self.per_window.items() if isinstance(c, Unmeasured))
        note = (
            f" Окна, где сравнение не определено и знак не проверен: {', '.join(undefined)}."
            if undefined
            else ""
        )
        if not decisive:
            return (
                "ни в одном окне превосходство не показано: вывод о пользе модели "
                "не опирается ни на что" + note
            )
        if not self.steady:
            return (
                f"знак превосходства МЕНЯЕТСЯ между окнами ({', '.join(decisive)}): "
                "вывод держится не на модели, а на выборе окна" + note
            )
        return f"знак превосходства одинаков во всех различимых окнах: {', '.join(decisive)}" + note

    def report_section(self) -> str:
        lines = ["## Устойчивость по окнам", ""]
        lines += [f"- {name}: {c}" for name, c in sorted(self.per_window.items())]
        if self.provenance:
            lines += ["", "Происхождение прогнозов:"]
            lines += [f"- {name}: {fit}" for name, fit in sorted(self.provenance.items())]
        lines += ["", self.statement()]
        return "\n".join(lines)


def stability_across_windows(
    ledger: SampleLedger,
    predictions: dict[str, Prediction],
    rule: BaselineRule,
    decision: str = "оценка устойчивости по окнам",
) -> Stability:
    """Сравнить модель с правилом в каждом окне отдельно.

    Заменяет требование N7. Прежняя формулировка велела переобучать модель на
    нескольких окнах истории — это обучение, и ядру оно не принадлежит. Здесь
    требуется предъявить предсказания ПО ОКНАМ и измерить, держится ли знак:
    доказательство, а не оркестровка.

    Прогнозы окна принимаются только с записью обучения, выданной журналом, и
    только если модель училась не на оценочных строках окна и не на метках,
    известных после его начала (DS-010). Кейсы 3–5 мерили ранние окна моделью,
    обученной на последнем, и N7 объявил знак устойчивым (класс 19 журнала
    повторов). Проверяется состав и время, а не число моделей: одна модель,
    обученная до самого раннего окна, проходит во всех.

    Все окна проверяются до расходования первого: отказ не тратит выборки.
    Расходует окна как выборки выбора: смотреть пооконные метрики и решать по
    ним — это выбор, а не аудит.
    """
    for window, prediction in predictions.items():
        if type(prediction) is not Prediction:
            raise ValueError(
                f"окно {window!r}: ожидается Prediction с записью обучения, получено "
                f"{_type_name(type(prediction))} — происхождение прогнозов неизвестно"
            )
        _require_series(f"оценки окна {window!r}", prediction.scores)
        defects = ledger.provenance_defects(window, prediction.fit)
        if defects:
            raise ValueError(f"окно {window!r}: прогнозы не принимаются — " + "; ".join(defects))

    per_window: dict[str, Comparison | Unmeasured] = {}
    provenance: dict[str, str] = {}
    for window, prediction in predictions.items():
        scores = prediction.scores
        frame = _require_frame(ledger.checkout(window, Purpose.SELECTION, decision))
        if scores.len() != frame.height:
            raise ValueError(
                f"в окне {window!r} предсказаний {scores.len():,}, а строк "
                f"{frame.height:,}: оценки не выровнены"
            )

        labels, model, baseline = _observed(frame, scores, rule.score(frame))
        per_window[window] = _compare(
            "разрешающая способность", discrimination, model, baseline, labels
        )
        provenance[window] = str(prediction.fit)
    return Stability(per_window=per_window, provenance=provenance)
