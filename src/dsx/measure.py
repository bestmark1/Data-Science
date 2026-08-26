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

from dataclasses import dataclass
from enum import StrEnum
from typing import Annotated

import numpy as np
import polars as pl
from pydantic import BaseModel, ConfigDict, Field, model_validator

from dsx.label import LABEL
from dsx.samples import Purpose, SampleLedger

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
    ranking: Comparison
    calibration: Comparison

    def statement(self) -> str:
        """Вывод словами. Осторожный там, где интервал ноль не исключает."""
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
            "",
            self.statement(),
        ]
        return "\n".join(lines)


def measure_against_baseline(
    ledger: SampleLedger,
    sample: str,
    scores: pl.Series,
    rule: BaselineRule,
    decision: str = "итоговая оценка",
) -> Verdict:
    """Измерить на выборке, израсходовав её тем же действием.

    Данные берутся ТОЛЬКО через журнал: получить их иначе нельзя, и потому
    измерение не может обойти учёт. Повторный заход блокируется — посмотреть
    метрику, подкрутить порог и посмотреть снова здесь невозможно.
    """
    frame = ledger.checkout(sample, Purpose.MEASUREMENT, decision)
    assert isinstance(frame, pl.DataFrame)

    if scores.len() != frame.height:
        raise ValueError(
            f"предсказаний {scores.len():,}, а строк в выборке {frame.height:,}: "
            "оценки не выровнены по единице решения"
        )

    observable = frame[LABEL].is_not_null().to_numpy()
    labels = frame[LABEL].fill_null(0).to_numpy().astype(np.int64)[observable]
    model = scores.to_numpy().astype(np.float64)[observable]
    baseline = rule.score(frame).to_numpy().astype(np.float64)[observable]

    rank_model, rank_rule = discrimination(model, labels), discrimination(baseline, labels)
    cal_model, cal_rule = calibration_error(model, labels), calibration_error(baseline, labels)

    rng = np.random.default_rng(SEED)
    rank_diff = np.empty(BOOTSTRAP)
    cal_diff = np.empty(BOOTSTRAP)
    for i in range(BOOTSTRAP):
        pick = rng.integers(0, labels.size, labels.size)
        rank_diff[i] = discrimination(model[pick], labels[pick]) - discrimination(
            baseline[pick], labels[pick]
        )
        cal_diff[i] = calibration_error(model[pick], labels[pick]) - calibration_error(
            baseline[pick], labels[pick]
        )

    rank_low, rank_high = _interval(rank_diff)
    cal_low, cal_high = _interval(cal_diff)
    return Verdict(
        sample=sample,
        rows=int(observable.sum()),
        rule=rule,
        ranking=Comparison("разрешающая способность", rank_model, rank_rule, rank_low, rank_high),
        calibration=Comparison(
            "ошибка калибровки", cal_model, cal_rule, cal_low, cal_high, lower_is_better=True
        ),
    )


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

    @property
    def difference(self) -> float:
        return self.left - self.right

    @property
    def decisive(self) -> bool:
        return self.low > 0 or self.high < 0

    def __str__(self) -> str:
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
    frame = ledger.checkout(sample, Purpose.MEASUREMENT, decision)
    assert isinstance(frame, pl.DataFrame)

    for name, scores in ((left_name, left_scores), (right_name, right_scores)):
        if scores.len() != frame.height:
            raise ValueError(
                f"у постановки {name!r} предсказаний {scores.len():,}, а строк "
                f"в выборке {frame.height:,}: оценки не выровнены по единице решения"
            )

    observable = frame[LABEL].is_not_null().to_numpy()
    labels = frame[LABEL].fill_null(0).to_numpy().astype(np.int64)[observable]
    left = left_scores.to_numpy().astype(np.float64)[observable]
    right = right_scores.to_numpy().astype(np.float64)[observable]

    rng = np.random.default_rng(SEED)
    differences = np.empty(BOOTSTRAP)
    for i in range(BOOTSTRAP):
        pick = rng.integers(0, labels.size, labels.size)
        differences[i] = discrimination(left[pick], labels[pick]) - discrimination(
            right[pick], labels[pick]
        )

    low, high = _interval(differences)
    return Contrast(
        sample=sample,
        rows=int(observable.sum()),
        left_name=left_name,
        right_name=right_name,
        left=discrimination(left, labels),
        right=discrimination(right, labels),
        low=low,
        high=high,
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

    return sorted(found, key=lambda s: -abs(s.bias))


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
class Stability:
    """Знак превосходства по окнам: держится ли вывод во времени."""

    per_window: dict[str, Comparison]

    @property
    def signs(self) -> set[bool]:
        return {c.difference > 0 for c in self.per_window.values() if c.decisive}

    @property
    def steady(self) -> bool:
        """Один и тот же знак во всех окнах, где он вообще различим."""
        return len(self.signs) <= 1

    def statement(self) -> str:
        decisive = [name for name, c in self.per_window.items() if c.decisive]
        if not decisive:
            return (
                "ни в одном окне превосходство не показано: вывод о пользе модели "
                "не опирается ни на что"
            )
        if not self.steady:
            return (
                f"знак превосходства МЕНЯЕТСЯ между окнами ({', '.join(decisive)}): "
                "вывод держится не на модели, а на выборе окна"
            )
        return f"знак превосходства одинаков во всех различимых окнах: {', '.join(decisive)}"

    def report_section(self) -> str:
        lines = ["## Устойчивость по окнам", ""]
        lines += [f"- {name}: {c}" for name, c in sorted(self.per_window.items())]
        lines += ["", self.statement()]
        return "\n".join(lines)


def stability_across_windows(
    ledger: SampleLedger,
    scores_by_window: dict[str, pl.Series],
    rule: BaselineRule,
    decision: str = "оценка устойчивости по окнам",
) -> Stability:
    """Сравнить модель с правилом в каждом окне отдельно.

    Заменяет требование N7. Прежняя формулировка велела переобучать модель на
    нескольких окнах истории — это обучение, и ядру оно не принадлежит. Здесь
    требуется предъявить предсказания ПО ОКНАМ и измерить, держится ли знак:
    доказательство, а не оркестровка.

    Расходует окна как выборки выбора: смотреть пооконные метрики и решать по
    ним — это выбор, а не аудит.
    """
    per_window: dict[str, Comparison] = {}
    for window, scores in scores_by_window.items():
        frame = ledger.checkout(window, Purpose.SELECTION, decision)
        assert isinstance(frame, pl.DataFrame)
        if scores.len() != frame.height:
            raise ValueError(
                f"в окне {window!r} предсказаний {scores.len():,}, а строк "
                f"{frame.height:,}: оценки не выровнены"
            )

        observable = frame[LABEL].is_not_null().to_numpy()
        labels = frame[LABEL].fill_null(0).to_numpy().astype(np.int64)[observable]
        model = scores.to_numpy().astype(np.float64)[observable]
        baseline = rule.score(frame).to_numpy().astype(np.float64)[observable]

        rng = np.random.default_rng(SEED)
        differences = np.empty(BOOTSTRAP)
        for i in range(BOOTSTRAP):
            pick = rng.integers(0, labels.size, labels.size)
            differences[i] = discrimination(model[pick], labels[pick]) - discrimination(
                baseline[pick], labels[pick]
            )
        low, high = _interval(differences)
        per_window[window] = Comparison(
            "разрешающая способность",
            discrimination(model, labels),
            discrimination(baseline, labels),
            low,
            high,
        )
    return Stability(per_window=per_window)
