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
