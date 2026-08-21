"""Эмпирическая сверка объявлений с данными.

Проверки объявлений читают то, что написал человек, и потому бессильны против
ложного объявления: они видят то же ложное утверждение. Признак, вычисленный
из будущего, но объявленный доступным при принятии решения, проходит их
насквозь.

Поймать его можно только по силе связи с исходом: признак, знающий ответ,
предсказывает слишком хорошо для того, что якобы известно заранее.

Проверка не доказывает лик, а поднимает вопрос. Сильная связь бывает и
законной: она означает, что признак либо действительно ценен, либо содержит
ответ, и различить это может только человек, знающий процесс.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from dsx.checks.base import Context, NotApplicable, Signal
from dsx.evals.case import Finding
from dsx.label import LABEL, LabelError, compute
from dsx.outcome import OutcomeContractError
from dsx.roles import Availability, Role
from dsx.task import Premise


def separation(values: pl.Series, labels: pl.Series) -> float | None:
    """Насколько признак разделяет классы, от 0 до 0.5.

    Считается через ранговую статистику: ноль означает отсутствие связи,
    половина — идеальное разделение. Знак не важен, важна сила.
    """
    mask = values.is_not_null()
    values, labels = values.filter(mask), labels.filter(mask)
    if values.len() < 50 or values.n_unique() < 2:
        return None

    positives = int(labels.sum())
    negatives = labels.len() - positives
    if positives == 0 or negatives == 0:
        return None

    ranks = values.rank(method="average").to_numpy()
    rank_sum = float(ranks[labels.to_numpy().astype(bool)].sum())
    auc = (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)
    return abs(auc - 0.5)


@dataclass(frozen=True)
class ImplausibleSeparation:
    """N6. Признак разделяет классы слишком хорошо для доступного заранее.

    Порог задан абсолютным значением и превышением над остальными признаками:
    без второго условия проверка молчала бы там, где все признаки сильны, и
    кричала бы там, где все слабы.
    """

    requirement: str = "N6"
    premises: frozenset[Premise] = frozenset({Premise.BINARY_TARGET})
    detects: frozenset[Finding] = frozenset({Finding.FEATURE_AFTER_DECISION})

    floor: float = 0.35
    """Минимальная сила связи, ниже которой вопрос не поднимается."""

    excess: float = 2.5
    """Во сколько раз связь должна превосходить типичную среди признаков."""

    def run(self, context: Context) -> list[Signal]:
        schema = context.world.schema
        declared = [
            c for c in schema.by_role(Role.FEATURE) if c.availability is Availability.AT_DECISION
        ]
        if len(declared) < 2:
            return []

        try:
            frame = compute(context.world, context.outcome)
        except (LabelError, OutcomeContractError) as exc:
            # Контракт исхода нарушен — это уже отдельная находка, и метку
            # вычислить нечем. Молчать нельзя: пропуск попадает в отчёт.
            raise NotApplicable(f"исход не вычисляется: {exc}") from exc
        labels = frame[LABEL]

        strengths: dict[str, float] = {}
        for column in declared:
            if column.name not in frame.columns:
                continue
            series = frame[column.name]
            if not series.dtype.is_numeric():
                continue
            value = separation(series, labels)
            if value is not None:
                strengths[column.name] = value

        if len(strengths) < 2:
            return []

        typical = float(np.median(list(strengths.values())))
        signals = []
        for name, value in sorted(strengths.items(), key=lambda kv: -kv[1]):
            if value < self.floor:
                continue
            if typical > 0 and value < typical * self.excess:
                continue
            signals.append(
                Signal(
                    Finding.FEATURE_AFTER_DECISION,
                    f"признак {name!r} разделяет классы с силой {value:.2f} при типичной "
                    f"{typical:.2f} среди остальных. Объявлен доступным в момент решения — "
                    "проверьте, не вычислен ли он из исхода",
                    blocking=True,
                )
            )
        return signals


EMPIRICAL_CHECKS = [ImplausibleSeparation()]
