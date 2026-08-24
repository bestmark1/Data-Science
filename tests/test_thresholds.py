"""Пороги проверок против шума и против молчания (ревью Кодекса).

Проверка, срабатывающая по шуму, не отличается от отсутствия проверки.
Проверка, молчащая в самом крайнем случае, — хуже: её молчание читается
как чистота.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest

from dsx.checks.base import Context
from dsx.checks.drift import (
    ExpectedRateHolds,
    FeatureRelationStability,
    TargetRateStationarity,
    support_overlap,
)
from dsx.checks.empirical import ImplausibleSeparation
from dsx.evals.case import Finding
from dsx.evals.registry import BY_ID
from dsx.label import LABEL
from harness import context_for  # noqa: E402


def _window(rows: int, positive_rate: float, seed: int) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    labels = (rng.random(rows) < positive_rate).astype("int8")
    return pl.DataFrame({LABEL: labels, "feature": rng.random(rows)})


# --- N3: доля класса -------------------------------------------------------


def test_zero_share_in_one_window_is_not_silence() -> None:
    """Ноль против ненулевой доли — крайнее расхождение, а не отсутствие его."""
    context = context_for(BY_ID["clean-baseline"])
    parts = context.split.parts
    zeroed = parts[0].evaluate.with_columns(pl.lit(0, dtype=pl.Int8).alias(LABEL))
    parts[0] = type(parts[0])(window=parts[0].window, train=parts[0].train, evaluate=zeroed)

    signals = TargetRateStationarity().run(context)

    assert [s.finding for s in signals] == [Finding.NON_STATIONARY_TARGET]
    assert "бесконечное" in signals[0].detail


# --- N4: устойчивость связи ------------------------------------------------


def test_relation_within_noise_raises_no_alarm() -> None:
    """0.20 против 0.05 при шуме 0.15 — одно значение, измеренное дважды."""
    context = context_for(BY_ID["post-treatment-missingness"])

    assert FeatureRelationStability().run(context) == []


def test_relation_that_disappears_is_reported() -> None:
    """Ослабление сильной связи до шума — это «признак перестал работать»."""
    context = context_for(BY_ID["flipped-feature-relation"])

    signals = FeatureRelationStability().run(context)

    assert [s.finding for s in signals] == [Finding.UNSTABLE_FEATURE_RELATION]


# --- N5: сопоставимость поддержки ------------------------------------------


def test_diverged_constants_are_incomparable() -> None:
    """Признак-константа 0 в одном окне и 10 в другом сравнению не подлежит."""
    windows = [
        ("w0", pl.DataFrame({"f": [0.0] * 200, LABEL: [0] * 200})),
        ("w1", pl.DataFrame({"f": [10.0] * 200, LABEL: [0] * 200})),
    ]

    assert support_overlap(windows, "f") == 0.0


def test_matching_constants_are_comparable() -> None:
    windows = [
        ("w0", pl.DataFrame({"f": [7.0] * 200, LABEL: [0] * 200})),
        ("w1", pl.DataFrame({"f": [7.0] * 200, LABEL: [0] * 200})),
    ]

    assert support_overlap(windows, "f") == 1.0


# --- N6: слишком сильная связь ---------------------------------------------


def test_single_feature_containing_the_answer_is_caught() -> None:
    """Прежняя версия молчала: сравнивать было не с чем, и она выходила сразу."""
    bundle = BY_ID["feature-falsely-declared-available"]
    context = context_for(bundle)
    only = [c for c in context.world.schema.columns if c.name != "lead_days"]
    world = type(context.world)(
        frames=context.world.frames, schema=type(context.world.schema)(columns=only)
    )
    stripped = Context(world, context.outcome, context.task, context.split)

    signals = ImplausibleSeparation().run(stripped)

    assert Finding.FEATURE_AFTER_DECISION in {s.finding for s in signals}


# --- N13: ожидаемая доля ---------------------------------------------------


def test_zero_observed_against_declared_expectation_is_reported() -> None:
    context = context_for(BY_ID["clean-baseline"])
    outcome = context.outcome.model_copy(update={"expected_positive_rate": 0.2})
    for part in context.split.parts:
        part.evaluate = part.evaluate.with_columns(pl.lit(0, dtype=pl.Int8).alias(LABEL))

    signals = ExpectedRateHolds().run(Context(context.world, outcome, context.task, context.split))

    assert [s.finding for s in signals] == [Finding.RATE_CONTRADICTS_EXPECTATION]


# --- A11: концентрация пропусков -------------------------------------------


def test_single_missing_value_is_not_a_finding() -> None:
    """Один пропуск у редкого статуса давал стопроцентную концентрацию."""
    from dsx.checks.data import PostTreatmentMissingness

    context = context_for(BY_ID["clean-baseline"])
    frame = context.world.main
    spoiled = frame.with_columns(
        pl.when(pl.int_range(pl.len()) == 0).then(None).otherwise(pl.col("size")).alias("size")
    )
    world = context.world.replace_main(spoiled)

    signals = PostTreatmentMissingness().run(
        Context(world, context.outcome, context.task, context.split)
    )

    assert signals == []


@pytest.mark.parametrize("moment", [dt.datetime(2024, 1, 1)])
def test_harness_is_reachable(moment: dt.datetime) -> None:
    """Страховка: файл обязан видеть стенд, иначе тесты выше молча не запустятся."""
    assert context_for(BY_ID["clean-baseline"]).split is not None


# --- резерв как измерительный инструмент -----------------------------------


def _reserve_with_rate(context, rate: float, rows: int = 4000):
    """Подменить резерв выборкой с заданной долей класса."""
    import numpy as np

    rng = np.random.default_rng(3)
    labels = (rng.random(rows) < rate).astype("int8")
    context.split.reserved = pl.DataFrame({LABEL: labels, "__outcome_reason": ["observed"] * rows})
    return context


def test_reserve_with_a_shifted_base_rate_is_reported() -> None:
    """Инструмент, откалиброванный на другой популяции, смещает итог."""
    context = context_for(BY_ID["clean-baseline"])
    windows = float(
        pl.concat([p.evaluate for p in context.split.parts], how="vertical_relaxed")[LABEL].mean()
    )
    _reserve_with_rate(context, windows * 1.35)

    signals = TargetRateStationarity().run(context)

    assert [s.finding for s in signals] == [Finding.NON_STATIONARY_TARGET]
    assert "РЕЗЕРВЕ" in signals[0].detail
    assert signals[0].blocking


def test_reserve_with_the_same_base_rate_is_silent() -> None:
    """Отрицательный контроль: проверка не должна кричать на совпадающей доле."""
    context = context_for(BY_ID["clean-baseline"])
    windows = float(
        pl.concat([p.evaluate for p in context.split.parts], how="vertical_relaxed")[LABEL].mean()
    )
    _reserve_with_rate(context, windows)

    assert TargetRateStationarity().run(context) == []


def test_ratio_threshold_does_not_gate_the_reserve_comparison() -> None:
    """Кратность 1.35 порога 1.5 не достигает, но сорок стандартных ошибок —
    это разница, а не колебание."""
    context = context_for(BY_ID["clean-baseline"])
    windows = float(
        pl.concat([p.evaluate for p in context.split.parts], how="vertical_relaxed")[LABEL].mean()
    )
    _reserve_with_rate(context, windows * 1.35, rows=40000)

    signals = TargetRateStationarity().run(context)

    assert signals, "порог по кратности к резерву неприменим"
