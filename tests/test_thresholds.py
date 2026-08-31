"""Пороги проверок против шума и против молчания (ревью Кодекса).

Проверка, срабатывающая по шуму, не отличается от отсутствия проверки.
Проверка, молчащая в самом крайнем случае, — хуже: её молчание читается
как чистота.
"""

from __future__ import annotations

import dataclasses
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

    # Сигналов два, и оба верны: обнуление первого окна создаёт и бесконечную
    # кратность, и монотонный рост.
    assert {s.finding for s in signals} == {Finding.NON_STATIONARY_TARGET}
    assert any("бесконечное" in s.detail for s in signals)
    assert any("однонаправленно" in s.detail for s in signals)


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


# --- медленный однонаправленный дрейф --------------------------------------


def _windows_with_rates(context, rates: list[float], rows: int = 6000):
    import numpy as np

    rng = np.random.default_rng(7)
    for part, rate in zip(context.split.parts, rates, strict=False):
        labels = (rng.random(rows) < rate).astype("int8")
        part.evaluate = pl.DataFrame({LABEL: labels, "__outcome_reason": ["observed"] * rows})
    context.split.reserved = None
    return context


def test_slow_monotone_drift_is_reported() -> None:
    """Соседние окна различаются мало, а накопленный дрейф велик."""
    context = _windows_with_rates(context_for(BY_ID["clean-baseline"]), [0.056, 0.065, 0.076])

    signals = TargetRateStationarity().run(context)

    assert any("однонаправленно" in s.detail for s in signals), (
        "кратность 1.36 порога 1.5 не достигает, но направление и величина различимы"
    )


def test_unordered_wobble_is_not_a_trend() -> None:
    """Отрицательный контроль: колебание без направления дрейфом не является."""
    context = _windows_with_rates(context_for(BY_ID["clean-baseline"]), [0.065, 0.056, 0.064])

    assert not any("однонаправленно" in s.detail for s in TargetRateStationarity().run(context))


def test_tiny_monotone_difference_is_not_a_trend() -> None:
    """Направление есть, различимости нет: три окна упорядочиваются случайно
    с вероятностью около трети."""
    context = _windows_with_rates(context_for(BY_ID["clean-baseline"]), [0.0600, 0.0601, 0.0602])

    assert not any("однонаправленно" in s.detail for s in TargetRateStationarity().run(context))


# --- что дрейф делает ЗА последним окном ------------------------------------
#
# Прежняя формулировка обещала, что смещение тем сильнее, чем дальше от
# обучения. Обещание было о будущем, которого проверка не измеряла, хотя
# резерв лежит сразу за последним окном. Ниже проверяется, что теперь она
# говорит ровно то, что видела.


def test_trend_that_continues_into_the_reserve_is_named_as_continuing() -> None:
    context = _windows_with_rates(context_for(BY_ID["clean-baseline"]), [0.056, 0.065, 0.076])
    _reserve_with_rate(context, 0.090, rows=40000)

    detail = next(
        s.detail for s in TargetRateStationarity().run(context) if "однонаправленно" in s.detail
    )

    assert "направление продолжается" in detail
    assert "РАЗВЕРНУЛОСЬ" not in detail


def test_trend_that_reverses_in_the_reserve_is_named_as_reversed() -> None:
    """Четырнадцатый кейс: 45.9% → 47.6% → 52.1% по окнам, 42.1% в резерве."""
    # Окна крупные: при шести тысячах строк шум выборки сам переставляет
    # 45.9% и 47.6% местами, и монотонности не остаётся.
    context = _windows_with_rates(
        context_for(BY_ID["clean-baseline"]), [0.459, 0.476, 0.521], rows=40000
    )
    _reserve_with_rate(context, 0.421, rows=40000)

    detail = next(
        s.detail for s in TargetRateStationarity().run(context) if "однонаправленно" in s.detail
    )

    assert "РАЗВЕРНУЛОСЬ" in detail
    assert "тем сильнее" not in detail, "обещание о продолжении осталось в формулировке"


def test_trend_without_a_reserve_says_it_has_nothing_to_say() -> None:
    """Молчание означало бы, что направление продлевается. Оно так не означает."""
    context = _windows_with_rates(context_for(BY_ID["clean-baseline"]), [0.056, 0.065, 0.076])
    context.split.reserved = None

    detail = next(
        s.detail for s in TargetRateStationarity().run(context) if "однонаправленно" in s.detail
    )

    assert "сказать нечем" in detail


# --- порог кратности прибит с обеих сторон ----------------------------------
#
# Мутация исходника показала: `ratio` в N3 можно было сдвинуть с 1.5 на 3.0 или
# на 0.75, и весь набор из семисот двух тестов оставался зелёным. Стенд
# доказывал, что проверка ЕСТЬ, и молчал о том, где у неё граница: инжекторы
# подкладывают дефект с огромным запасом, и ни один кейс не стоял вплотную.
#
# Прибить число — значит поставить два случая по разные стороны от него.


def test_ratio_just_past_the_threshold_is_reported() -> None:
    """Кратность 1.7: больше полутора, меньше трёх.

    Убивает мутанта `ratio` 1.5 → 3.0: при ослабленном пороге этот дефект
    прошёл бы молча.
    """
    context = _windows_with_rates(
        context_for(BY_ID["clean-baseline"]), [0.05, 0.085, 0.06], rows=40000
    )
    context.split.reserved = None

    signals = TargetRateStationarity().run(context)

    assert any("различается между окнами" in s.detail for s in signals)


def test_ratio_just_under_the_threshold_is_silent() -> None:
    """Кратность 1.35: меньше полутора, больше 1.25.

    Убивает мутанта `ratio` 1.5 → 1.25: при ужесточённом пороге это колебание
    стало бы ложной тревогой. Отрицательный контроль к предыдущему тесту:
    вместе они говорят, что граница проходит именно там, где объявлена.

    Прежде здесь стояла кратность 1.2 против мутанта 0.75 — половины от нуля.
    Половина кратности кратностью не является и лежит вне области смысла, так
    что тест сторожил границу, перейти которую нельзя. Опора кратности —
    единица, и ослабление 1.5 идёт к 1.25, а от 1.2 она неотличима.
    """
    context = _windows_with_rates(
        context_for(BY_ID["clean-baseline"]), [0.05, 0.0675, 0.06], rows=40000
    )
    context.split.reserved = None

    signals = TargetRateStationarity().run(context)

    assert not any("различается между окнами" in s.detail for s in signals)


# --- порог вопиющей силы связи прибит снизу ---------------------------------
#
# `blatant = 0.45` поднимает вопрос независимо от остальных признаков. Мутация
# показала, что ослабить его до 0.225 можно было незаметно. Сверху мутанта нет
# и быть не может: сила связи по построению не превосходит 0.5, и порог 0.9
# просто выключил бы ветвь — область смысла этого не допускает.


def _features_with_strengths(context, strengths: dict[str, float], seed: int = 5):
    """Подменить объявленные признаки на связи ЗАДАННОЙ силы.

    Для двоичного признака сила равна (q1 - q0) / 2, где q — доля единиц среди
    своего класса. Отсюда q1 = 0.5 + сила, q0 = 0.5 - сила, и никакой подгонки
    не требуется: проверено счётом до третьего знака.
    """
    from dsx.label import compute

    labels = compute(context.world, context.outcome)[LABEL].to_numpy()
    rng = np.random.default_rng(seed)
    columns = []
    for name, strength in strengths.items():
        chance = np.where(labels == 1, 0.5 + strength, 0.5 - strength)
        columns.append(pl.Series(name, (rng.random(len(labels)) < chance).astype(float)))
    world = context.world.replace_main(context.world.main.with_columns(columns))
    # Context заморожен намеренно: подменять мир на месте — значит менять
    # предмет проверки после того, как она его получила.
    return dataclasses.replace(context, world=world)


def test_separation_just_under_blatant_is_silent() -> None:
    """Сила 0.40 при типичной 0.20: ниже 0.45, но выше 0.225.

    Убивает мутанта `blatant` 0.45 → 0.225. Признак не проходит и по второму
    условию — 0.40 меньше, чем 0.20 * 2.5, — поэтому единственное, что могло
    бы его выдать, это порог вопиющей силы.
    """
    context = _features_with_strengths(
        context_for(BY_ID["clean-baseline"]),
        {"lead_days": 0.40, "size": 0.20, "region": 0.20},
    )

    assert not ImplausibleSeparation().run(context)


def test_separation_above_blatant_is_reported() -> None:
    """Сила 0.48 при той же типичной 0.20 обязана быть названа.

    Отрицательный контроль к предыдущему: без него молчание объяснялось бы
    поломкой опоры, а не порогом.
    """
    context = _features_with_strengths(
        context_for(BY_ID["clean-baseline"]),
        {"lead_days": 0.48, "size": 0.20, "region": 0.20},
    )

    assert ImplausibleSeparation().run(context)
