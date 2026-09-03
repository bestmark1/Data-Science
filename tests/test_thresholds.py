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

from dsx.checks.base import Context, NotApplicable
from dsx.checks.drift import (
    ExpectedRateHolds,
    FeatureRelationStability,
    TargetRateStationarity,
    UnobservedCostIsNamed,
    support_overlap,
)
from dsx.checks.empirical import ImplausibleSeparation
from dsx.checks.split_checks import MeasuredPartsAreUsable, TrainingPartIsUsable
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
    outcome = context.outcome.model_copy(
        update={"expected_positive_rate": 0.2, "expected_within": 0.05}
    )
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


# --- порог ослабления связи прибит с обеих сторон ---------------------------
#
# `N4.ratio = 3.0` — во сколько раз связь признака с исходом может ослабнуть
# между окнами. Мутация показала, что сдвинуть его на 2.0 или на 5.0 можно
# было незаметно.


def _relation_across_windows(
    context, name: str, values: list[float], rows: int = 20000, seed: int = 5
):
    """Задать связь признака с исходом ОТДЕЛЬНО В КАЖДОМ ОКНЕ.

    Та же двоичная опора, что и у силы связи: q1 = 0.5 + a, q0 = 0.5 - a даёт
    связь ровно a. Здесь она нужна по окнам, потому что N4 сравнивает окна
    между собой, а не с общим уровнем.
    """
    rng = np.random.default_rng(seed)
    for part, value in zip(context.split.parts, values, strict=False):
        labels = (rng.random(rows) < 0.4).astype("int8")
        chance = np.where(labels == 1, 0.5 + value, 0.5 - value)
        feature = (rng.random(rows) < chance).astype(float)
        part.evaluate = pl.DataFrame(
            {LABEL: labels, "__outcome_reason": ["observed"] * rows, name: feature}
        )
    context.split.reserved = None
    return context


def test_relation_weakening_past_the_threshold_is_reported() -> None:
    """Ослабление вчетверо: больше трёх, меньше пяти.

    Убивает мутанта `ratio` 3.0 → 5.0.
    """
    context = _relation_across_windows(
        context_for(BY_ID["clean-baseline"]), "lead_days", [0.36, 0.30, 0.09]
    )

    assert any("раза" in signal.detail for signal in FeatureRelationStability().run(context))


def test_relation_weakening_under_the_threshold_is_silent() -> None:
    """Ослабление в два с половиной раза: меньше трёх, больше двух.

    Убивает мутанта `ratio` 3.0 → 2.0. Отрицательный контроль к предыдущему.
    """
    context = _relation_across_windows(
        context_for(BY_ID["clean-baseline"]), "lead_days", [0.36, 0.30, 0.144]
    )

    assert not any("раза" in signal.detail for signal in FeatureRelationStability().run(context))


def test_separation_exceeding_the_typical_one_is_reported() -> None:
    """Сила 0.40 при типичной 0.13 — превышение втрое.

    Убивает мутанта `excess` 2.5 → 4.0. Признак не вопиющий (0.40 меньше 0.45),
    поэтому назвать его может только второе условие — превышение остальных.

    Снизу `excess` прибит УЖЕ: случай 0.40 при типичной 0.20 обязан молчать, а
    при пороге 1.75 заговорил бы. Отдельный тест на это не нужен, и добавлять
    его значило бы изображать покрытие там, где оно есть.
    """
    context = _features_with_strengths(
        context_for(BY_ID["clean-baseline"]),
        {"lead_days": 0.40, "size": 0.13, "region": 0.13},
    )

    assert ImplausibleSeparation().run(context)


# --- порог цены ненаблюдаемости прибит с обеих сторон -----------------------
#
# `N16.share = 0.05` решил дело в четырнадцатом кейсе: пока ненаблюдаемых было
# 5.4%, проверка называла цену — границы, в которых лежит правда; после отброса
# испорченных строк доля упала до 1.01%, и проверка замолчала. Молчание было
# верным, но проходило оно ровно по этому числу.


def _unobserved_share(context, share: float, rows: int = 20000, seed: int = 7):
    """Сделать заданную долю исходов ненаблюдаемой во всех окнах."""
    rng = np.random.default_rng(seed)
    for part in context.split.parts:
        observed = (rng.random(rows) < 0.4).astype("int8")
        missing = rng.random(rows) < share
        labels = [
            None if gone else int(value) for value, gone in zip(observed, missing, strict=True)
        ]
        part.evaluate = pl.DataFrame(
            {
                LABEL: pl.Series(labels, dtype=pl.Int8),
                "__outcome_reason": ["observed"] * rows,
            }
        )
    context.split.reserved = None
    return context


def test_unobserved_share_past_the_threshold_is_priced() -> None:
    """Семь процентов: больше пяти, меньше десяти.

    Убивает мутанта `share` 0.05 → 0.1.
    """
    context = _unobserved_share(context_for(BY_ID["clean-baseline"]), 0.07)

    assert any("не наблюдается" in s.detail for s in UnobservedCostIsNamed().run(context))


def test_unobserved_share_under_the_threshold_is_silent() -> None:
    """Три с половиной процента: меньше пяти, больше двух с половиной.

    Убивает мутанта `share` 0.05 → 0.025. Цена ненаблюдаемости здесь мала, и
    называть её значило бы поднимать тревогу на ширине в три процентных пункта.
    """
    context = _unobserved_share(context_for(BY_ID["clean-baseline"]), 0.035)

    assert not any("не наблюдается" in s.detail for s in UnobservedCostIsNamed().run(context))


# --- пороги пригодности частей прибиты с обеих сторон -----------------------
#
# `P8.minimum` и `P9.minimum` равны сотне строк: ниже учиться не на чем и
# мерить нечего. Оба сдвигались вдвое незаметно.


def _train_rows(context, rows: int):
    for part in context.split.parts:
        part.train = part.train.head(rows)
    return context


def _observable_rows(context, rows: int):
    for part in context.split.parts:
        part.evaluate = part.evaluate.head(rows)
    if context.split.reserved is not None:
        context.split.reserved = context.split.reserved.head(rows)
    return context


def test_training_part_above_the_minimum_is_accepted() -> None:
    """Полтораста строк: больше сотни, меньше двух сотен.

    Убивает мутанта `P8.minimum` 100 → 200.
    """
    assert not TrainingPartIsUsable().run(_train_rows(context_for(BY_ID["clean-baseline"]), 150))


def test_training_part_below_the_minimum_is_refused() -> None:
    """Семьдесят пять строк: меньше сотни, больше полусотни.

    Убивает мутанта `P8.minimum` 100 → 50. Отрицательный контроль к предыдущему.
    """
    assert TrainingPartIsUsable().run(_train_rows(context_for(BY_ID["clean-baseline"]), 75))


def test_measured_part_above_the_minimum_is_accepted() -> None:
    """Убивает мутанта `P9.minimum` 100 → 200."""
    assert not MeasuredPartsAreUsable().run(
        _observable_rows(context_for(BY_ID["clean-baseline"]), 150)
    )


def test_measured_part_below_the_minimum_is_refused() -> None:
    """Убивает мутанта `P9.minimum` 100 → 50."""
    assert MeasuredPartsAreUsable().run(_observable_rows(context_for(BY_ID["clean-baseline"]), 75))


# --- порог дробности категорий прибит с обеих сторон ------------------------
#
# `MIN_PER_CATEGORY = 50` отсекает случай, где сила связи оказалась бы
# свойством дробности признака, а не самого признака: на пределе каждая
# категория описывает одну строку и разделяет идеально. Порог введён ради
# строковой утечки десятого кейса и до сих пор не держался ничем.


def _pure_categories(context, count: int, column: str = "region", seed: int = 3):
    """Строковый признак, чьи категории ЧИСТЫ по исходу.

    Половина категорий достаётся положительным строкам, половина —
    отрицательным, поэтому разделение идеальное, а вопрос остаётся один:
    посмотрит ли проверка на признак при такой дробности.
    """
    from dsx.label import compute

    labels = compute(context.world, context.outcome)[LABEL].to_numpy()
    rng = np.random.default_rng(seed)
    half = max(1, count // 2)
    values = [f"c{rng.integers(0, half) + (half if value == 1 else 0)}" for value in labels]
    world = context.world.replace_main(context.world.main.with_columns(pl.Series(column, values)))
    return dataclasses.replace(context, world=world)


def test_categories_large_enough_are_examined() -> None:
    """Шестьдесят категорий на четыре тысячи строк — по 67 на категорию.

    Больше полусотни, меньше сотни: убивает мутанта 50 → 100. Утечка идеальна,
    и промолчать проверка может только отказавшись смотреть.
    """
    assert ImplausibleSeparation().run(_pure_categories(context_for(BY_ID["clean-baseline"]), 60))


def test_categories_too_fine_are_not_examined() -> None:
    """Сто двадцать категорий — по 33 на категорию.

    Меньше полусотни, больше двадцати пяти: убивает мутанта 50 → 25. Утечка та
    же самая, и молчание здесь верно: при такой дробности идеальное разделение
    было бы свойством дробления, а не признака.
    """
    assert not ImplausibleSeparation().run(
        _pure_categories(context_for(BY_ID["clean-baseline"]), 120)
    )


# --- ожидание меряется в долях, а не в разах --------------------------------
#
# Шестнадцатый кейс: объявлено 0.85, наблюдается 54.2%. Ошибка автора в
# тридцать один процентный пункт прошла молча, потому что 0.85/0.542 = 1.57, а
# порог по кратности стоял на двойке.
#
# Шум сюда не годится и это проверено счётом: при 168 тысячах строк три
# стандартные ошибки составляют 0.36 п.п. Здесь измерение сравнивается с
# ДОГАДКОЙ, у которой выборочной ошибки нет вовсе.


def _expectation(context, expected: float, within: float):
    return Context(
        context.world,
        context.outcome.model_copy(
            update={"expected_positive_rate": expected, "expected_within": within}
        ),
        context.task,
        context.split,
    )


def _observed_rate(context) -> float:
    frames = [part.evaluate for part in context.split.parts]
    return float(pl.concat(frames, how="vertical_relaxed")[LABEL].mean())


def test_expectation_outside_its_tolerance_is_reported() -> None:
    """Случай шестнадцатого кейса: прежний прибор его пропускал."""
    context = context_for(BY_ID["clean-baseline"])
    observed = _observed_rate(context)

    signals = ExpectedRateHolds().run(_expectation(context, observed + 0.30, 0.10))

    assert [s.finding for s in signals] == [Finding.RATE_CONTRADICTS_EXPECTATION]
    assert "допуском 10.0%" in signals[0].detail


def test_expectation_inside_its_tolerance_is_silent() -> None:
    """Отрицательный контроль: ошибка в пределах объявленного — не находка."""
    context = context_for(BY_ID["clean-baseline"])
    observed = _observed_rate(context)

    assert not ExpectedRateHolds().run(_expectation(context, observed + 0.05, 0.10))


def test_the_old_ratio_instrument_would_have_stayed_silent() -> None:
    """Показ того, что именно чинится, а не рассказ о нём.

    Доли 0.85 и 0.55 отличаются на тридцать пунктов и в 1.55 раза. Порог по
    кратности стоял на двойке — и молчал; порог по долям в десять пунктов
    говорит.
    """
    high, low = 0.85, 0.55

    assert high / low < 2.0, "прежний прибор такое расхождение пропускал"
    assert abs(high - low) > 0.10, "новый прибор его называет"


def test_a_skipped_expectation_says_what_it_costs() -> None:
    """Пропуск, не называющий цены, читается как «проверять было нечего».

    Ожидаемая доля объявлена в двух формулярах из пятнадцати, и в остальных
    тринадцати молчит единственное, чем ядро видит ошибку построителя.
    Сообщение обязано это называть: молчаливый пропуск неотличим от проверки,
    которой нечего было делать.
    """
    context = context_for(BY_ID["clean-baseline"])
    without = context.outcome.model_copy(
        update={"expected_positive_rate": None, "expected_within": None}
    )

    with pytest.raises(NotApplicable, match="ПОСТРОИТЕЛЯ"):
        ExpectedRateHolds().run(Context(context.world, without, context.task, context.split))
