"""Измерение: разница с правилом интервалом, а не одинокая метрика.

История проекта начинается с вывода «модель лучше правила», перевернувшегося
трижды из-за протокола. Здесь проверяется, что такой вывод больше нельзя
получить дёшево: интервал, включающий ноль, обязан читаться как «не показано».
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest

from dsx.label import LABEL
from dsx.measure import (
    BaselineRule,
    RuleKind,
    calibration_error,
    discrimination,
    measure_against_baseline,
)
from dsx.policy import Blocked
from dsx.samples import Extent, Purpose, SampleLedger

ROWS = 4000


def world(seed: int = 5) -> pl.DataFrame:
    """Мир с известной истинной вероятностью.

    `risk` — честно калиброванная оценка, `signal` — та же величина до
    преобразования: ранжирует так же, а вероятностью не является.
    """
    rng = np.random.default_rng(seed)
    signal = rng.normal(size=ROWS)
    risk = 1 / (1 + np.exp(-(signal - 1.2)))
    label = (rng.random(ROWS) < risk).astype(np.int8)
    return pl.DataFrame(
        {LABEL: label, "signal": signal, "risk": risk, "noise": rng.normal(size=ROWS)}
    )


def ledger_with(frame: pl.DataFrame, name: str = "резерв") -> SampleLedger:
    sl = SampleLedger()
    sl.register(
        name,
        Extent(
            units=frozenset(str(i) for i in range(frame.height)),
            since=dt.datetime(2024, 1, 1),
            until=dt.datetime(2024, 6, 1),
        ),
        frame=frame,
    )
    return sl


CONSTANT = BaselineRule(kind=RuleKind.CONSTANT, constant=0.2)


# --- сами величины ---------------------------------------------------------


def test_discrimination_of_a_constant_is_a_half() -> None:
    """Без усреднения связок постоянное правило выглядело бы различающим."""
    labels = np.array([0, 1] * 50)

    assert discrimination(np.full(100, 0.3), labels) == pytest.approx(0.5)


def test_discrimination_of_the_answer_is_one() -> None:
    labels = np.array([0, 1] * 50)

    assert discrimination(labels.astype(float), labels) == pytest.approx(1.0)


def test_calibration_of_an_honest_constant_is_near_zero() -> None:
    labels = (np.arange(1000) < 200).astype(int)

    assert calibration_error(np.full(1000, 0.2), labels) < 0.01


def test_calibration_punishes_a_confident_liar() -> None:
    labels = (np.arange(1000) < 200).astype(int)

    assert calibration_error(np.full(1000, 0.9), labels) > 0.6


# --- измерение -------------------------------------------------------------


def test_measurement_spends_the_sample() -> None:
    frame = world()
    sl = ledger_with(frame)

    measure_against_baseline(sl, "резерв", frame["signal"], CONSTANT)

    assert sl.was_measured("резерв")


def test_second_measurement_is_blocked() -> None:
    """Посмотреть, подкрутить, посмотреть снова — здесь невозможно."""
    frame = world()
    sl = ledger_with(frame)
    measure_against_baseline(sl, "резерв", frame["signal"], CONSTANT)

    with pytest.raises(Blocked, match="P1"):
        measure_against_baseline(sl, "резерв", frame["noise"], CONSTANT)


def test_useful_model_beats_the_constant_rule() -> None:
    """Модель подаётся ВЕРОЯТНОСТЬЮ: сырой признак ранжирует так же, но
    вероятностью не является, и проверка калибровки это скажет."""
    frame = world()
    verdict = measure_against_baseline(ledger_with(frame), "резерв", frame["risk"], CONSTANT)

    assert verdict.ranking.decisive
    assert verdict.ranking.model_wins
    assert "превосходит правило" in verdict.statement()


def test_noise_does_not_beat_the_rule() -> None:
    """Отрицательный контроль: интервал обязан включить ноль."""
    frame = world()
    verdict = measure_against_baseline(ledger_with(frame), "резерв", frame["noise"], CONSTANT)

    assert not verdict.ranking.decisive
    assert "не показано" in verdict.statement()


def test_ranking_and_calibration_are_judged_apart() -> None:
    """Модель может ранжировать лучше и калиброваться хуже: склейка этих двух
    и позволяла перевороту вывода."""
    frame = world()

    verdict = measure_against_baseline(ledger_with(frame), "резерв", frame["signal"], CONSTANT)

    assert verdict.ranking.model_wins
    assert verdict.calibration.decisive and not verdict.calibration.model_wins
    assert "калибрована хуже" in verdict.statement()


def test_threshold_rule_is_a_real_competitor() -> None:
    frame = world()
    rule = BaselineRule(kind=RuleKind.THRESHOLD, feature="signal", threshold=1.2)

    verdict = measure_against_baseline(ledger_with(frame), "резерв", frame["noise"], rule)

    assert not verdict.ranking.model_wins, "шум не может превзойти осмысленный порог"


def test_misaligned_scores_are_refused() -> None:
    frame = world()

    with pytest.raises(ValueError, match="не выровнены"):
        measure_against_baseline(ledger_with(frame), "резерв", frame["signal"].head(10), CONSTANT)


def test_rule_without_its_fields_is_refused() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="назвать признак и порог"):
        BaselineRule(kind=RuleKind.THRESHOLD)

    with pytest.raises(ValidationError, match="назвать значение"):
        BaselineRule(kind=RuleKind.CONSTANT)


def test_measurement_is_reproducible() -> None:
    """Отчёт обязан воспроизводиться побайтово: интервал считается с зерном."""
    frame = world()
    first = measure_against_baseline(ledger_with(frame), "резерв", frame["signal"], CONSTANT)
    second = measure_against_baseline(ledger_with(frame), "резерв", frame["signal"], CONSTANT)

    assert first.report_section() == second.report_section()


def test_audit_before_measurement_does_not_block_it() -> None:
    frame = world()
    sl = ledger_with(frame)
    sl.checkout("резерв", Purpose.AUDIT, "проверки постановки")

    measure_against_baseline(sl, "резерв", frame["signal"], CONSTANT)

    assert sl.was_measured("резерв")


def test_verdict_reaches_the_report_and_binds_the_conclusion() -> None:
    """Заключение, полученное при другом измерении, обязано отличаться видимо."""
    from dsx.report import Study

    frame = world()
    study = Study(title="Проверочное измерение")
    study.measurement = measure_against_baseline(
        ledger_with(frame), "резерв", frame["risk"], CONSTANT
    )
    study.conclude("модель принята")
    before = study.conclusions[-1].protocol

    study.measurement = measure_against_baseline(
        ledger_with(frame), "резерв", frame["noise"], CONSTANT
    )

    assert "Измерение" in study.render()
    assert study.protocol_digest() != before
    assert study.is_stale
