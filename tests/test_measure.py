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
    BINS,
    BaselineRule,
    Costs,
    RuleKind,
    Unmeasured,
    average_precision,
    brier_score,
    calibration_error,
    discrimination,
    log_loss,
    measure_against_baseline,
    measure_contrast,
    net_benefit,
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
    # Входы модели объявляются всегда: без объявления измерение отказывает (P11).
    # Требование введено после того, как утечка, спрятанная ролью `ignored`,
    # подняла разрешающую способность на восемь пунктов при молчании ядра.
    sl.declare_features(["signal", "risk", "noise"])
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


# --- сегменты (N10) --------------------------------------------------------


def _with_segments(seed: int = 9) -> pl.DataFrame:
    """Мир, где модель систематически завышает риск для одной небольшой группы."""
    rng = np.random.default_rng(seed)
    group = np.where(rng.random(ROWS) < 0.08, "редкая", "обычная")
    base = rng.random(ROWS) * 0.3
    label = (rng.random(ROWS) < base).astype(np.int8)
    score = np.where(group == "редкая", np.clip(base + 0.4, 0, 1), base)
    return pl.DataFrame({LABEL: label, "group": group, "score": score, "size": rng.random(ROWS)})


def test_segments_are_ordered_by_bias_not_by_size() -> None:
    """Порядок по объёму прячет самое дорогое."""
    from dsx.measure import segment_bias

    frame = _with_segments()
    segments = segment_bias(frame, frame["score"], ["group"], min_rows=100)

    assert segments[0].value == "редкая", "самый смещённый сегмент обязан быть первым"
    assert segments[0].rows < segments[1].rows, "и он же меньше по объёму"
    assert segments[0].bias > 0.3


def test_small_segments_are_not_reported() -> None:
    """На них смещение неотличимо от случайности."""
    from dsx.measure import segment_bias

    frame = _with_segments()

    assert segment_bias(frame, frame["score"], ["group"], min_rows=ROWS) == []


def test_segment_section_names_the_ordering_rule() -> None:
    from dsx.measure import segment_bias, segment_section

    frame = _with_segments()
    rendered = segment_section(segment_bias(frame, frame["score"], ["group"], min_rows=100))

    assert "не по объёму" in rendered


# --- устойчивость по окнам (N7, P5) ----------------------------------------


def _windowed_ledger(flip: bool) -> tuple[SampleLedger, dict[str, pl.Series]]:
    """Три окна. При flip знак превосходства в третьем меняется на обратный."""
    sl = SampleLedger()
    sl.declare_features(["signal", "risk", "noise"])
    scores: dict[str, pl.Series] = {}
    for index, name in enumerate(("w0", "w1", "w2")):
        frame = world(seed=index + 1)
        sl.register(
            name,
            Extent(
                units=frozenset(f"{name}-{i}" for i in range(frame.height)),
                since=dt.datetime(2024, 1, 1),
                until=dt.datetime(2024, 6, 1),
            ),
            frame=frame,
        )
        useful = frame["risk"]
        scores[name] = (1 - useful) if (flip and name == "w2") else useful
    return sl, scores


def test_steady_sign_across_windows() -> None:
    from dsx.measure import stability_across_windows

    sl, scores = _windowed_ledger(flip=False)
    result = stability_across_windows(sl, scores, CONSTANT)

    assert result.steady
    assert "одинаков" in result.statement()


def test_flipping_sign_is_reported() -> None:
    """Вывод, держащийся на выборе окна, — не вывод."""
    from dsx.measure import stability_across_windows

    sl, scores = _windowed_ledger(flip=True)
    result = stability_across_windows(sl, scores, CONSTANT)

    assert not result.steady
    assert "МЕНЯЕТСЯ" in result.statement()


def test_window_stability_spends_the_windows() -> None:
    """Смотреть пооконные метрики и решать по ним — это выбор, а не аудит."""
    from dsx.measure import stability_across_windows

    sl, scores = _windowed_ledger(flip=False)
    stability_across_windows(sl, scores, CONSTANT)

    assert all(sl.is_spent(name) for name in ("w0", "w1", "w2"))


# --- число корзин достаточно мелко ------------------------------------------


def _miscalibrated(seed: int = 23, rows: int = 20000):
    """Оценки, систематически расходящиеся с наблюдаемой долей."""
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, rows)
    scores = np.clip(0.5 + 0.25 * (labels * 2 - 1) + rng.normal(0, 0.22, rows), 0.001, 0.999)
    return scores, labels


def test_calibration_error_has_converged_by_the_declared_bin_count() -> None:
    """Измельчение корзин вдвое не должно менять ответ.

    Это и есть свойство, ради которого выбрано число корзин, — а не само
    число. Проверять равенство десяти значило бы прибивать величину, о которой
    ничего не сказано; проверять сходимость значит сказать, ПОЧЕМУ десяти
    достаточно.

    Мутация показала асимметрию. При пяти корзинах ответ 0.094, при десяти
    0.108: огрубление ЗАНИЖАЕТ ошибку калибровки, потому что внутри широкой
    корзины расхождения разных знаков гасят друг друга. Измельчение сверх
    десяти не меняет ничего, и мутант в эту сторону выживает по праву — его
    выживание есть свидетельство сходимости, а не дыра.
    """
    scores, labels = _miscalibrated()

    coarse = calibration_error(scores, labels, bins=BINS)
    fine = calibration_error(scores, labels, bins=2 * BINS)

    assert abs(coarse - fine) < 0.005, (
        f"при {BINS} корзинах ответ {coarse:.4f}, при {2 * BINS} — {fine:.4f}: "
        "корзины слишком широки, и ошибка калибровки занижена"
    )


def test_coarser_bins_understate_the_calibration_error() -> None:
    """Отрицательный контроль: занижение при огрублении показано, а не заявлено.

    Без него предыдущий тест говорил бы «разница мала» и не отличал бы
    сошедшуюся сетку от нечувствительной меры.
    """
    scores, labels = _miscalibrated()

    assert calibration_error(scores, labels, bins=5) < calibration_error(scores, labels, bins=BINS)


# --- сравнение двух постановок на одной выборке -----------------------------
#
# `measure_contrast` работает с шестого кейса и вызывается двумя проектами, но
# НАБОРОМ ТЕСТОВ не исполнялся ни разу: замер достижимости показал его среди
# тридцати пяти таких функций. Каждый из трёх дефектов последних заходов лежал
# ровно в таком коде — вызываемом, но не прогоняемом.


def _contrast_ledger(rows: int, frame: pl.DataFrame) -> SampleLedger:
    """Журнал, ВЛАДЕЮЩИЙ данными: иначе выдать по checkout нечего.

    Владение здесь не подробность вызова, а суть учёта: единственный способ
    получить кадр — через журнал, и расход записывается тем же действием.
    """
    ledger = SampleLedger()
    ledger.declare_features(["сильная", "слабая"])
    ledger.register(
        "резерв",
        Extent(
            units=frozenset(str(index) for index in range(rows)),
            since=dt.datetime(2024, 1, 1),
            until=dt.datetime(2024, 12, 31),
        ),
        frame=frame,
    )
    return ledger


def test_contrast_spends_the_sample_once_for_two_models() -> None:
    """Две постановки сравниваются одним расходом выборки.

    В этом весь смысл парного сравнения: выдать выборку дважды значило бы
    израсходовать её дважды, и второе измерение перестало бы быть независимым.
    """
    rng = np.random.default_rng(3)
    rows = 4000
    labels = rng.integers(0, 2, rows)
    strong = np.clip(0.5 + 0.25 * (labels * 2 - 1) + rng.normal(0, 0.25, rows), 0.01, 0.99)
    weak = np.clip(0.5 + 0.05 * (labels * 2 - 1) + rng.normal(0, 0.25, rows), 0.01, 0.99)
    ledger = _contrast_ledger(rows, pl.DataFrame({LABEL: labels}))

    contrast = measure_contrast(
        ledger,
        "резерв",
        pl.Series("сильная", strong),
        pl.Series("слабая", weak),
        left_name="сильная",
        right_name="слабая",
    )

    assert len(ledger.accesses) == 1, "парное сравнение обязано расходовать выборку один раз"
    assert "сильная" in str(contrast)


def test_contrast_refuses_an_unregistered_sample() -> None:
    """Имя, придуманное на ходу, обходит учёт расхода."""
    rng = np.random.default_rng(5)
    scores = pl.Series(rng.random(500))

    with pytest.raises(Blocked, match="не зарегистрирована"):
        measure_contrast(SampleLedger(), "выдуманная", scores, scores, "a", "b")


# --- ядро обязано знать, чем питалась модель (P11) ---------------------------
#
# Пятнадцатый кейс: ядро проверяло объявленную СХЕМУ и не знало входов модели.
# Признаки жили отдельным списком в коде проекта, связи с формуляром не было.
# Опыт на закрытом кейсе: колонка, объявленная `role: ignored` и скормленная
# модели в обход, подняла разрешающую способность с 0.7178 до 0.8009, и сигналы
# ядра совпали ДОСЛОВНО. Дыра обесценивала N6, S10 и подложенные контроли разом.


def test_measurement_refuses_until_model_inputs_are_declared() -> None:
    """Необъявленные входы означают, что измерять нечего.

    Требование стоит на самом измерении, а не рядом с ним: механизм, который
    можно не позвать, в этом проекте ломался трижды.
    """
    frame = world()
    ledger = SampleLedger()
    ledger.register(
        "резерв",
        Extent(
            units=frozenset(str(i) for i in range(frame.height)),
            since=dt.datetime(2024, 1, 1),
            until=dt.datetime(2024, 6, 1),
        ),
        frame=frame,
    )

    with pytest.raises(Blocked, match="входы модели не объявлены"):
        measure_against_baseline(ledger, "резерв", frame["signal"], CONSTANT)


def test_declared_inputs_let_the_measurement_through() -> None:
    """Отрицательный контроль: объявление снимает отказ и ничего больше."""
    frame = world()
    ledger = ledger_with(frame)

    verdict = measure_against_baseline(ledger, "резерв", frame["signal"], CONSTANT)

    assert ledger.features == ("signal", "risk", "noise")
    assert verdict.rows > 0


# --- DS-008: метрики при дисбалансе, порог по стоимости --------------------
#
# Каждая величина сверена с ответом, посчитанным вручную, а не с другой
# библиотекой: прежде чем верить измерителю, проверь его на известном ответе.


def test_brier_and_log_loss_on_a_known_answer() -> None:
    """Ответ от ревью карты техник: y=[0,1], p=[0.25,0.75]."""
    labels = np.array([0, 1])
    scores = np.array([0.25, 0.75])

    assert brier_score(scores, labels) == pytest.approx(0.0625, abs=1e-12)
    assert log_loss(scores, labels) == pytest.approx(0.287682, abs=1e-6)


def test_log_loss_of_a_confident_mistake_is_infinite_not_clipped() -> None:
    """Оценка ровно 0 при случившемся исходе — бесконечность, а не 34.5 от подрезки."""
    assert np.isinf(log_loss(np.array([0.0, 1.0]), np.array([1, 1])))
    assert log_loss(np.array([0.0, 1.0]), np.array([0, 1])) == 0.0


def test_probability_metrics_are_undefined_for_non_probabilities() -> None:
    labels = np.array([0, 1])
    for scores in (np.array([-0.5, 0.5]), np.array([0.2, 1.5]), np.array([np.nan, 0.5])):
        assert np.isnan(brier_score(scores, labels))
        assert np.isnan(log_loss(scores, labels))
        assert np.isnan(net_benefit(scores, labels, 0.5))


def test_average_precision_on_a_known_answer() -> None:
    """По порогам сверху: 0.8 — точность 1, полнота 0.5; 0.4 — полнота не растёт;
    0.35 — точность 2/3, полнота 1. AP = 0.5·1 + 0.5·2/3 = 0.8333."""
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.4, 0.35, 0.8])

    assert average_precision(scores, labels) == pytest.approx(0.5 + 0.5 * 2 / 3, abs=1e-12)


def test_average_precision_of_a_constant_is_the_prevalence() -> None:
    """Все оценки равны — один порог: точность равна доле класса, полнота 1."""
    labels = np.array([1, 0, 0, 0, 1, 0, 0, 0, 0, 0])

    assert average_precision(np.full(10, 0.3), labels) == pytest.approx(0.2, abs=1e-12)
    assert np.isnan(average_precision(np.full(4, 0.3), np.zeros(4, dtype=int)))


def test_net_benefit_on_a_known_answer() -> None:
    """Порог 0.2 → цена ложного срабатывания в единицах верного p/(1−p) = 0.25.
    Действуем на трёх строках: два верных, одно ложное. (2 − 1·0.25) / 5 = 0.35."""
    labels = np.array([1, 1, 0, 0, 1])
    scores = np.array([0.9, 0.5, 0.3, 0.1, 0.1])

    assert net_benefit(scores, labels, 0.2) == pytest.approx((2 - 0.25) / 5, abs=1e-12)
    assert net_benefit(np.zeros(5), labels, 0.2) == 0.0


def test_costs_set_the_threshold_and_have_no_default() -> None:
    from pydantic import ValidationError

    assert Costs(false_positive=1, false_negative=4).threshold == pytest.approx(0.2)
    for bad in ({"false_positive": 0, "false_negative": 1}, {"false_positive": 1}, {}):
        with pytest.raises(ValidationError):
            Costs(**bad)


def test_the_verdict_matches_a_direct_count() -> None:
    """Числа отчёта — те же, что прямой счёт по формулам на том же мире."""
    frame = world()
    verdict = measure_against_baseline(ledger_with(frame), "резерв", frame["risk"], CONSTANT)
    labels = frame[LABEL].to_numpy().astype(float)
    risk = frame["risk"].to_numpy()

    assert verdict.prevalence == pytest.approx(labels.mean())
    assert verdict.brier.model == pytest.approx(np.mean((risk - labels) ** 2))
    assert verdict.brier.baseline == pytest.approx(np.mean((0.2 - labels) ** 2))
    expected = -np.mean(labels * np.log(risk) + (1 - labels) * np.log(1 - risk))
    assert verdict.log_loss.model == pytest.approx(expected)
    assert verdict.precision.baseline == pytest.approx(labels.mean()), "константа = доля класса"


def test_a_calibrated_model_beats_the_constant_on_every_probability_metric() -> None:
    frame = world()
    verdict = measure_against_baseline(
        ledger_with(frame),
        "резерв",
        frame["risk"],
        CONSTANT,
        costs=Costs(false_positive=1, false_negative=3),
    )

    for comparison in (verdict.precision, verdict.brier, verdict.log_loss, verdict.benefit):
        assert comparison.model_wins, str(comparison)


def test_noise_does_not_win_on_precision() -> None:
    """Отрицательный контроль: шумовая вероятность не превосходит постоянное правило по PR-AUC."""
    frame = world()
    noise = pl.Series(np.random.default_rng(3).random(frame.height))
    verdict = measure_against_baseline(ledger_with(frame), "резерв", noise, CONSTANT)

    assert not verdict.precision.model_wins


def test_a_threshold_rule_makes_log_loss_undefined_not_a_win() -> None:
    """Пороговое правило даёт 0 или 1: ошибившись, оно бесконечно неправо. Сравнение
    по log loss от этого не выигрывается моделью — оно не определено."""
    frame = world()
    rule = BaselineRule(kind=RuleKind.THRESHOLD, feature="signal", threshold=1.2)

    verdict = measure_against_baseline(ledger_with(frame), "резерв", frame["risk"], rule)

    assert isinstance(verdict.log_loss, Unmeasured)
    assert "бесконечно" in str(verdict.log_loss)
    assert not isinstance(verdict.brier, Unmeasured)


def test_raw_scores_leave_probability_metrics_undefined_and_say_so() -> None:
    frame = world()
    verdict = measure_against_baseline(ledger_with(frame), "резерв", frame["signal"], CONSTANT)

    assert isinstance(verdict.brier, Unmeasured)
    assert "не вероятности" in str(verdict.brier)
    assert verdict.ranking.model_wins, "ранжирование от этого не страдает"


def test_the_report_names_prevalence_and_missing_costs() -> None:
    frame = world()
    section = measure_against_baseline(
        ledger_with(frame), "резерв", frame["risk"], CONSTANT
    ).report_section()

    assert "доля класса" in section and "PR-AUC (average precision)" in section
    assert "стоимость ошибок не объявлена" in section
    assert "Brier" in section and "log loss" in section


def test_the_report_with_costs_shows_the_threshold_and_the_references() -> None:
    frame = world()
    section = measure_against_baseline(
        ledger_with(frame),
        "резерв",
        frame["risk"],
        CONSTANT,
        costs=Costs(false_positive=1, false_negative=4),
    ).report_section()

    assert "≥ 0.2000" in section
    assert "чистая польза при пороге 0.2000" in section
    assert "действовать всегда" in section and "никогда 0" in section


# --- ревью DS-008: неопределённое объявляется, а не прячется ---------------


def _tiny(labels, model, rule_value=0.2):
    frame = pl.DataFrame({LABEL: labels, "signal": model, "risk": model, "noise": model})
    rule = BaselineRule(kind=RuleKind.CONSTANT, constant=rule_value)
    return measure_against_baseline(ledger_with(frame), "резерв", frame["risk"], rule)


def test_a_bootstrap_with_undefined_resamples_is_declared_not_printed_as_nan() -> None:
    """Ревью `0dad089`: y=[0,1] — в 100 из 400 пересборок нет положительного
    класса, AP там не определён, а интервал печатался как [+nan; +nan]."""
    verdict = _tiny([0, 1], [0.2, 0.8])

    assert isinstance(verdict.precision, Unmeasured)
    assert "из 400 пересборок" in str(verdict.precision)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_scores_leave_the_report_whole(bad) -> None:
    """Ревью `0dad089`: NaN и бесконечность в оценках обрывали весь отчёт
    исключением sklearn."""
    verdict = _tiny([0, 1, 0, 1], [bad, 0.8, 0.3, 0.6])

    for comparison in (verdict.precision, verdict.brier, verdict.log_loss):
        assert isinstance(comparison, Unmeasured)
        assert "не конечны" in str(comparison)
    assert "## Измерение" in verdict.report_section()


def test_no_observed_labels_leave_the_report_whole() -> None:
    """Ревью `0dad089`: на `d19df7d` такой вызов давал отчёт, на `0dad089` — исключение."""
    verdict = _tiny([None, None], [0.2, 0.8])

    assert verdict.rows == 0
    for comparison in (verdict.precision, verdict.brier, verdict.log_loss):
        assert isinstance(comparison, Unmeasured)
        assert "нет строк с наблюдаемым исходом" in str(comparison)
    assert "## Измерение" in verdict.report_section()


def test_average_precision_of_non_finite_scores_is_undefined() -> None:
    assert np.isnan(average_precision(np.array([np.nan, 0.8]), np.array([0, 1])))


def test_costs_do_not_depend_on_their_scale() -> None:
    """Ревью `0dad089`: Costs(1e308, 1e308) давал порог 0.0 — сумма переполнялась."""
    assert Costs(false_positive=2, false_negative=2).threshold == 0.5
    assert Costs(false_positive=1e308, false_negative=1e308).threshold == 0.5
    assert Costs(false_positive=1, false_negative=1e308).threshold > 0


@pytest.mark.parametrize(("fp", "fn"), [(1e308, 1), (1e-300, 1e300), (1e300, 1e-300)])
def test_costs_whose_threshold_is_not_representable_are_refused(fp, fn) -> None:
    """Порог обязан лежать строго в (0, 1): 1.0 делил бы на ноль в чистой пользе,
    0.0 делал бы ложное срабатывание бесплатным. Подрезки нет — отказ."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="непредставим"):
        Costs(false_positive=fp, false_negative=fn)


def test_net_benefit_refuses_a_threshold_outside_the_open_interval() -> None:
    for threshold in (0.0, 1.0, -0.1, float("nan")):
        with pytest.raises(ValueError):
            net_benefit(np.array([0.5]), np.array([1]), threshold)


def test_the_report_calls_prevalence_the_constant_not_every_random_ranking() -> None:
    """Ревью `0dad089`, P3: у случайного ранжирования двух строк AP в среднем 0.75,
    а доля класса 0.5; доля — точный AP ПОСТОЯННОЙ оценки."""
    frame = world()
    section = measure_against_baseline(
        ledger_with(frame), "резерв", frame["risk"], CONSTANT
    ).report_section()

    assert "PR-AUC постоянной оценки" in section
    assert "столько PR-AUC даёт ранжирование" not in section


# --- второе ревью DS-008: входы метрик проверяются до счёта ----------------

METRICS = [average_precision, brier_score, log_loss, lambda s, y: net_benefit(s, y, 0.5)]


@pytest.mark.parametrize("metric", METRICS)
@pytest.mark.parametrize(
    "labels", [[np.nan, np.nan], [np.inf, 1.0], [2.0, 1.0], [-1.0, 0.0], [0.5, 1.0]]
)
def test_a_label_that_is_not_zero_or_one_is_refused(metric, labels) -> None:
    """Ревью `75a6f1c`: NaN-метка становилась отрицательным исходом, и log loss
    объявлял модель победителем. Неизвестный исход — не отрицательный класс."""
    with pytest.raises(ValueError, match="метк"):
        metric(np.array([0.2, 0.8]), np.array(labels))


@pytest.mark.parametrize("metric", METRICS)
@pytest.mark.parametrize(
    ("scores", "labels"),
    [([0.9, 0.9], [1]), ([0.9], [1, 0]), ([[0.9, 0.1]], [[1, 0]])],
)
def test_misaligned_or_nested_inputs_are_refused_not_broadcast(metric, scores, labels) -> None:
    """Ревью `75a6f1c`: одна метка растягивалась на две оценки, и чистая польза
    выходила 2.0 — два верных срабатывания на одну строку."""
    with pytest.raises(ValueError, match="выровнен|одномерн"):
        metric(np.array(scores), np.array(labels))


def test_nan_labels_in_the_sample_are_refused_before_the_cast() -> None:
    """Ревью `75a6f1c`: NaN в колонке исхода — не пусто для polars; приведение к
    целому давало мусорное число, которое считалось наблюдаемой строкой."""
    frame = pl.DataFrame(
        {
            LABEL: [np.nan, 1.0, np.nan, 0.0],
            "signal": [0.0] * 4,
            "risk": [0.0] * 4,
            "noise": [0.0] * 4,
        }
    )
    with pytest.raises(ValueError, match="метк"):
        measure_against_baseline(ledger_with(frame), "резерв", frame["risk"], CONSTANT)


@pytest.mark.parametrize(
    ("fp", "fn", "expected"),
    [(1e-10, 1e308, 1e-318), (5e-324, 1.0, 5e-324), (1.0, 1e-10, 1 / (1 + 1e-10))],
)
def test_a_representable_threshold_is_not_lost_to_overflow(fp, fn, expected) -> None:
    """Ревью `75a6f1c`: C_fn/C_fp переполнялось, и представимый порог ~1e-318
    отвергался как непредставимый. Считается устойчивой ветвью, сверено с Decimal."""
    from decimal import Decimal, getcontext

    getcontext().prec = 50
    exact = float(Decimal(fp) / (Decimal(fp) + Decimal(fn)))
    threshold = Costs(false_positive=fp, false_negative=fn).threshold

    assert threshold == exact
    assert threshold == pytest.approx(expected, rel=1e-9)


# --- третье ревью DS-008: порог точной дробью, типы входов -----------------


def _exact_threshold(fp: float, fn: float) -> float:
    from fractions import Fraction

    return float(Fraction(fp) / (Fraction(fp) + Fraction(fn)))


@pytest.mark.parametrize(("fp", "fn"), [(1.0, 1e-16), (1e16, 1.0), (1e308, 1e292)])
def test_a_representable_threshold_near_one_is_kept(fp, fn) -> None:
    """Ревью `e3c4788`: 1 + 1e-16 округлялось до 1, и порог 0.9999999999999999
    отвергался как непредставимый."""
    threshold = Costs(false_positive=fp, false_negative=fn).threshold

    assert threshold == _exact_threshold(fp, fn)
    assert 0 < threshold < 1


def test_the_threshold_is_the_exact_fraction_rounded_once() -> None:
    """Тысяча пар стоимостей на всём диапазоне double: порог равен точной дроби,
    округлённой один раз, а отказ бывает только там, где она округляется в 0 или 1."""
    from pydantic import ValidationError

    rng = np.random.default_rng(7)
    exponents = rng.uniform(-323, 308, size=(1000, 2))
    mantissas = rng.uniform(1, 10, size=(1000, 2))
    refused = 0
    for (e1, e2), (m1, m2) in zip(exponents, mantissas, strict=True):
        with np.errstate(over="ignore"):
            fp, fn = float(m1 * 10.0**e1), float(m2 * 10.0**e2)
        if not (0 < fp < np.inf and 0 < fn < np.inf):
            continue
        exact = _exact_threshold(fp, fn)
        if 0 < exact < 1:
            assert Costs(false_positive=fp, false_negative=fn).threshold == exact, (fp, fn)
        else:
            refused += 1
            with pytest.raises(ValidationError, match="непредставим"):
                Costs(false_positive=fp, false_negative=fn)
    assert refused > 0, "проба не дошла до непредставимых порогов"


@pytest.mark.parametrize("metric", METRICS)
@pytest.mark.parametrize(
    ("scores", "labels"),
    [
        (np.array([0.2 + 1j, 0.8 + 1j]), np.array([0, 1])),
        (np.array([0.2, 0.8]), np.array([0 + 0j, 1 + 0j])),
        (np.array(["0.2", "0.8"]), np.array([0, 1])),
        (np.array([0.2, 0.8], dtype=object), np.array([0, 1])),
    ],
)
def test_non_real_inputs_are_refused_not_truncated(metric, scores, labels) -> None:
    """Ревью `e3c4788`: комплексные оценки проходили, мнимая часть отбрасывалась,
    Brier выходил отрицательным и засчитывался победой."""
    with pytest.raises(ValueError, match="вещественн"):
        metric(scores, labels)


def test_a_complex_model_does_not_win_a_comparison() -> None:
    from dsx.measure import _compare

    with pytest.raises(ValueError, match="вещественн"):
        _compare(
            "Brier",
            brier_score,
            np.array([0.2 + 1j, 0.8 + 1j]),
            np.array([0.5, 0.5]),
            np.array([0, 1]),
        )


def test_boolean_probabilities_and_labels_are_numbers() -> None:
    """Ревью `e3c4788`: законные 0/1 в булевом виде роняли Brier TypeError."""
    scores, labels = np.array([False, True]), np.array([False, True])

    assert brier_score(scores, labels) == 0.0
    assert log_loss(scores, labels) == 0.0
    assert average_precision(scores, labels) == 1.0
    assert net_benefit(scores, labels, 0.5) == 0.5


# --- четвёртое ревью DS-008: проверяется исходное, приводится без потерь ---


@pytest.mark.parametrize("metric", METRICS)
@pytest.mark.parametrize("masked", ["scores", "labels"])
def test_a_mask_is_refused_not_dropped(metric, masked) -> None:
    """Ревью `cdf9638`: `np.asarray` снимал маску, и все замаскированные метки
    считались известными — Brier объявлял модель победителем."""
    scores = np.array([0.2, 0.8])
    labels = np.array([0, 1])
    if masked == "scores":
        scores = np.ma.masked_array(scores, mask=[True, True])
    else:
        labels = np.ma.masked_array(labels, mask=[True, True])

    with pytest.raises(ValueError, match="маск"):
        metric(scores, labels)


def test_large_integer_scores_keep_their_order_or_are_refused() -> None:
    """Ревью `cdf9638`: 2**53 и 2**53 + 1 в float64 совпадали, AP падал с 1 до 0.5."""
    labels = np.array([0, 1])
    with pytest.raises(ValueError, match="точност"):
        average_precision(np.array([2**53, 2**53 + 1], dtype=np.int64), labels)
    exact = np.array([2**60, 2**61], dtype=np.int64)  # представимы в float64 точно
    assert average_precision(exact, labels) == 1.0
    assert average_precision(np.array([3, 7], dtype=np.uint64), labels) == 1.0


@pytest.mark.parametrize(
    "scores",
    [
        pl.Series(
            [
                __import__("decimal").Decimal("0"),
                __import__("decimal").Decimal("1.00000000000000000001"),
            ]
            * 40
        ),
        pl.Series(["0.2", "0.8"] * 40),
    ],
    ids=["decimal", "string"],
)
def test_the_report_checks_scores_before_converting_them(scores) -> None:
    """Ревью `cdf9638`: `measure_against_baseline` приводил оценки к float64 раньше
    проверки — Decimal выше 1 округлялся до 1, и Brier объявлял модель лучше."""
    frame = pl.DataFrame(
        {LABEL: [0, 1] * 40, "signal": [0.0] * 80, "risk": [0.0] * 80, "noise": [0.0] * 80}
    )

    with pytest.raises(ValueError, match="вещественн"):
        measure_against_baseline(ledger_with(frame), "резерв", scores, CONSTANT)


def test_log_loss_keeps_small_positive_losses() -> None:
    """Ревью `cdf9638`: 1 − 1e-20 округлялось до 1, и потеря 1e-20 становилась −0.0."""
    from decimal import Decimal, getcontext

    getcontext().prec = 100
    exact = float(-(Decimal(1) - Decimal("1e-20")).ln())

    assert log_loss(np.array([1e-20]), np.array([0])) == exact
    assert np.isinf(log_loss(np.array([1.0]), np.array([0])))
    assert np.isinf(log_loss(np.array([0.0]), np.array([1])))
