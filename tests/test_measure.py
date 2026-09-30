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
