"""DS-010: происхождение прогнозов в измерении по окнам (класс 19 журнала повторов).

Кейсы 3–5 мерили ранние окна моделью, обученной на обучающей части последнего,
и N7 объявил знак устойчивым. Проверяется состав и время, а не число моделей.
"""

from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl
import pytest

from dsx.label import LABEL
from dsx.measure import BaselineRule, Prediction, RuleKind, stability_across_windows
from dsx.policy import Blocked
from dsx.samples import Extent, Fit, SampleLedger, Training, _epoch_microseconds

CONSTANT = BaselineRule(kind=RuleKind.CONSTANT, constant=0.3)
STARTS = {
    "w0": dt.datetime(2024, 1, 1),
    "w1": dt.datetime(2024, 3, 1),
    "w2": dt.datetime(2024, 5, 1),
}
ROWS = 200


def _evaluation(name: str) -> frozenset[str]:
    return frozenset(f"{name}-{i}" for i in range(ROWS))


def _frame(seed: int) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    risk = rng.uniform(0.05, 0.95, ROWS)
    return pl.DataFrame(
        {LABEL: (rng.uniform(size=ROWS) < risk).astype(int), "signal": risk, "risk": risk}
    )


def _training(units: frozenset[str], known_until: dt.datetime) -> Training:
    return Training(
        frame=pl.DataFrame({LABEL: [0, 1]}),
        rows=len(units),
        units=units,
        labels_known_until=_epoch_microseconds(known_until),
        labels_known_text=f"{known_until:%Y-%m-%d %H:%M%z}",
        zoned=known_until.utcoffset() is not None,
    )


def _ledger(trainings: dict[str, Training | None] | None = None) -> SampleLedger:
    """Три окна с расширяющимся обучением, как строит `split_by_windows`.

    Обучение окна — строки, чьи метки известны до его начала: у w1 и w2 в него
    входят оценочные строки ранних окон, и их метки известны уже после начала
    этих окон.
    """
    honest = {
        "w0": _training(frozenset({"история-0", "история-1"}), STARTS["w0"] - dt.timedelta(days=1)),
        "w1": _training(
            frozenset({"история-0", "история-1"}) | _evaluation("w0"),
            STARTS["w1"] - dt.timedelta(days=1),
        ),
        "w2": _training(
            frozenset({"история-0", "история-1"}) | _evaluation("w0") | _evaluation("w1"),
            STARTS["w2"] - dt.timedelta(days=1),
        ),
    }
    honest.update(trainings or {})
    sl = SampleLedger()
    sl.declare_features(["signal", "risk"])
    for index, (name, start) in enumerate(STARTS.items()):
        units = _evaluation(name)
        sl.register_window(
            name,
            start=start,
            extent=Extent(units=units, since=start, until=start + dt.timedelta(days=50)),
            evaluation=units,
            frame=_frame(index + 1),
            training=honest[name],
        )
    return sl


def _scores(sl: SampleLedger, name: str) -> pl.Series:
    return sl._frames[name]["risk"]  # noqa: SLF001 — тестовые оценки: сам риск


def _measure(sl: SampleLedger, fits: dict[str, Fit]):
    return stability_across_windows(
        sl, {w: Prediction(fit=f, scores=_scores(sl, w)) for w, f in fits.items()}, CONSTANT
    )


def test_a_model_per_window_trained_on_its_own_part_passes() -> None:
    sl = _ledger()
    result = _measure(sl, {w: sl.training(w, f"обучение на {w}") for w in STARTS})

    assert set(result.per_window) == set(STARTS)
    assert "обучающая часть окна w1" in result.report_section()


def test_one_model_trained_before_the_earliest_window_passes_everywhere() -> None:
    """Число моделей не проверяется: честная общая модель проходит."""
    sl = _ledger()
    earliest = sl.training("w0", "одна модель на самом раннем окне")
    result = _measure(sl, dict.fromkeys(STARTS, earliest))

    assert set(result.per_window) == set(STARTS)
    assert set(result.provenance.values()) == {str(earliest)}


def test_one_model_trained_on_the_last_window_is_refused_before_spending() -> None:
    """Кейсы 3–5: модель с последнего окна мерила все. Отказ называет окно, число
    общих строк и даты; ни одно окно не израсходовано."""
    sl = _ledger()
    last = sl.training("w2", "одна модель на последнем окне")

    with pytest.raises(ValueError) as refused:
        _measure(sl, dict.fromkeys(STARTS, last))

    message = str(refused.value)
    assert "окно 'w0'" in message
    assert f"{ROWS:,} из {ROWS:,} оценочных" in message
    assert "метки из будущего окна" in message
    assert not any(sl.is_spent(w) for w in STARTS)


def test_future_labels_without_shared_rows_are_refused() -> None:
    """Строки не пересекаются, но метки обучения известны после начала окна."""
    later = _training(frozenset({"чужие-0", "чужие-1"}), STARTS["w1"] - dt.timedelta(days=1))
    sl = _ledger({"w1": later})
    fit = sl.training("w1", "модель на обучении w1")

    assert sl.provenance_defects("w1", fit) == []
    defects = sl.provenance_defects("w0", fit)
    assert len(defects) == 1 and "метки из будущего окна" in defects[0]
    with pytest.raises(ValueError, match="метки из будущего"):
        _measure(sl, {"w0": fit})


def test_labels_known_exactly_at_the_start_are_refused() -> None:
    """Строго, как `split_by_windows`: обучение окна — метки, известные ДО начала."""
    edge = _training(frozenset({"граница"}), STARTS["w0"])
    sl = _ledger({"w0": edge})

    with pytest.raises(ValueError, match="метки из будущего"):
        _measure(sl, {"w0": sl.training("w0", "обучение на w0")})


def test_scores_without_a_fit_are_refused() -> None:
    sl = _ledger()
    with pytest.raises(ValueError, match="ожидается Prediction"):
        stability_across_windows(sl, {"w0": _scores(sl, "w0")}, CONSTANT)
    assert not sl.is_spent("w0")


def test_a_fit_the_ledger_did_not_issue_is_refused() -> None:
    sl = _ledger()
    forged = Fit(
        window="w0",
        frame=pl.DataFrame(),
        rows=1,
        units=frozenset({"x"}),
        labels_known_until=0,
        labels_known_text="1970-01-01 00:00",
        zoned=False,
    )
    foreign = _ledger().training("w0", "обучение в другом журнале")

    for fit in (forged, foreign):
        with pytest.raises(ValueError, match="не этим журналом"):
            _measure(sl, {"w0": fit})


def test_a_fit_not_yet_handed_out_is_refused() -> None:
    """Запись, взятая из журнала мимо `training`, обращения не записала."""
    sl = _ledger()
    hidden = sl._windows["w0"].fit  # noqa: SLF001 — обход, который должен не пройти

    with pytest.raises(ValueError, match="не этим журналом"):
        _measure(sl, {"w0": hidden})


def test_subclasses_of_prediction_and_fit_are_refused() -> None:
    class LoudPrediction(Prediction):
        pass

    class LoudFit(Fit):
        pass

    sl = _ledger()
    fit = sl.training("w0", "обучение на w0")
    with pytest.raises(ValueError, match="ожидается Prediction"):
        stability_across_windows(
            sl, {"w0": LoudPrediction(fit=fit, scores=_scores(sl, "w0"))}, CONSTANT
        )
    loud = LoudFit(
        window="w0",
        frame=fit.frame,
        rows=fit.rows,
        units=fit.units,
        labels_known_until=fit.labels_known_until,
        labels_known_text=fit.labels_known_text,
        zoned=fit.zoned,
    )
    with pytest.raises(ValueError, match="без записи обучения"):
        _measure(sl, {"w0": loud})


def test_an_empty_training_part_is_refused_when_handed_out() -> None:
    sl = _ledger()
    sl.register_window(
        "пустое",
        start=dt.datetime(2023, 1, 1),
        extent=None,
        evaluation=frozenset({"п-0"}),
        frame=_frame(9),
        training=None,
    )
    with pytest.raises(ValueError, match="пуста"):
        sl.training("пустое", "обучение")


@pytest.mark.parametrize("broken", ["короткие", "строки"])
def test_bad_scores_in_the_last_window_spend_no_window(broken) -> None:
    """Ревью `648a185`: длина и значения проверялись после `checkout(SELECTION)` —
    ошибка последнего окна тратила все предыдущие. Обычная ошибка данных, без
    враждебного кода: все окна проверяются до расходования первого."""
    sl = _ledger()
    predictions = {
        w: Prediction(fit=sl.training(w, f"обучение на {w}"), scores=_scores(sl, w)) for w in STARTS
    }
    bad = _scores(sl, "w2").head(1) if broken == "короткие" else pl.Series("risk", [".5"] * ROWS)
    predictions["w2"] = Prediction(fit=predictions["w2"].fit, scores=bad)

    with pytest.raises(ValueError):
        stability_across_windows(sl, predictions, CONSTANT)
    assert not any(sl.is_spent(w) for w in STARTS)
    assert not sl.selections("w0") and not sl.selections("w2")


@pytest.mark.parametrize("window", [[], {}, pl.Series([1])])
def test_a_forged_fit_with_odd_fields_is_refused_not_crashed(window) -> None:
    """Ревью `648a185`: поле `window` чужой записи служило ключом поиска до сверки
    тождества — непригодное значение роняло `TypeError`. Сначала тождество."""
    sl = _ledger()
    forged = Fit(
        window=window,
        frame=pl.DataFrame(),
        rows=1,
        units=frozenset({"x"}),
        labels_known_until=0,
        labels_known_text="1970-01-01 00:00",
        zoned=False,
    )
    with pytest.raises(ValueError, match="не этим журналом"):
        _measure(sl, {"w0": forged})


def test_a_forged_fit_is_refused_without_hashing_its_fields() -> None:
    touched: list[str] = []

    class LoudName(str):
        def __hash__(self):
            touched.append("__hash__")
            return super().__hash__()

    sl = _ledger()
    forged = Fit(
        window=LoudName("w0"),
        frame=pl.DataFrame(),
        rows=1,
        units=frozenset({"x"}),
        labels_known_until=0,
        labels_known_text="1970-01-01 00:00",
        zoned=False,
    )
    with pytest.raises(ValueError, match="не этим журналом"):
        _measure(sl, {"w0": forged})
    assert touched == []


@pytest.mark.parametrize(
    "field, value",
    [
        ("rows", []),
        ("rows", 0),
        ("rows", True),
        ("units", frozenset()),
        ("units", frozenset({1})),
        ("units", {"a"}),
        ("labels_known_until", "2024-01-01"),
        ("labels_known_until", dt.datetime(2023, 1, 1)),
        ("labels_known_until", True),
        ("labels_known_text", ""),
        ("zoned", 1),
        ("zoned", True),
    ],
)
def test_training_metadata_is_checked_at_registration(field, value) -> None:
    """Ревью `ab5d402`: `rows=[]` проходил регистрацию и ронял форматирование
    происхождения ПОСЛЕ расходования всех окон. Данные обучающей части
    проверяются, когда журнал их принимает."""
    fields = {
        "frame": pl.DataFrame({LABEL: [0, 1]}),
        "rows": 2,
        "units": frozenset({"a", "b"}),
        "labels_known_until": 0,
        "labels_known_text": "1970-01-01 00:00",
        "zoned": False,
    }
    fields[field] = value
    with pytest.raises(ValueError, match="обучающая часть окна"):
        _ledger({"w2": Training(**fields)})


def test_a_window_already_measured_spends_no_other_window() -> None:
    """Ревью `ab5d402`: P7 известен заранее, а отказ по второму окну наступал
    после расходования первого. Допустимость выбора проверяется до расхода."""
    sl = _ledger()
    fits = {w: sl.training(w, f"обучение на {w}") for w in ("w0", "w1")}
    sl.measure("w1", "итоговое измерение на w1")

    with pytest.raises(Blocked, match="P7"):
        _measure(sl, fits)
    assert not sl.selections("w0") and not sl.selections("w1")


def test_an_override_of_p7_still_lets_the_windows_be_spent() -> None:
    sl = _ledger()
    fits = {w: sl.training(w, f"обучение на {w}") for w in ("w0", "w1")}
    sl.measure("w1", "итоговое измерение на w1")
    sl._overrides.override("P7", reason="проверка механизма", author="тест")  # noqa: SLF001

    _measure(sl, fits)
    assert sl.selections("w0") and sl.selections("w1")


BERLIN = ZoneInfo("Europe/Berlin")


def _one_window(start: dt.datetime, known: dt.datetime) -> tuple[SampleLedger, Fit]:
    sl = SampleLedger()
    units = frozenset({"о-0"})
    sl.register_window(
        "w0",
        start=start,
        extent=Extent(units=units, since=start, until=start),
        evaluation=units,
        frame=_frame(1),
        training=_training(frozenset({"у-0"}), known),
    )
    return sl, sl.training("w0", "обучение")


def test_future_labels_across_the_clock_change_are_refused() -> None:
    """Ревью `da79e10`: при одном объекте пояса Python сравнивает местное время
    без `fold`. 02:15+01:00 (второй проход часа) позже 02:30+02:00 на 45 минут,
    а `>=` давал False — обучение на будущих метках принималось."""
    start = dt.datetime(2024, 10, 27, 2, 30, tzinfo=BERLIN, fold=0)
    known = dt.datetime(2024, 10, 27, 2, 15, tzinfo=BERLIN, fold=1)
    sl, fit = _one_window(start, known)

    defects = sl.provenance_defects("w0", fit)
    assert len(defects) == 1 and "метки из будущего" in defects[0]
    assert "+0100" in defects[0] and "+0200" in defects[0]


def test_honest_labels_across_the_clock_change_pass() -> None:
    """Обратное направление: 02:45+02:00 раньше 02:30+01:00 на 45 минут, и polars
    по `KNOWN_AT < start` включает такую строку в обучение окна."""
    start = dt.datetime(2024, 10, 27, 2, 30, tzinfo=BERLIN, fold=1)
    known = dt.datetime(2024, 10, 27, 2, 45, tzinfo=BERLIN, fold=0)
    sl, fit = _one_window(start, known)

    assert sl.provenance_defects("w0", fit) == []


def test_an_empty_decision_is_refused_before_reading_windows() -> None:
    """Ревью `da79e10`: пустое `decision` проходило первый проход и падало на
    записи выбора — после трёх обращений аудита."""
    sl = _ledger()
    fits = {w: sl.training(w, f"обучение на {w}") for w in STARTS}
    before = len(sl.accesses)

    with pytest.raises(ValueError, match="решение"):
        stability_across_windows(
            sl,
            {w: Prediction(fit=f, scores=_scores(sl, w)) for w, f in fits.items()},
            CONSTANT,
            decision="",
        )
    assert len(sl.accesses) == before


NEW_YORK = ZoneInfo("America/New_York")


@pytest.mark.parametrize(
    "known, start, future",
    [
        # Ревью `69074aa`: перевод в UTC уводил год 1 по Берлину (+00:53:28) в год 0.
        (dt.datetime(1, 1, 1, tzinfo=BERLIN), dt.datetime(1, 1, 2, tzinfo=BERLIN), False),
        # ...а начало окна в 9999 году по Нью-Йорку — в год 10000.
        (
            dt.datetime(9999, 12, 30, 12, tzinfo=NEW_YORK),
            dt.datetime(9999, 12, 31, 23, 59, 59, tzinfo=NEW_YORK),
            False,
        ),
        (
            dt.datetime(9999, 12, 31, 23, 59, 59, tzinfo=NEW_YORK),
            dt.datetime(9999, 12, 30, 12, tzinfo=NEW_YORK),
            True,
        ),
    ],
)
def test_moments_at_the_edges_of_the_calendar_are_compared(known, start, future) -> None:
    """Сравнение — целыми микросекундами в UTC, без промежуточного `datetime`
    за пределами 1..9999: polars такие моменты сравнивает, и ядро обязано тоже."""
    sl, fit = _one_window(start, known)
    defects = sl.provenance_defects("w0", fit)
    assert (len(defects) == 1 and "метки из будущего" in defects[0]) if future else defects == []


@pytest.mark.parametrize(
    "known, start",
    [
        (
            dt.datetime(2024, 10, 27, 2, 15, tzinfo=BERLIN, fold=1),
            dt.datetime(2024, 10, 27, 2, 30, tzinfo=BERLIN, fold=0),
        ),
        (
            dt.datetime(2024, 10, 27, 2, 45, tzinfo=BERLIN, fold=0),
            dt.datetime(2024, 10, 27, 2, 30, tzinfo=BERLIN, fold=1),
        ),
        (dt.datetime(1, 1, 1, tzinfo=BERLIN), dt.datetime(1, 1, 2, tzinfo=BERLIN)),
        (dt.datetime(2024, 1, 1), dt.datetime(2024, 1, 1, 0, 0, 0, 1)),
        (dt.datetime(2024, 1, 1, 0, 0, 0, 1), dt.datetime(2024, 1, 1)),
        (dt.datetime(2024, 1, 1, tzinfo=dt.UTC), dt.datetime(2024, 1, 1, tzinfo=dt.UTC)),
    ],
)
def test_the_time_condition_agrees_with_the_split(known, start) -> None:
    """Обучение окна — строки с `KNOWN_AT < start` в polars; проверка
    происхождения обязана говорить то же самое."""
    column = pl.Series("known", [known])
    in_training = column.to_frame().select(pl.col("known") < start).item()
    assert (column.dt.epoch("us")[0] < _epoch_microseconds(start)) is in_training


def _training_from_split(known: dt.datetime, cutoff: dt.datetime) -> Training | None:
    """Обучающая часть, построенная штатным `training_of` из отбора polars."""
    from dsx.evals.world import World, base_schema
    from dsx.split import KNOWN_AT, Part, Window, training_of

    frame = pl.DataFrame({"entity_id": ["a", "b"], LABEL: [0, 1], KNOWN_AT: [known, known]})
    train = frame.filter(pl.col(KNOWN_AT) < cutoff)
    part = Part(
        Window("история", cutoff, cutoff + dt.timedelta(microseconds=1)), train, pl.DataFrame()
    )
    return training_of(part, World({"main": frame}, base_schema()))


@pytest.mark.parametrize(
    "known, cutoff",
    [
        (
            dt.datetime(9999, 12, 31, 23, 59, 59, tzinfo=NEW_YORK),
            dt.datetime(9999, 12, 31, 23, 59, 59, 999998, tzinfo=NEW_YORK),
        ),
        (dt.datetime(1, 1, 1, tzinfo=BERLIN), dt.datetime(1, 1, 1, 0, 0, 1, tzinfo=BERLIN)),
        # Сложение с timedelta сбрасывает fold: граница переводится из UTC.
        (
            dt.datetime(2024, 10, 27, 2, 15, tzinfo=BERLIN, fold=1),
            dt.datetime(2024, 10, 27, 1, 16, tzinfo=dt.UTC).astimezone(BERLIN),
        ),
        (
            dt.datetime(2024, 10, 27, 2, 15, tzinfo=BERLIN, fold=0),
            dt.datetime(2024, 10, 27, 0, 16, tzinfo=dt.UTC).astimezone(BERLIN),
        ),
        (dt.datetime(2024, 1, 1, 12), dt.datetime(2024, 1, 1, 12, 0, 1)),
    ],
)
def test_training_of_keeps_the_moment_at_the_edges(known, cutoff) -> None:
    """Ревью `98ab50a`: `Series.max()` строил промежуточную дату в UTC и падал на
    9999-12-31 по Нью-Йорку, хотя polars обе строки включал в обучение. Момент
    берётся числом из polars — тем, что сравнивает отбор сплита."""
    training = _training_from_split(known, cutoff)

    assert training is not None and training.rows == 2
    assert training.labels_known_until == _epoch_microseconds(known)
    assert training.zoned is (known.tzinfo is not None)


def test_the_moment_comes_from_polars_not_from_python_zone_rules() -> None:
    """Ревью `bae53e9`: дата восстанавливалась через `ZoneInfo` Python, а его база
    правил поясов расходилась с polars — Ванкувер 2026-11-15 00:00 UTC: polars
    печатает 16:00-08:00, системная база macOS давала 16:00 смещение −07:00, и
    честное обучение отвергалось. Число и текст берутся из одного источника."""
    from dsx.split import _latest_moment

    utc = 1794700800000000
    column = pl.Series("known", [utc - 1, utc], dtype=pl.Int64).cast(
        pl.Datetime("us", "America/Vancouver")
    )
    known, text = _latest_moment(column)

    assert known == utc
    assert text == column.dt.to_string("%Y-%m-%d %H:%M%z")[1]


def test_future_labels_at_the_end_of_the_calendar_are_refused_through_the_split() -> None:
    """Обратный сценарий ревью целиком: обучение, отобранное polars до своего
    окна, предъявлено окну, начавшемуся раньше его меток."""
    known = dt.datetime(9999, 12, 31, 23, 59, 59, tzinfo=NEW_YORK)
    training = _training_from_split(known, known.replace(microsecond=999998))
    start = dt.datetime(9999, 12, 30, 12, tzinfo=NEW_YORK)

    sl = SampleLedger()
    units = frozenset({"о-0"})
    sl.register_window(
        "w0",
        start=start,
        extent=Extent(units=units, since=start, until=start),
        evaluation=units,
        frame=_frame(1),
        training=training,
    )
    defects = sl.provenance_defects("w0", sl.training("w0", "обучение"))
    assert len(defects) == 1 and "метки из будущего" in defects[0]
