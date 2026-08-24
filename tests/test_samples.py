"""Журнал расхода выборок (P1, P2, P7, S4)."""

from __future__ import annotations

import datetime as dt

import pytest
from pydantic import ValidationError

from dsx.policy import Blocked, OverrideLedger
from dsx.samples import Extent, Purpose, SampleLedger

ORIGIN = dt.datetime(2024, 1, 1)


def extent(prefix: str, count: int = 100, month: int = 1) -> Extent:
    """Непересекающийся состав: у каждой выборки свои единицы решения."""
    return Extent(
        units=frozenset(f"{prefix}-{i}" for i in range(count)),
        since=ORIGIN.replace(month=month),
        until=ORIGIN.replace(month=month + 1),
    )


def ledger(overrides: OverrideLedger | None = None) -> SampleLedger:
    sl = SampleLedger(overrides)
    for month, name in enumerate(("train", "valid", "test", "holdout"), start=1):
        sl.register(name, extent(name, month=month))
    return sl


def test_clean_protocol_passes() -> None:
    """Обучение, выбор на валидации, единственное измерение на тесте."""
    sl = ledger()

    sl.fit("train", "обучение модели")
    sl.select("valid", "выбор окна обучения")
    sl.select("valid", "выбор порога")
    sl.measure("test")

    assert sl.was_measured("test")
    assert not sl.is_spent("holdout")


def test_measuring_on_a_sample_used_for_selection_is_blocked() -> None:
    """Ровно эта ошибка трижды переворачивала вывод этапа 0."""
    sl = ledger()
    sl.select("test", "выбор порога")

    with pytest.raises(Blocked, match="P1"):
        sl.measure("test")


def test_block_message_lists_what_the_sample_was_spent_on() -> None:
    sl = ledger()
    sl.select("test", "выбор окна обучения")
    sl.select("test", "выбор калибровки")

    with pytest.raises(Blocked) as excinfo:
        sl.measure("test")

    message = str(excinfo.value)
    assert "выбор окна обучения" in message and "выбор калибровки" in message


def test_second_measurement_on_the_same_sample_is_blocked() -> None:
    sl = ledger()
    sl.measure("test")

    with pytest.raises(Blocked, match="P1"):
        sl.measure("test")


def test_fitting_does_not_spend_a_sample_as_an_instrument() -> None:
    """Обучение на выборке не мешает измерять на другой."""
    sl = ledger()
    sl.fit("train", "обучение")

    assert not sl.is_spent("train")
    sl.measure("test")


def test_selection_after_measurement_requires_a_fresh_sample() -> None:
    """P7: протокол, изменённый после просмотра метрик, обесценивает измерение."""
    sl = ledger()
    sl.measure("test")

    with pytest.raises(Blocked, match="P7"):
        sl.select("test", "передумал и меняю порог")


def test_selection_on_another_sample_after_measurement_is_allowed() -> None:
    sl = ledger()
    sl.measure("test")

    sl.select("holdout", "новое решение на свежей выборке")

    assert sl.is_spent("holdout")


def test_unregistered_sample_is_refused() -> None:
    """Имя, придуманное на ходу, обходит учёт: 'это был не тест, а dev'."""
    sl = ledger()

    with pytest.raises(Blocked, match="P2"):
        sl.select("dev", "переименовал выборку")


def test_refusal_lists_the_registered_samples() -> None:
    sl = ledger()

    with pytest.raises(Blocked) as excinfo:
        sl.measure("nope")

    assert "train" in str(excinfo.value) and "valid" in str(excinfo.value)


def test_recorded_override_permits_the_spent_measurement() -> None:
    overrides = OverrideLedger()
    overrides.override("P1", reason="осознанно принимаю смещённую оценку", author="автор")
    sl = ledger(overrides)
    sl.select("test", "выбор порога")

    sl.measure("test")

    assert sl.was_measured("test")


def test_override_of_p1_does_not_cover_p7() -> None:
    overrides = OverrideLedger()
    overrides.override("P1", reason="осознанно принимаю смещённую оценку", author="автор")
    sl = ledger(overrides)
    sl.measure("test")

    with pytest.raises(Blocked, match="P7"):
        sl.select("test", "меняю протокол после метрик")


def test_every_access_names_the_decision() -> None:
    """Обращение без названного решения неотличимо от необъяснимого расхода."""
    sl = ledger()

    with pytest.raises(ValidationError):
        sl.select("valid", "")


def test_unspent_samples_are_listed() -> None:
    sl = ledger()
    sl.select("valid", "выбор порога")

    assert sl.unspent() == ("holdout", "test", "train")


def test_report_shows_what_each_decision_rested_on() -> None:
    sl = ledger()
    sl.fit("train", "обучение модели")
    sl.select("valid", "выбор окна обучения")
    sl.measure("test")

    section = sl.report_section()

    assert "выбор окна обучения" in section
    assert Purpose.MEASUREMENT.value in section
    assert "holdout" in section, "нерасходованные выборки должны быть видны"


def test_empty_ledger_is_reported_explicitly() -> None:
    assert "не зафиксировано" in SampleLedger().report_section().lower()


# --- состав выборок, а не имена (F-6) --------------------------------------


def test_overlapping_samples_are_caught_despite_different_names() -> None:
    """На втором кейсе w0 и w2 назывались по-разному и делили все строки."""
    sl = SampleLedger()
    shared = extent("row", count=200)
    sl.register("w0", shared)
    sl.register("w2", shared)
    sl.select("w0", "выбор окна признаков")

    with pytest.raises(Blocked, match="P1"):
        sl.measure("w2")


def test_overlap_message_states_the_share() -> None:
    sl = SampleLedger()
    sl.register("w0", extent("row", count=100))
    sl.register("w2", extent("row", count=100))
    sl.fit("w0", "обучение")

    with pytest.raises(Blocked) as excinfo:
        sl.measure("w2")

    assert "100%" in str(excinfo.value)


def test_partial_overlap_is_enough_to_block() -> None:
    sl = SampleLedger()
    sl.register("w0", Extent(units=frozenset({"a", "b"}), since=ORIGIN, until=ORIGIN))
    sl.register("w2", Extent(units=frozenset({"b", "c"}), since=ORIGIN, until=ORIGIN))
    sl.select("w0", "выбор порога")

    with pytest.raises(Blocked, match="P1"):
        sl.measure("w2")


def test_disjoint_samples_pass() -> None:
    sl = SampleLedger()
    sl.register("valid", extent("v"))
    sl.register("test", extent("t"))
    sl.select("valid", "выбор порога")

    sl.measure("test")

    assert sl.was_measured("test")


def test_undeclared_content_blocks_instead_of_claiming_independence() -> None:
    """Молчание журнала об именах читалось бы как независимость выборок."""
    sl = SampleLedger()
    sl.register("valid")
    sl.register("test")
    sl.select("valid", "выбор порога")

    with pytest.raises(Blocked, match="P2"):
        sl.measure("test")


def test_first_measurement_needs_no_extent_when_nothing_was_touched() -> None:
    """Сравнивать не с чем: требовать состав здесь было бы обрядом."""
    sl = SampleLedger()
    sl.register("test")

    sl.measure("test")

    assert sl.was_measured("test")


def test_report_names_samples_without_declared_content() -> None:
    sl = SampleLedger()
    sl.register("test")
    sl.measure("test")

    assert "только по именам" in sl.report_section()


def test_measuring_the_same_content_twice_under_different_names_is_blocked() -> None:
    """Ровно тот обход, ради которого учёт по содержимому и вводился."""
    sl = SampleLedger()
    shared = extent("row", count=150)
    sl.register("a", shared)
    sl.register("b", shared)
    sl.measure("a")

    with pytest.raises(Blocked, match="P1"):
        sl.measure("b")


def test_fitting_on_the_measurement_rows_is_blocked() -> None:
    """Обучение на тех же строках делает измерение бессмысленным."""
    sl = SampleLedger()
    shared = extent("row", count=150)
    sl.register("train", shared)
    sl.register("test", shared)
    sl.fit("train", "обучение модели")

    with pytest.raises(Blocked, match="P1"):
        sl.measure("test")


# --- журнал владеет данными (совет второго Клода) --------------------------


def test_checkout_records_the_spend_by_the_same_action() -> None:
    """Чтение и запись расхода — одна операция, а не две."""
    import polars as pl

    sl = SampleLedger()
    frame = pl.DataFrame({"x": [1, 2, 3]})
    sl.register("test", extent("t"), frame=frame)

    got = sl.checkout("test", Purpose.MEASUREMENT, "итоговая оценка")

    assert got.equals(frame)
    assert sl.was_measured("test")


def test_second_checkout_for_measurement_is_blocked() -> None:
    """Посмотреть метрику, подкрутить порог, посмотреть снова — самое лёгкое
    действие, как только скоринг доступен."""
    import polars as pl

    sl = SampleLedger()
    sl.register("test", extent("t"), frame=pl.DataFrame({"x": [1]}))
    sl.checkout("test", Purpose.MEASUREMENT, "итоговая оценка")

    with pytest.raises(Blocked, match="P1"):
        sl.checkout("test", Purpose.MEASUREMENT, "ещё разок")


def test_selection_after_measurement_through_checkout_is_blocked() -> None:
    import polars as pl

    sl = SampleLedger()
    sl.register("test", extent("t"), frame=pl.DataFrame({"x": [1]}))
    sl.checkout("test", Purpose.MEASUREMENT, "итоговая оценка")

    with pytest.raises(Blocked, match="P7"):
        sl.checkout("test", Purpose.SELECTION, "подкручиваю порог")


def test_audit_does_not_spend_the_sample() -> None:
    """Аудит решений не принимает и потому измерение не смещает."""
    import polars as pl

    sl = SampleLedger()
    sl.register("test", extent("t"), frame=pl.DataFrame({"x": [1]}))

    sl.checkout("test", Purpose.AUDIT, "проверки постановки")

    assert not sl.is_spent("test")
    sl.checkout("test", Purpose.MEASUREMENT, "итоговая оценка")


def test_checkout_of_a_sample_without_data_is_refused() -> None:
    sl = SampleLedger()
    sl.register("test", extent("t"))

    with pytest.raises(Blocked, match="не владеет данными"):
        sl.checkout("test", Purpose.MEASUREMENT, "итоговая оценка")
