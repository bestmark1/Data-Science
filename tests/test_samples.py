"""Журнал расхода выборок (P1, P2, P7, S4)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from dsx.policy import Blocked, OverrideLedger
from dsx.samples import Purpose, SampleLedger


def ledger(overrides: OverrideLedger | None = None) -> SampleLedger:
    sl = SampleLedger(overrides)
    sl.register("train", "valid", "test", "holdout")
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
