"""Переносимый отчёт: заключение вместе с протоколом (S7, P6)."""

from __future__ import annotations

import datetime as dt

import pytest
from pydantic import ValidationError

from dsx.assumptions import Basis
from dsx.evals.registry import BY_ID
from dsx.report import Study
from dsx.samples import Extent
from harness import report_for


def study(title: str = "Проверочное исследование") -> Study:
    subject = Study(title=title)
    for name in ("train", "valid", "test"):
        subject.samples.register(name)
    return subject


def test_empty_study_says_so_rather_than_staying_silent() -> None:
    rendered = study().render()

    assert "Не сформулировано" in rendered
    assert "Обращений не зафиксировано" in rendered
    assert "Не зафиксировано" in rendered


def test_conclusion_carries_the_protocol_digest() -> None:
    subject = study()
    subject.samples.fit("train", "обучение")

    conclusion = subject.conclude("модель применима")

    assert conclusion.protocol
    assert f"`{conclusion.protocol}`" in subject.render()


def test_protocol_digest_changes_when_a_sample_is_spent() -> None:
    """Расход выборки способен изменить заключение, значит меняет и отпечаток."""
    subject = study()
    before = subject.protocol_digest()

    subject.samples.select("valid", "выбор порога")

    assert subject.protocol_digest() != before


def test_protocol_digest_changes_when_an_override_is_recorded() -> None:
    subject = study()
    before = subject.protocol_digest()

    subject.overrides.override("A8", reason="грануляция подтверждена владельцем", author="автор")

    assert subject.protocol_digest() != before


def test_conclusion_becomes_stale_when_the_protocol_moves() -> None:
    subject = study()
    subject.conclude("модель применима")

    subject.samples.select("valid", "выбор порога после заключения")

    assert subject.is_stale
    assert "Протокол изменился" in subject.render()


def test_fresh_conclusion_is_not_marked_stale() -> None:
    subject = study()
    subject.samples.select("valid", "выбор порога")
    subject.conclude("модель применима")

    assert not subject.is_stale
    assert "Протокол изменился" not in subject.render()


def test_history_is_kept_not_overwritten() -> None:
    """Вывод, менявшийся трижды, заслуживает иного доверия, чем полученный раз."""
    subject = study()
    subject.conclude("модель хуже правила")
    subject.samples.select("valid", "введено валидационное окно")
    subject.conclude("ни один кандидат не лучше")
    subject.samples.select("test", "скользящая оценка")
    subject.conclude("модель устойчиво лучше")

    rendered = subject.render()

    assert "История заключений" in rendered
    assert rendered.count("протокол `") >= 3
    assert subject.conclusion_changed_under_a_different_protocol


def test_repeated_identical_conclusion_is_not_flagged_as_changing() -> None:
    subject = study()
    subject.conclude("модель применима")
    subject.samples.select("valid", "уточнение порога")
    subject.conclude("модель применима")

    assert not subject.conclusion_changed_under_a_different_protocol


def test_findings_separate_blocking_from_advisory() -> None:
    subject = study()
    subject.checks = report_for(BY_ID["outcome-component-as-feature"])

    rendered = subject.render()

    assert "### Блокирующие" in rendered


def test_skipped_checks_are_listed_with_their_reason() -> None:
    subject = study()
    subject.checks = report_for(BY_ID["clean-baseline"])

    rendered = subject.render()

    assert "Непроведённые проверки" in rendered


def test_clean_world_reports_no_defects_explicitly() -> None:
    subject = study()
    subject.checks = report_for(BY_ID["clean-baseline"])

    assert "Проверки не нашли дефектов" in subject.render()


def test_assumptions_and_questions_reach_the_report() -> None:
    subject = study()
    subject.assumptions.record(
        "порядок событий процесса известен из опыта",
        basis=Basis.DOMAIN_KNOWLEDGE,
        author="автор",
        consequence="проверка монотонности проверяет неверный порядок",
    )

    rendered = subject.render()

    assert "Допущения" in rendered
    assert "Требуют подтверждения" in rendered


def test_sample_spending_reaches_the_report() -> None:
    subject = study()
    for name in ("train", "test"):
        subject.samples.register(
            name,
            Extent(
                units=frozenset(f"{name}-{i}" for i in range(10)),
                since=dt.datetime(2024, 1, 1),
                until=dt.datetime(2024, 2, 1),
            ),
        )
    subject.samples.fit("train", "обучение модели")
    subject.samples.measure("test")

    rendered = subject.render()

    assert "Расход выборок" in rendered
    assert "обучение модели" in rendered


def test_empty_statement_is_refused() -> None:
    with pytest.raises(ValidationError):
        study().conclude("")


def test_signal_order_does_not_depend_on_group_by() -> None:
    """Отчёт обязан воспроизводиться дословно, а не «по существу».

    Седьмой кейс дал два прогона, различающиеся местом одной строки: порядок
    после group_by в polars не определён. Заключение от этого не менялось, но
    отпечаток, которым заключение связано с протоколом, — менялся.
    """
    import polars as pl

    from dsx.checks.base import Context
    from dsx.checks.data import StatusEventConflict
    from dsx.evals.registry import BY_ID

    base = BY_ID["clean-baseline"]
    world = base.build()
    # Много разных статусов: чем их больше, тем вероятнее перестановка.
    frame = world.main.with_columns(
        pl.when(pl.arange(0, pl.len()) % 4 == 0)
        .then(None)
        .otherwise(pl.col("event_at"))
        .alias("event_at")
    )
    world = world.replace_main(frame)
    context = Context(world, base.outcome, None, None)

    runs = [[s.detail for s in StatusEventConflict().run(context)] for _ in range(5)]

    assert runs[0], "кейс должен давать хотя бы один сигнал, иначе тест пуст"
    assert all(r == runs[0] for r in runs), runs
    assert runs[0] == sorted(runs[0]), "порядок обязан быть определённым, а не случайным"
