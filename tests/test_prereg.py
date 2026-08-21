"""Предрегистрация второго кейса (S8)."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from dsx.prereg import (
    COUNTS_FOR_ACCEPTANCE,
    Amendment,
    PreregViolation,
    accepts,
    digest_dataset,
    register,
    verify,
)

AREAS = frozenset(
    {
        "контракт исхода",
        "временной сплит",
        "грануляция",
        "реестр допущений",
        "расход выборок",
        "перенос без правок ядра",
        "воспроизводимость",
        "новые типы дефектов",
    }
)


@pytest.fixture
def criteria(tmp_path: Path) -> Path:
    path = tmp_path / "prereg.md"
    path.write_text("P-1: перенос без правок ядра\nP-2: не более 12 часов\n", encoding="utf-8")
    return path


def test_registration_captures_the_criteria(criteria: Path) -> None:
    prereg = register(criteria, core_version="0.0.1")

    assert prereg.criteria_digest
    assert not prereg.data_contacted


def test_unchanged_criteria_verify(criteria: Path) -> None:
    verify(register(criteria, core_version="0.0.1"), criteria)


def test_edited_criteria_are_detected(criteria: Path) -> None:
    """Правка после фиксации перестаёт быть незаметной."""
    prereg = register(criteria, core_version="0.0.1")
    criteria.write_text("P-1: перенос допускает мелкие правки ядра\n", encoding="utf-8")

    with pytest.raises(PreregViolation, match="изменились после фиксации"):
        verify(prereg, criteria)


def test_violation_message_points_to_the_amendment_route(criteria: Path) -> None:
    prereg = register(criteria, core_version="0.0.1")
    criteria.write_text("другое\n", encoding="utf-8")

    with pytest.raises(PreregViolation) as excinfo:
        verify(prereg, criteria)

    assert "поправку" in str(excinfo.value)


def test_dataset_digest_changes_when_a_file_changes(tmp_path: Path) -> None:
    """Переход на другую версию датасета — способ обойти предрегистрацию."""
    first = tmp_path / "a.csv"
    second = tmp_path / "b.csv"
    first.write_text("id\n1\n", encoding="utf-8")
    second.write_text("id\n2\n", encoding="utf-8")
    before = digest_dataset([first, second])

    first.write_text("id\n9\n", encoding="utf-8")

    assert digest_dataset([first, second]) != before


def test_dataset_digest_is_order_independent(tmp_path: Path) -> None:
    first = tmp_path / "a.csv"
    second = tmp_path / "b.csv"
    first.write_text("x\n", encoding="utf-8")
    second.write_text("y\n", encoding="utf-8")

    assert digest_dataset([first, second]) == digest_dataset([second, first])


def test_amendment_requires_a_reason_and_affected_predictions() -> None:
    with pytest.raises(ValidationError):
        Amendment(
            what="сдвинул порог",
            why="так",
            affected_predictions=["P-2"],
            invalidates_conclusions=False,
            author="автор",
        )

    with pytest.raises(ValidationError):
        Amendment(
            what="сдвинул порог",
            why="двенадцати часов не хватило на разбор схемы",
            affected_predictions=[],
            invalidates_conclusions=False,
            author="автор",
        )


def test_amendment_marks_conclusions_stale_when_it_should() -> None:
    amendment = Amendment(
        what="изменён порог стоимости адаптации",
        why="двенадцати часов не хватило на разбор незнакомой схемы",
        affected_predictions=["P-2"],
        invalidates_conclusions=True,
        author="автор",
    )

    assert "прежние заключения устарели" in str(amendment)


# --- правило приёмки ------------------------------------------------------


def test_serious_finding_in_a_named_area_closes_the_requirement() -> None:
    assert accepts({"временной сплит": "blocker"}, AREAS)


def test_cosmetic_finding_does_not_close_it() -> None:
    """«Хоть что-то сломалось» оптимизирует под наличие поломки."""
    assert not accepts({"грануляция": "cosmetic"}, AREAS)


def test_report_only_finding_does_not_close_it() -> None:
    assert not accepts({"контракт исхода": "report-only"}, AREAS)


def test_serious_finding_outside_the_named_areas_does_not_close_it() -> None:
    """Поломка вне заданных областей не свидетельствует о переносимости."""
    assert not accepts({"оформление отчёта": "blocker"}, AREAS)


def test_no_findings_at_all_does_not_close_it() -> None:
    """Прохождение целиком означает слишком похожий кейс, а не успех."""
    assert not accepts({}, AREAS)


def test_assumption_gap_counts() -> None:
    assert accepts({"реестр допущений": "assumption-gap"}, AREAS)


def test_acceptance_levels_are_fixed_in_advance() -> None:
    assert COUNTS_FOR_ACCEPTANCE == frozenset({"blocker", "protocol-change", "assumption-gap"})
