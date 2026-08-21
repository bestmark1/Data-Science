"""Реестр допущений и порождение вопросов отрасли (S2, S3)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from dsx.assumptions import AssumptionRegistry, Basis


def registry_with_stage0_examples() -> AssumptionRegistry:
    """Пять решений этапа 0, каждое из которых было доменным знанием."""
    registry = AssumptionRegistry()
    registry.record(
        "порядок событий процесса: решение, подтверждение, передача, завершение",
        basis=Basis.DOMAIN_KNOWLEDGE,
        author="автор",
        consequence="проверка монотонности меток проверяет неверный порядок",
    )
    registry.record(
        "отменённый объект не считается опоздавшим",
        basis=Basis.DOMAIN_KNOWLEDGE,
        author="автор",
        consequence="модель учится предсказывать отмену вместо задержки",
    )
    registry.record(
        "доля положительного класса равна 8.48%",
        basis=Basis.DATA,
        author="автор",
        consequence="ничего: пересчитывается из данных",
        evidence="шаг 04",
    )
    registry.record(
        "крайний срок отгрузки ставится при оформлении",
        basis=Basis.OWNER,
        author="автор",
        consequence="признак оказывается post-treatment и даёт лик",
    )
    return registry


def test_domain_knowledge_becomes_a_question() -> None:
    """Скорость собственного ответа была принята за отсутствие вопроса."""
    registry = registry_with_stage0_examples()

    questions = registry.open_questions()

    assert len(questions) == 3
    assert any("порядок событий" in q for q in questions)


def test_data_derived_assumption_needs_no_asking() -> None:
    registry = registry_with_stage0_examples()

    derived = registry.by_basis(Basis.DATA)

    assert len(derived) == 1
    assert not derived[0].needs_asking


def test_owner_claim_still_needs_confirmation_elsewhere() -> None:
    """Ответ владельца верен для этого места и не переносится на следующее."""
    registry = registry_with_stage0_examples()

    assert registry.by_basis(Basis.OWNER)[0].needs_asking


def test_question_carries_the_consequence() -> None:
    """Вопрос без последствия невозможно приоритизировать."""
    registry = registry_with_stage0_examples()

    question = next(q for q in registry.open_questions() if "отменённый" in q)

    assert "предсказывать отмену" in question


def test_consequence_is_required() -> None:
    registry = AssumptionRegistry()

    with pytest.raises(ValidationError):
        registry.record("что-то принято", basis=Basis.DATA, author="автор", consequence="")


def test_statement_is_required() -> None:
    registry = AssumptionRegistry()

    with pytest.raises(ValidationError):
        registry.record("", basis=Basis.DATA, author="автор", consequence="сломается")


def test_report_lists_assumptions_and_questions_separately() -> None:
    section = registry_with_stage0_examples().report_section()

    assert "## Допущения" in section
    assert "Требуют подтверждения" in section
    assert section.count("| domain_knowledge |") == 2


def test_empty_registry_is_reported_explicitly() -> None:
    """Отсутствие записанных допущений — не то же самое, что их отсутствие."""
    assert "Не зафиксировано" in AssumptionRegistry().report_section()


def test_questions_are_generated_not_maintained_by_hand() -> None:
    """Список, ведомый вручную, расходится с тем, на чём анализ держится."""
    registry = AssumptionRegistry()
    registry.record(
        "новое доменное допущение",
        basis=Basis.DOMAIN_KNOWLEDGE,
        author="автор",
        consequence="что-то сломается",
    )

    assert len(registry.open_questions()) == 1
