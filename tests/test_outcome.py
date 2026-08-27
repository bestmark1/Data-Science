"""Контракт исхода (C3, C4, C6, A10, F-1)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from dsx.evals.world import build_world
from dsx.outcome import (
    ComparisonMode,
    MissingEventCause,
    MissingEventMeaning,
    OutcomeContractError,
    OutcomeDefinition,
    PositiveClass,
    validate_outcome,
)
from dsx.roles import ColumnSpec, Role, Schema, TemporalKind

BY_STATUS = MissingEventCause(
    name="событие не произошло",
    meaning=MissingEventMeaning.NOT_OCCURRED,
    status_value="completed",
)
UNKNOWABLE = MissingEventCause(
    name="объект выведен из эксплуатации",
    meaning=MissingEventMeaning.NOT_OCCURRED,
    assumption="вывода из эксплуатации в наблюдаемом периоде не было",
)


def definition(**overrides) -> OutcomeDefinition:
    payload = {
        "event_column": "event_at",
        "deadline_column": "deadline_on",
        "comparison": ComparisonMode.BY_DATE,
        "positive_class": PositiveClass.EVENT_AFTER_DEADLINE,
        "missing_causes": [BY_STATUS],
        "event_name": "событие",
        "deadline_name": "назначенный срок",
    }
    return OutcomeDefinition(**{**payload, **overrides})


# --- временная семантика ---------------------------------------------------


def test_by_date_comparison_is_accepted() -> None:
    validate_outcome(definition(), build_world().schema)


def test_direct_comparison_of_date_and_instant_is_refused() -> None:
    with pytest.raises(OutcomeContractError, match="хранит дату"):
        validate_outcome(definition(comparison=ComparisonMode.DIRECT), build_world().schema)


def test_refusal_names_the_consequence_not_just_the_types() -> None:
    with pytest.raises(OutcomeContractError) as excinfo:
        validate_outcome(definition(comparison=ComparisonMode.DIRECT), build_world().schema)

    assert "в тот же день" in str(excinfo.value)


def test_direct_comparison_is_fine_when_granularity_matches() -> None:
    schema = Schema(
        columns=[
            ColumnSpec(name="decided_at", role=Role.DECISION_TIME, temporal=TemporalKind.INSTANT),
            ColumnSpec(name="event_on", role=Role.OUTCOME_COMPONENT, temporal=TemporalKind.DATE),
            ColumnSpec(name="deadline_on", role=Role.DEADLINE, temporal=TemporalKind.DATE),
        ]
    )

    validate_outcome(definition(event_column="event_on", comparison=ComparisonMode.DIRECT), schema)


def test_unknown_column_is_refused() -> None:
    with pytest.raises(OutcomeContractError, match="отсутствует в схеме"):
        validate_outcome(definition(event_column="nope"), build_world().schema)


def test_component_without_granularity_is_refused() -> None:
    schema = Schema(
        columns=[
            ColumnSpec(name="decided_at", role=Role.DECISION_TIME, temporal=TemporalKind.INSTANT),
            ColumnSpec(name="event_at", role=Role.OUTCOME_COMPONENT),
            ColumnSpec(name="deadline_on", role=Role.DEADLINE, temporal=TemporalKind.DATE),
        ]
    )

    with pytest.raises(OutcomeContractError, match="обязана объявить"):
        validate_outcome(definition(), schema)


# --- причины отсутствия события (F-1) --------------------------------------


def test_empty_cause_list_is_refused() -> None:
    """Склейка причин в один класс добавила 11.3% ложных положительных."""
    with pytest.raises(ValidationError):
        definition(missing_causes=[])


def test_indistinguishable_cause_requires_an_assumption() -> None:
    """Неназванное допущение неотличимо от его отсутствия."""
    with pytest.raises(ValidationError, match="допущение"):
        MissingEventCause(name="датчики отключились", meaning=MissingEventMeaning.UNOBSERVED)


def test_blank_assumption_does_not_count() -> None:
    with pytest.raises(ValidationError):
        MissingEventCause(
            name="датчики отключились",
            meaning=MissingEventMeaning.UNOBSERVED,
            assumption="   ",
        )


def test_distinguishable_cause_needs_no_assumption() -> None:
    assert BY_STATUS.distinguishable
    assert BY_STATUS.assumption is None


def test_duplicate_status_values_are_refused() -> None:
    twin = MissingEventCause(
        name="другая причина",
        meaning=MissingEventMeaning.EXCLUDED,
        status_value="completed",
    )

    with pytest.raises(ValidationError, match="нескольких причин"):
        definition(missing_causes=[BY_STATUS, twin])


def test_contract_builds_without_any_status_column() -> None:
    """Второй кейс: статуса нет ни в одной из пяти таблиц."""
    contract = definition(missing_causes=[UNKNOWABLE])

    assert contract.distinguishable == ()
    assert len(contract.indistinguishable) == 1


def test_assumptions_are_exposed_for_the_registry() -> None:
    """Неразличимая причина обязана попасть в реестр, а не раствориться."""
    contract = definition(missing_causes=[UNKNOWABLE])

    assert contract.assumptions() == (UNKNOWABLE.assumption,)


def test_same_meaning_across_indistinguishable_causes_is_resolvable() -> None:
    another = MissingEventCause(
        name="отказ не зарегистрирован",
        meaning=MissingEventMeaning.NOT_OCCURRED,
        assumption="доля незарегистрированных пренебрежимо мала",
    )
    contract = definition(missing_causes=[UNKNOWABLE, another])

    assert not contract.conflated
    assert contract.fallback_meaning() is MissingEventMeaning.NOT_OCCURRED


def test_differing_meanings_are_reported_as_conflated() -> None:
    """Строку такой группы разметить нельзя: наблюдение это или цензура — неизвестно."""
    censored = MissingEventCause(
        name="наблюдение прервано",
        meaning=MissingEventMeaning.UNOBSERVED,
        assumption="доля прерванных наблюдений неизвестна",
    )
    contract = definition(missing_causes=[UNKNOWABLE, censored])

    assert contract.conflated
    assert contract.fallback_meaning() is None


def test_event_name_is_required() -> None:
    with pytest.raises(ValidationError):
        definition(event_name="")


def test_components_are_listed_for_leakage_control() -> None:
    assert definition().components() == frozenset({"event_at", "deadline_on"})


# --- estimand собирается, а не пишется --------------------------------------
#
# Второй кейс: объявлено «отказ в течение 30 дней», вычислялось «отказ позже
# 30 дней», ядро не возразило. Источников было два, и они разошлись.


def test_estimand_follows_the_direction_that_is_computed() -> None:
    """Переворот positive_class переворачивает предложение. В этом всё дело."""
    late = definition(positive_class=PositiveClass.EVENT_AFTER_DEADLINE).estimand
    early = definition(positive_class=PositiveClass.EVENT_WITHIN_DEADLINE).estimand

    assert "наступает позже" in late
    assert "не наступает вовсе" in late, "невозникшее событие тоже положительно"
    assert "наступает не позже" in early
    assert late != early


def test_estimand_names_what_the_missing_causes_mean() -> None:
    """Смысл причин — тоже вычисление, и он тоже попадает в предложение."""
    assert "means" not in definition().estimand
    assert "означает" in definition().estimand


def test_direction_word_in_a_name_is_refused() -> None:
    """Имя, высказывающее суждение, вернуло бы второй источник направления."""
    with pytest.raises(ValidationError, match="слова направления"):
        definition(event_name="отказ в течение 30 дней")
    with pytest.raises(ValidationError, match="слова направления"):
        definition(deadline_name="срок, после которого поздно")


def test_a_direction_word_inside_another_word_is_not_a_direction_word() -> None:
    """«до» живёт внутри «доставки», «более» — внутри «наиболее».

    Отрицательный контроль: запрет, срабатывающий на честных именах, сделал бы
    поле незаполнимым и вернул бы автора к вранью в обход.
    """
    assert definition(event_name="доставка заказа", deadline_name="обещанная дата").estimand
