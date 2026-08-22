"""Формат проверочного кейса.

Кейс знает правильный ответ по построению: дефект в данные закладывается
намеренно, поэтому вердикт механический и не является чьим-то суждением.

Negative control — полноправный вид кейса. Мир без дефекта, где срабатывание
является ошибкой. Без таких кейсов проверка, поднимающая тревогу всегда,
выглядит идеальной.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field


class Finding(StrEnum):
    """Вид дефекта, который проверка способна обнаружить.

    Перечисление растёт вместе с проверками. Строка вместо свободного текста —
    чтобы вердикт был сравнением множеств, а не сопоставлением формулировок.
    """

    UNDECLARED_TEMPORAL_KIND = "undeclared_temporal_kind"
    MIXED_TEMPORAL_COMPARISON = "mixed_temporal_comparison"
    UNDECLARED_AVAILABILITY = "undeclared_availability"
    FEATURE_AFTER_DECISION = "feature_after_decision"
    OUTCOME_COMPONENT_AS_FEATURE = "outcome_component_as_feature"
    SURROGATE_KEY_AS_ENTITY = "surrogate_key_as_entity"
    JOIN_WITHOUT_DECLARED_GRAIN = "join_without_declared_grain"
    ROW_INFLATION_ON_JOIN = "row_inflation_on_join"
    DUPLICATE_ROWS = "duplicate_rows"
    MISSING_PERIOD = "missing_period"
    TRUNCATED_TAIL = "truncated_tail"
    STATUS_TIMESTAMP_CONFLICT = "status_timestamp_conflict"
    POST_TREATMENT_MISSINGNESS = "post_treatment_missingness"
    ENTITY_OVERLAP_ACROSS_SPLITS = "entity_overlap_across_splits"
    LABEL_IMMATURITY = "label_immaturity"
    NON_STATIONARY_TARGET = "non_stationary_target"
    SAMPLE_ALREADY_SPENT = "sample_already_spent"
    PREMISE_MISMATCH = "premise_mismatch"
    UNDECLARED_FEATURE_WINDOW = "undeclared_feature_window"
    FEATURE_WINDOW_OVERLAP = "feature_window_overlap"
    NO_RESERVED_MEASUREMENT_SAMPLE = "no_reserved_measurement_sample"
    FEATURE_WINDOW_MISMATCH = "feature_window_mismatch"
    UNSTABLE_FEATURE_RELATION = "unstable_feature_relation"
    INCOMPARABLE_SUPPORT = "incomparable_support"
    DIRECTION_CONTRADICTS_DOMAIN = "direction_contradicts_domain"
    COMPETING_KINDS_COLLAPSED = "competing_kinds_collapsed"
    SIMULTANEITY_UNDECLARED = "simultaneity_undeclared"
    RATE_CONTRADICTS_EXPECTATION = "rate_contradicts_expectation"


class Expectation(BaseModel):
    """Что кейс ожидает от проверки."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    findings: frozenset[Finding] = frozenset()
    """Дефекты, заложенные в кейс намеренно. Пустое множество означает
    negative control: срабатывание на таком кейсе — ложная тревога."""

    caught_by: frozenset[str] = frozenset()
    """Требования, чьи проверки способны обнаружить этот дефект.

    Существует, потому что один и тот же вид дефекта возникает по разным
    механизмам. Признак, честно объявленный недоступным, ловится проверкой
    контракта; признак, ЛОЖНО объявленный доступным, — нет: проверка читает то
    же ложное утверждение. Без этого поля легко решить, что контрактная
    проверка покрывает случай, который она покрыть не может.
    """

    @property
    def is_negative_control(self) -> bool:
        return not self.findings


class Case(BaseModel):
    """Один проверочный кейс."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: Annotated[str, Field(pattern=r"^[a-z0-9-]{3,60}$")]
    title: Annotated[str, Field(min_length=1)]
    rationale: Annotated[str, Field(min_length=1)]
    """Чем кейс обоснован: какое трение этапа 0 он воспроизводит либо какую
    предпосылку проверяет."""

    expectation: Expectation


class Verdict(StrEnum):
    PASS = "pass"
    MISSED = "missed"
    """Дефект был заложен, но не найден."""

    FALSE_ALARM = "false_alarm"
    """Тревога там, где дефекта нет."""

    BOTH = "both"
    """И пропуск, и ложная тревога одновременно."""


class Outcome(BaseModel):
    """Результат прогона одного кейса."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str
    verdict: Verdict
    missed: frozenset[Finding] = frozenset()
    unexpected: frozenset[Finding] = frozenset()

    @property
    def ok(self) -> bool:
        return self.verdict is Verdict.PASS


def judge(case: Case, reported: frozenset[Finding]) -> Outcome:
    """Сравнить найденное с заложенным.

    Вердикт механический: правильный ответ известен по построению кейса, а не
    является мнением аналитика.
    """
    expected = case.expectation.findings
    missed = expected - reported
    unexpected = reported - expected

    if missed and unexpected:
        verdict = Verdict.BOTH
    elif missed:
        verdict = Verdict.MISSED
    elif unexpected:
        verdict = Verdict.FALSE_ALARM
    else:
        verdict = Verdict.PASS

    return Outcome(case_id=case.id, verdict=verdict, missed=missed, unexpected=unexpected)
