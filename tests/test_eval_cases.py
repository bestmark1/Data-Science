"""Набор проверочных кейсов и механика вердикта (этап 1)."""

from __future__ import annotations

import pytest

from dsx.evals import Finding, Verdict, judge
from dsx.evals.registry import ALL, BY_ID, NEGATIVE_CONTROLS


def test_registry_has_enough_cases() -> None:
    assert len(ALL) >= 12, "набор меньше двенадцати кейсов не покрывает наблюдённые трения"


def test_negative_controls_are_a_meaningful_share() -> None:
    """Без них проверка, поднимающая тревогу всегда, выглядит идеальной."""
    share = len(NEGATIVE_CONTROLS) / len(ALL)

    assert share >= 0.2, f"negative controls составляют {share:.0%}, нужно не меньше 20%"


def test_case_ids_are_unique() -> None:
    assert len(BY_ID) == len(ALL)


def test_every_case_states_its_rationale() -> None:
    for bundle in ALL:
        assert bundle.case.rationale.strip(), bundle.id


def test_every_declared_finding_is_covered_by_some_case() -> None:
    """Вид дефекта, который никто не воспроизводит, проверить нельзя."""
    covered = {f for b in ALL for f in b.case.expectation.findings}
    declared = set(Finding)
    uncovered = declared - covered

    # Часть видов относится к протоколу и появится вместе с ним.
    protocol_scope = {
        Finding.SAMPLE_ALREADY_SPENT,
        Finding.LABEL_IMMATURITY,
        Finding.JOIN_WITHOUT_DECLARED_GRAIN,
        Finding.ROW_INFLATION_ON_JOIN,
        Finding.UNDECLARED_AVAILABILITY,
        # Проверяется напрямую в test_premises.py: расхождение объявления с
        # данными — свойство пары "мир + объявленная задача", а не мира.
        Finding.PREMISE_MISMATCH,
        # Проверяется напрямую в test_split.py: отсутствие резерва — свойство
        # построенного сплита, а не данных. Мир при этом любой.
        Finding.NO_RESERVED_MEASUREMENT_SAMPLE,
        # Проверяется напрямую в test_project.py: незаявленная колонка —
        # свойство пары "таблица + объявленная схема", а не самой таблицы.
        Finding.UNDECLARED_COLUMN,
    }
    assert uncovered <= protocol_scope, f"не покрыты кейсами: {uncovered - protocol_scope}"


@pytest.mark.parametrize("bundle", ALL, ids=lambda b: b.id)
def test_case_builds_deterministically(bundle) -> None:
    first, second = bundle.build(), bundle.build()

    assert first.main.equals(second.main), "кейс недетерминирован и невоспроизводим"


@pytest.mark.parametrize("bundle", ALL, ids=lambda b: b.id)
def test_case_produces_non_empty_world(bundle) -> None:
    world = bundle.build()

    assert world.main.height > 0
    assert world.schema.decision_time is not None


@pytest.mark.parametrize("bundle", NEGATIVE_CONTROLS, ids=lambda b: b.id)
def test_silence_on_negative_control_is_a_pass(bundle) -> None:
    assert judge(bundle.case, frozenset()).verdict is Verdict.PASS


@pytest.mark.parametrize("bundle", NEGATIVE_CONTROLS, ids=lambda b: b.id)
def test_any_alarm_on_negative_control_is_a_false_alarm(bundle) -> None:
    outcome = judge(bundle.case, frozenset({Finding.DUPLICATE_ROWS}))

    assert outcome.verdict is Verdict.FALSE_ALARM
    assert not outcome.ok


def test_missed_and_unexpected_together_give_both() -> None:
    bundle = BY_ID["duplicate-rows"]

    outcome = judge(bundle.case, frozenset({Finding.MISSING_PERIOD}))

    assert outcome.verdict is Verdict.BOTH
    assert outcome.missed == frozenset({Finding.DUPLICATE_ROWS})
    assert outcome.unexpected == frozenset({Finding.MISSING_PERIOD})


def test_perfect_detector_would_pass_every_case() -> None:
    """Мысленный идеальный детектор проходит весь набор. Если нет — набор противоречив."""
    for bundle in ALL:
        outcome = judge(bundle.case, bundle.case.expectation.findings)
        assert outcome.ok, bundle.id


def test_always_alarming_detector_fails_the_negative_controls() -> None:
    """Проверка, поднимающая тревогу всегда, обязана провалить набор."""
    everything = frozenset(Finding)
    failures = [b.id for b in ALL if not judge(b.case, everything).ok]

    assert len(failures) == len(ALL), "набор не отличает детектор от паникёра"


@pytest.mark.parametrize(
    "bundle", [b for b in ALL if not b.case.expectation.is_negative_control], ids=lambda b: b.id
)
def test_injected_world_actually_differs_from_the_clean_one(bundle) -> None:
    """Кейс обязан доказать, что дефект внесён.

    Четыре раза подряд набор ловил ошибку не в проверке, а в инжекторе: дважды
    дефект отсутствовал вовсе. Изменение данных не означает внесения дефекта,
    но отсутствие изменений означает его отсутствие наверняка.

    Сличается не только мир. Дефект бывает не в данных: объявленная наугад
    предпосылка расходится с данными, не трогая их; отсутствие резерва и
    расположение окон — свойства протокола, а не таблицы. Требовать от таких кейсов изменённых
    данных значило бы запретить их вовсе.
    """
    clean = BY_ID["clean-baseline"]
    injected = bundle.build()

    differs = (
        not injected.main.equals(clean.build().main)
        or injected.schema != clean.build().schema
        or bundle.outcome != clean.outcome
        or bundle.declared_process != clean.declared_process
        or bundle.overlapping_windows != clean.overlapping_windows
        or bundle.reserve != clean.reserve
        or bundle.lifetime != clean.lifetime
        or bundle.target_kind != clean.target_kind
    )

    assert differs, "кейс с дефектом ничем не отличается от чистого: не внесено ничего"


def test_every_requirement_has_a_case_that_makes_it_fire() -> None:
    """Проверка, которую никто не заставлял сработать, не проверена.

    Десятый кейс: N6 смотрела только числовые признаки и пропускала строковые
    утечки. Слепота прожила десять кейсов и семь прогонов на настоящих данных,
    потому что подложенные утечки случайно оказывались числами, а кейса стенда
    именно на строковую не было.

    Отсюда правило: у каждого требования ядра есть кейс, который заставляет его
    возразить. Ноль блокирующих дефектов на настоящих данных этого не заменяет —
    он означает «на этих данных ничего не сломалось», а не «инструмент видит».
    """
    from dsx.checks import ALL_CHECKS

    required = {check.requirement for check in ALL_CHECKS}
    covered = {name for bundle in ALL for name in bundle.case.expectation.caught_by}
    naked = sorted(required - covered)

    assert not naked, f"нет кейса стенда, воспроизводящего дефект: {naked}"


def test_negative_controls_exist_for_the_bench_to_mean_anything() -> None:
    """Набор без отрицательных контролей не отличает проверку от паникёра."""
    controls = [b.id for b in ALL if b.case.expectation.is_negative_control]

    assert len(controls) >= 10, f"отрицательных контролей слишком мало: {len(controls)}"


def test_recurrence_ledger_does_not_overpromise() -> None:
    """Журнал повторов обязан говорить, чего он НЕ покрывает.

    Первая его версия обещала «против каждого урока стоит ответ», а покрывала
    три находки из тринадцати и пять требований из тридцати двух. Документ,
    обещающий больше, чем даёт, хуже отсутствующего: на него ссылаются как на
    полный.
    """
    from pathlib import Path

    ledger = (Path(__file__).resolve().parent.parent / "docs" / "recurrence-ledger.md").read_text(
        encoding="utf-8"
    )

    assert "не все" in ledger.lower(), "журнал не оговаривает своей неполноты"
    assert "Три слоя защиты" in ledger, "не сказано, чем защищено то, чего в журнале нет"
