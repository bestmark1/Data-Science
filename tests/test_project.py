"""Форма проекта и запуск по ней."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from dsx.evals.registry import BY_ID
from dsx.project import ProjectForm, load
from dsx.runner import RESERVE, run

FORM = textwrap.dedent("""
    title: "Проверочный проект"
    observed_until: 2025-01-01T00:00:00
    observed_until_source: "конец генерации проверочного мира"
    columns:
      - {name: entity_id, role: entity_id}
      - {name: decided_at, role: decision_time, temporal: instant}
      - {name: deadline_on, role: deadline, temporal: date, value_as_of: decided_at}
      - {name: event_at, role: outcome_component, temporal: instant}
      - {name: status, role: status}
      - name: lead_days
        role: feature
        availability: at_decision
        source_of_claim: "генератор мира"
        window_lookback_days: 0
        window_source: "код мира"
        value_as_of: decided_at
      - {name: size, role: ignored}
      - {name: region, role: ignored}
    outcome:
      event_column: event_at
      deadline_column: deadline_on
      comparison: by_date
      positive_class: event_after_deadline
      estimand: "событие позже назначенного срока"
      degenerate_beyond: 0.01
      missing_causes:
        - {name: "события не было", meaning: not_occurred, status_value: completed}
        - {name: "объект исключён", meaning: excluded, status_value: aborted}
        - {name: "событие ещё впереди", meaning: not_occurred, status_value: pending}
    task:
      target_kind: binary
      outcome_timing: delayed
      has_process: true
      is_stream: true
      object_lifetime: one_shot
    split:
      windows:
        - {name: w0, start_day: 300, stop_day: 345}
        - {name: w1, start_day: 345, stop_day: 390}
      reserve_from_day: 480
    assumptions:
      - statement: "срок назначается до решения"
        basis: owner
        author: "автор"
        consequence: "срок мог назначаться задним числом"
""")


def form(**overrides) -> ProjectForm:
    payload = yaml.safe_load(FORM)
    payload.update(overrides)
    return ProjectForm(**payload)


def test_form_loads_from_yaml(tmp_path: Path) -> None:
    path = tmp_path / "project.yaml"
    path.write_text(FORM, encoding="utf-8")

    assert load(path).title == "Проверочный проект"


def test_unknown_field_is_refused() -> None:
    """Опечатка в имени поля не должна проходить молча."""
    with pytest.raises(ValidationError):
        form(unknown_section={})


def test_window_without_a_source_is_refused() -> None:
    """Окно без источника — необеспеченное объявление (F-5, F-11)."""
    payload = yaml.safe_load(FORM)
    for column in payload["columns"]:
        if column.get("window_source"):
            del column["window_source"]

    with pytest.raises(ValueError, match="источника"):
        ProjectForm(**payload).schema_spec()


def test_reserve_behind_a_window_is_refused() -> None:
    """Резерв, в который заходит окно, независимым не является (F-9)."""
    payload = yaml.safe_load(FORM)
    payload["split"]["reserve_from_day"] = 310

    with pytest.raises(ValidationError, match="заходят за границу резерва"):
        ProjectForm(**payload)


def test_project_without_assumptions_is_refused() -> None:
    """Ноль записанных допущений означает, что их принимали молча."""
    payload = yaml.safe_load(FORM)
    payload["assumptions"] = []

    with pytest.raises(ValidationError):
        ProjectForm(**payload)


def test_direction_of_the_outcome_must_be_declared() -> None:
    """Умолчание здесь дало долю противоположного класса (F-10)."""
    payload = yaml.safe_load(FORM)
    del payload["outcome"]["positive_class"]

    with pytest.raises(ValidationError):
        ProjectForm(**payload)


def test_object_lifetime_must_be_declared() -> None:
    payload = yaml.safe_load(FORM)
    del payload["task"]["object_lifetime"]

    with pytest.raises(ValidationError):
        ProjectForm(**payload)


# --- запуск ----------------------------------------------------------------


def test_run_produces_split_reserve_and_report(tmp_path: Path) -> None:
    result = run(form(), BY_ID["clean-baseline"].build().main, tmp_path)

    assert [p.name for p in result.split.parts] == ["w0", "w1"]
    assert result.split.reserved_rows
    assert result.split.reserved is None, "данные резерва обязаны уйти в журнал"
    assert (tmp_path / "report.md").exists()


def test_clean_world_raises_no_blocking_signals() -> None:
    result = run(form(), BY_ID["clean-baseline"].build().main)

    assert not result.checks.blocking, [str(s) for s in result.checks.signals]


def test_reserve_is_registered_and_measurable() -> None:
    result = run(form(), BY_ID["clean-baseline"].build().main)

    result.samples.select("w0", "выбор порога")
    result.samples.measure(RESERVE)

    assert result.samples.was_measured(RESERVE)


def test_assumptions_from_the_form_reach_the_report() -> None:
    result = run(form(), BY_ID["clean-baseline"].build().main)

    assert "срок назначается до решения" in result.study.render()


def test_summary_names_features_whose_window_cannot_be_verified() -> None:
    result = run(form(), BY_ID["clean-baseline"].build().main)

    assert "lead_days" in result.summary()


def test_template_is_a_valid_shape_even_though_filled_with_placeholders() -> None:
    """Шаблон обязан загружаться: форма с опечаткой не поможет заполнить её."""
    import yaml as _yaml

    payload = _yaml.safe_load(Path("templates/project.yaml").read_text(encoding="utf-8"))

    ProjectForm(**payload)


# --- незаявленные колонки (ревью Кодекса) ----------------------------------


def test_undeclared_column_blocks() -> None:
    """Ядро видит только объявленное: незаявленная колонка невидима проверкам."""
    from dsx.evals.case import Finding

    payload = yaml.safe_load(FORM)
    payload["columns"] = [c for c in payload["columns"] if c["name"] != "region"]
    result = run(ProjectForm(**payload), BY_ID["clean-baseline"].build().main)

    findings = {s.finding for s in result.checks.signals}
    assert Finding.UNDECLARED_COLUMN in findings


def test_ignored_role_is_a_valid_answer() -> None:
    """«Колонка есть, и она не нужна» — ответ. Промолчать — нет."""
    from dsx.evals.case import Finding

    result = run(form(), BY_ID["clean-baseline"].build().main)

    assert Finding.UNDECLARED_COLUMN not in {s.finding for s in result.checks.signals}


def test_blank_evidence_is_refused() -> None:
    """Пробел неотличим от незаполненного поля, а выглядит заполненным."""
    payload = yaml.safe_load(FORM)
    payload["assumptions"][0]["consequence"] = "   "

    with pytest.raises(ValidationError, match="пробелами"):
        ProjectForm(**payload)


def test_duplicate_window_names_are_refused() -> None:
    payload = yaml.safe_load(FORM)
    payload["split"]["windows"][1]["name"] = payload["split"]["windows"][0]["name"]

    with pytest.raises(ValidationError, match="различны"):
        ProjectForm(**payload)


def test_measurement_time_must_play_its_role() -> None:
    """Иначе окно можно сверить само с собой, направив measured_at на решение."""
    payload = yaml.safe_load(FORM)
    for column in payload["columns"]:
        if column.get("role") == "feature":
            column["measured_at"] = "decided_at"

    with pytest.raises(ValueError, match="не объявлена ролью"):
        ProjectForm(**payload).schema_spec()


def test_cause_assumptions_reach_the_registry() -> None:
    """Допущения из контракта исхода — такие же допущения проекта."""
    payload = yaml.safe_load(FORM)
    payload["outcome"]["missing_causes"].append(
        {
            "name": "причина, неотличимая по данным",
            "meaning": "not_occurred",
            "assumption": "принимается, что таких строк немного",
        }
    )

    rendered = ProjectForm(**payload).registry().report_section()

    assert "таких строк немного" in rendered


# --- вопросы порождаются формой, а не догадливостью (ревью Кодекса) --------


def test_open_questions_come_from_the_form_not_from_the_registry() -> None:
    """Чтобы записать неизвестное вручную, надо уже понимать, где оно есть."""
    questions = form().unverifiable_declarations()

    assert questions, "форма обязана породить вопросы сама"
    joined = " ".join(questions)
    assert "lead_days" in joined, "объявленная доступность непроверяема и должна спрашиваться"
    assert "положительным исходом" in joined
    assert "объект" in joined
    assert "ДО просмотра метрик" in joined


def test_every_declared_available_feature_becomes_a_question() -> None:
    payload = yaml.safe_load(FORM)
    available = [c["name"] for c in payload["columns"] if c.get("availability") == "at_decision"]
    questions = " ".join(ProjectForm(**payload).unverifiable_declarations())

    assert all(name in questions for name in available)


def test_questions_reach_the_report() -> None:
    result = run(form(), BY_ID["clean-baseline"].build().main)

    rendered = result.study.render()

    assert "Вопросы владельцу данных" in rendered
    assert "Зелёный прогон не означает" in rendered


def test_declarations_bind_the_conclusion() -> None:
    """Смена горизонта или направления исхода обязана делать вывод устаревшим."""
    world = BY_ID["clean-baseline"].build().main
    first = run(form(), world)
    first.study.conclude("модель лучше правила")

    payload = yaml.safe_load(FORM)
    payload["outcome"]["positive_class"] = "event_within_deadline"
    second = ProjectForm(**payload)
    first.study.declarations = second.model_dump_json()

    assert first.study.is_stale, "объявления обязаны входить в отпечаток протокола"


def test_unsupported_task_kind_is_refused_by_the_runner() -> None:
    """Защита существовала, но не вызывалась."""
    from dsx.task import UnsupportedTask

    payload = yaml.safe_load(FORM)
    payload["task"]["target_kind"] = "regression"

    with pytest.raises(UnsupportedTask):
        run(ProjectForm(**payload), BY_ID["clean-baseline"].build().main)


def test_override_reaches_the_checks_from_the_working_path() -> None:
    """Журнал обходов существовал, но runner его не передавал."""
    from dsx.evals.case import Finding
    from dsx.policy import OverrideLedger

    payload = yaml.safe_load(FORM)
    payload["columns"] = [c for c in payload["columns"] if c["name"] != "region"]
    world = BY_ID["clean-baseline"].build().main

    blocked = run(ProjectForm(**payload), world)
    assert any(
        s.blocking and s.finding is Finding.UNDECLARED_COLUMN for s in blocked.checks.signals
    )

    ledger = OverrideLedger()
    ledger.override("S8", reason="колонка проверена вручную и не нужна", author="автор")
    passed = run(ProjectForm(**payload), world, overrides=ledger)

    assert not any(
        s.blocking and s.finding is Finding.UNDECLARED_COLUMN for s in passed.checks.signals
    )
    assert "колонка проверена вручную" in passed.study.render()
