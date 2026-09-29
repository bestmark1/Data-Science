"""Порядок кейса проверяется до вызова построителя.

В девятнадцатом кейсе слепой контроль был проведён ПОСЛЕ первого прогона,
вопреки §5а: автор нарушил порядок собственной рукой и записал это ошибкой.
Напоминание в инструкциях уже стояло и не помогло.

Главный тест здесь — отрицательный: при неподготовленном контроле построитель
не вызывается ВООБЩЕ. Отказ после сборки данных пришёл бы задним числом, когда
работа сделана.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from protocol import OutOfOrder, guarded, preflight, save_control_run  # noqa: E402
from protocol.preflight import (  # noqa: E402
    ANALOGS,
    CONTROL_RUN,
    EMPTY_BIT,
    FROM_CASE,
    SEALED,
    _control_roles,
    _form_identity,
    control_states,
    declared,
    fired_against_the_control_run,
    fired_against_the_ledger,
    lost_bets_against_verdict,
    read_control_run,
    unreachable_controls,
)

FORM = """
columns:
  - {name: at_night, role: feature}
  - {name: received_at, role: decision_time}
"""

CONTROLS = """
```yaml
project: проба
controls:
  - {name: К-1, column: at_night, role: feature, clean_role: ignored,
     expect: sentinel_as_value, basis: plant}
```
"""


ANALOGS_BLOCK = """
```yaml
project: проба
analogs:
  date: 2026-09-29
  queries:
    github: ["задержка доставки прогноз"]
    kaggle: ["delivery delay prediction"]
    openml: ["delivery delay"]
  found: []
  taken: []
  rejected: []
  pitfalls: []
```
"""
"""Годная запись об аналогах: «ничего не найдено», но с запросами по всем трём
источникам. Добавляется фикстурами по умолчанию, чтобы тесты других шагов
проверяли свой шаг, а не отказ по аналогам."""


def _prereg(tmp_path: Path, body: str, number: int = FROM_CASE, *, analogs=True) -> Path:
    path = tmp_path / f"prereg-case-{number}.md"
    if analogs and "project: проба" in body:
        body += ANALOGS_BLOCK
    path.write_text(body, encoding="utf-8")
    return path


def _project(tmp_path: Path, *, manifest=True, sealed=True, empty=True, analogs=True) -> Path:
    project = tmp_path / "проба"
    (project / "report").mkdir(parents=True)
    (project / "project.yaml").write_text(FORM, encoding="utf-8")
    if analogs:
        (project / ANALOGS).write_text(ANALOGS_BLOCK, encoding="utf-8")
    if manifest:
        (project / "manifest.yaml").write_text("source: проба\n", encoding="utf-8")
    if sealed:
        (project / SEALED).write_text("ОЖИДАЕМАЯ НАХОДКА: sentinel_as_value\n", encoding="utf-8")
    if empty:
        (project / EMPTY_BIT).write_text("Контроль был НЕПУСТЫМ.\n", encoding="utf-8")
    return project


def _digests(project: Path) -> str:
    """Отпечатки в том виде, в каком их вписывают в §5а."""
    bit = hashlib.sha256((project / EMPTY_BIT).read_bytes()).hexdigest()[:12]
    answer = hashlib.sha256((project / SEALED).read_bytes()).hexdigest()[:12]
    return (
        "```yaml\nproject: проба\n```\n\n"
        f"## 5а. Слепой контроль\n\n"
        f"Отпечаток бита пустоты: **`{bit}`**\n"
        f"Отпечаток ответа: **`{answer}`**\n"
    )


# --- Главное: построитель не вызывается --------------------------------------


def test_the_builder_is_not_called_without_a_sealed_control(tmp_path) -> None:
    """Слепой контроль не внесён — сборки данных не происходит вовсе.

    Построитель подменён счётчиком. Он обязан остаться нулевым: отказ, пришедший
    после сборки, означал бы, что данные уже прочитаны и обработаны.
    """
    project = _project(tmp_path, sealed=False, empty=False)
    prereg = _prereg(tmp_path, "```yaml\nproject: проба\n```\n\n## 5а. Слепой контроль\n")
    calls = 0

    def build():
        nonlocal calls
        calls += 1

    with pytest.raises(OutOfOrder) as отказ:
        preflight(project, prereg)
        build()

    assert calls == 0, "построитель вызван, хотя порядок нарушен"
    assert "слепой контроль не внесён" in str(отказ.value).lower()


def test_the_builder_is_not_called_without_collected_data(tmp_path) -> None:
    """Манифеста нет — данных нет, и говорить не о чем."""
    project = _project(tmp_path, manifest=False)
    prereg = _prereg(tmp_path, _digests(project))

    with pytest.raises(OutOfOrder) as отказ:
        preflight(project, prereg)

    assert "данные не собраны" in str(отказ.value)


def test_missing_digests_stop_the_run(tmp_path) -> None:
    """Отпечатки не вписаны — запись перестала бы быть обещанием.

    Вписанные ПОСЛЕ прогона, они ничего не удостоверяют: автор к тому моменту
    видел сигналы и мог подобрать заключение.
    """
    project = _project(tmp_path)
    prereg = _prereg(
        tmp_path,
        "```yaml\nproject: проба\n```\n\n## 5а\n\nОтпечаток бита пустоты: `—`\n",
    )

    with pytest.raises(OutOfOrder) as отказ:
        preflight(project, prereg)

    assert "не оба отпечатка" in str(отказ.value)


def test_a_digest_that_does_not_match_the_file_stops_the_run(tmp_path) -> None:
    """Отпечаток не сходится: вписан не тот либо файл изменён после запечатывания."""
    project = _project(tmp_path)
    prereg = _prereg(
        tmp_path,
        "```yaml\nproject: проба\n```\n\n## 5а\n\nОтпечаток бита пустоты: **`000000000000`**\n"
        "Отпечаток ответа: **`111111111111`**\n",
    )

    with pytest.raises(OutOfOrder) as отказ:
        preflight(project, prereg)

    assert "не сходится" in str(отказ.value)


# --- Снятие контроля до прогона ----------------------------------------------


def test_spent_control_before_the_control_run_is_refused(tmp_path) -> None:
    """`spent_controls`, вписанные заранее, не разрешают пропустить контроль.

    Это требование задания в чистом виде: отметка об отработке не должна
    заменять саму отработку. До контрольного прогона `control-run.md` не
    существует, и запись оказывается ничем не обеспечена.
    """
    project = _project(tmp_path)
    prereg = _prereg(
        tmp_path,
        _digests(project) + "\n```yaml\nproject: проба\nspent_controls:\n"
        "  - {name: К-1, fired: true, outcome: 'снят'}\n```\n",
    )

    with pytest.raises(OutOfOrder) as отказ:
        preflight(project, prereg)

    # Формулировка уточнена вслед за отказом: он теперь называет ПРИЧИНУ
    # непригодности, а не общее «нет либо не по схеме». Намерение прежнее.
    assert "файла доказательства нет" in str(отказ.value)


def test_a_prepared_case_passes(tmp_path) -> None:
    """Положительный контроль: подготовленный кейс не отклоняется.

    Проверка, отказывающая всегда, защищает не лучше отсутствующей.
    """
    project = _project(tmp_path)
    preflight(project, _prereg(tmp_path, _digests(project)))


# --- Совместимость с историческими кейсами -----------------------------------


def test_historical_cases_are_left_alone(tmp_path) -> None:
    """Кейсы до FROM_CASE не проверяются: доказательств у них нет.

    Достроить их задним числом значило бы выдумать свидетельство. Проверка
    молчит, а отсутствие доказательства называется в отчёте — не выдаётся за
    успех.
    """
    project = _project(tmp_path, manifest=False, sealed=False, empty=False)

    блок = "```yaml\nproject: проба\n```\n"
    preflight(project, _prereg(tmp_path, блок, number=FROM_CASE - 1))
    preflight(project, _prereg(tmp_path, блок, number=2))


# --- Сохранение отчёта контрольного прогона ----------------------------------


def test_the_control_run_report_is_saved_while_controls_stand(tmp_path) -> None:
    """Отчёт контрольного прогона сохраняется отдельно от рабочего.

    `report.md` перезаписывается следующим прогоном. На кейсе 20 это проверено
    прямо: в git лежит версия после снятия контролей, и находки К-1 в ней нет
    ни одной, хотя контроль сработал.
    """
    project = _project(tmp_path)
    prereg = _prereg(tmp_path, _digests(project) + CONTROLS)
    saved = save_control_run(project, prereg, ["sentinel_as_value"])

    assert saved == project / "report" / CONTROL_RUN
    assert "sentinel_as_value" in saved.read_text(encoding="utf-8")
    assert "case: 21" in saved.read_text(encoding="utf-8"), "доказательство обязано назвать кейс"


def test_the_control_run_report_is_not_overwritten(tmp_path) -> None:
    """Первый контрольный прогон и есть тот, о котором говорит `spent_controls`."""
    project = _project(tmp_path)
    prereg = _prereg(tmp_path, _digests(project) + CONTROLS)
    (project / "report" / CONTROL_RUN).write_text("первый\n", encoding="utf-8")

    assert save_control_run(project, prereg, ["duplicate_rows"]) is None
    assert (project / "report" / CONTROL_RUN).read_text(encoding="utf-8") == "первый\n"


def test_nothing_is_saved_when_controls_are_already_removed(tmp_path) -> None:
    """Отрицательный контроль: чистый прогон отчётом контрольного не считается.

    Роль колонки в формуляре разошлась с той, что требует контроль, — значит
    контроль снят, и сохранять нечего.
    """
    project = _project(tmp_path)
    (project / "project.yaml").write_text(
        "columns:\n  - {name: at_night, role: ignored}\n", encoding="utf-8"
    )
    prereg = _prereg(tmp_path, _digests(project) + CONTROLS)

    assert save_control_run(project, prereg, ["sentinel_as_value"]) is None
    assert not (project / "report" / CONTROL_RUN).is_file()


# --- Достижимость ожидаемой находки ------------------------------------------


CONTRADICTORY = """
```yaml
project: проба
controls:
  - {name: К-2, column: pump_installed, role: feature, clean_role: ignored,
     expect: value_revised_after_decision, basis: plant}
```
"""

FORM_WITH_VALUE_AS_OF = """
columns:
  - {name: received_at, role: decision_time}
  - {name: pump_installed, role: feature, value_as_of: received_at}
"""

FORM_WITH_HONEST_FIXING = """
columns:
  - {name: received_at, role: decision_time}
  - {name: pump_installed, role: feature, value_as_of: pump_installed}
"""


def test_a_control_whose_finding_is_unreachable_is_refused(tmp_path) -> None:
    """Случай двадцатого кейса: К-2 не мог сработать по построению.

    Контроль ждал `value_revised_after_decision` от колонки, объявленной с
    `value_as_of`, равным моменту решения, — а проверка S10 такой случай
    пропускает. Роль была свободна, сверка §5 промолчала: она знает про
    исполнимость и молчит про достижимость.
    """
    project = _project(tmp_path)
    (project / "project.yaml").write_text(FORM_WITH_VALUE_AS_OF, encoding="utf-8")
    prereg = _prereg(tmp_path, _digests(project) + CONTRADICTORY)

    with pytest.raises(OutOfOrder) as отказ:
        preflight(project, prereg)

    сказано = str(отказ.value)
    assert "недостижимой" in сказано
    assert "premise_check.py" in сказано, "отказ обязан называть строку, где проверка молчит"


def test_an_honestly_declared_control_is_not_refused(tmp_path) -> None:
    """Положительный контроль: честный момент фиксации проходит.

    Реестр противоречий не должен отклонять всё подряд — иначе он запретил бы
    контроль, который как раз и сработает.
    """
    project = _project(tmp_path)
    (project / "project.yaml").write_text(FORM_WITH_HONEST_FIXING, encoding="utf-8")

    preflight(project, _prereg(tmp_path, _digests(project) + CONTRADICTORY))


# --- Сверка `fired` с отчётом контрольного прогона ---------------------------


def _spent(fired: bool) -> str:
    return (
        "\n```yaml\nproject: проба\nspent_controls:\n"
        f"  - {{name: К-1, fired: {str(fired).lower()}, outcome: 'снят'}}\n```\n"
    )


def test_fired_true_without_the_finding_is_reported(tmp_path) -> None:
    """Объявлено «сработал», а находки в отчёте ядра нет.

    `fired` пишет автор, отчёт порождает ядро. Расхождение между ними — это
    расхождение утверждения с фактом, а не двух утверждений между собой.
    """
    project = _project(tmp_path)
    prereg = _prereg(tmp_path, _digests(project) + CONTROLS + _spent(True))
    save_control_run(project, prereg, ["duplicate_rows"])

    problems = fired_against_the_control_run(project, prereg)

    assert len(problems) == 1
    assert "объявлен сработавшим" in problems[0]


def test_fired_false_with_the_finding_is_reported(tmp_path) -> None:
    """Обратное: объявлено «промолчал», а ядро находку назвало."""
    project = _project(tmp_path)
    prereg = _prereg(tmp_path, _digests(project) + CONTROLS + _spent(False))
    save_control_run(project, prereg, ["sentinel_as_value"])

    problems = fired_against_the_control_run(project, prereg)

    assert len(problems) == 1
    assert "объявлен промолчавшим" in problems[0]


def test_fired_matching_the_report_is_silent(tmp_path) -> None:
    """Положительный контроль: запись автора совпала с выводом ядра."""
    project = _project(tmp_path)
    prereg = _prereg(tmp_path, _digests(project) + CONTROLS + _spent(True))
    save_control_run(project, prereg, ["sentinel_as_value"])

    assert not fired_against_the_control_run(project, prereg)


# --- Проигранная ставка против вердикта --------------------------------------


BET_AND_PREDICTION = """
```yaml
project: проба
promises:
  - {column: at_night, role: observation_reason}
predictions:
  - {id: P-8, rests_on: promises}
lost_bets:
  - {column: at_night, reason: "проиграна"}
```
"""


def _verdict(tmp_path: Path, итог: str) -> Path:
    path = tmp_path / "case21-verdict.md"
    path.write_text(
        f"| № | Предсказание | Итог |\n|---|---|---|\n| P-8 | обещания исполнены | {итог} |\n",
        encoding="utf-8",
    )
    return path


def test_a_lost_bet_declared_fulfilled_in_the_verdict_is_reported(tmp_path) -> None:
    """Ставка проиграна, а вердикт говорит «сбылось».

    Номер предсказания в механизм не зашит: связь идёт через `rests_on`, и какое
    именно предсказание опирается на обещания, говорит пре-регистрация.
    """
    prereg = _prereg(tmp_path, BET_AND_PREDICTION)

    problems = lost_bets_against_verdict(prereg, _verdict(tmp_path, "сбылось"))

    assert len(problems) == 1
    assert "ставка проиграна" in problems[0]


def test_a_lost_bet_matching_the_verdict_is_silent(tmp_path) -> None:
    """Положительный контроль: вердикт признаёт проигрыш."""
    prereg = _prereg(tmp_path, BET_AND_PREDICTION)

    assert not lost_bets_against_verdict(prereg, _verdict(tmp_path, "**НЕ СБЫЛОСЬ**"))


def test_a_lost_bet_without_a_declared_prediction_is_reported(tmp_path) -> None:
    """Связь проверить нечем, и молчать об этом нельзя."""
    prereg = _prereg(
        tmp_path,
        "```yaml\nproject: проба\npromises:\n  - {column: at_night, role: observation_reason}\n"
        "lost_bets:\n  - {column: at_night, reason: 'проиграна'}\n```\n",
    )

    problems = lost_bets_against_verdict(prereg, _verdict(tmp_path, "сбылось"))

    assert len(problems) == 1
    assert "rests_on" in problems[0]


# --- Промолчавший контроль против учёта --------------------------------------


def _ledger(tmp_path: Path, controls: str) -> Path:
    path = tmp_path / "stopping-rule.md"
    path.write_text(
        f"| Кейс | Блокирующих | Контроли | Отсчёт |\n|---|---|---|---|\n"
        f"| {FROM_CASE} — проба | 0 | {controls} | не начат |\n",
        encoding="utf-8",
    )
    return path


def test_a_silent_control_counted_as_success_is_reported(tmp_path) -> None:
    """Контроль промолчал, а учёт говорит об успехе.

    Проверка слабая намеренно: сильная потребовала бы машиночитаемого учёта, то
    есть переписывания всех прежних строк задним числом.
    """
    prereg = _prereg(tmp_path, CONTROLS + _spent(False))

    problems = fired_against_the_ledger(prereg, _ledger(tmp_path, "все контроли сработали"))

    assert len(problems) == 1
    assert "о молчании не говорит" in problems[0]


def test_a_ledger_that_admits_the_silence_is_accepted(tmp_path) -> None:
    """Положительный контроль: учёт признаёт молчание."""
    prereg = _prereg(tmp_path, CONTROLS + _spent(False))

    assert not fired_against_the_ledger(prereg, _ledger(tmp_path, "К-1 промолчал"))


# --- Регрессии по независимому ревью 9 сентября ------------------------------
#
# Шесть воспроизведений, показавших, что новый протокол не делал заявленного.
# Перенесены в проект: воспроизведение, живущее вне репозитория, защищает ровно
# до тех пор, пока о нём помнят.


def test_an_unrecognized_prereg_name_does_not_disable_the_guard(tmp_path) -> None:
    """Ошибка имени молча отключала ВЕСЬ механизм.

    `prereg-case-21-draft.md` не совпадал с образцом номера, номер выходил None,
    и это обрабатывалось как подтверждённый исторический кейс: preflight
    возвращал успех, не прочитав ни манифеста, ни файлов контроля.
    """
    project = _project(tmp_path, manifest=False, sealed=False, empty=False)
    draft = tmp_path / "prereg-case-21-draft.md"
    draft.write_text("черновик", encoding="utf-8")

    with pytest.raises(OutOfOrder) as отказ:
        preflight(project, draft)

    assert "не опознан как пре-регистрация" in str(отказ.value)


def test_a_missing_prereg_file_is_refused(tmp_path) -> None:
    """Пре-регистрации нет по пути — это дыра, а не совместимость."""
    project = _project(tmp_path)

    with pytest.raises(OutOfOrder) as отказ:
        preflight(project, tmp_path / "нет-такого.md")

    assert "пре-регистрации нет по пути" in str(отказ.value)


def test_a_prereg_of_another_project_is_refused(tmp_path) -> None:
    """Кейс не сверяется с чужой пре-регистрацией."""
    project = _project(tmp_path)
    bit = hashlib.sha256((project / EMPTY_BIT).read_bytes()).hexdigest()[:12]
    answer = hashlib.sha256((project / SEALED).read_bytes()).hexdigest()[:12]
    prereg = _prereg(
        tmp_path,
        "```yaml\nproject: чужой\n```\n\n"
        f"Отпечаток бита пустоты: **`{bit}`**\nОтпечаток ответа: **`{answer}`**\n",
    )

    with pytest.raises(OutOfOrder) as отказ:
        preflight(project, prereg)

    assert "объявляет проект" in str(отказ.value)


def test_declared_controls_must_stand_before_the_first_run(tmp_path) -> None:
    """Контроли объявлены, доказательства нет, а в формуляре их не стоит.

    Preflight пропускал такой запуск: первый прогон шёл БЕЗ контролей, и
    контрольного прогона не случалось вовсе — ни один сигнал не был бы отнесён к
    подложенному дефекту.
    """
    project = _project(tmp_path)
    (project / "project.yaml").write_text(
        "columns:\n  - {name: at_night, role: ignored}\n"
        "  - {name: received_at, role: decision_time}\n",
        encoding="utf-8",
    )
    prereg = _prereg(tmp_path, _digests(project) + CONTROLS)

    with pytest.raises(OutOfOrder) as отказ:
        preflight(project, prereg)

    assert "обязан идти СО ВСЕМИ" in str(отказ.value)


def test_an_empty_or_foreign_control_run_does_not_authorize_removal(tmp_path) -> None:
    """Пустое или чужое доказательство снятия не разрешает.

    Проверялось лишь существование файла. Пустой появляется копированием или из
    шаблона — злого умысла для этого не нужно.
    """
    project = _project(tmp_path)
    (project / "project.yaml").write_text(
        "columns:\n  - {name: at_night, role: ignored}\n"
        "  - {name: received_at, role: decision_time}\n",
        encoding="utf-8",
    )
    prereg = _prereg(tmp_path, _digests(project) + CONTROLS + _spent(True))
    (project / "report" / CONTROL_RUN).write_text("", encoding="utf-8")

    with pytest.raises(OutOfOrder) as пусто:
        preflight(project, prereg)
    assert "верхний уровень доказательства — не отображение" in str(пусто.value)

    (project / "report" / CONTROL_RUN).write_text(
        "case: 999\nform: aaaaaaaaaaaa\nmanifest: bbbbbbbbbbbb\nfindings: [sentinel_as_value]\n",
        encoding="utf-8",
    )
    with pytest.raises(OutOfOrder) as чужое:
        preflight(project, prereg)
    assert "относится к кейсу 999" in str(чужое.value)


def test_fired_is_checked_against_findings_the_core_produced(tmp_path) -> None:
    """Сверка работает с НАСТОЯЩИМ источником находок, а не с рукописным текстом.

    Прежняя версия искала имя находки в `report.md`, а тот печатает
    `signal.detail` — человеческую фразу. В отчёте двадцатого кейса строк
    `sentinel_as_value` ноль вхождений, и сверка не работала вовсе; unit-тесты
    этого не показывали, потому что фикстуры писались в том же формате, в каком
    сверка искала.

    Здесь доказательство порождается `save_control_run` из находок, которые
    вернуло ЯДРО на синтетическом мире стенда.
    """
    from dsx.evals.case import Finding
    from dsx.evals.registry import BY_ID
    from harness import report_for

    checks = report_for(BY_ID["sentinel-as-value"])
    assert Finding.SENTINEL_AS_VALUE in checks.findings, "мир стенда перестал давать находку"

    project = _project(tmp_path)
    prereg = _prereg(tmp_path, _digests(project) + CONTROLS + _spent(True))
    save_control_run(project, prereg, checks.findings)

    assert not fired_against_the_control_run(project, prereg)

    # И обратное: то же доказательство при объявленном молчании — расхождение.
    prereg_false = _prereg(tmp_path, _digests(project) + CONTROLS + _spent(False), number=FROM_CASE)
    problems = fired_against_the_control_run(project, prereg_false)
    assert problems and "объявлен промолчавшим" in problems[0]


def test_the_audit_walks_real_project_artifacts() -> None:
    """Связи проверяются обходом НАСТОЯЩИХ материалов, а не только в unit-тестах.

    Пока сверки вызывались лишь из синтетических тестов, противоречие в живой
    пре-регистрации не сделало бы общую команду красной: помощник был написан, а
    защита — нет.
    """
    from protocol.preflight import audit_case_artifacts

    problems = audit_case_artifacts(ROOT / "docs", ROOT / "projects")

    assert not problems, "; ".join(problems)


def test_the_audit_catches_a_contradiction_in_temporary_artifacts(tmp_path) -> None:
    """Тот же обход на подставленных материалах обязан краснеть.

    Проверка обхода, который всегда молчит, неотличима от отсутствующей: сейчас
    кейсов новее FROM_CASE нет, и на настоящих материалах он законно пуст.
    """
    from protocol.preflight import audit_case_artifacts

    docs = tmp_path / "docs"
    projects = tmp_path / "projects"
    docs.mkdir()
    project = projects / "проба"
    (project / "report").mkdir(parents=True)
    (project / "project.yaml").write_text(FORM, encoding="utf-8")
    (docs / f"prereg-case-{FROM_CASE}.md").write_text(
        "```yaml\nproject: проба\ncontrols:\n"
        "  - {name: К-1, column: at_night, role: feature, clean_role: ignored,\n"
        "     expect: sentinel_as_value, basis: plant}\n"
        "spent_controls:\n  - {name: К-1, fired: true, outcome: 'снят'}\n```\n",
        encoding="utf-8",
    )

    problems = audit_case_artifacts(docs, projects)

    assert problems, "обход не заметил снятия без доказательства прогона"
    assert "доказательства прогона нет" in problems[0]


# --- Регрессии по ПОВТОРНОМУ ревью 9 сентября --------------------------------
#
# Первая редакция исправлений закрыла две находки из пяти и породила новую.
# Ревьюер воспроизвёл 17 сценариев; ниже те, что защищают исправленное.


@pytest.mark.parametrize("findings", ["null", "{sentinel_as_value: false}", "'строка'"])
def test_evidence_off_schema_is_refused(tmp_path, findings) -> None:
    """Негодный документ становился ПОЛОЖИТЕЛЬНЫМ доказательством.

    Reader проверял только тип dict и наличие ключа. `findings: null` давал
    пустое множество, отображение — множество ключей. При `fired: false` сверка
    молчала, потому что находок «не было».
    """
    project = _project(tmp_path)
    prereg = _prereg(tmp_path, _digests(project) + CONTROLS + _spent(False))
    (project / "project.yaml").write_text(
        "columns:\n  - {name: at_night, role: ignored}\n"
        "  - {name: received_at, role: decision_time}\n",
        encoding="utf-8",
    )
    (project / "report" / CONTROL_RUN).write_text(
        f"case: 21\nform: aaaaaaaaaaaa\nmanifest: bbbbbbbbbbbb\nfindings: {findings}\n",
        encoding="utf-8",
    )

    with pytest.raises(OutOfOrder):
        preflight(project, prereg)


def test_evidence_without_fingerprints_is_refused(tmp_path) -> None:
    """Отпечатки записывались и НИКОГДА не читались — обещание без обеспечения."""
    project = _project(tmp_path)
    prereg = _prereg(tmp_path, _digests(project) + CONTROLS + _spent(True))
    (project / "report" / CONTROL_RUN).write_text(
        "case: 21\nfindings: [sentinel_as_value]\n", encoding="utf-8"
    )

    with pytest.raises(OutOfOrder):
        preflight(project, prereg)


def test_only_part_of_the_controls_standing_is_refused(tmp_path) -> None:
    """`any` выдавал один поставленный контроль за все объявленные."""
    project = _project(tmp_path)
    (project / "project.yaml").write_text(
        "columns:\n  - {name: at_night, role: feature}\n"
        "  - {name: chain, role: ignored}\n"
        "  - {name: received_at, role: decision_time}\n",
        encoding="utf-8",
    )
    prereg = _prereg(
        tmp_path,
        _digests(project) + "\n```yaml\nproject: проба\ncontrols:\n"
        "  - {name: К-1, column: at_night, role: feature, clean_role: ignored,\n"
        "     expect: sentinel_as_value, basis: plant}\n"
        "  - {name: К-2, column: chain, role: group_id, clean_role: ignored,\n"
        "     expect: undeclared_group, basis: plant}\n"
        "```\n",
    )

    with pytest.raises(OutOfOrder) as отказ:
        preflight(project, prereg)

    assert "К-2" in str(отказ.value)
    assert "СО ВСЕМИ" in str(отказ.value)


def test_a_clean_run_without_declared_removal_is_refused(tmp_path) -> None:
    """Контроли сняты, доказательство есть, а §5 об этом не знает.

    Исход контролей нигде не записан, и вердикт опёрся бы на память автора.
    """
    project = _project(tmp_path)
    prereg_with = _prereg(tmp_path, _digests(project) + CONTROLS)
    save_control_run(project, prereg_with, ["sentinel_as_value"])
    (project / "project.yaml").write_text(
        "columns:\n  - {name: at_night, role: ignored}\n"
        "  - {name: received_at, role: decision_time}\n",
        encoding="utf-8",
    )

    with pytest.raises(OutOfOrder) as отказ:
        preflight(project, prereg_with)

    assert "не объявлены" in str(отказ.value)


def test_removal_declared_while_controls_still_stand_is_refused(tmp_path) -> None:
    """Снятие объявлено, а колонка осталась в контрольной роли."""
    project = _project(tmp_path)
    prereg = _prereg(tmp_path, _digests(project) + CONTROLS + _spent(True))
    save_control_run(project, prereg, ["sentinel_as_value"])

    with pytest.raises(OutOfOrder) as отказ:
        preflight(project, prereg)

    assert "всё ещё стоят" in str(отказ.value)


def test_a_symlink_may_not_pass_one_case_for_another(tmp_path) -> None:
    """`prereg-case-20.md` — ссылка на кейс 21: имя говорило одно, цель другое."""
    project = _project(tmp_path)
    настоящая = _prereg(tmp_path, _digests(project) + CONTROLS)
    ссылка = tmp_path / "prereg-case-20.md"
    ссылка.symlink_to(настоящая)

    with pytest.raises(OutOfOrder) as отказ:
        preflight(project, ссылка)

    assert "расходится с целью" in str(отказ.value)


def test_a_protocol_block_without_project_is_refused(tmp_path) -> None:
    """Контроли в блоке без `project` парсер не видел вовсе.

    Preflight считал, что контролей нет, и пропускал прогон без них.
    """
    project = _project(tmp_path)
    prereg = _prereg(
        tmp_path,
        _digests(project)
        + "\n```yaml\nproject: проба\n```\n"
        + "\n```yaml\ncontrols:\n  - {name: К-1, column: at_night, role: feature,\n"
        "     clean_role: ignored, expect: sentinel_as_value, basis: plant}\n```\n",
    )

    with pytest.raises(OutOfOrder) as отказ:
        preflight(project, prereg)

    assert "без `project`" in str(отказ.value)


def test_conflicting_project_declarations_are_refused(tmp_path) -> None:
    """`project: чужой`, следом `project: проба` — конфликт перезаписывался молча."""
    project = _project(tmp_path)
    prereg = _prereg(
        tmp_path,
        _digests(project) + "\n```yaml\nproject: чужой\n```\n" + CONTROLS,
    )

    with pytest.raises(OutOfOrder) as отказ:
        preflight(project, prereg)

    assert "повторено с другим значением" in str(отказ.value)


def test_the_check_command_keeps_the_cause_of_a_long_collection_error(
    monkeypatch, capsys, tmp_path
) -> None:
    """Регрессия, которую прошлый раз ЗАБЫЛИ перенести, вопреки отчёту.

    Причина ошибки сборки стоит в начале вывода, а хвост занимают
    предупреждения. Прежняя версия печатала последние 4000 знаков и теряла имя
    отсутствующего модуля — то самое, ради чего диагностику и чинили.

    Корень подменён на временный. Прежде тест писал лог в КОРЕНЬ РЕПОЗИТОРИЯ и
    затем удалял его: настоящий `.check-collection-error.log`, оставшийся от
    неудачного прогона, уничтожался запуском набора. Замечено третьим ревью как
    побочный эффект. Проверка не смеет трогать рабочее дерево.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location("check", ROOT / "tools" / "check.py")
    check = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(check)

    вывод = (
        "ImportError while loading conftest\n"
        "ModuleNotFoundError: No module named 'missing_fixture_module'\n"
        + "\n".join(f"warning {i}: длинное предупреждение" for i in range(400))
    )
    answers = iter([(2, вывод), (0, "All checks passed!")])
    monkeypatch.setattr(check, "_run", lambda argv: next(answers))
    monkeypatch.setattr(check, "ROOT", tmp_path)

    assert check.main() == 1
    напечатано = capsys.readouterr().out

    assert "missing_fixture_module" in напечатано, "первопричина потеряна в хвосте"
    assert (tmp_path / ".check-collection-error.log").is_file(), "полный вывод не сохранён"


def test_a_prereg_without_a_declared_project_is_refused(tmp_path) -> None:
    """Историческое исключение не выдаётся по одному имени файла.

    Прежде ветка `< FROM_CASE` возвращала управление ДО чтения `project`, и
    документ чужого кейса под старым номером отключал защиту целиком.
    """
    project = _project(tmp_path)
    prereg = _prereg(tmp_path, "## 5а\n\nбез машиночитаемого блока\n")

    with pytest.raises(OutOfOrder) as отказ:
        preflight(project, prereg)

    assert "не объявляет `project`" in str(отказ.value)


# --- Разбор трёх спорных сценариев повторного ревью --------------------------
#
# Ревьюер оставил три сценария красными, и объяснения «контракт изменился» мало:
# устойчивый хеш сам по себе не доказывает, что законное снятие разрешено, а
# всякая иная правка формы поймана. Здесь проверяется свойство, а не хеш;
# исходное намерение сценариев сохранено, изменён только способ его проверки.

RICH_FORM = """
observed_until: 2026-01-01
split:
  kind: time
  column: received_at
columns:
  - {name: at_night, role: feature}
  - {name: received_at, role: decision_time}
  - {name: outcome, role: outcome}
"""

REMOVED = RICH_FORM.replace("{name: at_night, role: feature}", "{name: at_night, role: ignored}")


def _with_form(tmp_path: Path, form: str) -> tuple[Path, Path]:
    """Кейс с богатой формой: в ней есть срок, сплит и неконтрольные колонки."""
    project = _project(tmp_path)
    (project / "project.yaml").write_text(form, encoding="utf-8")
    prereg = _prereg(tmp_path, _digests(project) + CONTROLS)
    return project, prereg


def test_a_lawful_removal_does_not_change_the_form_fingerprint(tmp_path) -> None:
    """Снятие контроля разрешено: отпечаток формы от него не меняется.

    Это половина требования. Вторая половина — соседний тест: всё прочее он
    обязан ловить. Устойчивость хеша без неё означала бы просто слепой хеш.
    """
    project, prereg = _with_form(tmp_path, RICH_FORM)
    columns = _control_roles(prereg)
    до = _form_identity(project / "project.yaml", columns)

    (project / "project.yaml").write_text(REMOVED, encoding="utf-8")

    assert _form_identity(project / "project.yaml", columns) == до


def test_a_lawful_removal_passes_the_whole_path(tmp_path) -> None:
    """И то же самое целиком: доказательство сохранено, контроль снят, отказа нет."""
    project, prereg = _with_form(tmp_path, RICH_FORM)
    save_control_run(project, prereg, ["sentinel_as_value"])

    (project / "project.yaml").write_text(REMOVED, encoding="utf-8")
    preflight(project, _prereg(tmp_path, _digests(project) + CONTROLS + _spent(True)))


@pytest.mark.parametrize(
    "что, форма",
    [
        ("добавлена колонка", REMOVED + "  - {name: extra, role: feature}\n"),
        ("изменён срок наблюдения", REMOVED.replace("2026-01-01", "2026-02-01")),
        ("изменён сплит", REMOVED.replace("kind: time", "kind: random")),
        (
            "изменена роль неконтрольной колонки",
            REMOVED.replace("name: outcome, role: outcome", "name: outcome, role: ignored"),
        ),
        (
            "удалена неконтрольная колонка",
            REMOVED.replace("  - {name: outcome, role: outcome}\n", ""),
        ),
    ],
)
def test_any_other_change_to_the_form_is_caught(tmp_path, что: str, форма: str) -> None:
    """Изменение значимого поля формы обнаруживается, хотя контроль снят законно.

    Каждый случай — снятие контроля ПЛЮС одна правка. Если бы отпечаток был
    устойчив ко всему подряд, проверка проходила бы: она не проходит.
    """
    project, prereg = _with_form(tmp_path, RICH_FORM)
    save_control_run(project, prereg, ["sentinel_as_value"])

    (project / "project.yaml").write_text(форма, encoding="utf-8")
    prereg = _prereg(tmp_path, _digests(project) + CONTROLS + _spent(True))

    assert (
        _form_identity(project / "project.yaml", _control_roles(prereg))
        != read_control_run(project)["form"]
    ), f"отпечаток не заметил, что {что}"

    with pytest.raises(OutOfOrder) as отказ:
        preflight(project, prereg)

    assert "форма изменилась" in str(отказ.value), что


def test_unreadable_evidence_stops_before_the_builder_and_names_the_cause(tmp_path) -> None:
    """Испорченные байты доказательства: штатный отказ, а не `UnicodeDecodeError`.

    Сценарий ревьюера ждал `UnicodeDecodeError`, и намерение его было не в
    классе исключения, а в двух вещах: прогон обязан прекратиться и обязан
    прекратиться ДО построителя. Обе проверяются здесь прямо — счётчиком вызовов
    и текстом причины. Отказ, называющий «нет либо не по схеме», намерения не
    выполнял бы: он неотличим от отсутствующего файла.
    """
    project, prereg = _with_form(tmp_path, RICH_FORM)
    save_control_run(project, prereg, ["sentinel_as_value"])
    (project / "project.yaml").write_text(REMOVED, encoding="utf-8")
    prereg = _prereg(tmp_path, _digests(project) + CONTROLS + _spent(True))
    (project / "report" / CONTROL_RUN).write_bytes(b"\xff")

    вызовов = 0

    def build() -> str:
        nonlocal вызовов
        вызовов += 1
        return "данные собраны"

    # Через `guarded`, а не «preflight, а следующей строкой build». Вторая
    # запись не проверяла ничего: исключение прерывает блок само, и счётчик
    # остался бы нулевым при любой реализации preflight, включая пустую.
    with pytest.raises(OutOfOrder) as отказ:
        guarded(project, prereg, build)

    assert вызовов == 0, "построитель вызван при нечитаемом доказательстве"
    assert "не текст в UTF-8" in str(отказ.value)
    assert read_control_run(project) is None


@pytest.mark.parametrize(
    "содержимое, причина",
    [
        (b"[broken", "не разбирается как YAML"),
        (b"- a\n- b\n", "не отображение"),
        (b"case: '21'\nfindings: []\nform: a\nmanifest: b\n", "числового `case`"),
        (b"case: 21\nfindings: null\nform: a\nmanifest: b\n", "не список имён находок"),
        (b"case: 21\nfindings: []\n", "отпечатков `form` и `manifest`"),
    ],
)
def test_every_kind_of_unusable_evidence_names_its_own_cause(
    tmp_path, содержимое: bytes, причина: str
) -> None:
    """Причины различаются между собой, а не сливаются в одну общую фразу."""
    project, prereg = _with_form(tmp_path, RICH_FORM)
    save_control_run(project, prereg, ["sentinel_as_value"])
    (project / "project.yaml").write_text(REMOVED, encoding="utf-8")
    prereg = _prereg(tmp_path, _digests(project) + CONTROLS + _spent(True))
    (project / "report" / CONTROL_RUN).write_bytes(содержимое)

    with pytest.raises(OutOfOrder) as отказ:
        preflight(project, prereg)

    assert причина in str(отказ.value)


def test_valid_digests_without_a_project_block_are_refused(tmp_path) -> None:
    """Отпечатки верные, машиночитаемого `project` нет — прогон не начинается.

    Пре-регистрация здесь строится ПОМИМО общей фикстуры `_digests`: та сама
    объявляет `project`, и сценарий с её помощью проверял бы не то, что назван.
    Первая проверка теста — что удалённое поле не вернулось: без неё тест
    зеленел бы по любой другой причине.
    """
    project = _project(tmp_path)
    бит = hashlib.sha256((project / EMPTY_BIT).read_bytes()).hexdigest()[:12]
    ответ = hashlib.sha256((project / SEALED).read_bytes()).hexdigest()[:12]
    prereg = _prereg(
        tmp_path,
        "## 5а. Слепой контроль\n\n"
        f"Отпечаток бита пустоты: **`{бит}`**\n"
        f"Отпечаток ответа: **`{ответ}`**\n",
    )

    assert declared(prereg).get("project") is None, "поле вернулось — сценарий проверяет не то"

    with pytest.raises(OutOfOrder) as отказ:
        preflight(project, prereg)

    assert "не объявляет `project`" in str(отказ.value)
    assert "отпечаток" not in str(отказ.value).lower(), "отказ пришёл по другой причине"


def test_the_evidence_lives_through_the_real_path(tmp_path) -> None:
    """Создание → сохранение → чтение → сверка, без рукописных фикстур посередине.

    Находки берутся у ЯДРА на мире стенда, записываются `save_control_run`,
    читаются `read_control_run` и сверяются `fired_against_the_control_run`.
    Оба круга ревью показали одно: самопроверка на собственных фикстурах
    пропускает ошибки соединения звеньев, а не самих звеньев.

    Отпечаток формы здесь НЕ хеш файла — сценарий ревьюера ждал именно его.
    Хеш файла запрещал бы законное снятие контроля; свойство, ради которого
    отпечаток вообще нужен, проверяют два теста выше.
    """
    from dsx.evals.case import Finding
    from dsx.evals.registry import BY_ID
    from harness import report_for

    project, prereg = _with_form(tmp_path, RICH_FORM)
    preflight(project, prereg)  # первый прогон: контроли стоят

    checks = report_for(BY_ID["sentinel-as-value"])
    assert Finding.SENTINEL_AS_VALUE in checks.findings, "мир стенда перестал давать находку"
    saved = save_control_run(project, prereg, checks.findings)

    evidence = read_control_run(project)
    assert evidence is not None
    assert evidence["case"] == FROM_CASE
    assert Finding.SENTINEL_AS_VALUE.value in evidence["findings"]
    assert (
        evidence["manifest"]
        == hashlib.sha256((project / "manifest.yaml").read_bytes()).hexdigest()[:12]
    )
    assert evidence["form"] == _form_identity(project / "project.yaml", _control_roles(prereg))

    было = saved.read_bytes()
    assert save_control_run(project, prereg, []) is None, "первый прогон перезаписан вторым"
    assert saved.read_bytes() == было

    (project / "project.yaml").write_text(REMOVED, encoding="utf-8")
    снят = _prereg(tmp_path, _digests(project) + CONTROLS + _spent(True))
    preflight(project, снят)
    assert not fired_against_the_control_run(project, снят)

    ложь = _prereg(tmp_path, _digests(project) + CONTROLS + _spent(False))
    assert fired_against_the_control_run(project, ложь), "ложное «промолчал» не замечено"


# --- Регрессии по ТРЕТЬЕМУ ревью 9 сентября ----------------------------------


def test_the_builder_runs_when_the_order_is_kept(tmp_path) -> None:
    """Положительный контроль к `guarded`: на подготовленном кейсе построитель ЗОВЁТСЯ.

    Без него проверка «построитель не вызван» проходила бы и для обёртки,
    которая не зовёт его никогда. Отрицательная половина без положительной
    неотличима от сломанного механизма.
    """
    project, prereg = _with_form(tmp_path, RICH_FORM)
    вызовов = 0

    def build() -> str:
        nonlocal вызовов
        вызовов += 1
        return "данные собраны"

    assert guarded(project, prereg, build) == "данные собраны"
    assert вызовов == 1


def test_the_builder_is_not_reached_when_the_order_is_broken(tmp_path) -> None:
    """И обратное на том же пути: слепого контроля нет — построитель не зовётся."""
    project = _project(tmp_path, sealed=False, empty=False)
    prereg = _prereg(tmp_path, "```yaml\nproject: проба\n```\n\n## 5а. Слепой контроль\n")
    вызовов = 0

    def build() -> None:
        nonlocal вызовов
        вызовов += 1

    with pytest.raises(OutOfOrder):
        guarded(project, prereg, build)

    assert вызовов == 0


@pytest.mark.parametrize("роль", ["outcome", "decision_time", "weight"])
def test_a_control_column_moved_to_another_role_is_not_a_removal(tmp_path, роль: str) -> None:
    """Подмена роли контрольной колонки — не снятие, и отпечаток её обязан видеть.

    Блокирующая находка третьего ревью: `_form_identity` опускала роль
    контрольной колонки БЕЗУСЛОВНО, а `preflight` считал снятием всякое
    несовпадение роли. Перевод `feature → outcome` проходил чистым прогоном, не
    меняя отпечатка, — при том что смысл колонки менялся полностью.
    """
    project, prereg = _with_form(tmp_path, RICH_FORM)
    save_control_run(project, prereg, ["sentinel_as_value"])

    (project / "project.yaml").write_text(
        RICH_FORM.replace("{name: at_night, role: feature}", f"{{name: at_night, role: {роль}}}"),
        encoding="utf-8",
    )
    prereg = _prereg(tmp_path, _digests(project) + CONTROLS + _spent(True))

    assert control_states(prereg, project / "project.yaml") == {"К-1": "изменён"}
    assert (
        _form_identity(project / "project.yaml", _control_roles(prereg))
        != read_control_run(project)["form"]
    ), "отпечаток не заметил подмены роли"

    with pytest.raises(OutOfOrder) as отказ:
        preflight(project, prereg)

    assert "не снятие контроля" in str(отказ.value)


def test_a_control_removed_to_ignored_is_a_removal(tmp_path) -> None:
    """Отрицательный контроль к предыдущему: `ignored` по-прежнему снятие."""
    project, prereg = _with_form(tmp_path, RICH_FORM)
    save_control_run(project, prereg, ["sentinel_as_value"])
    (project / "project.yaml").write_text(REMOVED, encoding="utf-8")
    prereg = _prereg(tmp_path, _digests(project) + CONTROLS + _spent(True))

    assert control_states(prereg, project / "project.yaml") == {"К-1": "снят"}
    preflight(project, prereg)


def test_the_audit_checks_the_fingerprints_of_the_evidence(tmp_path) -> None:
    """Обход обязан ловить доказательство с выдуманными отпечатками.

    Прежде `audit_case_artifacts` их не читал вовсе: документ с `form: x` и
    `manifest: y` проходил обход молча. `preflight` такое ловит, но зелёный
    обход сам по себе о привязке доказательства к форме и данным не говорил.
    """
    from protocol.preflight import audit_case_artifacts

    docs = tmp_path / "docs"
    projects = tmp_path / "projects"
    docs.mkdir()
    project = projects / "проба"
    (project / "report").mkdir(parents=True)
    (project / "project.yaml").write_text(REMOVED, encoding="utf-8")
    (project / "manifest.yaml").write_text("source: проба\n", encoding="utf-8")
    (project / "report" / CONTROL_RUN).write_text(
        "case: 21\nfindings: [sentinel_as_value]\nform: x\nmanifest: y\n", encoding="utf-8"
    )
    (docs / f"prereg-case-{FROM_CASE}.md").write_text(
        "```yaml\nproject: проба\ncontrols:\n"
        "  - {name: К-1, column: at_night, role: feature, clean_role: ignored,\n"
        "     expect: sentinel_as_value, basis: plant}\n"
        "spent_controls:\n  - {name: К-1, fired: true, outcome: 'снят'}\n```\n",
        encoding="utf-8",
    )

    problems = audit_case_artifacts(docs, projects)

    assert any("отпечаток манифеста не совпадает" in item for item in problems), problems
    assert any("получено на другой форме" in item for item in problems), problems


# --- Регрессии по ЧЕТВЁРТОМУ ревью 9 сентября --------------------------------

IGNORED_CONTROL = """
```yaml
project: проба
controls:
  - {name: К-1, column: at_night, role: ignored, clean_role: feature,
     expect: sentinel_as_value, basis: plant}
```
"""

HIDDEN = RICH_FORM.replace("{name: at_night, role: feature}", "{name: at_night, role: ignored}")


def test_a_control_planted_as_ignored_has_a_lawful_way_back(tmp_path) -> None:
    """`ignored` — законная РОЛЬ КОНТРОЛЯ, а не только маркер его снятия.

    Спрятать колонку, которая должна быть признаком, — подложенный дефект: так
    объявлен К-1 в `test_prereg_promises`. Прежняя версия зашивала снятие в
    константу `ignored`, и такому контролю пути к состоянию «снят» не
    оставалось вовсе: возврат колонки в `feature` отвергался как подмена роли.
    Найдено четвёртым ревью.
    """
    project = _project(tmp_path)
    (project / "project.yaml").write_text(HIDDEN, encoding="utf-8")
    prereg = _prereg(tmp_path, _digests(project) + IGNORED_CONTROL)

    assert control_states(prereg, project / "project.yaml") == {"К-1": "стоит"}
    save_control_run(project, prereg, ["sentinel_as_value"])

    # Снятие: колонка возвращается к честной роли — и это НЕ `ignored`.
    (project / "project.yaml").write_text(RICH_FORM, encoding="utf-8")
    prereg = _prereg(tmp_path, _digests(project) + IGNORED_CONTROL + _spent(True))

    assert control_states(prereg, project / "project.yaml") == {"К-1": "снят"}
    preflight(project, prereg)


def test_a_control_without_a_clean_role_is_refused(tmp_path) -> None:
    """Умолчание запрещено: без исходной роли снятие неотличимо от подмены.

    Угадать честную роль колонки ядру нечем. `ignored` в качестве умолчания
    совпадал бы с честным ответом в части случаев и молча врал бы в остальных —
    ровно то, что правило «умолчание, совпадающее с честным ответом, запрещено»
    и запрещает.
    """
    project = _project(tmp_path)
    prereg = _prereg(
        tmp_path,
        _digests(project) + "\n```yaml\nproject: проба\ncontrols:\n"
        "  - {name: К-1, column: at_night, role: feature,\n"
        "     expect: sentinel_as_value, basis: plant}\n```\n",
    )

    with pytest.raises(OutOfOrder) as отказ:
        preflight(project, prereg)

    assert "без исходной роли" in str(отказ.value)
    assert "К-1: нет `clean_role`" in str(отказ.value)


def test_a_clean_role_equal_to_the_control_role_is_refused(tmp_path) -> None:
    """Совпадение двух ролей делает стоящий контроль неотличимым от снятого."""
    project = _project(tmp_path)
    prereg = _prereg(
        tmp_path,
        _digests(project) + "\n```yaml\nproject: проба\ncontrols:\n"
        "  - {name: К-1, column: at_night, role: feature, clean_role: feature,\n"
        "     expect: sentinel_as_value, basis: plant}\n```\n",
    )

    with pytest.raises(OutOfOrder) as отказ:
        preflight(project, prereg)

    assert "совпадает с ролью контроля" in str(отказ.value)


@pytest.mark.parametrize(
    ("clean_role", "yaml_value"),
    [(None, "null"), ("", "''"), ("not_a_role", "not_a_role")],
)
def test_an_invalid_clean_role_stops_before_the_builder(
    tmp_path, clean_role, yaml_value: str
) -> None:
    """`clean_role` обязан быть существующей Role, а не просто ключом в YAML.

    `null`, пустая строка и неизвестное имя прежде проходили до построителя.
    Счётчик проверяет именно границу `guarded`: следующий вызов после отдельного
    `preflight` не доказывал бы, что построитель недостижим при отказе.
    """
    project = _project(tmp_path)
    prereg = _prereg(
        tmp_path,
        _digests(project) + "\n```yaml\nproject: проба\ncontrols:\n"
        "  - {name: К-1, column: at_night, role: feature, "
        f"clean_role: {yaml_value}, expect: sentinel_as_value, basis: plant}}\n```\n",
    )
    вызовов = 0

    def build() -> None:
        nonlocal вызовов
        вызовов += 1

    with pytest.raises(OutOfOrder) as отказ:
        guarded(project, prereg, build)

    assert вызовов == 0, f"построитель вызван при clean_role={clean_role!r}"
    assert "контроли объявлены без исходной роли" in str(отказ.value)


def test_invalid_clean_role_stops_a_removed_control_after_saved_evidence(tmp_path) -> None:
    """Удаление `role` после сохранения не проходит через `clean_role: null`.

    Доказательство создано и прочитано по корректному объявлению. Затем
    отдельная испорченная версия §5 одновременно удаляет роль колонки и меняет
    `clean_role` на `null`: отказ обязан прийти до состояний и построителя.
    """
    project, prereg = _with_form(tmp_path, RICH_FORM)
    preflight(project, prereg)
    save_control_run(project, prereg, ["sentinel_as_value"])
    assert read_control_run(project) is not None

    (project / "project.yaml").write_text(
        RICH_FORM.replace("{name: at_night, role: feature}", "{name: at_night}"),
        encoding="utf-8",
    )
    malformed = _prereg(
        tmp_path,
        _digests(project)
        + CONTROLS.replace("clean_role: ignored", "clean_role: null")
        + _spent(True),
    )
    вызовов = 0

    def build() -> None:
        nonlocal вызовов
        вызовов += 1

    with pytest.raises(OutOfOrder) as отказ:
        guarded(project, malformed, build)

    assert вызовов == 0, "построитель вызван при удалённой роли и clean_role: null"
    assert "не строка роли" in str(отказ.value)


def test_a_role_swap_is_named_even_when_the_finding_is_also_unreachable(tmp_path) -> None:
    """Подмена роли отвергается ДО проверки достижимости, как обещает спецификация.

    Прежде `unreachable_controls` стоял первым, и подмена, совпавшая с
    недостижимой находкой, получала отказ «недостижима». Отказ приходил —
    но не тот, и вёл разбирающегося не туда.
    """
    project = _project(tmp_path)
    (project / "project.yaml").write_text(
        "columns:\n  - {name: received_at, role: decision_time}\n"
        "  - {name: pump_installed, role: outcome, value_as_of: received_at}\n",
        encoding="utf-8",
    )
    prereg = _prereg(tmp_path, _digests(project) + CONTRADICTORY)

    состояния = control_states(prereg, project / "project.yaml")
    assert состояния == {"К-2": "изменён"}
    assert unreachable_controls(prereg, project / "project.yaml"), "случай перестал быть двойным"

    with pytest.raises(OutOfOrder) as отказ:
        preflight(project, prereg)

    assert "не снятие контроля" in str(отказ.value)
    assert "недостижим" not in str(отказ.value)


def test_a_boolean_case_is_not_an_integer(tmp_path) -> None:
    """`case: true` — не номер кейса, хотя `bool` в Python подкласс `int`.

    Документ признавался годным, и отказ приходил позже и о другом: «относится
    к кейсу True». Ветвь схемы обещала «case не целое» и обещания не исполняла.
    """
    project, prereg = _with_form(tmp_path, RICH_FORM)
    save_control_run(project, prereg, ["sentinel_as_value"])
    (project / "project.yaml").write_text(REMOVED, encoding="utf-8")
    prereg = _prereg(tmp_path, _digests(project) + CONTROLS + _spent(True))
    (project / "report" / CONTROL_RUN).write_text(
        "case: true\nfindings: []\nform: a\nmanifest: b\n", encoding="utf-8"
    )

    assert read_control_run(project) is None

    with pytest.raises(OutOfOrder) as отказ:
        preflight(project, prereg)

    assert "числового `case`" in str(отказ.value)


def test_the_diagnostic_test_does_not_disturb_an_existing_log(monkeypatch, capsys, tmp_path):
    """Законно лежащий лог неудачного прогона переживает запуск набора.

    `.gitignore` хранит `.check-collection-error.log` между прогонами намеренно:
    это диагностика последнего отказа. Прежняя проверка требовала его
    ОТСУТСТВИЯ и падала, когда он законно есть, — то есть ровно тогда, когда
    он нужнее всего. Найдено четвёртым ревью.
    """
    import importlib.util

    лог = ROOT / ".check-collection-error.log"
    было = лог.read_bytes() if лог.is_file() else None
    подложено = "вывод прошлого неудачного прогона".encode()
    лог.write_bytes(подложено)
    try:
        spec = importlib.util.spec_from_file_location("check", ROOT / "tools" / "check.py")
        check = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(check)
        monkeypatch.setattr(check, "_run", lambda argv: (2, "ImportError: нет модуля"))
        monkeypatch.setattr(check, "ROOT", tmp_path)

        assert check.main() == 1
        capsys.readouterr()

        assert лог.read_bytes() == подложено, "лог тронут проверкой"
    finally:
        if было is None:
            лог.unlink(missing_ok=True)
        else:
            лог.write_bytes(было)


# --- Поиск аналогов: два этапа, с двадцать первого кейса ----------------------
#
# Первый этап (другие наборы) опечатывается вместе с пре-регистрацией, второй
# (тот же набор) пишется после опечатывания. Проверяется форма записи, а не
# честность поиска — этот предел назван в `analogs_defects`.


def _ready(tmp_path: Path, *, analogs_in_prereg=True, analogs_file=True) -> tuple[Path, Path]:
    project = _project(tmp_path, analogs=analogs_file)
    prereg = _prereg(tmp_path, _digests(project) + CONTROLS, analogs=analogs_in_prereg)
    return project, prereg


def test_the_builder_is_not_called_without_the_first_stage_of_analogs(tmp_path) -> None:
    project, prereg = _ready(tmp_path, analogs_in_prereg=False)
    calls = 0

    def build():
        nonlocal calls
        calls += 1

    with pytest.raises(OutOfOrder) as отказ:
        guarded(project, prereg, build)

    assert calls == 0, "построитель вызван без первого этапа поиска аналогов"
    assert "аналогах на других наборах" in str(отказ.value)


def test_nothing_found_without_queries_is_not_a_search(tmp_path) -> None:
    """«Ничего не найдено» с пустыми запросами неотличимо от «не искал»."""
    project, prereg = _ready(tmp_path)
    prereg.write_text(
        prereg.read_text(encoding="utf-8").replace('openml: ["delivery delay"]', "openml: []"),
        encoding="utf-8",
    )

    with pytest.raises(OutOfOrder) as отказ:
        preflight(project, prereg)

    assert "queries.openml" in str(отказ.value)


def test_a_found_analog_needs_a_decision(tmp_path) -> None:
    project, prereg = _ready(tmp_path)
    (project / ANALOGS).write_text(
        ANALOGS_BLOCK.replace(
            "found: []",
            "found:\n    - {url: 'https://kaggle.com/x',\n"
            "       reference_result: {metric: roc_auc, value: 0.91, split: random,"
            " population: 'заявки 2019–2021'}}",
        ),
        encoding="utf-8",
    )

    with pytest.raises(OutOfOrder) as отказ:
        preflight(project, prereg)

    assert "найден, но не взят и не отвергнут" in str(отказ.value)


def test_a_reference_result_names_its_conditions(tmp_path) -> None:
    """Опорный результат без сплита сравнивать не с чем."""
    project, prereg = _ready(tmp_path)
    (project / ANALOGS).write_text(
        ANALOGS_BLOCK.replace(
            "found: []",
            "found:\n    - {url: 'https://kaggle.com/x',\n"
            "       reference_result: {metric: roc_auc, value: 0.91}}",
        ).replace(
            "rejected: []",
            "rejected:\n    - {url: 'https://kaggle.com/x', reason: 'случайный сплит'}",
        ),
        encoding="utf-8",
    )

    with pytest.raises(OutOfOrder) as отказ:
        preflight(project, prereg)

    assert "reference_result.split" in str(отказ.value)


def test_the_second_stage_is_required_after_sealing(tmp_path) -> None:
    project, prereg = _ready(tmp_path, analogs_file=False)

    with pytest.raises(OutOfOrder) as отказ:
        preflight(project, prereg)

    assert "второй этап поиска аналогов не записан" in str(отказ.value)


def test_the_second_stage_belongs_to_this_project(tmp_path) -> None:
    project, prereg = _ready(tmp_path)
    (project / ANALOGS).write_text(
        ANALOGS_BLOCK.replace("project: проба", "project: чужой"), encoding="utf-8"
    )

    with pytest.raises(OutOfOrder) as отказ:
        preflight(project, prereg)

    assert "объявляет проект 'чужой'" in str(отказ.value)


def test_complete_analogs_let_the_builder_run(tmp_path) -> None:
    """Обратный исход: при полной записи обоих этапов построитель зовётся."""
    project, prereg = _ready(tmp_path)
    (project / ANALOGS).write_text(
        ANALOGS_BLOCK.replace(
            "found: []",
            "found:\n    - {url: 'https://kaggle.com/x',\n"
            "       reference_result: {metric: roc_auc, value: 0.91, split: random,"
            " population: 'заявки 2019–2021'}}",
        ).replace(
            "rejected: []",
            "rejected:\n    - {url: 'https://kaggle.com/x', reason: 'случайный сплит'}",
        ),
        encoding="utf-8",
    )

    assert guarded(project, prereg, lambda: "построено") == "построено"


def test_historical_cases_do_not_need_analogs(tmp_path) -> None:
    """Правило вводится вперёд: у кейсов до двадцать первого записи нет и не будет."""
    project = _project(tmp_path, analogs=False)
    prereg = _prereg(
        tmp_path, "```yaml\nproject: проба\n```\n", number=FROM_CASE - 1, analogs=False
    )

    preflight(project, prereg)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("found", "null"),
        ("taken", "null"),
        ("rejected", "null"),
        ("pitfalls", "null"),
        ("found", "'нет'"),
        ("date", "true"),
        ("date", "'вчера'"),
    ],
)
def test_a_wrong_type_is_not_an_empty_answer(tmp_path, field, value) -> None:
    """Независимое ревью PR #1: `null` вместо списка проходил как пустой список.

    Валидатор проверял наличие поля и трактовал негодный тип как «ничего»:
    запись `found: null, taken: null, rejected: null, pitfalls: null, date: true`
    с непустыми запросами пропускала построитель. Умолчание, совпадающее с
    честным ответом, — ровно то, что главное правило проекта запрещает.
    """
    project, prereg = _ready(tmp_path)
    default = {
        "found": "[]",
        "taken": "[]",
        "rejected": "[]",
        "pitfalls": "[]",
        "date": "2026-09-29",
    }
    (project / ANALOGS).write_text(
        ANALOGS_BLOCK.replace(f"{field}: {default[field]}", f"{field}: {value}"), encoding="utf-8"
    )
    calls = 0

    def build():
        nonlocal calls
        calls += 1

    with pytest.raises(OutOfOrder) as отказ:
        guarded(project, prereg, build)

    assert calls == 0, f"построитель вызван при {field}: {value}"
    assert field in str(отказ.value)


def test_the_reviewers_record_is_refused(tmp_path) -> None:
    """Запись ревьюера целиком, как она была предъявлена."""
    project, prereg = _ready(tmp_path)
    record = ANALOGS_BLOCK
    for field in ("found", "taken", "rejected", "pitfalls"):
        record = record.replace(f"{field}: []", f"{field}: null")
    (project / ANALOGS).write_text(
        record.replace("date: 2026-09-29", "date: true"), encoding="utf-8"
    )

    with pytest.raises(OutOfOrder):
        guarded(project, prereg, lambda: "построено")


def test_a_stray_analogs_block_does_not_touch_a_historical_case(tmp_path) -> None:
    """Независимое ревью PR #1: на 4637616 такой документ проходил — и проходит снова."""
    prereg = _prereg(
        tmp_path,
        "```yaml\nproject: проба\n```\n\n```yaml\nanalogs: {}\n```\n",
        number=20,
        analogs=False,
    )

    assert declared(prereg)["project"] == "проба"


def test_an_analogs_block_without_project_is_refused_from_case_21(tmp_path) -> None:
    prereg = _prereg(
        tmp_path, "```yaml\nproject: проба\n```\n\n```yaml\nanalogs: {}\n```\n", analogs=False
    )

    with pytest.raises(OutOfOrder) as отказ:
        declared(prereg)

    assert "без `project`" in str(отказ.value)


def test_a_boolean_is_not_a_link_or_a_reason(tmp_path) -> None:
    """`url: true` и `reason: true` проходили проверкой на истинность."""
    project, prereg = _ready(tmp_path)
    (project / ANALOGS).write_text(
        ANALOGS_BLOCK.replace("found: []", "found:\n    - {url: true}").replace(
            "rejected: []", "rejected:\n    - {url: true, reason: true}"
        ),
        encoding="utf-8",
    )

    with pytest.raises(OutOfOrder) as отказ:
        guarded(project, prereg, lambda: "построено")

    assert "found.0.url" in str(отказ.value)


FOUND_WITH_DECISION = (
    "found:\n    - url: 'https://kaggle.com/x'\n      reference_result: {REF}\n"
    "  taken: []\n  rejected:\n    - {url: 'https://kaggle.com/x', reason: 'случайный сплит'}"
)
GOOD_REFERENCE = "{metric: roc_auc, value: 0.91, split: random, population: 'заявки 2019–2021'}"


def _with_reference(reference: str) -> str:
    return ANALOGS_BLOCK.replace(
        "found: []\n  taken: []\n  rejected: []",
        FOUND_WITH_DECISION.replace("{REF}", reference),
    )


@pytest.mark.parametrize(
    ("label", "record"),
    [
        ("дата без дефисов", ANALOGS_BLOCK.replace("date: 2026-09-29", "date: '20260929'")),
        ("дата-время", ANALOGS_BLOCK.replace("date: 2026-09-29", "date: 2026-09-29 10:00:00")),
        (
            "опорный результат без содержания",
            _with_reference("{metric: true, value: false, split: []}"),
        ),
        (
            "нет population",
            _with_reference("{metric: roc_auc, value: 0.91, split: random}"),
        ),
        (
            "сплит не из шаблона",
            _with_reference(GOOD_REFERENCE.replace("split: random", "split: случайный")),
        ),
        (
            "значение — bool",
            _with_reference(GOOD_REFERENCE.replace("value: 0.91", "value: true")),
        ),
        (
            "опечатка в поле",
            ANALOGS_BLOCK.replace("rejected: []", "rejected: []\n  rejeted: []"),
        ),
        ("запрос — число", ANALOGS_BLOCK.replace('openml: ["delivery delay"]', "openml: [42]")),
        (
            "значение — NaN",
            _with_reference(GOOD_REFERENCE.replace("value: 0.91", "value: .nan")),
        ),
    ],
)
def test_form_without_content_is_refused(tmp_path, label, record) -> None:
    """Повторное ревью PR #2: заплаты по одному полю пропускали следующие дыры.

    `date: '20260929'` проходил, потому что `date.fromisoformat` принимает запись
    без дефисов; `reference_result: {metric: true, value: false, split: []}` —
    потому что проверялось наличие полей, а не их типы. Корень — ручная проверка
    поле за полем; теперь форма задана строгой схемой.
    """
    project, prereg = _ready(tmp_path)
    (project / ANALOGS).write_text(record, encoding="utf-8")
    calls = 0

    def build():
        nonlocal calls
        calls += 1

    with pytest.raises(OutOfOrder):
        guarded(project, prereg, build)

    assert calls == 0, f"построитель вызван: {label}"


def test_a_full_reference_result_is_accepted(tmp_path) -> None:
    """Обратный исход: полная запись шаблона проходит."""
    project, prereg = _ready(tmp_path)
    (project / ANALOGS).write_text(_with_reference(GOOD_REFERENCE), encoding="utf-8")

    assert guarded(project, prereg, lambda: "построено") == "построено"


def _decisions(found: str, taken: str = "[]", rejected: str = "[]") -> str:
    return ANALOGS_BLOCK.replace(
        "found: []\n  taken: []\n  rejected: []",
        f"found: {found}\n  taken: {taken}\n  rejected: {rejected}",
    )


@pytest.mark.parametrize(
    ("label", "record"),
    [
        (
            "ссылка — не URL",
            _decisions("[{url: 'x'}]", rejected="[{url: 'x', reason: 'проверен'}]"),
        ),
        (
            "ссылка без хоста с точкой",
            _decisions("[{url: 'https://x'}]", rejected="[{url: 'https://x', reason: 'проверен'}]"),
        ),
        (
            "ссылка не http",
            _decisions(
                "[{url: 'ftp://kaggle.com/x'}]",
                rejected="[{url: 'ftp://kaggle.com/x', reason: 'проверен'}]",
            ),
        ),
        (
            "решение о ненайденном",
            _decisions("[]", taken="[{url: 'https://kaggle.com/x', reason: 'взят'}]"),
        ),
        (
            "взят и отвергнут сразу",
            _decisions(
                "[{url: 'https://kaggle.com/x'}]",
                taken="[{url: 'https://kaggle.com/x', reason: 'взят'}]",
                rejected="[{url: 'https://kaggle.com/x', reason: 'отвергнут'}]",
            ),
        ),
        (
            "найден дважды",
            _decisions(
                "[{url: 'https://kaggle.com/x'}, {url: 'https://kaggle.com/x'}]",
                rejected="[{url: 'https://kaggle.com/x', reason: 'проверен'}]",
            ),
        ),
    ],
)
def test_decisions_answer_for_what_was_found(tmp_path, label, record) -> None:
    """Третье ревью PR #2: `url: 'x'` проходил как ссылка, а решение в `taken` —
    при пустом `found`: согласованность проверялась лишь от найденного к решению.
    """
    project, prereg = _ready(tmp_path)
    (project / ANALOGS).write_text(record, encoding="utf-8")
    calls = 0

    def build():
        nonlocal calls
        calls += 1

    with pytest.raises(OutOfOrder):
        guarded(project, prereg, build)

    assert calls == 0, f"построитель вызван: {label}"


GOOD_LINKS = [
    "https://kaggle.com/x",
    "http://www.openml.org/t/1",
    "https://github.com/a/b?x=1#y",
    "https://Kaggle.COM/x",
    "https://kaggle.com:443/x",
    "https://пример.рф/x",
]
BAD_LINKS = [
    "x",
    "https://x",
    "ftp://kaggle.com/x",
    "https://kaggle .com/x",
    "https://.com",
    "https://a.b:badport/x",
    "https://a.b:99999/x",
    "https://a..b/x",
    "https://-a.com",
    "https://a-.com",
    "https://a_b.com",
    "https://1.2.3.4/x",
    "https://a.123",
]


@pytest.mark.parametrize("link", GOOD_LINKS)
def test_a_link_parsed_by_the_standard_is_accepted(link) -> None:
    from protocol.preflight import _web_link

    assert _web_link(link) == link


@pytest.mark.parametrize("link", BAD_LINKS)
def test_a_link_the_standard_rejects_is_refused(link) -> None:
    """Четвёртое ревью PR #2: порт `badport` и пустая часть хоста `a..b` проходили.

    Ручная проверка «схема и точка в хосте» — та же заплата поле за полем, что
    уже дважды подводила. Ссылка разбирается стандартным разборщиком (порт,
    схема, IPv4, IDNA), имя хоста — по RFC 1123.
    """
    from protocol.preflight import _web_link

    with pytest.raises(ValueError):
        _web_link(link)


def test_the_reviewers_bad_port_stops_the_builder(tmp_path) -> None:
    project, prereg = _ready(tmp_path)
    link = "https://a.b:badport/x"
    (project / ANALOGS).write_text(
        _decisions(f"[{{url: '{link}'}}]", rejected=f"[{{url: '{link}', reason: 'проверен'}}]"),
        encoding="utf-8",
    )

    with pytest.raises(OutOfOrder):
        guarded(project, prereg, lambda: "построено")


def _two_found(first: str, second: str) -> str:
    return _decisions(
        f"[{{url: '{first}'}}, {{url: '{second}'}}]",
        taken=f"[{{url: '{first}', reason: 'взят'}}]",
        rejected=f"[{{url: '{second}', reason: 'отвергнут'}}]",
    )


SAME_LINK = [
    ("https://kaggle.com/x", "https://Kaggle.COM/x"),
    ("https://kaggle.com/x", "HTTPS://kaggle.com/x"),
    ("https://kaggle.com/x", "https://kaggle.com:443/x"),
    ("https://kaggle.com/x", "https://kaggle.com/a/../x"),
    ("https://kaggle.com/~x", "https://kaggle.com/%7Ex"),
    ("https://kaggle.com/a%3ab", "https://kaggle.com/a%3Ab"),
    ("https://kaggle.com/x", "https://kaggle.com/x#обзор"),
    ("https://пример.рф/x", "https://xn--e1afmkfd.xn--p1ai/x"),
]
DIFFERENT_LINK = [
    ("https://kaggle.com/x", "https://kaggle.com/y"),
    ("https://kaggle.com/x", "https://kaggle.com/x/"),
    ("https://kaggle.com/x", "http://kaggle.com/x"),
    ("https://kaggle.com/x?a=1", "https://kaggle.com/x?a=2"),
]


@pytest.mark.parametrize(("first", "second"), SAME_LINK)
def test_one_link_written_twice_is_a_repeat(tmp_path, first, second) -> None:
    """Пятое ревью PR #2: `https://kaggle.com/x` и `https://Kaggle.COM/x` проходили
    двумя аналогами — разбор нормализовал ссылку, а повторы сверялись по исходной строке.

    Тождество — синтаксическая нормализация RFC 3986 §6.2.2 без фрагмента.
    """
    project, prereg = _ready(tmp_path)
    (project / ANALOGS).write_text(_two_found(first, second), encoding="utf-8")

    with pytest.raises(OutOfOrder) as отказ:
        guarded(project, prereg, lambda: "построено")

    assert "дважды" in str(отказ.value)


@pytest.mark.parametrize(("first", "second"), DIFFERENT_LINK)
def test_different_links_are_not_merged(tmp_path, first, second) -> None:
    """Обратный исход: то, что стандарт тождеством не считает, остаётся разным.

    http и https, слэш на конце — одна ли это страница, знает сервер, а не форма.
    """
    project, prereg = _ready(tmp_path)
    (project / ANALOGS).write_text(_two_found(first, second), encoding="utf-8")

    assert guarded(project, prereg, lambda: "построено") == "построено"


@pytest.mark.parametrize(
    ("label", "record"),
    [
        (
            "учётные данные в ссылке",
            _decisions(
                "[{url: 'https://user:pass@kaggle.com/x'}]",
                rejected="[{url: 'https://user:pass@kaggle.com/x', reason: 'проверен'}]",
            ),
        ),
        ("дата поиска в будущем", ANALOGS_BLOCK.replace("date: 2026-09-29", "date: 2999-01-01")),
    ],
)
def test_what_the_record_itself_rules_out(tmp_path, label, record) -> None:
    """Найдено автором при перечитывании после пятого ревью PR #2, до шестого.

    Ссылка с логином и паролем — не ссылка на аналог, а чужой доступ; дата поиска
    позже сегодняшней невозможна. Оба видны по самой записи, без чтения страниц.
    """
    project, prereg = _ready(tmp_path)
    (project / ANALOGS).write_text(record, encoding="utf-8")

    with pytest.raises(OutOfOrder):
        guarded(project, prereg, lambda: "построено")


LONG_HOST = ".".join(["a" * 60] * 5) + ".com"


@pytest.mark.parametrize(
    ("label", "record"),
    [
        (
            "запрос из управляющего символа",
            ANALOGS_BLOCK.replace('openml: ["delivery delay"]', 'openml: ["\\0"]'),
        ),
        (
            "запрос из невидимого символа",
            ANALOGS_BLOCK.replace('openml: ["delivery delay"]', 'openml: ["\\u200b"]'),
        ),
        (
            "причина из управляющего символа",
            _decisions(
                "[{url: 'https://kaggle.com/x'}]",
                rejected='[{url: "https://kaggle.com/x", reason: "\\0"}]',
            ),
        ),
    ],
)
def test_text_without_text_is_refused(tmp_path, label, record) -> None:
    """Шестое ревью PR #2 (пункт 4): `"\\0"` и `"\\u200b"` проходили как непустой текст."""
    project, prereg = _ready(tmp_path)
    (project / ANALOGS).write_text(record, encoding="utf-8")

    with pytest.raises(OutOfOrder):
        guarded(project, prereg, lambda: "построено")


@pytest.mark.parametrize(
    "link",
    [
        "https://kaggle.com/%GG",
        "https://kaggle.com/%",
        "https://kaggle.com\\evil",
        "https://kag​gle.com/x",
        "https://kaggle.com/a​b",
        f"https://{LONG_HOST}/x",
    ],
)
def test_a_link_the_parser_would_repair_is_refused(link) -> None:
    """Шестое ревью PR #2 (пункты 5–7): разборщик чинил ссылку молча, а хранилась исходная.

    Неверный процент, обратная косая черта, невидимый символ; и хост длиннее 253
    знаков — DNS-имя длиннее 255 октетов (RFC 1035) при допустимых частях.
    """
    from protocol.preflight import _web_link

    with pytest.raises(ValueError):
        _web_link(link)


def test_a_host_of_the_maximum_length_is_accepted() -> None:
    from protocol.preflight import _web_link

    host = ".".join(["a" * 63] * 3 + ["b" * 61])
    assert len(host) == 253
    assert _web_link(f"https://{host}/x")


CONTRADICTION = ANALOGS_BLOCK.replace("project: проба\nanalogs:", "analogs:").replace(
    "  pitfalls: []\n```", "  pitfalls: ['утечка через статус']\nproject: проба\n```"
)
"""Второй блок того же проекта с другим содержимым: `project` не первой строкой."""


@pytest.mark.parametrize(
    ("label", "extra"),
    [
        ("project не первой строкой", CONTRADICTION),
        ("битый YAML", "\n```yaml\nproject: проба\nanalogs: [\n```\n"),
        (
            "повтор ключа",
            ANALOGS_BLOCK.replace(
                "  found: []", "  found:\n    - {url: 'https://kaggle.com/x'}\n  found: []"
            ),
        ),
        ("ключ слияния", "\n```yaml\nбаза: &b {project: проба}\n<<: *b\n```\n"),
        ("ограда тильдами", CONTRADICTION.replace("```yaml", "~~~yaml").replace("```", "~~~")),
        ("ограда заглавными", CONTRADICTION.replace("```yaml", "```YAML")),
        ("ограда с отступом", CONTRADICTION.replace("```yaml", "  ```yaml")),
    ],
)
def test_a_hidden_block_is_refused_from_case_21(tmp_path, label, extra) -> None:
    """Шестое ревью PR #2, пункты 1–3: разбор блоков пропускал то, что не понимал.

    Блок, где `project:` не первой строкой, битый YAML и повтор ключа прятали
    запись от проверки, и построитель вызывался. Нестандартная ограда блока —
    та же дыра, найдена автором при перечитывании.
    """
    project, prereg = _ready(tmp_path)
    record = ANALOGS_BLOCK if label == "повтор ключа" else ANALOGS_BLOCK + extra
    if label == "повтор ключа":
        record = extra
    (project / ANALOGS).write_text(record, encoding="utf-8")
    calls = 0

    def build():
        nonlocal calls
        calls += 1

    with pytest.raises(OutOfOrder):
        guarded(project, prereg, build)

    assert calls == 0, f"построитель вызван: {label}"


def test_project_after_the_controls_is_read_from_case_21(tmp_path) -> None:
    """Та же дыра в контролях: блок `controls` с `project` второй строкой не читался."""
    prereg = _prereg(
        tmp_path,
        "```yaml\ncontrols:\n  - {name: К-1, column: at_night}\nproject: проба\n```\n",
        analogs=False,
    )

    assert declared(prereg)["controls"][0]["name"] == "К-1"


def test_a_historical_case_keeps_its_old_reading(tmp_path) -> None:
    """Опечатанные документы 2–20 разбираются по-старому: битый блок молча пропускается."""
    prereg = _prereg(
        tmp_path,
        "```yaml\nproject: проба\n```\n\n```yaml\nanalogs: [\n```\n",
        number=20,
        analogs=False,
    )

    assert declared(prereg) == {"project": "проба"}


@pytest.mark.parametrize("text", ["\\uFE0F", "\\u034F", "\\u200d\\uFE0F"])
def test_a_lone_mark_is_not_text(tmp_path, text) -> None:
    """Седьмое ревью PR #2: невидимые метки (категория Mn) проходили как текст.

    Перечень запрещённого всегда неполон — текст теперь обязан нести хотя бы
    одну букву, цифру, знак препинания или символ.
    """
    project, prereg = _ready(tmp_path)
    (project / ANALOGS).write_text(
        ANALOGS_BLOCK.replace('openml: ["delivery delay"]', f'openml: ["{text}"]'), encoding="utf-8"
    )

    with pytest.raises(OutOfOrder):
        guarded(project, prereg, lambda: "построено")


def test_ordinary_text_in_any_script_is_text(tmp_path) -> None:
    project, prereg = _ready(tmp_path)
    (project / ANALOGS).write_text(
        ANALOGS_BLOCK.replace('openml: ["delivery delay"]', 'openml: ["задержка 🙂 délai"]'),
        encoding="utf-8",
    )

    assert guarded(project, prereg, lambda: "построено") == "построено"


def test_a_link_with_a_mark_the_parser_would_encode_is_refused() -> None:
    from protocol.preflight import _web_link

    with pytest.raises(ValueError):
        _web_link("https://kaggle.com/a️b")


def test_a_link_in_cyrillic_is_accepted() -> None:
    from protocol.preflight import _web_link

    assert _web_link("https://пример.рф/путь/к-аналогу")


QUOTED_BLOCK = "\n> ```yaml\n> project: проба\n> analogs: {}\n> ```\n"


@pytest.mark.parametrize(
    ("label", "extra"),
    [
        ("блок в цитате", QUOTED_BLOCK),
        ("блок-список", "\n```yaml\n- project: проба\n  analogs: {}\n```\n"),
        (
            "строчная ограда и тильды",
            "\nсм. ```yaml\nproject: проба\n```\n" + CONTRADICTION.replace("```", "~~~"),
        ),
        ("единственная ограда с отступом", None),
        ("ограда из четырёх", None),
        ("незакрытый блок", "\n```yaml\nproject: проба\n"),
    ],
)
def test_only_one_way_to_fence_a_block(tmp_path, label, extra) -> None:
    """Седьмое ревью PR #2: ограды искались шаблонами, и блок в цитате, блок-список,
    счёт оград при строчном ```yaml пропускали запись. Теперь грамматика одна:
    блок открывается строкой ```язык с начала строки и закрывается строкой ```;
    любая другая строка с ``` или ~~~ — отказ.
    """
    project, prereg = _ready(tmp_path)
    if label == "единственная ограда с отступом":
        record = ANALOGS_BLOCK.replace("\n```yaml", "\n  ```yaml")
    elif label == "ограда из четырёх":
        record = ANALOGS_BLOCK.replace("```", "````")
    else:
        record = ANALOGS_BLOCK + extra
    (project / ANALOGS).write_text(record, encoding="utf-8")

    with pytest.raises(OutOfOrder):
        guarded(project, prereg, lambda: "построено")


def test_other_code_blocks_are_allowed(tmp_path) -> None:
    project, prereg = _ready(tmp_path)
    (project / ANALOGS).write_text(
        "Запрос выполнен так:\n\n```bash\ncurl https://kaggle.com\n```\n" + ANALOGS_BLOCK,
        encoding="utf-8",
    )

    assert guarded(project, prereg, lambda: "построено") == "построено"
