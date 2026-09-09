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

from protocol import OutOfOrder, preflight, save_control_run  # noqa: E402
from protocol.preflight import (  # noqa: E402
    CONTROL_RUN,
    EMPTY_BIT,
    FROM_CASE,
    SEALED,
    fired_against_the_control_run,
    fired_against_the_ledger,
    lost_bets_against_verdict,
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
  - {name: К-1, column: at_night, role: feature, expect: sentinel_as_value, basis: plant}
```
"""


def _prereg(tmp_path: Path, body: str, number: int = FROM_CASE) -> Path:
    path = tmp_path / f"prereg-case-{number}.md"
    path.write_text(body, encoding="utf-8")
    return path


def _project(tmp_path: Path, *, manifest=True, sealed=True, empty=True) -> Path:
    project = tmp_path / "проба"
    (project / "report").mkdir(parents=True)
    (project / "project.yaml").write_text(FORM, encoding="utf-8")
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

    assert "доказательства контрольного прогона нет" in str(отказ.value)


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
  - {name: К-2, column: pump_installed, role: feature,
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
    assert "не по схеме" in str(пусто.value)

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
        "  - {name: К-1, column: at_night, role: feature,\n"
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
        "  - {name: К-1, column: at_night, role: feature,\n"
        "     expect: sentinel_as_value, basis: plant}\n"
        "  - {name: К-2, column: chain, role: group_id, expect: undeclared_group, basis: plant}\n"
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
        "     expect: sentinel_as_value, basis: plant}\n```\n",
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


def test_the_check_command_keeps_the_cause_of_a_long_collection_error(monkeypatch, capsys) -> None:
    """Регрессия, которую прошлый раз ЗАБЫЛИ перенести, вопреки отчёту.

    Причина ошибки сборки стоит в начале вывода, а хвост занимают
    предупреждения. Прежняя версия печатала последние 4000 знаков и теряла имя
    отсутствующего модуля — то самое, ради чего диагностику и чинили.
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

    assert check.main() == 1
    напечатано = capsys.readouterr().out

    assert "missing_fixture_module" in напечатано, "первопричина потеряна в хвосте"
    (ROOT / ".check-collection-error.log").unlink(missing_ok=True)


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
