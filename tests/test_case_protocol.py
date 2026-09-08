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
    project = tmp_path / "проект"
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
    prereg = _prereg(tmp_path, "## 5а. Слепой контроль\n")
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
    prereg = _prereg(tmp_path, "## 5а. Слепой контроль\n\nОтпечаток бита пустоты: `—`\n")

    with pytest.raises(OutOfOrder) as отказ:
        preflight(project, prereg)

    assert "не оба отпечатка" in str(отказ.value)


def test_a_digest_that_does_not_match_the_file_stops_the_run(tmp_path) -> None:
    """Отпечаток не сходится: вписан не тот либо файл изменён после запечатывания."""
    project = _project(tmp_path)
    prereg = _prereg(
        tmp_path,
        "## 5а\n\nОтпечаток бита пустоты: **`000000000000`**\n"
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

    assert "отчёта контрольного прогона нет" in str(отказ.value)


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

    preflight(project, _prereg(tmp_path, "пусто", number=FROM_CASE - 1))
    preflight(project, _prereg(tmp_path, "пусто", number=2))


# --- Сохранение отчёта контрольного прогона ----------------------------------


def test_the_control_run_report_is_saved_while_controls_stand(tmp_path) -> None:
    """Отчёт контрольного прогона сохраняется отдельно от рабочего.

    `report.md` перезаписывается следующим прогоном. На кейсе 20 это проверено
    прямо: в git лежит версия после снятия контролей, и находки К-1 в ней нет
    ни одной, хотя контроль сработал.
    """
    project = _project(tmp_path)
    prereg = _prereg(tmp_path, _digests(project) + CONTROLS)
    (project / "report" / "report.md").write_text("сигнал: sentinel_as_value\n", encoding="utf-8")

    saved = save_control_run(project, prereg)

    assert saved == project / "report" / CONTROL_RUN
    assert "sentinel_as_value" in saved.read_text(encoding="utf-8")


def test_the_control_run_report_is_not_overwritten(tmp_path) -> None:
    """Первый контрольный прогон и есть тот, о котором говорит `spent_controls`."""
    project = _project(tmp_path)
    prereg = _prereg(tmp_path, _digests(project) + CONTROLS)
    (project / "report" / CONTROL_RUN).write_text("первый\n", encoding="utf-8")
    (project / "report" / "report.md").write_text("второй\n", encoding="utf-8")

    assert save_control_run(project, prereg) is None
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
    (project / "report" / "report.md").write_text("чисто\n", encoding="utf-8")

    assert save_control_run(project, prereg) is None
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
    (project / "report" / CONTROL_RUN).write_text("никаких находок\n", encoding="utf-8")

    problems = fired_against_the_control_run(project, prereg)

    assert len(problems) == 1
    assert "объявлен сработавшим" in problems[0]


def test_fired_false_with_the_finding_is_reported(tmp_path) -> None:
    """Обратное: объявлено «промолчал», а ядро находку назвало."""
    project = _project(tmp_path)
    prereg = _prereg(tmp_path, _digests(project) + CONTROLS + _spent(False))
    (project / "report" / CONTROL_RUN).write_text("sentinel_as_value: 124\n", encoding="utf-8")

    problems = fired_against_the_control_run(project, prereg)

    assert len(problems) == 1
    assert "объявлен промолчавшим" in problems[0]


def test_fired_matching_the_report_is_silent(tmp_path) -> None:
    """Положительный контроль: запись автора совпала с выводом ядра."""
    project = _project(tmp_path)
    prereg = _prereg(tmp_path, _digests(project) + CONTROLS + _spent(True))
    (project / "report" / CONTROL_RUN).write_text("sentinel_as_value: 124\n", encoding="utf-8")

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
