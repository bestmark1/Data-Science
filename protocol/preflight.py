"""Порядок кейса, проверяемый до вызова построителя.

ЗАЧЕМ ОТДЕЛЬНО ОТ ЯДРА. `dsx` переносим: он принимает форму и данные и о
исследовательском протоколе не знает. Порядок кейсов — свойство этого проекта, а
не инструмента, и держать его в ядре значило бы обязать всякого, кто возьмёт
`dsx`, вести пре-регистрации и слепые контроли.

ЗАЧЕМ ВООБЩЕ. В девятнадцатом кейсе слепой контроль был проведён ПОСЛЕ первого
прогона, вопреки §5а, — автор нарушил порядок собственной рукой и записал это
ошибкой. Напоминание в инструкциях от повторения не спасает: оно уже там было.

ЧЕГО ЭТО НЕ ДАЁТ. Штатный путь защищён, обход — нет. Оператор с доступом к коду
вызовет построитель напрямую, и никакая проверка внутри репозитория этому не
помешает. Обещать большее нельзя.

СОВМЕСТИМОСТЬ. Проверки применяются с кейса, названного в `FROM_CASE`. У кейсов
2–20 отчёта контрольного прогона не существует, и достроить его задним числом
означало бы выдумать доказательство. Такие кейсы пропускаются с прямым словом
«не проверено», а не с молчанием.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

FROM_CASE = 21
"""Кейс, с которого действует проверка порядка.

Тот же приём, что `GROUNDS_FROM_CASE` в сверке пре-регистраций: правило вводится
вперёд, опечатанные материалы прошлых кейсов не переписываются, а недостающие
доказательства называются недостающими.
"""

CASE_NUMBER = re.compile(r"prereg-case-(\d+)\.md$")
BLOCK = re.compile(r"```yaml\n(project:.*?)\n```", re.DOTALL)
DIGEST = re.compile(r"Отпечаток (бита пустоты|ответа): \*\*`([0-9a-f]{12})`\*\*")

SEALED = "blind-control.sealed.md"
EMPTY_BIT = "blind-control.empty.md"
CONTROL_RUN = "control-run.md"


class OutOfOrder(SystemExit):
    """Шаг протокола пропущен. Наследует SystemExit: прогон обязан прекратиться.

    Возбуждается ДО вызова построителя — иначе данные будут собраны и
    обработаны, а отказ придёт задним числом, когда работа уже сделана.
    """


def _case_number(prereg: Path) -> int | None:
    found = CASE_NUMBER.search(prereg.name)
    return int(found.group(1)) if found else None


def declared(prereg: Path) -> dict:
    """Машиночитаемые объявления пре-регистрации — все блоки сразу."""
    merged: dict = {}
    for found in BLOCK.finditer(prereg.read_text(encoding="utf-8")):
        block = yaml.safe_load(found.group(1))
        for key, value in block.items():
            if key in merged and isinstance(value, list):
                merged[key] = merged[key] + value
            else:
                merged[key] = value
    return merged


def _controls_are_standing(prereg: Path, form_path: Path) -> bool:
    """Стоят ли контроли §5 в формуляре прямо сейчас.

    Прогон с внесённым контролем — контрольный, и его отчёт обязан сохраниться
    отдельно: `report.md` перезаписывается следующим же прогоном, и к моменту
    записи вердикта от контрольного прогона не остаётся ничего, кроме прозы.
    """
    block = declared(prereg)
    form = yaml.safe_load(form_path.read_text(encoding="utf-8"))
    roles = {column["name"]: column.get("role") for column in form.get("columns", [])}
    return any(
        control.get("column") and roles.get(control["column"]) == control.get("role")
        for control in block.get("controls", [])
    )


CONTRADICTIONS = {
    "value_revised_after_decision": (
        "value_as_of",
        "проверка S10 пропускает случай, когда объявленный момент фиксации равен "
        "моменту решения: `checks/premise_check.py` — «if fixed == decision.name: "
        "continue». Находка недостижима по построению",
    ),
}
"""Известные противоречия «ожидаемая находка ↔ объявление, при котором проверка молчит».

РЕЕСТР, А НЕ АНАЛИЗАТОР. Здесь ровно то, на чём проект уже обжёгся, — сейчас
один случай, найденный двадцатым кейсом. Копировать условия молчания всех
проверок в отдельный модуль запрещено: копия разойдётся с оригиналом при первой
же правке проверки, и разойдётся молча.

Пополняется, когда противоречие найдено кейсом. Не раньше.

НАЗВАННЫЙ ПРЕДЕЛ: реестр проверяет ОБЪЯВЛЕНИЕ, а не данные. Достижимость,
зависящая от значений, им не проверяется, и предварительная проверка объявления
не является доказательством того, что проверка сработает на настоящих данных.
"""


def unreachable_controls(prereg: Path, form_path: Path) -> list[str]:
    """Контроли, чья ожидаемая находка недостижима при их же объявлении.

    Двадцатый кейс: К-2 требовал `value_revised_after_decision` от колонки,
    объявленной с `value_as_of`, равным моменту решения. Роль была свободна,
    сверка §5 промолчала — она проверяет ИСПОЛНИМОСТЬ, — и контроль всё равно
    не мог сработать. Третий случай одного рода: кейсы 13, 18, 20.
    """
    form = yaml.safe_load(form_path.read_text(encoding="utf-8"))
    columns = {column["name"]: column for column in form.get("columns", [])}
    decision = next(
        (c["name"] for c in form.get("columns", []) if c.get("role") == "decision_time"), None
    )
    broken: list[str] = []
    for control in declared(prereg).get("controls", []):
        rule = CONTRADICTIONS.get(control.get("expect"))
        if rule is None or not control.get("column"):
            continue
        field, why = rule
        column = columns.get(control["column"])
        if column and decision and column.get(field) == decision:
            broken.append(
                f"{control['name']}: ждёт {control['expect']!r} от {control['column']!r}, "
                f"но {field}={decision!r} — {why}"
            )
    return broken


def fired_against_the_control_run(project: Path, prereg: Path) -> list[str]:
    """Расхождения между объявленным `fired` и отчётом контрольного прогона.

    `fired` пишет автор, а отчёт порождает ядро. Совпадение двух записей автора
    доказательством не является; совпадение записи автора с выводом ядра —
    является.
    """
    block = declared(prereg)
    spent = {item["name"]: item for item in block.get("spent_controls", [])}
    if not spent:
        return []
    report = project / "report" / CONTROL_RUN
    if not report.is_file():
        return [f"объявлено снятие контролей, а отчёта контрольного прогона нет: {report}"]

    text = report.read_text(encoding="utf-8")
    problems: list[str] = []
    for control in block.get("controls", []):
        record = spent.get(control["name"])
        if record is None or not isinstance(record.get("fired"), bool):
            continue
        seen = str(control.get("expect", "")) in text
        if record["fired"] and not seen:
            problems.append(
                f"{control['name']}: объявлен сработавшим, но находки "
                f"{control.get('expect')!r} в отчёте контрольного прогона нет"
            )
        if not record["fired"] and seen:
            problems.append(
                f"{control['name']}: объявлен промолчавшим, а находка "
                f"{control.get('expect')!r} в отчёте контрольного прогона есть"
            )
    return problems


def preflight(project: Path, prereg: Path) -> None:
    """Проверить порядок. Возбуждает OutOfOrder, называя недостающий шаг."""
    number = _case_number(prereg)
    if number is None or number < FROM_CASE:
        return  # исторический кейс: доказательств нет, и достраивать их нельзя

    if not (project / "manifest.yaml").is_file():
        raise OutOfOrder(
            f"ОТКАЗ: данные не собраны — нет {project / 'manifest.yaml'}. Построитель не вызывался."
        )

    sealed = project / SEALED
    if not sealed.is_file():
        raise OutOfOrder(
            f"ОТКАЗ: слепой контроль не внесён — нет {sealed}. §5а требует провести "
            "его ДО первого прогона. Построитель не вызывался."
        )

    text = prereg.read_text(encoding="utf-8")
    written = {kind: value for kind, value in DIGEST.findall(text)}
    if len(written) < 2:
        raise OutOfOrder(
            "ОТКАЗ: в §5а вписаны не оба отпечатка слепого контроля. Вписать их "
            "нужно ДО прогона: после него запись перестаёт быть обещанием. "
            "Построитель не вызывался."
        )

    import hashlib

    actual = {
        "бита пустоты": hashlib.sha256((project / EMPTY_BIT).read_bytes()).hexdigest()[:12]
        if (project / EMPTY_BIT).is_file()
        else None,
        "ответа": hashlib.sha256(sealed.read_bytes()).hexdigest()[:12],
    }
    for kind, value in written.items():
        if actual.get(kind) != value:
            raise OutOfOrder(
                f"ОТКАЗ: отпечаток {kind} в §5а не сходится с файлом контроля. "
                "Либо вписан не тот, либо файл изменён после запечатывания. "
                "Построитель не вызывался."
            )

    unreachable = unreachable_controls(prereg, project / "project.yaml")
    if unreachable:
        raise OutOfOrder(
            "ОТКАЗ: контроль требует находки, недостижимой при его же объявлении. "
            + "; ".join(unreachable)
            + ". Построитель не вызывался."
        )

    # Снятие контроля объявлено — значит контрольный прогон должен был
    # состояться. До него `control-run.md` не существует, и запись `spent_controls`,
    # сделанная заранее, обязана быть отвергнута: иначе она разрешает пропустить
    # сам контроль.
    if declared(prereg).get("spent_controls") and not (project / "report" / CONTROL_RUN).is_file():
        raise OutOfOrder(
            f"ОТКАЗ: объявлено снятие контролей, а отчёта контрольного прогона нет "
            f"({project / 'report' / CONTROL_RUN}). Снятие без прогона означает, что "
            "контроль пропущен. Построитель не вызывался."
        )


def save_control_run(project: Path, prereg: Path) -> Path | None:
    """Сохранить отчёт контрольного прогона, если контроли стоят в формуляре.

    Вызывается ПОСЛЕ прогона. `report.md` перезаписывается каждым следующим
    прогоном; на кейсе 20 это проверено прямо: в git лежит версия после снятия
    контролей, и находки К-1 в ней нет ни одной, хотя контроль сработал.

    Пишется один раз. Повторный контрольный прогон отчёта не затирает: первый
    и есть тот, о котором говорит `spent_controls`.
    """
    number = _case_number(prereg)
    if number is None or number < FROM_CASE:
        return None
    if not _controls_are_standing(prereg, project / "project.yaml"):
        return None

    report = project / "report" / "report.md"
    target = project / "report" / CONTROL_RUN
    if target.is_file() or not report.is_file():
        return None
    target.write_text(report.read_text(encoding="utf-8"), encoding="utf-8")
    return target


VERDICT_ROW = re.compile(r"^\|\s*(P-\d+)\s*\|[^|]*\|\s*([^|]+?)\s*\|", re.MULTILINE)
STOPPING_ROW = re.compile(r"^\| (\d+) — [^|]*\|[^|]*\|([^|]*)\|", re.MULTILINE)


def lost_bets_against_verdict(prereg: Path, verdict: Path) -> list[str]:
    """Проигранная ставка обязана быть видна в разрешении предсказаний.

    Номер предсказания в механизм НЕ зашивается: связь идёт через объявление
    `predictions` с полем `rests_on`, а какое именно предсказание опирается на
    обещания, говорит сама пре-регистрация.

    Проверяется одно: при непустом `lost_bets` предсказание, опирающееся на
    `promises`, в вердикте не разрешено сбывшимся.

    ЧЕГО НЕ ДАЁТ: механизм читает слово в колонке итога и не понимает
    содержания. «Не сбылось» можно написать и там, где всё исполнено.
    """
    block = declared(prereg)
    if not block.get("lost_bets") or not verdict.is_file():
        return []
    resting = {
        item["id"] for item in block.get("predictions", []) if item.get("rests_on") == "promises"
    }
    if not resting:
        return [
            "объявлены проигравшие ставки, но ни одно предсказание не объявило "
            "`rests_on: promises` — связь с вердиктом проверить нечем"
        ]

    outcomes = dict(VERDICT_ROW.findall(verdict.read_text(encoding="utf-8")))
    problems: list[str] = []
    for number in sorted(resting):
        итог = outcomes.get(number, "")
        if not итог:
            problems.append(f"{number} опирается на обещания, но в вердикте не разрешено")
        elif итог.lower().lstrip("*").startswith("сбылось"):
            problems.append(
                f"{number} опирается на обещания, ставка проиграна, а в вердикте записано {итог!r}"
            )
    return problems


def fired_against_the_ledger(prereg: Path, ledger: Path) -> list[str]:
    """Промолчавший контроль не может числиться в учёте полным успехом.

    Проверка СЛАБАЯ намеренно, и это признано при согласовании: сильная
    потребовала бы машиночитаемого учёта правила остановки, то есть
    переписывания всех прежних строк задним числом.

    Читается колонка «Контроли» строки этого кейса: при `fired: false` она
    обязана говорить о молчании. Слова о молчании механизм знает по списку —
    он не понимает прозу, он ищет в ней признак.
    """
    block = declared(prereg)
    silent = [
        item["name"] for item in block.get("spent_controls", []) if item.get("fired") is False
    ]
    if not silent or not ledger.is_file():
        return []
    number = _case_number(prereg)
    for found, controls in STOPPING_ROW.findall(ledger.read_text(encoding="utf-8")):
        if int(found) != number:
            continue
        if any(слово in controls.lower() for слово in ("промолчал", "не сработал", "молчал")):
            return []
        return [
            f"контроли {silent} объявлены промолчавшими, а строка учёта кейса {number} "
            f"о молчании не говорит: {controls.strip()!r}"
        ]
    return [f"контроли {silent} промолчали, а строки кейса {number} в учёте нет вовсе"]
