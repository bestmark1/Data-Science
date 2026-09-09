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

import hashlib
import re
from collections.abc import Iterable
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
        return [f"объявлено снятие контролей, а доказательства прогона нет: {report}"]

    evidence = read_control_run(project)
    if evidence is None:
        return [f"доказательство контрольного прогона нечитаемо или пусто: {report}"]
    if evidence.get("case") != _case_number(prereg):
        return [
            f"доказательство относится к кейсу {evidence.get('case')!r}, "
            f"а сверяется кейс {_case_number(prereg)!r}"
        ]

    found = set(evidence.get("findings") or [])
    problems: list[str] = []
    for control in block.get("controls", []):
        record = spent.get(control["name"])
        if record is None or not isinstance(record.get("fired"), bool):
            continue
        seen = str(control.get("expect", "")) in found
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
    if not prereg.is_file():
        raise OutOfOrder(
            f"ОТКАЗ: пре-регистрации нет по пути {prereg}. Ошибка имени или пути молча "
            "отключала бы весь механизм — это не совместимость, а дыра. "
            "Построитель не вызывался."
        )

    number = _case_number(prereg)
    if number is None:
        raise OutOfOrder(
            f"ОТКАЗ: {prereg.name!r} не опознан как пре-регистрация кейса — ожидается "
            "имя вида `prereg-case-N.md`. Историческое исключение применяется только к "
            "ПОДТВЕРЖДЁННОМУ номеру, а не к нераспознанному документу. "
            "Построитель не вызывался."
        )

    if number < FROM_CASE:
        # Исторический кейс: доказательств у него нет, достраивать их нельзя.
        # Это НЕ обычный успех проверки, и говорится об этом вслух.
        print(
            f"  протокол: кейс {number} старше {FROM_CASE} — порядок НЕ ПРОВЕРЕН, "
            "доказательств у него не существует"
        )
        return

    declared_project = declared(prereg).get("project")
    if declared_project and declared_project != project.name:
        raise OutOfOrder(
            f"ОТКАЗ: пре-регистрация объявляет проект {declared_project!r}, а прогоняется "
            f"{project.name!r}. Сверять кейс с чужой пре-регистрацией нельзя. "
            "Построитель не вызывался."
        )

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

    block = declared(prereg)
    evidence = read_control_run(project)
    standing = _controls_are_standing(prereg, project / "project.yaml")
    named = [c["name"] for c in block.get("controls", []) if c.get("column")]

    # СНЯТИЕ объявлено — значит прогон ЧИСТЫЙ, и контроли стоять уже не должны.
    # Проверяется доказательство: оно обязано существовать, читаться и относиться
    # к этому кейсу. Пустой или чужой файл появляется копированием, без всякой
    # правки кода.
    if block.get("spent_controls"):
        if evidence is None:
            raise OutOfOrder(
                f"ОТКАЗ: объявлено снятие контролей, а доказательства контрольного "
                f"прогона нет либо оно нечитаемо ({project / 'report' / CONTROL_RUN}). "
                "Снятие без прогона означает, что контроль пропущен. "
                "Построитель не вызывался."
            )
        if evidence.get("case") != number:
            raise OutOfOrder(
                f"ОТКАЗ: доказательство контрольного прогона относится к кейсу "
                f"{evidence.get('case')!r}, а прогоняется {number}. Отчёт другого кейса "
                "снятия не разрешает. Построитель не вызывался."
            )
    # ПЕРВЫЙ ПРОГОН. Снятие не объявлено, доказательства нет — значит контроли
    # обязаны СТОЯТЬ в формуляре. Иначе первый прогон пройдёт без них, и
    # контрольного прогона не случится вовсе: ни один сигнал не будет отнесён к
    # подложенному дефекту.
    elif named and evidence is None and not standing:
        raise OutOfOrder(
            f"ОТКАЗ: контроли {named} объявлены в §5, доказательства контрольного "
            "прогона нет, а в формуляре они не стоят. Первый прогон обязан идти С "
            "контролями. Построитель не вызывался."
        )


def _fingerprint(path: Path) -> str:
    """Отпечаток файла — двенадцать знаков, как у слепого контроля."""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12] if path.is_file() else ""


def save_control_run(project: Path, prereg: Path, findings: Iterable[object]) -> Path | None:
    """Записать доказательство контрольного прогона — списком находок, не прозой.

    ПОЧЕМУ НЕ КОПИЯ ОТЧЁТА. Первая версия копировала `report.md` и искала в нём
    имя находки. Отчёт печатает `signal.detail` — человеческую фразу, — а не
    `finding.value`: в настоящем отчёте двадцатого кейса строк `sentinel_as_value`
    и `non_stationary_target` ноль вхождений. Сверка не работала вовсе, и unit-тесты
    этого не показывали, потому что фикстуры писались в том же формате, в каком
    сверка искала. Класс 1 журнала повторов, седьмой случай.

    Здесь берутся `Finding` прямо из результата прогона — устойчивые
    идентификаторы, порождённые ядром.

    ПРИВЯЗКА. Доказательство хранит номер кейса, отпечатки формы и манифеста.
    Отчёт другого кейса или прогон по другой форме опознаётся и не принимается:
    пустой либо чужой файл может появиться копированием, без всякой
    злонамеренной правки кода.
    """
    number = _case_number(prereg)
    if number is None or number < FROM_CASE:
        return None
    if not _controls_are_standing(prereg, project / "project.yaml"):
        return None

    target = project / "report" / CONTROL_RUN
    if target.is_file():
        return None  # первый контрольный прогон и есть тот, о котором говорит §5

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        yaml.safe_dump(
            {
                "case": number,
                "form": _fingerprint(project / "project.yaml"),
                "manifest": _fingerprint(project / "manifest.yaml"),
                "findings": sorted(str(getattr(f, "value", f)) for f in findings),
            },
            allow_unicode=True,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return target


def read_control_run(project: Path) -> dict | None:
    """Прочитать доказательство. None — его нет или оно нечитаемо."""
    path = project / "report" / CONTROL_RUN
    if not path.is_file():
        return None
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError:
        return None
    return loaded if isinstance(loaded, dict) and "findings" in loaded else None


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


def audit_case_artifacts(docs: Path, projects: Path) -> list[str]:
    """Обход НАСТОЯЩИХ материалов кейсов: связи проверяются здесь, а не в тестах.

    До этого обхода сверки `fired`, `lost_bets` и учёта существовали только как
    функции, вызываемые из синтетических unit-тестов. Противоречие в живой
    пре-регистрации не сделало бы общую команду красной: помощник был написан, а
    защита — нет. Разница между «функция существует» и «правило действует» ровно
    в этом обходе.

    Кейсы старше `FROM_CASE` пропускаются: доказательств у них не существует, и
    достраивать их задним числом нельзя.
    """
    problems: list[str] = []
    ledger = docs / "stopping-rule.md"
    for prereg in sorted(docs.glob("prereg-case-*.md")):
        number = _case_number(prereg)
        if number is None or number < FROM_CASE:
            continue
        block = declared(prereg)
        name = block.get("project")
        if not name:
            problems.append(f"{prereg.name}: не объявлен project — сверить кейс не с чем")
            continue
        project = projects / name
        if not project.is_dir():
            continue  # формуляр ещё не написан: обещание не наступило

        problems += [
            f"{prereg.name}: {item}" for item in fired_against_the_control_run(project, prereg)
        ]
        problems += [f"{prereg.name}: {item}" for item in fired_against_the_ledger(prereg, ledger)]
        problems += [
            f"{prereg.name}: {item}"
            for item in lost_bets_against_verdict(prereg, docs / f"case{number}-verdict.md")
        ]
        problems += [
            f"{prereg.name}: {item}"
            for item in unreachable_controls(prereg, project / "project.yaml")
        ]
    return problems
