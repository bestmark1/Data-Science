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


ANY_BLOCK = re.compile(r"```yaml\n(.*?)\n```", re.DOTALL)


def declared(prereg: Path) -> dict:
    """Машиночитаемые объявления пре-регистрации — все блоки сразу.

    Блок протокола обязан нести `project:`. Повторное ревью показало, чем грозит
    обратное: контроли, объявленные в блоке без `project`, парсер не видел вовсе,
    и preflight считал, что контролей нет.
    """
    text = prereg.read_text(encoding="utf-8")
    for found in ANY_BLOCK.finditer(text):
        try:
            block = yaml.safe_load(found.group(1))
        except yaml.YAMLError:
            continue
        if isinstance(block, dict) and any(key in block for key in PROTOCOL_KEYS):
            if "project" not in block:
                raise OutOfOrder(
                    f"ОТКАЗ: в {prereg.name} есть машиночитаемый блок протокола "
                    f"({sorted(set(block) & set(PROTOCOL_KEYS))}) без `project`. "
                    "Такой блок не был бы связан с кейсом и потерялся бы молча. "
                    "Построитель не вызывался."
                )

    merged: dict = {}
    for found in BLOCK.finditer(text):
        block = yaml.safe_load(found.group(1))
        for key, value in block.items():
            if key in merged and merged[key] != value:
                if isinstance(value, list):
                    merged[key] = merged[key] + value
                    continue
                # Молчаливая перезапись позволяла объявить `project: чужой`, а
                # следующим блоком — `project: свой`, и пройти проверку. Общий
                # парсер §5 такой конфликт отвергает; здесь он отвергался молча.
                raise OutOfOrder(
                    f"ОТКАЗ: объявление {key!r} повторено с другим значением "
                    f"({merged[key]!r} и {value!r}). Построитель не вызывался."
                )
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
    named = [c for c in block.get("controls", []) if c.get("column")]
    if not named:
        return False
    # ВСЕ, а не любой. Прежняя версия брала `any`, и один поставленный контроль
    # выдавал за поставленные все: прогон с половиной контролей считался
    # контрольным, а вторая половина не испытывалась вовсе.
    return all(roles.get(c["column"]) == c.get("role") for c in named)


def controls_partly_standing(prereg: Path, form_path: Path) -> list[str]:
    """Контроли, объявленные с колонкой, но НЕ поставленные в формуляр."""
    block = declared(prereg)
    form = yaml.safe_load(form_path.read_text(encoding="utf-8"))
    roles = {column["name"]: column.get("role") for column in form.get("columns", [])}
    return [
        c["name"]
        for c in block.get("controls", [])
        if c.get("column") and roles.get(c["column"]) != c.get("role")
    ]


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
        return [
            f"доказательство контрольного прогона негодно — {control_run_defect(project)}: {report}"
        ]
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

    # Симлинк не смеет выдавать один кейс за другой: имя ссылки и имя цели
    # обязаны говорить об одном номере.
    resolved = prereg.resolve()
    if _case_number(prereg) != _case_number(resolved):
        raise OutOfOrder(
            f"ОТКАЗ: {prereg.name} указывает на {resolved.name} — номер кейса в имени "
            "ссылки расходится с целью. Построитель не вызывался."
        )
    prereg = resolved
    number = _case_number(prereg)
    if number is None:
        raise OutOfOrder(
            f"ОТКАЗ: {prereg.name!r} не опознан как пре-регистрация кейса — ожидается "
            "имя вида `prereg-case-N.md`. Историческое исключение применяется только к "
            "ПОДТВЕРЖДЁННОМУ номеру, а не к нераспознанному документу. "
            "Построитель не вызывался."
        )

    # ИДЕНТИЧНОСТЬ ПРОВЕРЯЕТСЯ ДО исторического исключения. Иначе документ
    # чужого кейса, поданный под старым номером, отключал защиту целиком —
    # ровно то, что нашло повторное ревью.
    declared_project = declared(prereg).get("project")
    if declared_project is None:
        raise OutOfOrder(
            f"ОТКАЗ: {prereg.name} не объявляет `project` машиночитаемо. Связать "
            "пре-регистрацию с прогоняемым кейсом нечем, и историческое исключение "
            "по одному лишь имени файла не выдаётся. Построитель не вызывался."
        )
    if declared_project != project.name:
        raise OutOfOrder(
            f"ОТКАЗ: пре-регистрация объявляет проект {declared_project!r}, а прогоняется "
            f"{project.name!r}. Сверять кейс с чужой пре-регистрацией нельзя. "
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
    named = [c["name"] for c in block.get("controls", []) if c.get("column")]

    # ПЕРЕХОДЫ. Проверяется не наличие файлов, а допустимость самого перехода.
    # Повторное ревью показало четыре состояния, которые прежняя версия
    # пропускала: половина контролей, чистый прогон без объявленного снятия,
    # снятие при стоящих контролях и ложный `fired`.
    missing = controls_partly_standing(prereg, project / "project.yaml")

    if block.get("spent_controls"):
        # СНЯТИЕ. Прогон чистый: доказательство обязано существовать, быть по
        # схеме, относиться к этому кейсу — и форма обязана ОТЛИЧАТЬСЯ от
        # контрольной, иначе контроли на самом деле не сняты.
        if evidence is None:
            raise OutOfOrder(
                f"ОТКАЗ: объявлено снятие контролей, а доказательство контрольного прогона "
                f"негодно — {control_run_defect(project)} "
                f"({project / 'report' / CONTROL_RUN}). "
                "Снятие без прогона означает, что контроль пропущен. "
                "Построитель не вызывался."
            )
        if evidence.get("case") != number:
            raise OutOfOrder(
                f"ОТКАЗ: доказательство контрольного прогона относится к кейсу "
                f"{evidence.get('case')!r}, а прогоняется {number}. Отчёт другого кейса "
                "снятия не разрешает. Построитель не вызывался."
            )
        if evidence.get("manifest") != _fingerprint(project / "manifest.yaml"):
            raise OutOfOrder(
                "ОТКАЗ: доказательство получено на других данных — отпечаток манифеста "
                "не совпадает. Контрольный прогон и чистый обязаны идти по одной "
                "выгрузке. Построитель не вызывался."
            )
        if evidence.get("form") != _form_identity(
            project / "project.yaml", _control_columns(prereg)
        ):
            raise OutOfOrder(
                "ОТКАЗ: форма изменилась не только снятием контролей — доказательство "
                "получено на другой форме. Отпечаток считается БЕЗ ролей контрольных "
                "колонок, поэтому законное снятие его не меняет, а всякая иная правка "
                "меняет. Построитель не вызывался."
            )
        if not missing:
            raise OutOfOrder(
                "ОТКАЗ: объявлено снятие контролей, а в формуляре они всё ещё стоят. "
                "Прогон не чистый. Построитель не вызывался."
            )
        problems = fired_against_the_control_run(project, prereg)
        if problems:
            raise OutOfOrder(
                "ОТКАЗ: объявленный исход контролей расходится с доказательством "
                "прогона: " + "; ".join(problems) + ". Построитель не вызывался."
            )
    elif named and evidence is None:
        # ПЕРВЫЙ ПРОГОН. Все объявленные контроли обязаны стоять — не часть.
        if missing:
            raise OutOfOrder(
                f"ОТКАЗ: контроли {missing} объявлены в §5, доказательства контрольного "
                "прогона нет, а в формуляре они не стоят. Первый прогон обязан идти СО "
                "ВСЕМИ объявленными контролями. Построитель не вызывался."
            )
    elif named and evidence is not None and missing:
        # Контроли сняты, доказательство есть, а снятие не объявлено. Прогон
        # выглядит чистым, но §5 об этом не знает: исход контролей нигде не
        # записан, и вердикт будет опираться на память автора.
        raise OutOfOrder(
            f"ОТКАЗ: контроли {missing} сняты с формуляра, доказательство прогона есть, "
            "а `spent_controls` в пре-регистрации не объявлены. Снятие обязано быть "
            "записано вместе с исходом. Построитель не вызывался."
        )


PROTOCOL_KEYS = ("controls", "promises", "spent_controls", "lost_bets", "predictions")


def _form_identity(form_path: Path, control_columns: set[str]) -> str:
    """Отпечаток формы БЕЗ ролей контрольных колонок.

    Законное снятие контроля меняет форму — и сравнивать её отпечаток целиком
    нельзя: чистый прогон обязан отличаться от контрольного. Но отличаться он
    обязан ТОЛЬКО этим.

    Поэтому роли контрольных колонок из отпечатка исключаются. Он одинаков до и
    после снятия и расходится при любой другой правке: добавленной колонке,
    смене срока, другом сплите. Так доказательство привязано к форме, а переход
    «контрольный → чистый» остаётся разрешённым.
    """
    form = yaml.safe_load(form_path.read_text(encoding="utf-8")) or {}
    columns = [
        {
            k: v
            for k, v in column.items()
            if not (column.get("name") in control_columns and k == "role")
        }
        for column in form.get("columns", [])
    ]
    skeleton = {**{k: v for k, v in form.items() if k != "columns"}, "columns": columns}
    return hashlib.sha256(
        yaml.safe_dump(skeleton, allow_unicode=True, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]


def _control_columns(prereg: Path) -> set[str]:
    return {c["column"] for c in declared(prereg).get("controls", []) if c.get("column")}


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
                "form": _form_identity(project / "project.yaml", _control_columns(prereg)),
                "manifest": _fingerprint(project / "manifest.yaml"),
                "findings": sorted(str(getattr(f, "value", f)) for f in findings),
            },
            allow_unicode=True,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return target


IDENTIFIER = re.compile(r"^[a-z][a-z0-9_]*$")


def read_control_run(project: Path) -> dict | None:
    """Прочитать доказательство. None — его нет, оно нечитаемо или не по схеме.

    СХЕМА ПРОВЕРЯЕТСЯ, а не предполагается. Первая версия принимала любой
    словарь с ключом `findings`, и повторное ревью показало, чем это кончается:
    `findings: null` превращался в пустое множество, `findings: {a: false}` — в
    множество ключей. Негодный документ становился ПОЛОЖИТЕЛЬНЫМ доказательством:
    при `fired: false` сверка молчала, потому что находок «не было».

    Пустой список — законное доказательство промолчавшего контроля. Отсутствие
    списка, `null` и отображение — не список, и это разные вещи.
    """
    return _control_run(project / "report" / CONTROL_RUN)[0]


def control_run_defect(project: Path) -> str:
    """Почему доказательство не годится — одной фразой для текста отказа.

    Отказ, называющий «нет либо не по схеме», не различает отсутствующий файл,
    испорченные байты и годный YAML без отпечатков. Прекратить прогон мало:
    причина должна быть названа, иначе отказ читается как поломка инструмента.
    """
    return _control_run(project / "report" / CONTROL_RUN)[1]


def _control_run(path: Path) -> tuple[dict | None, str]:
    """Доказательство и причина непригодности. Пустая причина — документ годен."""
    if not path.is_file():
        return None, "файла доказательства нет"
    try:
        raw = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return None, "файл есть, но он не текст в UTF-8 — читать доказательство нечем"
    try:
        loaded = yaml.safe_load(raw)
    except yaml.YAMLError:
        return None, "файл есть, но он не разбирается как YAML"
    if not isinstance(loaded, dict):
        return None, "верхний уровень доказательства — не отображение"
    if not isinstance(loaded.get("case"), int):
        return None, "в доказательстве нет числового `case` — с каким кейсом его сверять, неясно"
    findings = loaded.get("findings")
    if not isinstance(findings, list) or not all(
        isinstance(item, str) and IDENTIFIER.match(item) for item in findings
    ):
        return None, (
            "`findings` — не список имён находок; `null` и отображение списком не являются "
            "и обращались бы в пустое множество, то есть в положительное доказательство"
        )
    if not all(
        isinstance(loaded.get(field), str) and loaded.get(field) for field in ("form", "manifest")
    ):
        return None, "в доказательстве нет непустых отпечатков `form` и `manifest`"
    return loaded, ""


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
