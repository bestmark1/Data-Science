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

import datetime
import hashlib
import re
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictFloat,
    StrictInt,
    StringConstraints,
    ValidationError,
    field_validator,
)

from dsx.roles import Role

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
    keys = _protocol_keys(prereg)
    for found in ANY_BLOCK.finditer(text):
        try:
            block = yaml.safe_load(found.group(1))
        except yaml.YAMLError:
            continue
        if isinstance(block, dict) and any(key in block for key in keys):
            if "project" not in block:
                raise OutOfOrder(
                    f"ОТКАЗ: в {prereg.name} есть машиночитаемый блок протокола "
                    f"({sorted(set(block) & set(keys))}) без `project`. "
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


STANDING, REMOVED, CHANGED = "стоит", "снят", "изменён"

CLEAN_ROLE = "clean_role"
"""Поле §5: роль, которую колонка носит БЕЗ контроля.

Прежде снятием считался переход в `ignored`, зашитый в код одной константой.
Четвёртое ревью показало, что это неверно с двух сторон сразу:

* `ignored` — законная РОЛЬ КОНТРОЛЯ: спрятать колонку, которая должна быть
  признаком, и есть подложенный дефект (`test_prereg_promises`, К-1). Такому
  контролю пути к состоянию «снят» не оставалось вовсе;
* честная роль колонки после снятия не обязана быть `ignored`: возврат
  `observation_reason → feature` — законное снятие, а отвергался как подмена.

Угадать честную роль ядру нечем: `ignored` в качестве умолчания совпадал бы с
честным ответом в части случаев и молча врал бы в остальных. Поэтому она
ОБЪЯВЛЯЕТСЯ, и объявление обязательно — контроль без него не принимается.
"""


def controls_without_clean_role(prereg: Path) -> list[str]:
    """Контроли с колонкой, но без допустимой объявленной исходной роли.

    Умолчание здесь запрещено: оно совпадало бы с честным ответом там, где
    колонка и правда снимается в `ignored`, и было бы неотличимо от
    невнимательности во всех прочих случаях.
    """
    broken = []
    for control in declared(prereg).get("controls", []):
        if not control.get("column"):
            continue
        if CLEAN_ROLE not in control:
            broken.append(f"{control['name']}: нет `{CLEAN_ROLE}`")
            continue

        clean_role = control[CLEAN_ROLE]
        if not isinstance(clean_role, str):
            broken.append(f"{control['name']}: `{CLEAN_ROLE}` не строка роли ({clean_role!r})")
            continue
        try:
            Role(clean_role)
        except ValueError:
            broken.append(
                f"{control['name']}: `{CLEAN_ROLE}` {clean_role!r} не входит в `dsx.roles.Role`"
            )
            continue
        if clean_role == control.get("role"):
            broken.append(
                f"{control['name']}: `{CLEAN_ROLE}` совпадает с ролью контроля "
                f"({control.get('role')!r}) — стоящий контроль неотличим от снятого"
            )
    return broken


def control_states(prereg: Path, form_path: Path) -> dict[str, str]:
    """Состояние каждого объявленного контроля: стоит, снят или изменён.

    Три состояния, а не два. Прежде их было два — «роль совпала» и «не
    совпала», — и второе накрывало собой и законное снятие, и подмену роли на
    любую другую. Подмена называется своим именем и отвергается.

    Контроль без объявленной исходной роли состояния не получает: угадывать её
    нечем, и `controls_without_clean_role` отвергает такой контроль раньше.
    """
    form = yaml.safe_load(form_path.read_text(encoding="utf-8")) or {}
    roles = {column["name"]: column.get("role") for column in form.get("columns", [])}
    states: dict[str, str] = {}
    for control in declared(prereg).get("controls", []):
        if not control.get("column") or CLEAN_ROLE not in control:
            continue
        роль = roles.get(control["column"])
        if роль == control.get("role"):
            states[control["name"]] = STANDING
        elif роль == control[CLEAN_ROLE]:
            states[control["name"]] = REMOVED
        else:
            states[control["name"]] = CHANGED
    return states


def _controls_are_standing(prereg: Path, form_path: Path) -> bool:
    """Стоят ли контроли §5 в формуляре прямо сейчас.

    Прогон с внесённым контролем — контрольный, и его отчёт обязан сохраниться
    отдельно: `report.md` перезаписывается следующим же прогоном, и к моменту
    записи вердикта от контрольного прогона не остаётся ничего, кроме прозы.
    """
    states = control_states(prereg, form_path)
    if not states:
        return False
    # ВСЕ, а не любой. Прежняя версия брала `any`, и один поставленный контроль
    # выдавал за поставленные все: прогон с половиной контролей считался
    # контрольным, а вторая половина не испытывалась вовсе.
    return all(state == STANDING for state in states.values())


def controls_partly_standing(prereg: Path, form_path: Path) -> list[str]:
    """Контроли, объявленные с колонкой, но НЕ стоящие в формуляре."""
    return [name for name, state in control_states(prereg, form_path).items() if state != STANDING]


def controls_changed_beyond_removal(prereg: Path, form_path: Path) -> list[str]:
    """Контроли, чья колонка получила роль, не равную ни объявленной, ни `ignored`."""
    return [name for name, state in control_states(prereg, form_path).items() if state == CHANGED]


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


def evidence_against_artifacts(project: Path, prereg: Path) -> list[str]:
    """Отпечатки доказательства против НАСТОЯЩИХ файлов кейса.

    Отдельной функцией, потому что читателей двое: `preflight` перед прогоном и
    `audit_case_artifacts` при общей проверке. Третье ревью показало, зачем это
    нужно: обход принимал доказательство с `form: x` и `manifest: y` и возвращал
    пустой список. Зелёный обход при таком доказательстве не говорил о привязке
    к форме и данным ничего.

    Годность самого документа здесь НЕ проверяется — это забота вызывающего:
    негодное доказательство и доказательство от другой формы суть разные отказы
    и должны называться по-разному.
    """
    evidence = read_control_run(project)
    if evidence is None:
        return []
    problems: list[str] = []
    if evidence.get("case") != _case_number(prereg):
        problems.append(
            f"доказательство контрольного прогона относится к кейсу "
            f"{evidence.get('case')!r}, а прогоняется {_case_number(prereg)!r} — "
            "отчёт другого кейса снятия не разрешает"
        )
    if evidence.get("manifest") != _fingerprint(project / "manifest.yaml"):
        problems.append(
            "доказательство получено на других данных — отпечаток манифеста не совпадает; "
            "контрольный прогон и чистый обязаны идти по одной выгрузке"
        )
    if evidence.get("form") != _form_identity(project / "project.yaml", _control_roles(prereg)):
        problems.append(
            "форма изменилась не только снятием контролей — доказательство получено на "
            "другой форме; отпечаток слеп лишь к роли контрольной колонки, пока та равна "
            f"объявленной в §5 либо `{CLEAN_ROLE}`"
        )
    return problems


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


ANALOGS = "analogs.md"
"""Второй этап поиска аналогов: решения на ТОМ ЖЕ наборе, после опечатывания."""

ANALOG_SOURCES = ("github", "kaggle", "openml")

ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")

_Text = Annotated[str, StringConstraints(strict=True, strip_whitespace=True, min_length=1)]
"""Непустая строка. `true`, число или пробелы на месте текста — не текст."""

_Asked = Annotated[list[_Text], Field(min_length=1)]


class _Form(BaseModel):
    """Строгая форма: точные типы, лишние поля — ошибка (опечатка в имени тоже)."""

    model_config = ConfigDict(strict=True, extra="forbid")


class _Queries(_Form):
    github: _Asked
    kaggle: _Asked
    openml: _Asked


class _Reference(_Form):
    """Опорный результат вместе с условиями, в которых он получен."""

    metric: _Text
    value: StrictInt | Annotated[StrictFloat, Field(allow_inf_nan=False)]
    split: Literal["random", "time", "group", "unknown"]
    population: _Text


class _Found(_Form):
    url: _Text
    reference_result: _Reference | None = None
    """Может отсутствовать: не всякий аналог публикует результат."""


class _Decision(_Form):
    url: _Text
    reason: _Text


class _Analogs(_Form):
    date: object
    queries: _Queries
    found: list[_Found]
    taken: list[_Decision]
    rejected: list[_Decision]
    pitfalls: list[_Text]

    @field_validator("date")
    @classmethod
    def _iso_date(cls, value: object) -> object:
        """Дата ГГГГ-ММ-ДД. YAML читает её как `date`; строку — только этого вида.

        `date.fromisoformat` принимает и '20260929', поэтому вид сверяется отдельно.
        `datetime` — подкласс `date` и отвергается: нужна дата, а не момент.
        """
        if type(value) is datetime.date:
            return value
        if isinstance(value, str) and ISO_DATE.fullmatch(value):
            datetime.date.fromisoformat(value)
            return value
        raise ValueError("не дата ГГГГ-ММ-ДД")


def analogs_defects(block: object) -> list[str]:
    """Чем запись об аналогах не дотягивает до доказательства поиска.

    ДВА ЭТАПА. Первый — аналоги на ДРУГИХ наборах той же задачи — пишется в
    пре-регистрацию и опечатывается вместе с ней: дата, вписанная автором, ничего
    бы не удостоверила, а опечатывание уже есть. Второй — решения на том же
    наборе — читается только после опечатывания: чужие ноутбуки показывают
    значения колонок и известные утечки, и предсказания подгонялись бы под них.

    ЧЕГО ЭТО НЕ ДАЁТ. Проверяется форма записи, а не честность поиска. Запросы
    можно выдумать, найденное — не записать. Пустой список запросов, однако, не
    проходит: «ничего не найдено» без запросов неотличимо от «не искал».

    ФОРМА — СХЕМОЙ, А НЕ ЗАПЛАТАМИ. Первая версия проверяла наличие полей и
    читала `null` как пустой список; ревью PR #1 нашло это, и типы были
    залатаны поле за полем. Повторное ревью PR #2 нашло следующие дыры того же
    рода — `date: '20260929'` и `reference_result: {metric: true, value: false,
    split: []}` без `population`. Заплата на каждое поле оставляет непроверенным
    соседнее, поэтому форма задана строгой схемой: точные типы, «ничего» — это
    `[]`, лишние и опечатанные поля — ошибка.
    """
    if not isinstance(block, dict):
        return ["блока `analogs` нет"]
    try:
        record = _Analogs.model_validate(block)
    except ValidationError as exc:
        return [
            f"`{'.'.join(str(part) for part in error['loc'])}`: {error['msg']}"
            for error in exc.errors()
        ]
    decided = {entry.url for entry in (*record.taken, *record.rejected)}
    return [
        f"аналог {item.url} найден, но не взят и не отвергнут"
        for item in record.found
        if item.url not in decided
    ]


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

    первый_этап = analogs_defects(declared(prereg).get("analogs"))
    if первый_этап:
        raise OutOfOrder(
            f"ОТКАЗ: в {prereg.name} нет годной записи об аналогах на других наборах — "
            f"{'; '.join(первый_этап)}. Первый этап поиска опечатывается вместе с "
            "пре-регистрацией. Построитель не вызывался."
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

    записка = project / ANALOGS
    if not записка.is_file():
        raise OutOfOrder(
            f"ОТКАЗ: второй этап поиска аналогов не записан — нет {записка}. Решения "
            "на том же наборе читаются после опечатывания и до первого прогона. "
            "Построитель не вызывался."
        )
    второй = declared(записка)
    if второй.get("project") != project.name:
        raise OutOfOrder(
            f"ОТКАЗ: {записка.name} объявляет проект {второй.get('project')!r}, а "
            f"прогоняется {project.name!r}. Построитель не вызывался."
        )
    второй_этап = analogs_defects(второй.get("analogs"))
    if второй_этап:
        raise OutOfOrder(
            f"ОТКАЗ: запись второго этапа поиска аналогов негодна — "
            f"{'; '.join(второй_этап)}. Построитель не вызывался."
        )

    # ПОРЯДОК ВЕТВЕЙ ЗДЕСЬ ЗНАЧИМ, и он проверяется тестами. Сначала — пригодно ли
    # объявление контроля вообще (есть исходная роль), затем — в каком состоянии
    # колонка, и лишь затем — достижима ли ожидаемая находка.
    #
    # Прежде `unreachable_controls` стоял первым, и подмена роли, совпавшая с
    # недостижимой находкой, получала отказ «недостижима» вместо «не снятие
    # контроля». Спецификация при этом обещала обратное — «до всех прочих
    # ветвей». Найдено четвёртым ревью.
    безосновательные = controls_without_clean_role(prereg)
    if безосновательные:
        raise OutOfOrder(
            f"ОТКАЗ: контроли объявлены без исходной роли — {'; '.join(безосновательные)}. "
            f"Без `{CLEAN_ROLE}` снятие контроля неотличимо от подмены роли, а угадать "
            "честную роль колонки ядру нечем. Построитель не вызывался."
        )

    подменены = controls_changed_beyond_removal(prereg, project / "project.yaml")
    if подменены:
        raise OutOfOrder(
            f"ОТКАЗ: контрольным колонкам {подменены} назначена роль, не равная ни "
            f"объявленной в §5, ни `{CLEAN_ROLE}`. Это не снятие контроля, а смена "
            "смысла колонки: прогон пойдёт по другой форме, чем объявлено. "
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
        расхождения = evidence_against_artifacts(project, prereg)
        if расхождения:
            raise OutOfOrder("ОТКАЗ: " + "; ".join(расхождения) + ". Построитель не вызывался.")
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


def guarded(project: Path, prereg: Path, build: Callable[[], object]) -> object:
    """Штатный путь кейса: порядок проверяется, и лишь потом зовётся построитель.

    ЗАЧЕМ ОТДЕЛЬНАЯ ФУНКЦИЯ. «Построитель не вызывается» до сих пор держалось на
    дисциплине автора: он обязан был позвать `preflight` сам и раньше сборки.
    Тест со счётчиком, ставивший вызов построителя строкой НИЖЕ `preflight`
    внутри `pytest.raises`, не проверял ничего: исключение прерывает блок само,
    и счётчик остался бы нулевым при любой реализации, включая пустую. Найдено
    третьим ревью.

    Здесь порядок задан кодом: `build` недостижим иначе как через `preflight`, и
    это можно проверить обоими исходами — отказом и разрешением.

    НАЗВАННЫЙ ПРЕДЕЛ. Ни один кейс этой функцией пока не пользуется: кейсы 2–20
    старше механизма, двадцать первый не начат. И обход остаётся возможен —
    построитель зовётся напрямую тем, у кого есть доступ к коду.
    """
    preflight(project, prereg)
    return build()


PROTOCOL_KEYS = ("controls", "promises", "spent_controls", "lost_bets", "predictions", "analogs")

FORWARD_KEYS = ("analogs",)
"""Ключи, введённые с `FROM_CASE`: в документах старших кейсов блоком протокола не считаются.

Независимое ревью PR #1: добавление `analogs` в `PROTOCOL_KEYS` действовало при
разборе ДО исторического выхода, и пре-регистрация двадцатого кейса с блоком
`analogs: {}` без `project` получала отказ, которого прежде не было. В настоящих
документах 2–20 такого блока нет, но правило вводится вперёд, а не назад.
"""


def _protocol_keys(path: Path) -> tuple[str, ...]:
    """Ключи протокола для документа: у пре-регистраций старше `FROM_CASE` — без новых."""
    number = _case_number(path)
    if number is not None and number < FROM_CASE:
        return tuple(key for key in PROTOCOL_KEYS if key not in FORWARD_KEYS)
    return PROTOCOL_KEYS


def _form_identity(form_path: Path, control_roles: dict[str, str]) -> str:
    """Отпечаток формы, слепой РОВНО к одному разрешённому переходу.

    Законное снятие контроля меняет форму — и сравнивать её отпечаток целиком
    нельзя: чистый прогон обязан отличаться от контрольного. Но отличаться он
    обязан ТОЛЬКО этим.

    Поэтому роль контрольной колонки опускается, но лишь пока она равна одной из
    ДВУХ объявленных: роли контроля либо исходной роли `clean_role`. Всякая
    третья в отпечаток входит и его меняет. Прежняя версия опускала роль
    контрольной колонки безусловно, и перевод `feature → outcome` проходил как
    снятие — блокирующая находка третьего ревью.
    """
    form = yaml.safe_load(form_path.read_text(encoding="utf-8")) or {}
    columns = []
    for column in form.get("columns", []):
        разрешённые = control_roles.get(column.get("name"))
        снимаемая = разрешённые is not None and column.get("role") in разрешённые
        columns.append({k: v for k, v in column.items() if not (снимаемая and k == "role")})
    skeleton = {**{k: v for k, v in form.items() if k != "columns"}, "columns": columns}
    return hashlib.sha256(
        yaml.safe_dump(skeleton, allow_unicode=True, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]


def _control_roles(prereg: Path) -> dict[str, tuple]:
    """Колонка контроля → пара ролей, между которыми переход разрешён."""
    return {
        c["column"]: (c.get("role"), c.get(CLEAN_ROLE))
        for c in declared(prereg).get("controls", [])
        if c.get("column") and CLEAN_ROLE in c
    }


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
                "form": _form_identity(project / "project.yaml", _control_roles(prereg)),
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
    # `bool` — подкласс `int`, и `case: true` проходил проверку на целое. Ветвь
    # обещала «case не целое» и обещания не исполняла: документ признавался
    # годным, а отказ приходил позже и о другом. Найдено четвёртым ревью.
    case = loaded.get("case")
    if isinstance(case, bool) or not isinstance(case, int):
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
            f"{prereg.name}: {item}" for item in evidence_against_artifacts(project, prereg)
        ]
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
