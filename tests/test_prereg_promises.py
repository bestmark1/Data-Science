"""Обещания и контроли пре-регистрации исполняются формуляром.

Класс 14 журнала повторов, трижды: кейсы 13 и 18 обещали контролю роль, которая
уже была занята, кейс 16 обещал направление колонке, неизвестной в момент
решения. Каждый раз пре-регистрация обещала колонку и её свойство, формуляр
обещания не исполнял, и не замечал этого никто, кроме автора.

Механизм узок намеренно. Сверять два документа целиком нельзя: пре-регистрация
обязана оставаться человеческим текстом — её читает человек и опечатывает до
работы. Но обещания О КОЛОНКАХ можно записать машиночитаемо, и это ровно то, что
дважды расходилось. Ни популяция, ни пороги, ни оси за пять кейсов не расходились
ни разу, и обещаний о них здесь нет.

Как записать обещание. В пре-регистрацию добавляется огороженный блок:

    ```yaml
    project: fire-response
    promises:
      - {column: at_night, role: feature, direction: increases}
      - {column: original_priority, role: observation_reason}
    ```

Свойств два — роль и направление, — потому что расходились именно они. Появится
третий род расхождения — добавится третье свойство, не раньше.

КОНТРОЛИ §5 добавлены после восемнадцатого кейса. Класс повторился ТРЕТИЙ раз
и второй раз одинаково: кейсы 13 и 18 оба объявили контролем колонку, роль
которой уже занята контрактом исхода. Первая версия механизма этого не видела —
она сверяла только блок обещаний §3а, а контроли лежат в §5 и в блок не входили.

Контроль объявляет колонку, роль, которую он ей даёт, и ожидаемую находку.
Формуляр обязан роль исполнить. Объявить контроль НЕИСПОЛНИМЫМ можно, но только
когда расхождение действительно есть: отговорка, прикрывающая исполнимый
контроль, — то же необеспеченное объявление, и она ловится отдельно.

Ожидаемая находка сверяется с каталогом ядра. Контроль, ждущий того, чего ядро
не умеет находить, не может сработать никогда, и молчание по нему неотличимо от
слепоты.

Проверка живёт тестом, а не инструментом: инструмент, который надо не забыть
позвать, в этом проекте ломался трижды.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from dsx.evals.case import Finding
from dsx.project import load

ROOT = Path(__file__).resolve().parents[1]
PREREGS = ROOT / "docs"

BLOCK = re.compile(r"```yaml\n(project:.*?)\n```", re.DOTALL)


def _declared_block(text: str) -> dict | None:
    """Машиночитаемые объявления пре-регистрации. None — их нет.

    Блоков может быть несколько: обещания живут в §3а, контроли в §5. Брать
    первый и молчать об остальных значило бы терять объявленное — ровно то, из-за
    чего механизм и не увидел повтора в восемнадцатом кейсе.
    """
    merged: dict = {}
    for found in BLOCK.finditer(text):
        block = yaml.safe_load(found.group(1))
        for key, value in block.items():
            if key in merged and merged[key] != value:
                if isinstance(value, list):
                    merged[key] = merged[key] + value
                    continue
                raise ValueError(f"объявление {key!r} повторено с другим значением")
            merged[key] = value
    return merged or None


def _unfulfilled(block: dict, form) -> list[str]:
    """Обещания, которых формуляр не исполнил."""
    declared = {column.name: column for column in form.columns}
    broken: list[str] = []
    # Пре-регистрация вправе не давать обещаний о колонках и дать одни контроли:
    # блоки живут в разных параграфах и друг друга не требуют.
    for promise in block.get("promises", []):
        name = promise["column"]
        column = declared.get(name)
        if column is None:
            broken.append(f"{name!r} обещана, но в формуляре не объявлена вовсе")
            continue
        if "role" in promise and column.role.value != promise["role"]:
            broken.append(
                f"{name!r} обещана ролью {promise['role']!r}, а объявлена {column.role.value!r}"
            )
        if "direction" in promise:
            actual = column.direction.value if column.direction else None
            if actual != promise["direction"]:
                broken.append(
                    f"{name!r} обещана направлением {promise['direction']!r}, "
                    f"а объявлена {actual!r}"
                )
    return broken


def _broken_controls(block: dict, form) -> list[str]:
    """Контроли §5, которых формуляр не исполнил.

    Контроль без колонки не сверяется: он говорит о значениях в данных, а не об
    объявлении, и формуляру исполнять нечего.
    """
    declared = {column.name: column for column in form.columns}
    broken: list[str] = []
    for control in block.get("controls", []):
        name = control["name"]
        column_name = control.get("column")
        if column_name is None:
            continue
        column = declared.get(column_name)
        wanted = control.get("role")
        actual = column.role.value if column is not None else None
        excuse = control.get("unfulfilled")
        if actual == wanted:
            if excuse:
                # Отговорка, прикрывающая исполнимый контроль, страшнее
                # неисполненного контроля: она объявляет невозможность, которой
                # нет, и снимает вопрос, не ответив на него.
                broken.append(
                    f"{name}: объявлен неисполнимым ({excuse!r}), но {column_name!r} "
                    f"ДЕЙСТВИТЕЛЬНО объявлена ролью {wanted!r} — отговорка не обеспечена"
                )
        elif not excuse:
            # Колонки может не быть вовсе — построитель волен её переименовать, и
            # ядро о переименовании не знает. Сказать про такую «объявлена None»
            # значит назвать отсутствие значением.
            found = (
                f"а объявлена {actual!r}"
                if column is not None
                else "а в формуляре не объявлена вовсе"
            )
            broken.append(f"{name}: требует {column_name!r} ролью {wanted!r}, {found}")
    return broken


def _unknown_findings(block: dict) -> list[str]:
    """Контроли, ждущие того, чего ядро находить не умеет."""
    known = {finding.value for finding in Finding}
    return [
        f"{control['name']}: ждёт находки {control.get('expect')!r}, которой в каталоге ядра нет"
        for control in block.get("controls", [])
        if control.get("expect") not in known
    ]


def test_every_promise_in_a_sealed_prereg_is_kept() -> None:
    """Обещанное до данных исполняется формуляром, написанным после."""
    checked, pending = 0, []
    for path in sorted(PREREGS.glob("prereg-case-*.md")):
        block = _declared_block(path.read_text(encoding="utf-8"))
        if block is None:
            continue
        form_path = ROOT / "projects" / block["project"] / "project.yaml"
        # Пре-регистрация опечатывается ДО того, как проект существует: в этом
        # весь её смысл. Обещание о непостроенном формуляре не нарушено — оно
        # не наступило, и путать одно с другим значит объявлять нарушением
        # порядок работы.
        #
        # Молчать о таких тоже нельзя: обещание, которое никогда не наступит,
        # неотличимо от исполненного. Поэтому они называются вслух.
        if not form_path.is_file():
            pending.append(f"{path.name} -> {block['project']}")
            continue
        broken = _unfulfilled(block, load(form_path)) + _broken_controls(block, load(form_path))
        assert not broken, f"{path.name}: {'; '.join(broken)}"
        checked += 1
    print(f"обещаний сверено: {checked}, ещё не наступило: {len(pending)}")
    for item in pending:
        print(f"  ждёт формуляра: {item}")


# --- проверка самой проверки ------------------------------------------------
#
# Механизм, которому нечего проверять, неотличим от сломанного. Ниже он
# запускается на выдуманных пре-регистрациях, где ответ известен.

FULFILLED = """
```yaml
project: fire-response
promises:
  - {column: at_night, role: feature, direction: increases}
  - {column: original_priority, role: observation_reason}
```
"""

ABSENT = """
```yaml
project: fire-response
promises:
  - {column: number_of_alarms, role: feature, direction: increases}
```
"""

WRONG_ROLE = """
```yaml
project: fire-response
promises:
  - {column: available_at, role: feature}
```
"""


@pytest.fixture
def form():
    return load(ROOT / "projects" / "fire-response" / "project.yaml")


def test_a_kept_promise_is_silent(form) -> None:
    assert not _unfulfilled(_declared_block(FULFILLED), form)


def test_a_column_promised_but_never_declared_is_reported(form) -> None:
    """Случай шестнадцатого кейса: направление обещано величине, которой в
    формуляре нет — она неизвестна в момент решения."""
    broken = _unfulfilled(_declared_block(ABSENT), form)

    assert broken == ["'number_of_alarms' обещана, но в формуляре не объявлена вовсе"]


def test_a_role_that_differs_is_reported(form) -> None:
    """Случай тринадцатого кейса: обещанная роль занята другим объявлением."""
    broken = _unfulfilled(_declared_block(WRONG_ROLE), form)

    assert broken == ["'available_at' обещана ролью 'feature', а объявлена 'ignored'"]


def test_a_prereg_without_promises_is_not_a_promise_of_nothing() -> None:
    """Отсутствие блока означает, что обещаний не давали, а не что они пусты."""
    assert _declared_block("# Пре-регистрация\n\nобычный текст без блока\n") is None


def test_a_promise_about_an_unbuilt_project_is_not_broken(tmp_path) -> None:
    """Пре-регистрация опечатывается ДО того, как проект существует.

    Механизм сломался ровно на этом при первом настоящем применении: блок
    восемнадцатого кейса был вписан при опечатывании, формуляра ещё не было, и
    сверка падала с FileNotFoundError. Обещание о непостроенном формуляре не
    нарушено — оно не наступило.
    """
    assert not (tmp_path / "project.yaml").is_file()


# --- контроли §5 ------------------------------------------------------------

CONTROL_KEPT = """
```yaml
project: fire-response
controls:
  - {name: К-1, column: available_at, role: ignored, expect: value_revised_after_decision}
  - {name: К-2, expect: sentinel_as_value}
```
"""

CONTROL_ROLE_TAKEN = """
```yaml
project: fire-response
controls:
  - {name: К-1, column: available_at, role: feature, expect: value_revised_after_decision}
```
"""

CONTROL_HONESTLY_UNFULFILLED = """
```yaml
project: fire-response
controls:
  - {name: К-1, column: available_at, role: feature, expect: value_revised_after_decision,
     unfulfilled: "роль занята другим объявлением"}
```
"""

CONTROL_FALSE_EXCUSE = """
```yaml
project: fire-response
controls:
  - {name: К-1, column: available_at, role: ignored, expect: value_revised_after_decision,
     unfulfilled: "роль занята другим объявлением"}
```
"""

CONTROL_INVENTED_FINDING = """
```yaml
project: fire-response
controls:
  - {name: К-1, column: available_at, role: ignored, expect: выдуманная_находка}
```
"""


def test_a_control_the_form_fulfils_is_silent(form) -> None:
    assert not _broken_controls(_declared_block(CONTROL_KEPT), form)


def test_a_control_demanding_a_taken_role_is_reported(form) -> None:
    """Случай кейсов 13 и 18: контроль требует роль, которой у колонки нет."""
    broken = _broken_controls(_declared_block(CONTROL_ROLE_TAKEN), form)

    assert broken == ["К-1: требует 'available_at' ролью 'feature', а объявлена 'ignored'"]


def test_an_honestly_unfulfilled_control_is_allowed(form) -> None:
    """Неисполнимость записать можно — она и есть честный исход кейса 18."""
    assert not _broken_controls(_declared_block(CONTROL_HONESTLY_UNFULFILLED), form)


def test_an_excuse_covering_a_fulfillable_control_is_reported(form) -> None:
    """Отговорка, прикрывающая исполнимый контроль, — необеспеченное объявление.

    Без этой проверки механизм обходился бы одним словом: приписать
    `unfulfilled` любому контролю и не исполнять ни одного.
    """
    broken = _broken_controls(_declared_block(CONTROL_FALSE_EXCUSE), form)

    assert len(broken) == 1
    assert "отговорка не обеспечена" in broken[0]


def test_a_control_awaiting_an_unknown_finding_is_reported() -> None:
    """Контроль, ждущий несуществующей находки, не сработает никогда."""
    unknown = _unknown_findings(_declared_block(CONTROL_INVENTED_FINDING))

    assert unknown == ["К-1: ждёт находки 'выдуманная_находка', которой в каталоге ядра нет"]


def test_every_control_awaits_a_finding_the_core_knows() -> None:
    """По всем опечатанным пре-регистрациям сразу: сверка не требует формуляра."""
    unknown: list[str] = []
    for path in sorted(PREREGS.glob("prereg-case-*.md")):
        block = _declared_block(path.read_text(encoding="utf-8"))
        if block is None:
            continue
        unknown += [f"{path.name}: {item}" for item in _unknown_findings(block)]
    assert not unknown, "; ".join(unknown)


CONTROL_ABOUT_A_MISSING_COLUMN = """
```yaml
project: fire-response
controls:
  - {name: К-1, column: number_of_alarms, role: feature, expect: value_revised_after_decision}
```
"""


def test_a_control_about_a_column_the_form_lacks_is_reported(form) -> None:
    """Случай восемнадцатого кейса: контроль назвал имя из ИСТОЧНИКА.

    Построитель волен переименовать колонку, и ядро о переименовании не знает.
    Расхождение всё равно названо, но названо тем, чем оно является, —
    отсутствием, а не ролью `None`.
    """
    broken = _broken_controls(_declared_block(CONTROL_ABOUT_A_MISSING_COLUMN), form)

    assert broken == [
        "К-1: требует 'number_of_alarms' ролью 'feature', а в формуляре не объявлена вовсе"
    ]
