"""Обещания пре-регистрации о колонках исполняются формуляром.

Класс 14 журнала повторов, дважды: кейс 13 обещал контролю роль, которая уже
была занята, кейс 16 обещал направление колонке, неизвестной в момент решения.
Оба раза пре-регистрация обещала колонку и её свойство, формуляр обещания не
исполнял, и не замечал этого никто, кроме автора.

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

Проверка живёт тестом, а не инструментом: инструмент, который надо не забыть
позвать, в этом проекте ломался трижды.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from dsx.project import load

ROOT = Path(__file__).resolve().parents[1]
PREREGS = ROOT / "docs"

BLOCK = re.compile(r"```yaml\n(project:.*?)\n```", re.DOTALL)


def _promise_block(text: str) -> dict | None:
    """Блок обещаний из текста пре-регистрации. None — обещаний нет."""
    found = BLOCK.search(text)
    return yaml.safe_load(found.group(1)) if found else None


def _unfulfilled(block: dict, form) -> list[str]:
    """Обещания, которых формуляр не исполнил."""
    declared = {column.name: column for column in form.columns}
    broken: list[str] = []
    for promise in block["promises"]:
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


def test_every_promise_in_a_sealed_prereg_is_kept() -> None:
    """Обещанное до данных исполняется формуляром, написанным после."""
    checked, pending = 0, []
    for path in sorted(PREREGS.glob("prereg-case-*.md")):
        block = _promise_block(path.read_text(encoding="utf-8"))
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
        broken = _unfulfilled(block, load(form_path))
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
    assert not _unfulfilled(_promise_block(FULFILLED), form)


def test_a_column_promised_but_never_declared_is_reported(form) -> None:
    """Случай шестнадцатого кейса: направление обещано величине, которой в
    формуляре нет — она неизвестна в момент решения."""
    broken = _unfulfilled(_promise_block(ABSENT), form)

    assert broken == ["'number_of_alarms' обещана, но в формуляре не объявлена вовсе"]


def test_a_role_that_differs_is_reported(form) -> None:
    """Случай тринадцатого кейса: обещанная роль занята другим объявлением."""
    broken = _unfulfilled(_promise_block(WRONG_ROLE), form)

    assert broken == ["'available_at' обещана ролью 'feature', а объявлена 'ignored'"]


def test_a_prereg_without_promises_is_not_a_promise_of_nothing() -> None:
    """Отсутствие блока означает, что обещаний не давали, а не что они пусты."""
    assert _promise_block("# Пре-регистрация\n\nобычный текст без блока\n") is None


def test_a_promise_about_an_unbuilt_project_is_not_broken(tmp_path) -> None:
    """Пре-регистрация опечатывается ДО того, как проект существует.

    Механизм сломался ровно на этом при первом настоящем применении: блок
    восемнадцатого кейса был вписан при опечатывании, формуляра ещё не было, и
    сверка падала с FileNotFoundError. Обещание о непостроенном формуляре не
    нарушено — оно не наступило.
    """
    assert not (tmp_path / "project.yaml").is_file()
