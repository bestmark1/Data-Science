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


def _lost_bets(block: dict) -> dict[str, str]:
    """Обещания, объявленные ПРОИГРАВШИМИ, и причина каждого.

    Обещание §3а — ставка, и пре-регистрация называет её ставкой прямо. Ставка,
    которая не может проиграть, ставкой не является: механизм, требующий, чтобы
    все обещания исполнились, запрещает их давать о том, чего автор не знает, —
    а именно о таком их и дают.

    Двадцатый кейс: `location_type` обещан признаком, а оказался следствием
    исхода — значение «Well (Construction Report)» означает, что запись создана
    по отчёту о строительстве. Ядро нашло связь силой 0.50 при типичной 0.03.
    Исполнить обещание значило бы оставить утечку в выборке.

    Запись о проигрыше вносится ПОСЛЕ кейса отдельным блоком и опечатанного
    объявления не трогает. Обеспеченность проверяется так же, как у отговорки
    контроля: объявить ставку проигравшей можно, только когда формуляр
    действительно разошёлся с ней.
    """
    return {
        item["column"]: str(item.get("reason", "")).strip() for item in block.get("lost_bets", [])
    }


def _unfulfilled(block: dict, form) -> list[str]:
    """Обещания, которых формуляр не исполнил."""
    declared = {column.name: column for column in form.columns}
    lost = _lost_bets(block)
    broken: list[str] = []
    promised = {promise["column"] for promise in block.get("promises", [])}

    # Проигрыш, объявленный об обещании, которого не было. Проверку он не
    # отключает, но остаётся записью, за которой ничего не стоит, — а такие
    # записи проект запрещает наравне с ложными отговорками.
    for name in lost:
        if name not in promised:
            broken.append(f"{name!r} объявлена проигравшей ставкой, но обещания о ней не было")

    for promise in block.get("promises", []):
        name = promise["column"]
        column = declared.get(name)
        if column is None:
            broken.append(f"{name!r} обещана, но в формуляре не объявлена вовсе")
            continue

        # Расхождения считаются ПО ВСЕМ свойствам сразу, и лишь потом решается,
        # прикрыто ли расхождение объявленным проигрышем. Прежняя версия
        # смотрела только на роль: обещание, разошедшееся НАПРАВЛЕНИЕМ при
        # верной роли, она объявляла исполненным — то есть выдавала ложную
        # тревогу о самом проигрыше и молчала о настоящем расхождении.
        differences: list[str] = []
        if "role" in promise and column.role.value != promise["role"]:
            differences.append(
                f"{name!r} обещана ролью {promise['role']!r}, а объявлена {column.role.value!r}"
            )
        if "direction" in promise:
            actual = column.direction.value if column.direction else None
            if actual != promise["direction"]:
                differences.append(
                    f"{name!r} обещана направлением {promise['direction']!r}, "
                    f"а объявлена {actual!r}"
                )

        if lost.get(name):
            if not differences:
                broken.append(
                    f"{name!r} объявлена проигравшей ставкой, но формуляр обещание ИСПОЛНИЛ"
                )
            continue  # ставка объявлена проигравшей, и причина записана
        broken += differences
    return broken


def _spent_controls(block: dict) -> dict[str, str]:
    """Контроли, отработавшие и снятые с колонки, и исход каждого.

    Контроль живёт ОДИН прогон: он вносится в формуляр, ядро о нём возражает,
    после чего роль возвращается на место и делается чистый прогон. Объявление
    же остаётся в пре-регистрации навсегда, и сверка, не знающая о снятии,
    краснеет вечно — начиная со следующего дня после кейса.

    До двадцатого кейса этого не замечали: контроли кейса 19 колонок не
    называли, а кейсы 13 и 18 объявили свои неисполнимыми. Здесь К-2 назвал
    колонку, отработал и был снят — и сверка потребовала вернуть его обратно.

    Обеспеченность та же, что у отговорки: объявить контроль отработавшим
    можно, только если формуляр действительно с ним разошёлся.
    """
    return {
        item["name"]: str(item.get("outcome", "")).strip()
        for item in block.get("spent_controls", [])
    }


def _controls_without_verdict(block: dict) -> list[str]:
    """Снятые контроли, чей исход записан только прозой.

    `outcome` — человеческий текст, и «сработал» с «промолчал» для машины в нём
    одинаковы. Контроль К-2 двадцатого кейса промолчал; проза это говорит,
    учёту правила остановки — нечем прочитать.

    Отсюда обязательное `fired: true|false` рядом с прозой. Оно не заменяет
    объяснения и не проверяет его правдивость — оно делает исход СЧИТЫВАЕМЫМ,
    чтобы «контроль снят» перестало быть неотличимо от «контроль сработал».
    """
    missing: list[str] = []
    for item in block.get("spent_controls", []):
        if not isinstance(item.get("fired"), bool):
            missing.append(
                f"{item['name']}: снятие объявлено, но не сказано, СРАБОТАЛ ли контроль — "
                "нужно поле fired: true|false рядом с outcome"
            )
    return missing


def _broken_controls(block: dict, form) -> list[str]:
    """Контроли §5, которых формуляр не исполнил.

    Контроль без колонки не сверяется: он говорит о значениях в данных, а не об
    объявлении, и формуляру исполнять нечего.
    """
    declared = {column.name: column for column in form.columns}
    spent = _spent_controls(block)
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
            if spent.get(name):
                broken.append(
                    f"{name}: объявлен отработавшим и снятым, но {column_name!r} "
                    f"ВСЁ ЕЩЁ объявлена ролью {wanted!r} — снятие не обеспечено"
                )
            if excuse:
                # Отговорка, прикрывающая исполнимый контроль, страшнее
                # неисполненного контроля: она объявляет невозможность, которой
                # нет, и снимает вопрос, не ответив на него.
                broken.append(
                    f"{name}: объявлен неисполнимым ({excuse!r}), но {column_name!r} "
                    f"ДЕЙСТВИТЕЛЬНО объявлена ролью {wanted!r} — отговорка не обеспечена"
                )
        elif spent.get(name):
            continue  # контроль отработал и снят с колонки, исход записан
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


GROUNDS_FROM_CASE = 19
"""Кейс, с которого контроль обязан объявлять основание применимости.

Задним числом правило не применяется: пре-регистрации 13–18 опечатаны без него,
и переписывать опечатанное — тот самый подбор под результат.
"""

CASE_NUMBER = re.compile(r"prereg-case-(\d+)\.md$")
COUNTED = re.compile(r"^counted:.*\d")


def _groundless_controls(block: dict) -> list[str]:
    """Контроли, о применимости которых ничего не объявлено.

    Счёт по кейсам 8–18, где контроли объявлялись: хотя бы один оказался
    НЕПРИМЕНИМ в шести кейсах из одиннадцати — 10, 11, 12, 15, 17, 18. Каждый
    раз это был контроль, опиравшийся на свойство данных, —
    «отсутствие, записанное значением», «строки с исходом раньше приёма», — то
    есть надежда, а не действие. Автор объявлял его до данных и узнавал после
    сбора, что подкладывать было нечего.

    Механизм сверки §5 такого не видел: он проверяет ИСПОЛНИМОСТЬ — свободна ли
    роль, — и молчит о ПРИМЕНИМОСТИ, которая живёт в значениях.

    Отсюда два законных основания, и третьего нет:

    * `plant` — контроль вносится действием автора и применим по построению;
    * `counted: ...` — контроль опирается на свойство данных, и это свойство
      сосчитано ДО опечатывания. Счёт без числа счётом не является.

    Контроль, для которого ни то ни другое невозможно, не объявляется вовсе:
    объявленный, он тратит место в §5 и создаёт видимость испытания.
    """
    groundless: list[str] = []
    for control in block.get("controls", []):
        basis = str(control.get("basis", "")).strip()
        if basis == "plant":
            continue
        if COUNTED.match(basis):
            continue
        groundless.append(
            f"{control['name']}: основание применимости не объявлено ({basis!r}); "
            "нужно 'plant' либо 'counted: <счёт с числом, сделанный до опечатывания>'"
        )
    return groundless


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


# --- Основание применимости контроля ----------------------------------------

CONTROL_PLANTED = """
```yaml
project: fire-response
controls:
  - {name: К-1, column: at_night, role: feature, expect: value_revised_after_decision,
     basis: plant}
```
"""

CONTROL_COUNTED = """
```yaml
project: fire-response
controls:
  - {name: К-2, expect: sentinel_as_value,
     basis: "counted: 'None' в районах — 121 строка, счёт до опечатывания"}
```
"""

CONTROL_HOPEFUL = """
```yaml
project: fire-response
controls:
  - {name: К-2, expect: sentinel_as_value}
```
"""

CONTROL_COUNT_WITHOUT_A_NUMBER = """
```yaml
project: fire-response
controls:
  - {name: К-2, expect: sentinel_as_value, basis: "counted: такие значения в данных есть"}
```
"""


def test_a_planted_control_needs_no_count() -> None:
    """Контроль, вносимый действием, применим по построению."""
    assert not _groundless_controls(_declared_block(CONTROL_PLANTED))


def test_a_counted_control_is_grounded_by_its_number() -> None:
    """Контроль о свойстве данных законен, когда свойство сосчитано заранее."""
    assert not _groundless_controls(_declared_block(CONTROL_COUNTED))


def test_a_hopeful_control_is_reported() -> None:
    """Контроль без основания — надежда, и в шести кейсах из одиннадцати она не сбылась."""
    groundless = _groundless_controls(_declared_block(CONTROL_HOPEFUL))

    assert len(groundless) == 1
    assert "основание применимости не объявлено" in groundless[0]


def test_a_count_without_a_number_is_not_a_count() -> None:
    """Утверждение требует показанного расчёта: «такие значения есть» им не является."""
    assert len(_groundless_controls(_declared_block(CONTROL_COUNT_WITHOUT_A_NUMBER))) == 1


def test_every_control_since_case_19_declares_its_grounds() -> None:
    """Рабочий путь: пре-регистрации с девятнадцатого кейса обязаны нести основание.

    Прежние опечатаны без этого правила и задним числом не переписываются.
    """
    groundless: list[str] = []
    for path in sorted(PREREGS.glob("prereg-case-*.md")):
        number = CASE_NUMBER.search(path.name)
        if number is None or int(number.group(1)) < GROUNDS_FROM_CASE:
            continue
        block = _declared_block(path.read_text(encoding="utf-8"))
        if block is None:
            continue
        groundless += [f"{path.name}: {item}" for item in _groundless_controls(block)]
    assert not groundless, "; ".join(groundless)


# --- Проигравшая ставка и отработавший контроль ------------------------------
#
# Двадцатый кейс: обещание §3а названо ставкой, а механизм требовал, чтобы все
# ставки выигрывали. Контроль живёт один прогон, а объявление о нём — вечно.

LOST_BET = """
```yaml
project: fire-response
promises:
  - {column: at_night, role: observation_reason}
lost_bets:
  - {column: at_night, reason: "проиграна: колонка оказалась следствием исхода"}
```
"""

LOST_BET_WITHOUT_REASON = """
```yaml
project: fire-response
promises:
  - {column: at_night, role: observation_reason}
lost_bets:
  - {column: at_night}
```
"""

BET_DECLARED_LOST_BUT_KEPT = """
```yaml
project: fire-response
promises:
  - {column: at_night, role: feature}
lost_bets:
  - {column: at_night, reason: "проиграна"}
```
"""


def test_a_lost_bet_with_a_reason_is_silent(form) -> None:
    """Ставка вправе проиграть: иначе её нельзя делать о неизвестном."""
    assert not _unfulfilled(_declared_block(LOST_BET), form)


def test_a_lost_bet_without_a_reason_is_reported(form) -> None:
    """Проигрыш без причины — объявление, которое ничего не объясняет."""
    broken = _unfulfilled(_declared_block(LOST_BET_WITHOUT_REASON), form)

    assert len(broken) == 1
    assert "обещана ролью" in broken[0]


def test_a_bet_declared_lost_while_kept_is_reported(form) -> None:
    """Отрицательный контроль: проигрыш объявлен там, где расхождения нет.

    Такая запись снимает вопрос, не ответив на него, — то же необеспеченное
    объявление, что и отговорка у исполнимого контроля.
    """
    broken = _unfulfilled(_declared_block(BET_DECLARED_LOST_BUT_KEPT), form)

    assert len(broken) == 1
    assert "формуляр обещание ИСПОЛНИЛ" in broken[0]


SPENT_CONTROL = """
```yaml
project: fire-response
controls:
  - {name: К-1, column: at_night, role: observation_reason,
     expect: value_revised_after_decision, basis: plant}
spent_controls:
  - {name: К-1, outcome: "сработал и снят: колонка возвращена в feature"}
```
"""

SPENT_CONTROL_STILL_IN_PLACE = """
```yaml
project: fire-response
controls:
  - {name: К-1, column: at_night, role: feature,
     expect: value_revised_after_decision, basis: plant}
spent_controls:
  - {name: К-1, outcome: "сработал и снят"}
```
"""


def test_a_spent_control_is_silent(form) -> None:
    """Контроль отработал, роль возвращена — сверке возражать не о чем."""
    assert not _broken_controls(_declared_block(SPENT_CONTROL), form)


def test_a_control_declared_spent_while_still_in_place_is_reported(form) -> None:
    """Отрицательный контроль: снятие объявлено, а контроль стоит.

    Тогда прогон идёт с внесённым дефектом, а отчёт называет его снятым — и
    измеренное описывает испорченную выборку.
    """
    broken = _broken_controls(_declared_block(SPENT_CONTROL_STILL_IN_PLACE), form)

    assert len(broken) == 1
    assert "снятие не обеспечено" in broken[0]


# --- Обходы, найденные проверкой механизма после кейса 20 --------------------
#
# Оба состояния ниже механизм пропускал: первое молча, второе — с ложным
# сигналом о том, что обещание исполнено.

LOST_BET_WITHOUT_A_PROMISE = """
```yaml
project: fire-response
promises:
  - {column: at_night, role: feature}
lost_bets:
  - {column: чего-не-обещали, reason: "проиграна"}
```
"""

LOST_BET_ON_DIRECTION = """
```yaml
project: fire-response
promises:
  - {column: at_night, role: feature, direction: decreases}
lost_bets:
  - {column: at_night, reason: "проиграна: связь оказалась обратной"}
```
"""


def test_a_lost_bet_about_a_promise_never_made_is_reported(form) -> None:
    """Запись о проигрыше там, где ставки не делали, ничего не прикрывает.

    Проверку она не отключает, но остаётся объявлением, за которым не стоит
    ничего, — и такие записи проект запрещает наравне с ложными отговорками.
    """
    broken = _unfulfilled(_declared_block(LOST_BET_WITHOUT_A_PROMISE), form)

    assert len(broken) == 1
    assert "обещания о ней не было" in broken[0]


def test_a_bet_lost_by_direction_is_silent(form) -> None:
    """Ставка проигрывается любым свойством, не только ролью.

    Прежняя версия смотрела на роль: при верной роли и разошедшемся направлении
    она объявляла обещание ИСПОЛНЕННЫМ — то есть выдавала ложную тревогу о
    проигрыше и молчала о настоящем расхождении.
    """
    assert not _unfulfilled(_declared_block(LOST_BET_ON_DIRECTION), form)


SPENT_WITHOUT_FIRED = """
```yaml
project: fire-response
spent_controls:
  - {name: К-1, outcome: "снят после прогона"}
```
"""

SPENT_WITH_FIRED = """
```yaml
project: fire-response
spent_controls:
  - {name: К-1, outcome: "промолчал: находка недостижима при этом объявлении", fired: false}
```
"""


def test_a_spent_control_must_say_whether_it_fired() -> None:
    """«Снят» не должно быть неотличимо от «сработал».

    Контроль К-2 двадцатого кейса промолчал, и это сказано прозой. Учёту
    правила остановки прозу не прочитать, а условие 2 держится именно на том,
    сработали ли контроли.
    """
    missing = _controls_without_verdict(_declared_block(SPENT_WITHOUT_FIRED))

    assert len(missing) == 1
    assert "СРАБОТАЛ ли контроль" in missing[0]


def test_a_spent_control_with_a_verdict_is_silent() -> None:
    """Промолчавший контроль записывается промолчавшим, и это законно."""
    assert not _controls_without_verdict(_declared_block(SPENT_WITH_FIRED))


def test_every_spent_control_in_the_docs_says_whether_it_fired() -> None:
    """Рабочий путь: все записи о снятии в пре-регистрациях несут исход."""
    missing: list[str] = []
    for path in sorted(PREREGS.glob("prereg-case-*.md")):
        block = _declared_block(path.read_text(encoding="utf-8"))
        if block is None:
            continue
        missing += [f"{path.name}: {item}" for item in _controls_without_verdict(block)]
    assert not missing, "; ".join(missing)
