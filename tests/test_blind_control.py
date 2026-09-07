"""Слепой контроль достижим с рабочего пути.

Инструмент был написан, закоммичен и НИ РАЗУ не запущен: запуск блокировался
разрешением. При первом настоящем запуске он отказал на первом же шаге — искал
перечень находок по имени файла `finding*.py`, которого не существует.

Это шестой случай класса «механизм построен и не проверен на достижимость».
Здесь он закрывается тем, чем только и закрывается: вызовом с рабочего пути.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

TOOL = Path(__file__).resolve().parents[1] / "tools" / "blind_control.py"


def _tool():
    spec = importlib.util.spec_from_file_location("blind_control", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_findings_catalogue_is_found() -> None:
    """Каталог находок обязан находиться, иначе выбирать второй стороне не из чего."""
    catalogue = _tool()._findings_catalogue()

    assert catalogue is not None, "перечень находок не найден: инструмент откажет на первом шаге"
    assert "class Finding(" in catalogue.read_text(encoding="utf-8")


def test_patch_is_extracted_from_a_noisy_stream() -> None:
    """Вывод второй стороны несёт счётчик токенов и прочий шум."""
    module = _tool()
    stream = "думаю...\ndiff --git a/x b/x\n--- a/x\n+++ b/x\n@@ -1 +1 @@\n-a\n+b\ntokens used\n17"

    patch = module._extract_patch(stream)

    assert patch.startswith("diff --git")
    assert "tokens used" not in patch
    assert patch.endswith("+b\n")


def test_nothing_is_extracted_when_there_is_no_patch() -> None:
    """Отрицательный контроль: молчание не должно выглядеть патчем."""
    assert _tool()._extract_patch("никакого патча тут нет") == ""


NEW_FILE_PATCH = """diff --git a/sealed.md b/sealed.md
new file mode 100644
index 0000000..e69de29
--- /dev/null
+++ b/sealed.md
@@ -0,0 +1,2 @@
+ОЖИДАЕМАЯ НАХОДКА: duplicate_rows
+объяснение
"""


def test_a_patch_that_creates_a_file_survives_extraction() -> None:
    """Патч, создающий файл, несёт `new file mode` второй строкой.

    Первая версия извлекателя знала три начала строки, обрывалась на этой и
    оставляла ровно 111 байт заголовка. `git apply` отказывал, и слепой
    контроль не мог состояться ни разу.
    """
    patch = _tool()._extract_patch("думаю...\n" + NEW_FILE_PATCH + "tokens used\n42")

    assert "new file mode 100644" in patch
    assert "+ОЖИДАЕМАЯ НАХОДКА: duplicate_rows" in patch
    assert "tokens used" not in patch


def test_an_empty_context_line_does_not_truncate_the_patch() -> None:
    """Пустая строка — тело патча: контекстная строка пустой строки файла."""
    stream = "diff --git a/x b/x\n--- a/x\n+++ b/x\n@@ -1,3 +1,3 @@\n a\n\n-b\n+c\n"

    patch = _tool()._extract_patch(stream)

    assert patch.endswith("+c\n")
    assert len(patch.splitlines()) == len(stream.splitlines())


def test_an_extracted_patch_actually_applies(tmp_path) -> None:
    """Извлечение проверяется применением, а не похожестью на патч.

    Оба прежних дефекта прошли мимо тестов именно потому, что тесты сверяли
    строки, а не результат.
    """
    import subprocess

    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    patch_file = tmp_path / "p.patch"
    patch_file.write_text(_tool()._extract_patch(NEW_FILE_PATCH), encoding="utf-8")

    done = subprocess.run(
        ["git", "apply", "--recount", "p.patch"], cwd=tmp_path, capture_output=True, text=True
    )

    assert done.returncode == 0, done.stderr
    assert (tmp_path / "sealed.md").read_text(encoding="utf-8").startswith("ОЖИДАЕМАЯ")


def test_the_empty_bit_is_sealed_separately_from_the_defect(tmp_path) -> None:
    """Две ступени раскрытия, и первая не выдаёт дефекта.

    Пятнадцатый кейс: молчание ядра означало сразу два события — «контроль был
    пуст» и «дефект подложен и не увиден». Различить их было нечем, и заключение
    автора оказалось неразрешимым по форме.

    Первая ступень отвечает ровно на один вопрос: было ли что подкладывать.
    """
    module = _tool()

    assert module.EMPTY_BIT != module.SEALED

    bit = tmp_path / module.EMPTY_BIT
    bit.write_text("Контроль был ПУСТЫМ.\n\nСоль: abc\n", encoding="utf-8")
    import hashlib

    digest = hashlib.sha256(bit.read_bytes()).hexdigest()[:12]

    assert module._verify(tmp_path, digest, module.EMPTY_BIT) == 0
    assert module._verify(tmp_path, "нетакого", module.EMPTY_BIT) == 1


def test_a_tampered_answer_is_refused(tmp_path) -> None:
    """Отпечаток обеспечивает одно: ответ нельзя переписать задним числом."""
    module = _tool()
    sealed = tmp_path / module.SEALED
    sealed.write_text("ОЖИДАЕМАЯ НАХОДКА: duplicate_rows\n", encoding="utf-8")
    import hashlib

    digest = hashlib.sha256(sealed.read_bytes()).hexdigest()[:12]
    sealed.write_text("ОЖИДАЕМАЯ НАХОДКА: что угодно другое\n", encoding="utf-8")

    assert module._verify(tmp_path, digest) == 1


BAD_COUNTS_PATCH = """diff --git a/sealed.md b/sealed.md
new file mode 100644
--- /dev/null
+++ b/sealed.md
@@ -0,0 +1,9 @@
+ОЖИДАЕМАЯ НАХОДКА: duplicate_rows
+вторая строка
"""


def test_apply_survives_a_hand_written_hunk_header(tmp_path, monkeypatch) -> None:
    """Вторая сторона пишет патч руками, и счётчики строк у неё сбиваются.

    Здесь заголовок куска обещает девять строк, а их две. Без `--recount`
    `git apply` отказывает. Спорить об этом после отказа нельзя: чтобы поправить
    патч, его пришлось бы прочитать, и слепой контроль перестал бы быть слепым.

    Функция дважды отказывала на живом патче и до сих пор не была прогнана ни
    одним тестом — она стояла в списке недостижимых с рабочего пути.
    """
    import subprocess

    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    patch = tmp_path / "p.patch"
    patch.write_text(BAD_COUNTS_PATCH, encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    assert _tool()._apply(patch) is True
    assert (tmp_path / "sealed.md").read_text(encoding="utf-8").startswith("ОЖИДАЕМАЯ")


def test_apply_refuses_a_patch_that_fits_nothing(tmp_path, monkeypatch) -> None:
    """Отрицательный контроль: перебор способов не должен принимать что попало.

    Патч правит строку файла, которого нет. Приняв такое, программа сообщила бы
    об успешном контроле там, где его не было.
    """
    import subprocess

    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    patch = tmp_path / "p.patch"
    patch.write_text(
        "diff --git a/нет.md b/нет.md\n--- a/нет.md\n+++ b/нет.md\n@@ -1 +1 @@\n-было\n+стало\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    assert _tool()._apply(patch) is False


# --- Ограничение выбора второй стороны ----------------------------------
#
# Два кейса подряд она выбрала `duplicate_rows`. Третий такой выбор сделал бы
# слепое испытание неотличимым от авторского контроля на том же дефекте.


CATALOGUE = """
class Finding(StrEnum):
    DUPLICATE_ROWS = "duplicate_rows"
    MISSING_PERIOD = "missing_period"
    SENTINEL_AS_VALUE = "sentinel_as_value"


class Verdict(StrEnum):
    PASS = "pass"
    MISSED = "missed"
"""


def _repo(tmp_path: Path) -> Path:
    catalogue = tmp_path / "src" / "dsx" / "evals"
    catalogue.mkdir(parents=True)
    (catalogue / "case.py").write_text(CATALOGUE, encoding="utf-8")
    (tmp_path / "projects").mkdir()
    return tmp_path


def _case(root: Path, name: str, expected: str) -> Path:
    project = root / "projects" / name
    project.mkdir()
    (project / "blind-control.sealed.md").write_text(
        f"ОЖИДАЕМАЯ НАХОДКА: {expected}\n\nпояснение второй стороны\n", encoding="utf-8"
    )
    return project


def test_tried_items_are_collected_from_the_sealed_files(tmp_path, monkeypatch) -> None:
    """Список испытанного берётся из ответов, а не из ведомости.

    Ведомость, заполняемая руками, в этом проекте уже дважды оказывалась
    объявленной действующей и незаполненной: учёт ошибок автора и учёт правила
    остановки. Третьего раза механизм не переживёт.
    """
    module = _tool()
    root = _repo(tmp_path)
    _case(root, "прошлый", "duplicate_rows")
    _case(root, "позапрошлый", "`sentinel_as_value` — подложено значением-заглушкой")
    monkeypatch.setattr(module, "ROOT", root)

    tried, complaints = module._already_tried(
        root / "projects" / "новый", module._finding_names(module._findings_catalogue())
    )

    assert tried == ["duplicate_rows", "sentinel_as_value"]
    assert complaints == []


def test_an_empty_control_does_not_spend_a_catalogue_item(tmp_path, monkeypatch) -> None:
    """Пустой контроль ничего не испытал — и не сужает выбор.

    Иначе пятнадцатый кейс, где подкладывать было нечего, отнял бы у следующих
    пункт каталога, который никто не проверял.
    """
    module = _tool()
    root = _repo(tmp_path)
    _case(root, "пустой", "нет, это пустой контроль.")
    monkeypatch.setattr(module, "ROOT", root)

    tried, complaints = module._already_tried(
        root / "projects" / "новый", module._finding_names(module._findings_catalogue())
    )

    assert tried == []
    assert complaints == []


def test_the_current_case_does_not_exclude_itself(tmp_path, monkeypatch) -> None:
    """Свой собственный запечатанный файл в список не идёт."""
    module = _tool()
    root = _repo(tmp_path)
    свой = _case(root, "текущий", "duplicate_rows")
    monkeypatch.setattr(module, "ROOT", root)

    tried, _ = module._already_tried(свой, module._finding_names(module._findings_catalogue()))

    assert tried == []


def test_an_unreadable_expectation_is_said_aloud(tmp_path, monkeypatch) -> None:
    """Умолчание, совпадающее с честным ответом, запрещено.

    Строка ожидаемой находки, из которой имя не разобралось, неотличима от
    отсутствия испытаний — и молча выглядела бы как «исключать нечего».
    """
    module = _tool()
    root = _repo(tmp_path)
    _case(root, "невнятный", "строка про какой-то дефект без имени каталога")
    monkeypatch.setattr(module, "ROOT", root)

    tried, complaints = module._already_tried(
        root / "projects" / "новый", module._finding_names(module._findings_catalogue())
    )

    assert tried == []
    assert len(complaints) == 1
    assert "невнятный" in complaints[0]


def test_verdict_names_are_not_catalogue_items(tmp_path) -> None:
    """Отрицательный контроль на разбор каталога.

    Рядом с перечнем находок лежит перечень вердиктов. Прихватив его, программа
    исключила бы из выбора `pass` и `missed` — пункты, которых в каталоге нет.
    """
    module = _tool()
    root = _repo(tmp_path)

    names = module._finding_names(root / "src" / "dsx" / "evals" / "case.py")

    assert names == {"duplicate_rows", "missing_period", "sentinel_as_value"}


def _intercept(module, monkeypatch) -> list[str]:
    """Перехватить задание, которое ушло бы второй стороне."""
    sent: list[str] = []

    class _Done:
        returncode = 1
        stdout = stderr = ""

    def _run(argv, **kwargs):
        sent.append(argv[-1])
        return _Done()

    monkeypatch.setattr(module.subprocess, "run", _run)
    return sent


def test_the_exclusion_reaches_the_task(tmp_path, monkeypatch) -> None:
    """Проверка достижимости: список обязан дойти до второй стороны.

    Функция, собирающая исключения правильно и не попадающая в задание, — это
    механизм, построенный и не проверенный на достижимости. Шесть случаев этого
    класса уже записаны в журнале повторов.
    """
    module = _tool()
    root = _repo(tmp_path)
    _case(root, "прошлый", "duplicate_rows")
    monkeypatch.setattr(module, "ROOT", root)
    monkeypatch.setattr(module.secrets, "randbelow", lambda _: 1)  # непустой контроль
    sent = _intercept(module, monkeypatch)

    module._plant(root / "projects" / "новый")

    assert len(sent) == 1
    assert "- duplicate_rows" in sent[0]
    assert "выбирать их НЕЛЬЗЯ" in sent[0]


def test_nothing_is_excluded_when_nothing_was_tried(tmp_path, monkeypatch) -> None:
    """Отрицательный контроль: без прежних кейсов задание не несёт запретов."""
    module = _tool()
    root = _repo(tmp_path)
    monkeypatch.setattr(module, "ROOT", root)
    monkeypatch.setattr(module.secrets, "randbelow", lambda _: 1)
    sent = _intercept(module, monkeypatch)

    module._plant(root / "projects" / "новый")

    assert "выбирать их НЕЛЬЗЯ" not in sent[0]


# --- Отказ второй стороны, названный вслух -----------------------------------
#
# Девятнадцатый кейс: исчерпанный лимит и процесс, повисший на чтении stdin,
# выглядели одинаково — «ОТКАЗ: код 1». Час ушёл на то, чтобы их различить.


def test_a_technical_refusal_is_named() -> None:
    """Причина, которую можно назвать, называется заранее написанной фразой."""
    module = _tool()

    assert "лимит" in module._technical_cause("ERROR: You've hit your usage limit.")
    assert "ввода" in module._technical_cause("Reading additional input from stdin...")
    assert "авторизована" in module._technical_cause("not authenticated")


def test_an_unknown_refusal_does_not_leak_the_stream() -> None:
    """Отрицательный контроль: чужой текст наружу не выходит.

    В потоке мог быть патч. Печатается фраза о нераспознанной причине, а не то,
    что вторая сторона написала.
    """
    module = _tool()
    said = module._technical_cause("diff --git a/build.py b/build.py\n+подлог")

    assert "diff" not in said
    assert "подлог" not in said
    assert "не распознана" in said


def test_the_second_party_is_called_with_a_closed_stdin(tmp_path, monkeypatch) -> None:
    """Codex CLI при живом stdin ждёт ввода и висит, пока его не убьют.

    Задание передаётся аргументом, читать ему нечего. Проверка достижимости:
    без закрытого ввода слепой контроль не состоится ни разу.
    """
    import subprocess

    module = _tool()
    root = _repo(tmp_path)
    monkeypatch.setattr(module, "ROOT", root)
    monkeypatch.setattr(module.secrets, "randbelow", lambda _: 1)
    seen: list[object] = []

    class _Done:
        returncode = 1
        stdout = stderr = ""

    def _run(argv, **kwargs):
        seen.append(kwargs.get("stdin"))
        return _Done()

    monkeypatch.setattr(module.subprocess, "run", _run)
    module._plant(root / "projects" / "новый")

    assert seen == [subprocess.DEVNULL]
