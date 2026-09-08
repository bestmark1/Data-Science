"""Учёты мер не отстают от кейсов.

Правило объявлено действующим, ведомость заведена — и три кейса подряд в неё не
попали. Прочесть состояние правила было негде, а документ выглядел мерой.

Ровно то же случилось с учётом ошибок автора: заведён и не заполнялся до
восемнадцатого кейса. Механизм, требующий, чтобы автор помнил дописать, есть
необеспеченное объявление и защищает хуже своего отсутствия — на него
ссылаются как на действующий.

Здесь это закрывается единственным, чем закрывается: набор тестов краснеет,
пока строка не дописана.
"""

from __future__ import annotations

import re
from pathlib import Path

DOCS = Path(__file__).resolve().parents[1] / "docs"
LEDGER = DOCS / "stopping-rule.md"
ERRORS = DOCS / "author-errors.md"

CASE_ROW = re.compile(r"^\| (\d+) — ", re.MULTILINE)
ERROR_ROW = re.compile(r"^\| \*{0,2}(\d+)\*{0,2} \|", re.MULTILINE)
VERDICT = re.compile(r"^case(\d+)-verdict\.md$")
CONDITION = re.compile(
    r"(\d+) требований из (\d+) покрыто, (\d+)\s*\n?кейс\S* стенда, (\d+) отрицательных контролей"
)


def _ledger() -> str:
    return LEDGER.read_text(encoding="utf-8")


def test_every_case_with_a_verdict_is_in_the_ledger() -> None:
    """У каждого кейса с вердиктом есть строка учёта.

    Отсчёт правила остановки — это состояние, читаемое из таблицы. Кейс, не
    попавший в неё, делает состояние нечитаемым: непонятно, идёт отсчёт или
    обнулён.
    """
    ledger = _ledger()
    counted = {int(n) for n in CASE_ROW.findall(ledger)}
    assert counted, "в учёте нет ни одной строки кейса"

    # Таблица начинается с седьмого кейса: до него действовало прежнее правило,
    # и его учёт остался в описании, а не в этой ведомости.
    first = min(counted)
    with_verdict = {
        int(m.group(1))
        for path in DOCS.iterdir()
        if (m := VERDICT.match(path.name)) and int(m.group(1)) >= first
    }

    missing = sorted(with_verdict - counted)
    assert not missing, f"кейс проведён, а в учёте правила остановки его нет: {missing}"


def test_condition_one_matches_a_direct_count() -> None:
    """Числа полноты стенда сверяются счётом, а не переписываются на глаз.

    Объявление, которое можно проверить, обязано проверяться. Прежняя запись
    отстала на четыре кейса: 35 требований и 52 кейса стенда против нынешних.
    """
    from dsx.checks import ALL_CHECKS
    from dsx.evals.registry import ALL, NEGATIVE_CONTROLS

    required = {check.requirement for check in ALL_CHECKS}
    covered = {name for bundle in ALL for name in bundle.case.expectation.caught_by}

    declared = CONDITION.findall(_ledger())
    assert declared, "состояние условия 1 в учёте не найдено или записано иначе"
    # Последняя запись — текущая; прежние датированы своими кейсами.
    got = tuple(int(n) for n in declared[-1])

    assert got == (len(required & covered), len(required), len(ALL), len(NEGATIVE_CONTROLS)), (
        "числа условия 1 разошлись с прямым счётом"
    )


def test_every_recent_case_is_in_the_author_error_ledger() -> None:
    """Учёт ошибок автора — вторая мера, объявленная и не заполнявшаяся.

    Она разбирает кейсы с тринадцатого; кейс, проведённый под ней и в неё не
    попавший, делает и её долю «названных машиной» невычислимой.
    """
    counted = {int(n) for n in ERROR_ROW.findall(ERRORS.read_text(encoding="utf-8"))}
    assert counted, "в учёте ошибок автора нет ни одной строки кейса"

    first = min(counted)
    with_verdict = {
        int(m.group(1))
        for path in DOCS.iterdir()
        if (m := VERDICT.match(path.name)) and int(m.group(1)) >= first
    }

    missing = sorted(with_verdict - counted)
    assert not missing, f"кейс проведён, а в учёте ошибок автора его нет: {missing}"


# --- Handoff для другой модели ----------------------------------------------
#
# Файл читает модель, которая не была в переписке. Устарев, он не просто
# бесполезен — он врёт тому, кто не может его проверить.

HANDOFF = DOCS / "handoff.md"

CASES_DONE = re.compile(r"Кейсов проведено: (\d+)")
BENCH = re.compile(r"(\d+) кейса, (\d+) требований из (\d+) покрыто, (\d+) отрицательных контролей")


def test_handoff_names_the_last_case() -> None:
    """Номер последнего кейса сверяется с вердиктами, а не с памятью автора."""
    text = HANDOFF.read_text(encoding="utf-8")
    declared = CASES_DONE.search(text)
    assert declared, "в handoff не найдено число проведённых кейсов"

    latest = max(int(m.group(1)) for path in DOCS.iterdir() if (m := VERDICT.match(path.name)))

    assert int(declared.group(1)) == latest, (
        f"handoff говорит о {declared.group(1)} кейсах, а последний вердикт — {latest}-й"
    )


def test_handoff_bench_numbers_match_a_direct_count() -> None:
    """Счёт стенда в шапке — тот же, что у реестра.

    Эти три числа меняются от каждой правки ядра, и именно они первыми
    устаревают в документе, который пишут руками.
    """
    from dsx.checks import ALL_CHECKS
    from dsx.evals.registry import ALL, NEGATIVE_CONTROLS

    required = {check.requirement for check in ALL_CHECKS}
    covered = {name for bundle in ALL for name in bundle.case.expectation.caught_by}

    declared = BENCH.search(HANDOFF.read_text(encoding="utf-8"))
    assert declared, "в handoff не найден счёт стенда или он записан иначе"
    got = tuple(int(number) for number in declared.groups())

    assert got == (len(ALL), len(required & covered), len(required), len(NEGATIVE_CONTROLS)), (
        "числа стенда в handoff разошлись с прямым счётом"
    )


def test_handoff_keeps_what_was_rejected() -> None:
    """Раздел отвергнутого — то, ради чего файл заведён.

    Возражение владельца при проектировании: без него другой агент предложит
    решение, которое уже разбирали и закрыли, и потратит на это токены. Пустой
    раздел неотличим от отсутствующего.
    """
    text = HANDOFF.read_text(encoding="utf-8")

    assert "## Отвергнутое" in text, "в handoff нет раздела отвергнутых решений"
    body = text.split("## Отвергнутое", 1)[1].split("\n## ", 1)[0]
    rows = [line for line in body.splitlines() if line.startswith("| ") and "---" not in line]

    assert len(rows) > 1, "раздел отвергнутого пуст: одна шапка таблицы без строк"
