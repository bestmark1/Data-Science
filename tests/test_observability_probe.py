"""Проба заполненности читает разрез, а не среднее.

Восемнадцатый кейс споткнулся дважды. Сперва по имени колонки было заключено,
что она несёт значения, — она оказалась пуста у всех строк. Потом счёт непустых
был прочитан и дал 48.5% при объявленном пороге 20%, а набор всё равно оказался
негодным: с июля 2023 событие не записано НИ РАЗУ.

Средняя заполненность обрыва не показывает. Программа читает разрез по месяцам
ДО опечатывания — там, где ядро ещё не может помочь.

Сеть здесь не трогается: проверяется разбор ответа и то, что провал назван
вслух. Сам запрос проверен на рабочем пути — на наборе `bdjm-n7q4`, где он
показал семь месяцев подряд с нулём.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

TOOL = Path(__file__).resolve().parents[1] / "tools" / "observability_probe.py"


def _tool():
    spec = importlib.util.spec_from_file_location("observability_probe", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rows(*pairs):
    return [
        {"month": f"2023-{i + 1:02d}-01T00:00:00.000", "rows": str(n), "filled": str(f)}
        for i, (n, f) in enumerate(pairs)
    ]


def test_a_lapse_inside_the_period_is_named(capsys) -> None:
    """Случай кейса 18: среднее выше порога, а внутри ноль."""
    rows = _rows((1000, 900), (1000, 900), (1000, 0), (1000, 0))

    code = _tool().report(rows, floor=0.2)
    out = capsys.readouterr().out

    assert code == 2, "провал обязан менять код возврата, а не только текст"
    assert "МЕСЯЦЕВ НИЖЕ 20%: 2" in out
    assert "45.0%" in out, "средняя заполненность обязана печататься рядом"


def test_an_even_period_is_silent(capsys) -> None:
    rows = _rows((1000, 900), (1000, 850), (1000, 880))

    code = _tool().report(rows, floor=0.2)

    assert code == 0
    assert "обрыва записи нет" in capsys.readouterr().out


def test_an_empty_answer_is_not_a_clean_verdict(capsys) -> None:
    """Пустой ответ означает неверный запрос, а не отсутствие обрыва.

    Молчание здесь неотличимо от честного «всё ровно», и различать их обязана
    программа, а не читатель.
    """
    code = _tool().report([], floor=0.2)

    assert code == 1
    assert "проверьте имена колонок" in capsys.readouterr().out


def test_a_month_exactly_at_the_floor_is_not_a_lapse(capsys) -> None:
    """Порог объявлен как «не менее»: ровно на нём — ещё не провал."""
    code = _tool().report(_rows((1000, 200), (1000, 900)), floor=0.2)

    assert code == 0


def test_a_column_name_that_could_carry_a_query_is_refused() -> None:
    """Имя подставляется в запрос строкой, поэтому проверяется на входе."""
    module = _tool()
    module._valid("createddate")

    for bad in ("created date", "a' OR '1", "count(*)", ""):
        with pytest.raises(SystemExit):
            module._valid(bad)
