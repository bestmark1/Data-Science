"""Проба заглушек достижима с рабочего пути и не спрашивает автора о колонках.

Девятнадцатый кейс: правило применимости контроля требовало счёта до
опечатывания, счёт был сделан — и соврал, потому что список колонок для него
автор составлял суждением. Двенадцать из семнадцати; заглушка лежала в
тринадцатой.

Инструмент, который надо не забыть позвать правильно, в этом проекте ломался
шесть раз (класс 1 журнала повторов). Здесь проверяется то, что от сети не
зависит: список заглушек берётся у ЯДРА, а не повторяется в инструменте, и
имена колонок не собираются в запрос без проверки.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

TOOL = Path(__file__).resolve().parents[1] / "tools" / "sentinel_probe.py"


def _tool():
    spec = importlib.util.spec_from_file_location("sentinel_probe", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_sentinel_list_comes_from_the_core() -> None:
    """Контроль обязан ждать того, что ядро умеет находить.

    Свой список в инструменте разошёлся бы с ядром молча, и проба объявляла бы
    применимым контроль, на который ядро не ответит.
    """
    from dsx.checks.data import SENTINELS

    assert _tool().SENTINELS is SENTINELS


def test_a_column_name_that_could_break_the_query_is_refused() -> None:
    """Запрос собирается строкой, и имя колонки в него попадает как есть."""
    module = _tool()

    assert module._valid("respondent_name") == "respondent_name"
    for bad in ("name'; drop", "name)", "name value", ""):
        with pytest.raises(SystemExit):
            module._valid(bad)


def test_service_columns_are_skipped_and_said_aloud(monkeypatch) -> None:
    """Socrata подмешивает свои поля, и на них проба падала.

    `:@computed_region_nku6_53ud` в наборе двадцатого кейса завершал программу
    до первого счёта: имя со скобкой и двоеточием не проходит проверку, а
    проверка завершает работу. Проба была испытана накануне на наборе из
    семнадцати обычных колонок, где служебных полей не было, — шестой случай
    класса «механизм не проверен на достижимость».

    Служебные поля отсеиваются явно и называются вслух: молчаливый пропуск
    колонки неотличим от недосмотра, а проба заведена именно против недосмотра
    в списке колонок.
    """
    module = _tool()
    monkeypatch.setattr(
        module,
        "_get",
        lambda url: {
            "columns": [
                {"fieldName": "permit"},
                {"fieldName": ":@computed_region_nku6_53ud"},
                {"fieldName": "county"},
                {"fieldName": ":id"},
            ]
        },
    )

    own, service = module.columns("data.colorado.gov", "wumm-7awb")

    assert own == ["permit", "county"]
    assert service == [":@computed_region_nku6_53ud", ":id"]
