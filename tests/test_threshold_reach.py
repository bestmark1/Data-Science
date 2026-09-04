"""Пороги, до которых стенд не достаёт по устройству.

`N4.floor` сравнивается не с константой, а с шумом выборки: `max(floor, шум)`.
Шум равен `3 × 0.29 / sqrt(smaller)`, где `smaller` — меньшая из двух групп в
окне. При объявленных 0.05 порог начинает участвовать только когда `smaller`
превышает три сотни; миры стенда несут по три сотни СТРОК в окне и шум 0.145 —
втрое выше порога. Мутант, переживший прогон стенда, свидетельствует о размере
мира, а не о беззащитности числа.

Расчёт записан в `docs/recurrence-ledger.md`, раздел «Выживший мутант у порога,
до которого стенд не достаёт».

**Чем этот тест НЕ является.** Он не мир стенда. Стенд доказывает, что проверка
находит НАСТОЯЩИЙ дефект; здесь проверяется только арифметика границы — что
объявленное число действительно решает исход. Это меньше, и заменой миру оно не
служит.

Почему не завести большой мир в стенде: он замедлил бы каждый прогон каждого
кейса, а поднятая этим доля прибитых была бы подгонкой стенда под измеритель —
то же необеспеченное объявление, только адресованное собственной мере.
"""

from __future__ import annotations

import dataclasses
import sys

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, "tests")

from dsx.checks import ALL_CHECKS
from dsx.checks.base import run_checks
from dsx.checks.drift import SIGMA, _association_error, _windows_with_labels
from dsx.evals.registry import BY_ID, build_world
from dsx.label import LABEL
from harness import context_for

ROWS = 120_000
"""Объём, при котором шум опускается ниже 0.025 и порог 0.05 начинает решать.

Меньше нельзя: при 12 000 строк шум равен 0.069 и перекрывает обе стороны
мутации, отчего тест не отличал бы 0.05 от 0.025.
"""


def _flipping_world(delta: float, seed: int = 7):
    """Мир, где связь признака с исходом меняет ЗНАК, а величина задана.

    Величина связи управляется сдвигом распределения у опоздавших: она нужна
    малой и точной, чтобы лечь между двумя мутантами порога.
    """
    world = build_world(rows=ROWS)
    rng = np.random.default_rng(seed)
    frame = world.main
    late = (
        (frame["event_at"].dt.date() > frame["deadline_on"].dt.date()).fill_null(False).to_numpy()
    )
    midpoint = np.datetime64(frame["decided_at"].quantile(0.66))
    sign = np.where(frame["decided_at"].to_numpy() >= midpoint, -1.0, 1.0)
    values = rng.normal(0, 1, frame.height) + late * delta * sign
    return world.replace_main(frame.with_columns(pl.Series("size", values)))


def _context(delta: float):
    bundle = dataclasses.replace(BY_ID["clean-baseline"], build=lambda: _flipping_world(delta))
    return context_for(bundle)


@pytest.fixture(scope="module")
def weak():
    """Связи около 0.03: ниже объявленного порога, выше половинного."""
    return _context(0.10)


@pytest.fixture(scope="module")
def moderate():
    """Связи около 0.07: выше объявленного порога, ниже удвоенного."""
    return _context(0.22)


def _fires(context) -> bool:
    """Проверка запускается со СВОИМ объявленным порогом.

    Первая версия подставляла порог через `dataclasses.replace` и проверяла
    поведение при переданном числе. Объявленное значение в ней не участвовало
    вовсе, и оба мутанта пережили прогон: тест мерил арифметику, ничего не
    говоря о том, чему порог равен. Механизм, проверяющий себя своим же
    параметром, защищает ровно так же, как необеспеченное объявление.
    """
    found = {s.finding.value for s in run_checks(list(ALL_CHECKS), context).signals}
    return "unstable_feature_relation" in found


def test_the_world_is_large_enough_for_the_floor_to_matter(weak) -> None:
    """Проверка самой проверки: без этого тест мерил бы шум, а не порог.

    Замер достижимости однажды врал вдвое, пока его не проверили на функции,
    чей ответ известен. Здесь известен шум: он обязан быть НИЖЕ обеих сторон
    мутации, иначе `max(floor, шум)` вернёт шум в любом случае.
    """
    windows = _windows_with_labels(weak)
    noise = [SIGMA * _association_error(frame[LABEL]) for _, frame in windows]

    assert max(noise) < 0.025, f"шум {max(noise):.3f} перекрывает порог: тест мерил бы не то"


def test_a_weak_flip_is_silent(weak) -> None:
    """Связи около 0.03 объявлены шумом, и знак их ничего не значит.

    Порог, опущенный вдвое, объявил бы находкой то, что объявлено шумом, — и
    этот тест упал бы. Так умирает мутант вниз.
    """
    assert not _fires(weak)


def test_a_moderate_flip_speaks(moderate) -> None:
    """Связи около 0.07 сильнее порога, и смена знака между окнами — находка.

    Порог, поднятый вдвое, потерял бы настоящую смену знака, — и этот тест упал
    бы. Так умирает мутант вверх.
    """
    assert _fires(moderate)
