"""Пороги, до которых стенд не достаёт по устройству.

`N4.floor` и `N8.floor` сравниваются не с константой, а с шумом выборки: `max(floor, шум)`.
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

import dsx.evals.injectors as inj
from dsx.checks import ALL_CHECKS
from dsx.checks.base import run_checks
from dsx.checks.drift import SIGMA, _association_error, _windows_with_labels
from dsx.evals.registry import BY_ID, build_world
from dsx.label import LABEL
from dsx.roles import Direction
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


# --- N8.floor ---------------------------------------------------------------
#
# Тот же порог 0.05 и то же устройство `max(floor, шум)`, но связь считается по
# ОБЪЕДИНЁННОМУ кадру всех окон: строк втрое больше, шум ниже, и мир нужен вдвое
# меньше — шестьдесят тысяч вместо ста двадцати.
#
# Знак связи здесь постоянный, в отличие от N4: проверяется не смена знака между
# окнами, а противоречие данных ОБЪЯВЛЕННОМУ доменному направлению.

N8_ROWS = 60_000
"""Объём, при котором шум по объединённому кадру равен 0.019 — ниже обеих
сторон мутации. При сорока тысячах он равен 0.024 и вплотную подходит к
половинному порогу, отчего тест перестал бы различать 0.05 и 0.025."""


def _contradicting_world(delta: float, seed: int = 11):
    """Признак растёт с исходом, а объявлен убывающим.

    Величина связи задаётся сдвигом распределения у опоздавших: она нужна малой
    и точной, чтобы лечь между двумя мутантами порога.
    """
    world = build_world(rows=N8_ROWS)
    rng = np.random.default_rng(seed)
    frame = world.main
    late = (
        (frame["event_at"].dt.date() > frame["deadline_on"].dt.date()).fill_null(False).to_numpy()
    )
    values = rng.normal(0, 1, frame.height) + late * delta
    world = world.replace_main(frame.with_columns(pl.Series("size", values)))
    return inj.declared_direction(world, "size", Direction.DECREASES)


def _context_n8(delta: float):
    bundle = dataclasses.replace(BY_ID["clean-baseline"], build=lambda: _contradicting_world(delta))
    return context_for(bundle)


@pytest.fixture(scope="module")
def faint_contradiction():
    """Связь около 0.04: ниже объявленного порога, выше половинного."""
    return _context_n8(0.10)


@pytest.fixture(scope="module")
def plain_contradiction():
    """Связь около 0.07: выше объявленного порога, ниже удвоенного."""
    return _context_n8(0.22)


def _contradicts(context) -> bool:
    """Как и выше, проверка идёт со СВОИМ объявленным порогом."""
    found = {s.finding.value for s in run_checks(list(ALL_CHECKS), context).signals}
    return "direction_contradicts_domain" in found


def test_the_joined_frame_is_large_enough_for_the_floor_to_matter(
    faint_contradiction,
) -> None:
    """Проверка самой проверки: шум обязан быть ниже обеих сторон мутации."""
    windows = _windows_with_labels(faint_contradiction)
    joined = pl.concat([frame for _, frame in windows], how="vertical_relaxed")
    noise = SIGMA * _association_error(joined[LABEL])

    assert noise < 0.025, f"шум {noise:.3f} перекрывает порог: тест мерил бы не то"


def test_a_faint_contradiction_is_silent(faint_contradiction) -> None:
    """Связь около 0.04 слабее порога: спорить с доменным знанием нечем.

    Порог, опущенный вдвое, объявил бы находкой то, что объявлено шумом, — и
    этот тест упал бы. Так умирает мутант вниз.
    """
    assert not _contradicts(faint_contradiction)


def test_a_plain_contradiction_speaks(plain_contradiction) -> None:
    """Связь около 0.07 сильнее порога и противоречит объявленному направлению.

    Порог, поднятый вдвое, потерял бы настоящее противоречие, — и этот тест упал
    бы. Так умирает мутант вверх.
    """
    assert _contradicts(plain_contradiction)


# --- N4.strong --------------------------------------------------------------
#
# Порог живёт в ветви ОСЛАБЛЕНИЯ связи, и до неё не доходит ни один кейс стенда:
# все они срабатывают через смену ЗНАКА, а знак проверяется раньше и уводит
# выполнение в сторону. Поэтому ветвь не была задействована ни разу, и оба
# мутанта переживали прогон.
#
# Условий здесь три сразу, и мир обязан удовлетворить все: сильнейшая связь выше
# `max(strong, шум)`, кратность сильнейшей к слабейшей не ниже трёх, а разрыв
# между ними выше совместного шума. Первая попытка дала кратность 2.5 и молчала
# при любом пороге — переход силы пришёлся на середину среднего окна и размыл
# его. Граница сдвинута на стык окон.

FADE_AT = 0.71
"""Доля периода, на которой связь слабеет. Совпадает с границей между вторым и
третьим окном стенда: иначе переход попадает ВНУТРЬ окна, связь в нём
усредняется, и кратность не дотягивает до требуемой."""


def _fading_world(strong_delta: float, weak_delta: float, rows: int, seed: int = 13):
    """Связь одного знака, слабеющая к последнему окну.

    Знак намеренно один: разные знаки увели бы проверку в ветвь смены знака, и
    порог ослабления снова остался бы непроверенным.
    """
    world = build_world(rows=rows)
    rng = np.random.default_rng(seed)
    frame = world.main
    late = (
        (frame["event_at"].dt.date() > frame["deadline_on"].dt.date()).fill_null(False).to_numpy()
    )
    edge = np.datetime64(frame["decided_at"].quantile(FADE_AT))
    delta = np.where(frame["decided_at"].to_numpy() >= edge, weak_delta, strong_delta)
    values = rng.normal(0, 1, frame.height) + late * delta
    return world.replace_main(frame.with_columns(pl.Series("size", values)))


def _context_fade(strong_delta: float, weak_delta: float, rows: int):
    bundle = dataclasses.replace(
        BY_ID["clean-baseline"],
        build=lambda: _fading_world(strong_delta, weak_delta, rows),
    )
    return context_for(bundle)


@pytest.fixture(scope="module")
def marked_fade():
    """Связь падает с 0.19 до 0.05: сильнейшая выше объявленного порога."""
    return _context_fade(0.70, 0.05, 30_000)


@pytest.fixture(scope="module")
def slight_fade():
    """Связь падает с 0.10 до 0.01: сильнейшая ниже объявленного порога.

    Мир вдвое больше: разрыв здесь 0.09, и шум обязан быть заметно меньше него,
    иначе условие «разрыв выше совместного шума» не выполнится ни при каком
    пороге и тест перестанет различать мутантов.
    """
    return _context_fade(0.35, 0.03, 60_000)


def test_a_marked_fade_speaks(marked_fade) -> None:
    """Связь 0.19 сильнее объявленных 0.15, и её падение вчетверо — находка.

    Порог, поднятый вдвое, объявил бы саму связь слабой и до падения не дошёл, —
    и этот тест упал бы. Так умирает мутант вверх.
    """
    assert _fires(marked_fade)


def test_a_slight_fade_is_silent(slight_fade) -> None:
    """Связь 0.10 слабее объявленных 0.15: ослабление шума — не находка.

    Порог, опущенный вдвое, объявил бы находкой падение связи, которая и в
    лучшем окне была слабой, — и этот тест упал бы. Так умирает мутант вниз.
    """
    assert not _fires(slight_fade)
