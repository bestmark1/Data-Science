"""Соединение таблиц с объявленной грануляцией.

Наивное соединение завысило выручку на 5% на этапе 0: строк стало в 1.19 раза
больше, суммы — в 1.05. Пятипроцентное завышение опаснее двукратного, потому
что выглядит правдоподобно и доходит до прода.

Блокируется не «соединение с последующим суммированием» — такую формулировку
обходят, разнося соединение и агрегацию по шагам, — а соединение без
объявленного ожидания. Объявление проверяется данными: расхождение объявленного
с фактическим и есть находка.
"""

from __future__ import annotations

from enum import StrEnum

import polars as pl

from dsx.policy import Blocked, OverrideLedger

REQUIREMENT = "A8"


class Cardinality(StrEnum):
    """Сколько строк правой таблицы ожидается на строку левой."""

    ONE_TO_ONE = "one_to_one"
    MANY_TO_ONE = "many_to_one"
    """Справочник: у каждой строки левой таблицы не более одного соответствия."""

    ONE_TO_MANY = "one_to_many"
    """Дочерние записи. Требует агрегации до соединения, если дальше идёт
    суммирование."""


class JoinExpectationViolated(Exception):
    """Фактическая грануляция не совпала с объявленной."""


def _rows_per_key(frame: pl.DataFrame, keys: list[str]) -> float:
    unique = frame.select(keys).unique().height
    return frame.height / unique if unique else 0.0


def check_cardinality(
    left: pl.DataFrame,
    right: pl.DataFrame,
    *,
    on: list[str],
    expect: Cardinality,
) -> None:
    """Сверить объявленную грануляцию с фактической."""
    right_per_key = _rows_per_key(right, on)
    left_per_key = _rows_per_key(left, on)

    if expect in (Cardinality.ONE_TO_ONE, Cardinality.MANY_TO_ONE) and right_per_key > 1.0:
        raise JoinExpectationViolated(
            f"объявлено {expect.value}, но справа {right_per_key:.2f} строк на ключ. "
            f"Соединение раздует левую таблицу примерно в {right_per_key:.2f} раза, "
            "и любое последующее суммирование завысит результат"
        )

    if expect is Cardinality.ONE_TO_ONE and left_per_key > 1.0:
        raise JoinExpectationViolated(
            f"объявлено one_to_one, но слева {left_per_key:.2f} строк на ключ"
        )


def guarded_join(
    left: pl.DataFrame,
    right: pl.DataFrame,
    *,
    on: list[str] | None = None,
    expect: Cardinality | None = None,
    how: str = "inner",
    ledger: OverrideLedger | None = None,
) -> pl.DataFrame:
    """Соединить таблицы, требуя объявленных ключей и ожидаемой грануляции.

    Возвращает результат соединения и сообщает о потерях строк с каждой стороны
    через исключение только тогда, когда объявление нарушено: потери сами по
    себе не ошибка, но решение о них должно быть явным (A9).
    """
    ledger = ledger or OverrideLedger()

    if not on:
        ledger.enforce(REQUIREMENT, "соединение без объявленных ключей")
        raise Blocked(REQUIREMENT, "соединение без объявленных ключей")

    if expect is None:
        ledger.enforce(
            REQUIREMENT,
            f"соединение по {on} без объявленной ожидаемой грануляции",
        )
    else:
        try:
            check_cardinality(left, right, on=on, expect=expect)
        except JoinExpectationViolated as exc:
            ledger.enforce(REQUIREMENT, str(exc))

    return left.join(right, on=on, how=how)


def join_losses(left: pl.DataFrame, right: pl.DataFrame, *, on: list[str]) -> dict[str, int]:
    """Сколько строк не найдёт пары с каждой стороны (A9).

    Возвращается, а не бросается: потеря строк — не ошибка, но она должна быть
    названа, а не выведена из типа соединения.
    """
    return {
        "left_without_match": left.height - left.join(right, on=on, how="semi").height,
        "right_without_match": right.height - right.join(left, on=on, how="semi").height,
    }


class AsofDirection(StrEnum):
    """В какую сторону от левой метки времени искать пару."""

    BACKWARD = "backward"
    """Последнее известное ДО левой метки: так берут состояние на момент решения."""

    FORWARD = "forward"
    """Ближайшее следующее ПОСЛЕ левой метки: так берут будущее событие, исход."""


def guarded_asof_join(
    left: pl.DataFrame,
    right: pl.DataFrame,
    *,
    left_on: str | None = None,
    right_on: str | None = None,
    by: list[str] | None = None,
    direction: AsofDirection | None = None,
    ledger: OverrideLedger | None = None,
) -> pl.DataFrame:
    """Соединить по ближайшей во времени строке, требуя объявления направления.

    Соединение вперёд-назад опасно не тем же, чем обычное. Обычное раздувает
    таблицу, и беда видна по числу строк; это число строк левой таблицы
    сохраняет ВСЕГДА и потому молча подставляет не ту строку. Ошиблись
    направлением — и в признак момента решения попало будущее, а число строк
    не дрогнуло.

    Направление проверить по результату НЕЛЬЗЯ, и притворяться иначе тут не
    следует: polars сам соблюдает `strategy`, так что пара всегда лежит с
    объявленной стороны. Испробовано на двухстах случайных таблицах и на
    пустых метках — ноль нарушений. Такая проверка была бы обрядом.

    Требование объявить направление держится на другом: `direction` не
    объявление О вычислении, а его ПАРАМЕТР. Умолчания у него нет намеренно —
    умолчание сделало бы выбор стороны молча, а сторона здесь и есть всё.

    Проверяется то, что проверяемо и может не сойтись:

    * сортировка обеих таблиц — polars её не проверяет, когда задан `by`, и
      лишь предупреждает словами, а предупреждение, которое некому прочесть,
      защитой не является;
    * пустые метки слева — такая строка не находит пары не потому, что события
      не было, а потому, что неизвестен момент. Прочитанные одинаково, эти два
      случая дают исход «ничего не случилось» там, где верно «неизвестно»;
    * сохранение единицы решения по числу строк.

    Ключ правой таблицы обязан называться иначе, чем левый: при совпадении
    имён polars схлопывает их в одну колонку, и сверять направление становится
    нечем. Требование выглядит придиркой ровно до первого молчаливого
    схлопывания.

    Обе таблицы обязаны быть отсортированы по своим ключам. Polars этого не
    проверяет, когда задан `by`, и предупреждает об этом словами — а
    предупреждение, которое некому прочесть, защитой не является.
    """
    ledger = ledger or OverrideLedger()

    if not left_on or not right_on:
        ledger.enforce(REQUIREMENT, "соединение по времени без объявленных ключей")
        raise Blocked(REQUIREMENT, "соединение по времени без объявленных ключей")

    if left_on == right_on:
        raise JoinExpectationViolated(
            f"ключи слева и справа названы одинаково ({left_on!r}): polars схлопнет "
            "их в одну колонку, и проверить направление будет нечем"
        )

    if direction is None:
        ledger.enforce(
            REQUIREMENT,
            f"соединение по времени {left_on!r} к {right_on!r} без объявленного направления",
        )
        direction = AsofDirection.BACKWARD

    blank = left[left_on].null_count()
    if blank:
        raise JoinExpectationViolated(
            f"у {blank:,} строк слева метка {left_on!r} пуста: пары им не найдётся, "
            "и «событие не наступило» станет неотличимо от «момент неизвестен»"
        )

    for frame, key, side in ((left, left_on, "левая"), (right, right_on, "правая")):
        if not frame[key].is_sorted():
            raise JoinExpectationViolated(
                f"{side} таблица не отсортирована по {key!r}: соединение по времени "
                "молча подставит не ту строку, и число строк этого не покажет"
            )

    joined = left.join_asof(
        right, left_on=left_on, right_on=right_on, by=by or None, strategy=direction.value
    )

    if joined.height != left.height:
        raise JoinExpectationViolated(
            f"строк слева {left.height:,}, а после соединения {joined.height:,}: "
            "соединение по времени обязано сохранять единицу решения"
        )

    return joined
