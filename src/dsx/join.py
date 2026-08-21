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
