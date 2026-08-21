"""Совместимость временных колонок при сравнении.

Сравнение даты с моментом времени пометило опоздавшими 1292 заказа, приехавших
в обещанный день — 16.5% положительного класса. Ошибка невидима глазами: обе
колонки выглядят как даты.
"""

from __future__ import annotations

from dsx.roles import ColumnSpec, TemporalKind


class IncompatibleComparison(Exception):
    """Сравнение временных колонок несовместимой грануляции."""


def check_comparable(left: ColumnSpec, right: ColumnSpec) -> None:
    """Проверить, что две временные колонки допустимо сравнивать напрямую.

    Разрешено сравнивать одинаковые грануляции. Дата с моментом времени
    сравнивается только после явного приведения обеих сторон к дате.
    """
    for column in (left, right):
        if column.temporal is None:
            raise IncompatibleComparison(
                f"колонка {column.name!r} не объявила временную грануляцию"
            )

    if left.temporal is not right.temporal:
        date_side = left if left.temporal is TemporalKind.DATE else right
        instant_side = right if date_side is left else left
        raise IncompatibleComparison(
            f"{date_side.name!r} хранит дату, {instant_side.name!r} — момент времени. "
            "Прямое сравнение пометит событие в тот же день как произошедшее позже. "
            "Приведите обе стороны к дате явно."
        )
