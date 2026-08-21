"""Соединение с объявленной грануляцией (A8, A9)."""

from __future__ import annotations

import polars as pl
import pytest

from dsx.evals.world import build_child_frame, build_world
from dsx.join import (
    Cardinality,
    JoinExpectationViolated,
    check_cardinality,
    guarded_join,
    join_losses,
)
from dsx.policy import Blocked, OverrideLedger

PARENTS = pl.DataFrame({"entity_id": ["a", "b", "c"], "size": [1, 2, 3]})
CHILDREN = pl.DataFrame({"entity_id": ["a", "a", "b"], "amount": [10.0, 20.0, 5.0]})
LOOKUP = pl.DataFrame({"entity_id": ["a", "b", "c"], "region": ["n", "s", "e"]})


def test_join_without_declared_keys_is_blocked() -> None:
    with pytest.raises(Blocked, match="A8"):
        guarded_join(PARENTS, LOOKUP)


def test_join_without_declared_cardinality_is_blocked() -> None:
    with pytest.raises(Blocked, match="A8"):
        guarded_join(PARENTS, LOOKUP, on=["entity_id"])


def test_declared_one_to_one_join_passes() -> None:
    result = guarded_join(PARENTS, LOOKUP, on=["entity_id"], expect=Cardinality.ONE_TO_ONE)

    assert result.height == PARENTS.height


def test_hidden_fanout_is_blocked() -> None:
    """Наивное соединение завысило выручку на 5%: строк x1.19, сумм x1.05."""
    with pytest.raises(Blocked, match="A8"):
        guarded_join(PARENTS, CHILDREN, on=["entity_id"], expect=Cardinality.ONE_TO_ONE)


def test_block_message_predicts_the_inflation() -> None:
    with pytest.raises(Blocked) as excinfo:
        guarded_join(PARENTS, CHILDREN, on=["entity_id"], expect=Cardinality.MANY_TO_ONE)

    message = str(excinfo.value)
    assert "раздует" in message and "завысит" in message


def test_declared_one_to_many_is_allowed() -> None:
    """Объявленное раздувание — осознанное решение, а не ошибка."""
    result = guarded_join(PARENTS, CHILDREN, on=["entity_id"], expect=Cardinality.ONE_TO_MANY)

    assert result.height == 3


def test_recorded_override_permits_the_join() -> None:
    ledger = OverrideLedger()
    ledger.override("A8", reason="раздувание учтено ниже по конвейеру", author="автор")

    result = guarded_join(
        PARENTS, CHILDREN, on=["entity_id"], expect=Cardinality.ONE_TO_ONE, ledger=ledger
    )

    assert result.height == 3


def test_blocking_the_join_is_not_bypassed_by_splitting_steps() -> None:
    """Формулировка "join плюс суммирование" обходится разнесением по шагам."""
    with pytest.raises(Blocked):
        guarded_join(PARENTS, CHILDREN, on=["entity_id"], expect=Cardinality.MANY_TO_ONE)


def test_duplicated_left_side_breaks_one_to_one() -> None:
    doubled = pl.concat([PARENTS, PARENTS])

    with pytest.raises(JoinExpectationViolated, match="слева"):
        check_cardinality(doubled, LOOKUP, on=["entity_id"], expect=Cardinality.ONE_TO_ONE)


def test_losses_are_reported_not_raised() -> None:
    """Потеря строк — не ошибка, но решение о ней должно быть явным (A9)."""
    losses = join_losses(PARENTS, CHILDREN, on=["entity_id"])

    assert losses["left_without_match"] == 1
    assert losses["right_without_match"] == 0


def test_real_world_children_trigger_the_block() -> None:
    world = build_world(rows=500, seed=3)
    children = build_child_frame(world)

    with pytest.raises(Blocked, match="A8"):
        guarded_join(world.main, children, on=["entity_id"], expect=Cardinality.ONE_TO_ONE)


def test_aggregating_before_the_join_satisfies_one_to_one() -> None:
    """Правильный порядок: агрегировать до соединения, не после."""
    world = build_world(rows=500, seed=3)
    children = build_child_frame(world)
    aggregated = children.group_by("entity_id").agg(pl.col("amount").sum())

    result = guarded_join(world.main, aggregated, on=["entity_id"], expect=Cardinality.ONE_TO_ONE)

    assert result.height == world.main.height
