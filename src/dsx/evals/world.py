"""Родовой синтетический мир для проверочных кейсов.

Имена намеренно не отраслевые: entity, decided_at, deadline_on. Мир, списанный
с конкретной отрасли, проверял бы не ядро, а совпадение с этой отраслью.

Мир детерминирован при заданном зерне: повторный вызов даёт те же байты, иначе
кейсы не воспроизводятся.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import polars as pl

from dsx.roles import Availability, ColumnSpec, Role, Schema, TemporalKind

ORIGIN = dt.datetime(2024, 1, 1)
SOURCE = "генератор мира"


@dataclass(frozen=True)
class World:
    """Данные кейса вместе с объявленной схемой."""

    frames: dict[str, pl.DataFrame]
    schema: Schema

    @property
    def main(self) -> pl.DataFrame:
        return self.frames["main"]

    def replace_main(self, frame: pl.DataFrame) -> World:
        return World(frames={**self.frames, "main": frame}, schema=self.schema)


def base_schema() -> Schema:
    return Schema(
        columns=[
            ColumnSpec(name="entity_id", role=Role.ENTITY_ID),
            ColumnSpec(name="decided_at", role=Role.DECISION_TIME, temporal=TemporalKind.INSTANT),
            # Срок назначается в момент решения и в этом мире не пересматривается:
            # момент фиксации совпадает с моментом решения.
            ColumnSpec(
                name="deadline_on",
                role=Role.DEADLINE,
                temporal=TemporalKind.DATE,
                value_as_of="decided_at",
            ),
            ColumnSpec(
                name="event_at",
                role=Role.OUTCOME_COMPONENT,
                temporal=TemporalKind.INSTANT,
            ),
            ColumnSpec(name="status", role=Role.STATUS),
            ColumnSpec(
                name="lead_days",
                role=Role.FEATURE,
                value_as_of="decided_at",
                availability=Availability.AT_DECISION,
                source_of_claim=SOURCE,
            ),
            ColumnSpec(
                name="size",
                role=Role.FEATURE,
                value_as_of="decided_at",
                availability=Availability.AT_DECISION,
                source_of_claim=SOURCE,
            ),
            ColumnSpec(
                name="region",
                role=Role.FEATURE,
                value_as_of="decided_at",
                availability=Availability.AT_DECISION,
                source_of_claim=SOURCE,
            ),
        ]
    )


def build_world(rows: int = 4000, days: int = 540, seed: int = 7) -> World:
    """Собрать чистый мир без заложенных дефектов.

    Исход зависит от обещанного срока монотонно: чем короче срок, тем выше риск
    не успеть. Зависимость закладывается намеренно, чтобы кейсы на дрейф могли
    её ломать.
    """
    rng = np.random.default_rng(seed)

    offsets = rng.integers(0, days, rows)
    decided = [
        ORIGIN + dt.timedelta(days=int(d), hours=int(h))
        for d, h in zip(offsets, rng.integers(6, 22, rows), strict=True)
    ]
    lead = rng.integers(3, 40, rows)
    size = np.round(rng.gamma(2.0, 30.0, rows), 2)
    region = rng.choice(["north", "south", "east", "west"], rows, p=[0.4, 0.3, 0.2, 0.1])

    risk = 1.0 / (1.0 + lead / 6.0)
    late = rng.random(rows) < risk * 0.55

    deadline = [
        d.date() + dt.timedelta(days=int(days_ahead))
        for d, days_ahead in zip(decided, lead, strict=True)
    ]
    event = [
        dt.datetime.combine(dl, dt.time(12, 0))
        + dt.timedelta(days=int(rng.integers(1, 9)) if is_late else -int(rng.integers(0, 3)))
        for dl, is_late in zip(deadline, late, strict=True)
    ]

    frame = pl.DataFrame(
        {
            "entity_id": [f"e{i:06d}" for i in range(rows)],
            "decided_at": decided,
            "deadline_on": [dt.datetime.combine(d, dt.time.min) for d in deadline],
            "event_at": event,
            "status": ["completed"] * rows,
            "lead_days": lead,
            "size": size,
            "region": region,
        }
    ).sort("decided_at")

    return World(frames={"main": frame}, schema=base_schema())


def build_child_frame(
    world: World, per_parent: tuple[int, int] = (1, 3), seed: int = 11
) -> pl.DataFrame:
    """Дочерняя таблица с несколькими строками на сущность.

    Нужна кейсам про join: агрегировать до соединения или после — вопрос,
    на котором в этапе 0 выручка завысилась на 5%.
    """
    rng = np.random.default_rng(seed)
    ids, amounts = [], []
    for entity in world.main["entity_id"].to_list():
        for _ in range(int(rng.integers(per_parent[0], per_parent[1] + 1))):
            ids.append(entity)
            amounts.append(float(np.round(rng.gamma(2.0, 15.0), 2)))
    return pl.DataFrame({"entity_id": ids, "amount": amounts})
