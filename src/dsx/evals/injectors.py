"""Инжекторы дефектов: портят чистый мир известным способом.

Каждый инжектор возвращает изменённый мир. Что именно испорчено, знает кейс,
поэтому вердикт проверки механический, а не является суждением.

Инжектор обязан ломать ровно один вид дефекта. Инжектор, портящий данные
сразу несколькими способами, делает вердикт неинтерпретируемым.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl

from dsx.evals.world import World
from dsx.roles import Availability, ColumnSpec, Role, Schema, TemporalKind

SOURCE = "инжектор"


def _replace_column(schema: Schema, column: ColumnSpec) -> Schema:
    return Schema(columns=[column if c.name == column.name else c for c in schema.columns])


def _add_column(schema: Schema, column: ColumnSpec) -> Schema:
    return Schema(columns=[*schema.columns, column])


def drop_temporal_declaration(world: World, name: str = "event_at") -> World:
    """Убрать объявленную грануляцию у компонента исхода.

    Ядро не должно угадывать грануляцию эвристикой: угадывание даёт ложные
    срабатывания, которые пользователь научится затыкать.

    Целью выбран компонент исхода, а не роль с обязательной грануляцией: иначе
    пришлось бы менять роль, и инжектор сломал бы сразу несколько вещей.
    """
    column = world.schema.get(name)
    assert column is not None
    stripped = ColumnSpec(name=column.name, role=column.role, availability=column.availability)
    return World(frames=world.frames, schema=_replace_column(world.schema, stripped))


def feature_declared_after_decision(world: World) -> World:
    """Добавить признак, честно объявленный появляющимся после решения.

    Ловится проверкой контракта: декларация говорит правду.
    """
    frame = world.main.with_columns(
        (pl.col("event_at") - pl.col("decided_at")).dt.total_days().alias("actual_days")
    )
    column = ColumnSpec(
        name="actual_days",
        role=Role.FEATURE,
        availability=Availability.AFTER,
        source_of_claim=SOURCE,
    )
    return World(
        frames={**world.frames, "main": frame},
        schema=_add_column(world.schema, column),
    )


def feature_from_the_future(world: World) -> World:
    """Добавить признак, ЛОЖНО объявленный доступным в момент решения.

    Значение известно только по факту наступления события, но декларация
    утверждает обратное. Проверка деклараций здесь бессильна: она читает то же
    ложное утверждение. Нужна эмпирическая сверка силы связи с исходом (N6).
    """
    # Просрочка относительно срока, а не длительность: длительность сама по себе
    # ответа не содержит, потому что исход зависит от неё В СРАВНЕНИИ со сроком.
    frame = world.main.with_columns(
        (pl.col("event_at").dt.date() - pl.col("deadline_on").dt.date())
        .dt.total_days()
        .alias("overrun_days")
    )
    column = ColumnSpec(
        name="overrun_days",
        role=Role.FEATURE,
        availability=Availability.AT_DECISION,
        source_of_claim=SOURCE,
    )
    return World(
        frames={**world.frames, "main": frame},
        schema=_add_column(world.schema, column),
    )


def outcome_component_as_feature(world: World) -> World:
    """Объявить компонент исхода обычным признаком.

    Лик, который проверки данных увидеть не могут: связь возникает в формуле
    метки, а не в данных.
    """
    column = ColumnSpec(
        name="event_at",
        role=Role.FEATURE,
        temporal=TemporalKind.INSTANT,
        availability=Availability.AT_DECISION,
        source_of_claim=SOURCE,
    )
    return World(frames=world.frames, schema=_replace_column(world.schema, column))


def surrogate_key_as_entity(world: World, per_object: int = 3, seed: int = 21) -> World:
    """Подменить идентификатор объекта суррогатным ключом строки.

    Естественный ключ делается крупнее строки: несколько наблюдений относятся
    к одному объекту. Без этого подмена не создаёт условия — в чистом мире
    одна строка и есть один объект.
    """
    rng = np.random.default_rng(seed)
    rows = world.main.height
    objects = max(1, rows // per_object)
    natural = [f"o{int(i):06d}" for i in rng.integers(0, objects, rows)]

    frame = world.main.with_columns(
        pl.Series("row_key", [f"r{i:06d}" for i in range(rows)]),
        pl.Series("entity_id", natural),
    )
    schema = Schema(
        columns=[
            ColumnSpec(name="row_key", role=Role.ENTITY_ID),
            *[c for c in world.schema.columns if c.name != "entity_id"],
            ColumnSpec(name="entity_id", role=Role.NATURAL_KEY),
        ]
    )
    return World(frames={**world.frames, "main": frame}, schema=schema)


def duplicate_rows(world: World, share: float = 0.15, seed: int = 3) -> World:
    """Продублировать часть строк целиком."""
    rng = np.random.default_rng(seed)
    count = int(world.main.height * share)
    idx = rng.choice(world.main.height, count, replace=False)
    extra = world.main[idx.tolist()]
    return world.replace_main(pl.concat([world.main, extra]))


def missing_period(world: World, months: int = 1) -> World:
    """Вырезать целый месяц из середины периода."""
    lo = world.main["decided_at"].min()
    hi = world.main["decided_at"].max()
    start = lo + (hi - lo) / 2
    stop = start + dt.timedelta(days=30 * months)
    return world.replace_main(
        world.main.filter((pl.col("decided_at") < start) | (pl.col("decided_at") >= stop))
    )


def truncated_tail(world: World, days: int = 45, keep: float = 0.1, seed: int = 5) -> World:
    """Обрушить объём в конце периода, не выбрасывая дни целиком.

    Прореживание по дням оставляет каждый день представленным: иначе инжектор
    порождал бы ещё и пропущенные периоды, и вердикт стал бы неинтерпретируемым.
    """
    rng = np.random.default_rng(seed)
    edge = world.main["decided_at"].max() - dt.timedelta(days=days)
    head = world.main.filter(pl.col("decided_at") < edge)
    tail = world.main.filter(pl.col("decided_at") >= edge)
    if tail.is_empty():
        return world

    kept = []
    for (_day,), chunk in tail.group_by(pl.col("decided_at").dt.date(), maintain_order=True):
        take = max(1, int(round(chunk.height * keep)))
        idx = rng.choice(chunk.height, take, replace=False)
        kept.append(chunk[idx.tolist()])

    return world.replace_main(pl.concat([head, *kept]))


def status_timestamp_conflict(world: World, count: int = 40, seed: int = 9) -> World:
    """Оставить статус завершения там, где метка события отсутствует."""
    rng = np.random.default_rng(seed)
    idx = set(rng.choice(world.main.height, count, replace=False).tolist())
    frame = world.main.with_columns(
        pl.when(pl.int_range(pl.len()).is_in(list(idx)))
        .then(None)
        .otherwise(pl.col("event_at"))
        .alias("event_at")
    )
    return world.replace_main(frame)


def post_treatment_missingness(world: World, seed: int = 13) -> World:
    """Сделать пропуск признака следствием исхода.

    Пропуск объясняется статусом, а не свойством объекта: импутация нулём
    превращает его в признак из будущего.
    """
    rng = np.random.default_rng(seed)
    late = pl.col("event_at").dt.date() > pl.col("deadline_on").dt.date()
    drop = pl.Series(rng.random(world.main.height) < 0.7)
    frame = world.main.with_columns(
        pl.when(late & drop).then(None).otherwise(pl.col("size")).alias("size"),
        pl.when(late & drop).then(pl.lit("aborted")).otherwise(pl.col("status")).alias("status"),
    )
    return world.replace_main(frame)


def non_stationary_target(world: World, seed: int = 17) -> World:
    """Сломать связь признака с исходом во второй половине периода.

    В первой половине короткий срок означает высокий риск, во второй связь
    исчезает. Именно это на этапе 0 трижды переворачивало вывод.
    """
    rng = np.random.default_rng(seed)
    midpoint = (
        world.main["decided_at"].min()
        + (world.main["decided_at"].max() - world.main["decided_at"].min()) / 2
    )
    frame = world.main.with_columns(pl.Series("_roll", rng.random(world.main.height)))
    shifted = frame.with_columns(
        pl.when(pl.col("decided_at") >= midpoint)
        .then(
            pl.when(pl.col("_roll") < 0.08)
            .then(pl.col("deadline_on").dt.offset_by("3d"))
            .otherwise(pl.col("deadline_on").dt.offset_by("30d"))
        )
        .otherwise(pl.col("deadline_on"))
        .alias("deadline_on")
    ).drop("_roll")
    return world.replace_main(shifted)


def entity_overlap(world: World, share: float = 0.1, seed: int = 19) -> World:
    """Повторить часть объектов позже, ВНУТРИ того же периода.

    Сдвиг за пределы периода порождал бы разрыв и разрежённый хвост — инжектор
    ломал бы три вещи вместо одной.
    """
    rng = np.random.default_rng(seed)
    span = world.main["decided_at"].max() - world.main["decided_at"].min()
    early = world.main.filter(pl.col("decided_at") < world.main["decided_at"].min() + span / 2)
    count = min(early.height, int(world.main.height * share))
    idx = rng.choice(early.height, count, replace=False)

    repeats = early[idx.tolist()].with_columns(
        (pl.col("decided_at") + pl.duration(days=int(span.days // 2))).alias("decided_at")
    )
    return world.replace_main(pl.concat([world.main, repeats]).sort("decided_at"))


def late_maturing_labels(
    world: World, share: float = 0.2, ahead: int = 400, seed: int = 29
) -> World:
    """Отодвинуть срок далеко вперёд у части объектов по всему периоду.

    Их исход станет известен уже после конца наблюдения. Затрагивается доля
    объектов, а не хвост периода: незрелость — свойство самих объектов, а не
    того, куда пришлось оценочное окно, и проверка не должна зависеть от
    расположения окна.
    """
    rng = np.random.default_rng(seed)
    picked = pl.Series(rng.random(world.main.height) < share)
    frame = world.main.with_columns(
        # Срок далеко впереди И события ещё нет: исход не наблюдаем. Одного
        # сдвига срока мало — при наступившем событии метка уже известна.
        pl.when(picked)
        .then(pl.col("deadline_on").dt.offset_by(f"{ahead}d"))
        .otherwise(pl.col("deadline_on"))
        .alias("deadline_on"),
        pl.when(picked).then(None).otherwise(pl.col("event_at")).alias("event_at"),
        # Свой статус: объект, событие которого ещё не наступило, отличается от
        # завершённого. Иначе один статус имел бы событие то есть, то нет — а
        # это уже другой дефект (A5), и вердикт стал бы неинтерпретируемым.
        pl.when(picked).then(pl.lit("pending")).otherwise(pl.col("status")).alias("status"),
    )
    return world.replace_main(frame)
