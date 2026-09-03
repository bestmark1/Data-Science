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
from dsx.roles import (
    Availability,
    ColumnSpec,
    Direction,
    FeatureWindow,
    Role,
    Schema,
    TemporalKind,
)

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
        value_as_of="decided_at",
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
        value_as_of="decided_at",
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
        value_as_of="decided_at",
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


def non_stationary_target(world: World, seed: int = 17, at_fraction: float = 0.72) -> World:
    """Сломать связь признака с исходом во второй половине периода.

    В первой половине короткий срок означает высокий риск, во второй связь
    исчезает. Именно это на этапе 0 трижды переворачивало вывод.
    """
    rng = np.random.default_rng(seed)
    midpoint = _moment_at(world.main, at_fraction)
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

    # Сдвигаются ВСЕ времена строки, а не только момент решения. Первая версия
    # двигала решение вперёд, оставляя событие на месте, и порождала второй
    # дефект: событие раньше решения. Нашла это проверка A13, добавленная
    # позже, — то есть инжектор полтора кейса ломал две вещи вместо одной.
    shift = pl.duration(days=int(span.days // 2))
    repeats = early[idx.tolist()].with_columns(
        (pl.col("decided_at") + shift).alias("decided_at"),
        (pl.col("event_at") + shift).alias("event_at"),
        (pl.col("deadline_on") + shift).alias("deadline_on"),
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


def repeated_object(world: World, per_object: int = 3, seed: int = 23) -> World:
    """Сделать объект долгоживущим: несколько решений на один объект.

    Это не дефект, а условие. Пересечение окон признаков невозможно там, где
    объект получает одно решение, поэтому кейсы про окна строятся на нём.
    """
    rng = np.random.default_rng(seed)
    rows = world.main.height
    objects = max(1, rows // per_object)
    natural = [f"obj{int(i):06d}" for i in rng.integers(0, objects, rows)]

    frame = world.main.with_columns(pl.Series("object_key", natural))
    schema = Schema(
        columns=[*world.schema.columns, ColumnSpec(name="object_key", role=Role.NATURAL_KEY)]
    )
    return World(frames={**world.frames, "main": frame}, schema=schema)


def declare_feature_windows(
    world: World,
    lookback_days: float,
    lag_days: float = 0.0,
    clock: str = "decided_at",
) -> World:
    """Объявить окно у всех признаков.

    Само по себе объявление дефектом не является: оно лишь делает видимым то,
    как признак посчитан. Дефектом становится ширина окна, при которой окна
    двух решений одного объекта пересекаются.

    `clock` — по какому времени строка отбирается в окно. По умолчанию момент
    решения: так окно видит только известное.
    """
    window = FeatureWindow(
        lookback_days=lookback_days, lag_days=lag_days, source_of_claim=SOURCE, clock=clock
    )
    columns = [
        c.model_copy(update={"window": window}) if c.role is Role.FEATURE else c
        for c in world.schema.columns
    ]
    return World(frames=world.frames, schema=Schema(columns=columns))


def stale_measurements(world: World, age_days: int = 3, lookback_days: float = 0.0) -> World:
    """Значения признаков получены раньше, чем утверждает объявленное окно.

    Окно нулевой длины утверждает, что значение получено в момент решения.
    Здесь оно получено на несколько дней раньше — ровно тот случай, который
    ядро не видело до F-11.
    """
    moment = world.schema.decision_time.name
    frame = world.main.with_columns(
        (pl.col(moment) - pl.duration(days=age_days)).alias("measured_at")
    )
    window = FeatureWindow(lookback_days=lookback_days, source_of_claim=SOURCE, clock=moment)
    columns = [
        c.model_copy(update={"window": window, "measured_at": "measured_at"})
        if c.role is Role.FEATURE
        else c
        for c in world.schema.columns
    ]
    columns.append(
        ColumnSpec(name="measured_at", role=Role.MEASURED_AT, temporal=TemporalKind.INSTANT)
    )
    return World(frames={**world.frames, "main": frame}, schema=Schema(columns=columns))


def _moment_at(frame: pl.DataFrame, fraction: float):
    """Момент внутри периода наблюдения по доле от его длины.

    Перелом должен приходиться МЕЖДУ оценочными окнами, иначе все окна
    оказываются по одну сторону от него и контраста не возникает.
    """
    lo, hi = frame["decided_at"].min(), frame["decided_at"].max()
    return lo + (hi - lo) * fraction


def flipped_feature_relation(
    world: World, feature: str = "lead_days", at_fraction: float = 0.72
) -> World:
    """Развернуть связь признака с исходом во второй половине периода.

    Значения зеркалятся относительно медианы: распределение признака остаётся
    прежним, доля класса тоже, меняется только знак связи. Отличается от
    non_stationary_target тем, что не трогает долю положительного класса —
    иначе нельзя проверить, что N4 видит смену знака сама по себе.
    """
    frame = world.main
    midpoint = _moment_at(frame, at_fraction)
    median = float(frame[feature].median())
    return world.replace_main(
        frame.with_columns(
            pl.when(pl.col("decided_at") >= midpoint)
            .then(2 * median - pl.col(feature))
            .otherwise(pl.col(feature))
            .alias(feature)
        )
    )


def disjoint_support(
    world: World, feature: str = "size", shift: float = 1000.0, at_fraction: float = 0.71
) -> World:
    """Развести значения признака по окнам так, что они не пересекаются.

    Сравнивать связь между окнами при такой поддержке нельзя: сравниваются
    разные участки шкалы, а не разные времена.

    Доля, на которой разводится носитель, совпадает с ГРАНИЦЕЙ между окнами
    стенда. Иначе разрыв попадает внутрь окна, носитель в нём смешивается, и
    кейс воспроизводит половину дефекта — ровно тот промах, что дважды случился
    в десятом кейсе. Прежнее значение 0.72 совпадало со старой границей; после
    перевода окон на доли периода граница стала 0.71.
    """
    frame = world.main
    midpoint = _moment_at(frame, at_fraction)
    return world.replace_main(
        frame.with_columns(
            pl.when(pl.col("decided_at") >= midpoint)
            .then(pl.col(feature) + shift)
            .otherwise(pl.col(feature))
            .alias(feature)
        )
    )


def declared_direction(world: World, feature: str, direction: Direction) -> World:
    """Объявить доменное направление связи признака с исходом."""
    columns = [
        c.model_copy(update={"direction": direction}) if c.name == feature else c
        for c in world.schema.columns
    ]
    return World(frames=world.frames, schema=Schema(columns=columns))


def competing_event_kinds(world: World, kinds: int = 4, seed: int = 29) -> World:
    """Разметить событие по видам: наступает не более одного из нескольких.

    Само по себе дефектом не является — это устройство предметной области.
    Дефектом становится молчаливое сведение видов в один исход: строка, где
    наступил конкурирующий вид, не отрицательна, а неизвестна.
    """
    rng = np.random.default_rng(seed)
    frame = world.main
    labels = [f"kind{i + 1}" for i in range(kinds)]
    drawn = [labels[int(i)] for i in rng.integers(0, kinds, frame.height)]

    frame = frame.with_columns(
        pl.when(pl.col("event_at").is_not_null())
        .then(pl.Series("event_kind", drawn))
        .otherwise(None)
        .alias("event_kind")
    )
    schema = Schema(
        columns=[*world.schema.columns, ColumnSpec(name="event_kind", role=Role.EVENT_KIND)]
    )
    return World(frames={**world.frames, "main": frame}, schema=schema)


def excluded_from_population(world: World, share: float = 0.12, seed: int = 31) -> World:
    """Часть объектов не предполагалась к обработке: события нет, статус объявлен.

    Не дефект, а условие. До него ветви разметки «исключён из популяции» и
    «наблюдение оборвано» не выполнялись ни разу: во всех мирах событие было у
    каждой строки, и код, различающий причины отсутствия, оставался мёртвым.
    """
    rng = np.random.default_rng(seed)
    frame = world.main
    picked = pl.Series("_pick", rng.random(frame.height) < share)
    return world.replace_main(
        frame.with_columns(picked)
        .with_columns(
            pl.when(pl.col("_pick")).then(None).otherwise(pl.col("event_at")).alias("event_at"),
            pl.when(pl.col("_pick"))
            .then(pl.lit("aborted"))
            .otherwise(pl.col("status"))
            .alias("status"),
        )
        .drop("_pick")
    )


def event_before_decision(world: World, count: int = 25, seed: int = 37) -> World:
    """Датировать событие раньше момента решения.

    Исходом решения такое событие быть не может. Прежде правило сравнения
    проверяло только верхнюю границу, и строка молча становилась положительной.
    """
    rng = np.random.default_rng(seed)
    frame = world.main
    picked = np.zeros(frame.height, dtype=bool)
    picked[rng.choice(frame.height, size=count, replace=False)] = True
    return world.replace_main(
        frame.with_columns(pl.Series("_pick", picked))
        .with_columns(
            pl.when(pl.col("_pick"))
            .then(pl.col("decided_at").dt.offset_by("-5d"))
            .otherwise(pl.col("event_at"))
            .alias("event_at")
        )
        .drop("_pick")
    )


def outcome_at_decision(world: World, share: float = 0.4, seed: int = 53) -> World:
    """Датировать событие ТЕМ ЖЕ моментом, что и решение.

    Не невозможно, а бесполезно: предсказывать нечего, исход уже наступил.
    Семнадцатый кейс: у 67.5% заявок отметка выполнения требований стояла днём
    подачи, и назвать это было нечем — доля класса не отличает редкий исход от
    известного заранее.
    """
    rng = np.random.default_rng(seed)
    frame = world.main
    picked = rng.random(frame.height) < share
    return world.replace_main(
        frame.with_columns(pl.Series("_pick", picked))
        .with_columns(
            pl.when(pl.col("_pick"))
            .then(pl.col("decided_at"))
            .otherwise(pl.col("event_at"))
            .alias("event_at")
        )
        .drop("_pick")
    )


def sentinel_as_value(world: World, token: str = "NA", share: float = 0.3, seed: int = 41) -> World:
    """Записать отсутствие строкой вместо пустого значения.

    Возникает не в данных, а при их чтении: читатель CSV принимает «NA»
    значением, и колонка выглядит заполненной. На четвёртом кейсе так вышло с
    датой отмены подписки, и доля класса получилась 100% вместо 22.1%.
    """
    rng = np.random.default_rng(seed)
    frame = world.main
    picked = rng.random(frame.height) < share
    return world.replace_main(
        frame.with_columns(pl.Series("_pick", picked))
        .with_columns(
            pl.when(pl.col("_pick")).then(pl.lit(token)).otherwise(pl.col("region")).alias("region")
        )
        .drop("_pick")
    )


def grouped_objects(world: World, per_group: int = 3, seed: int = 43) -> World:
    """Объединить объекты в группы: сеть, работодатель, филиал.

    Само по себе дефектом не является — это устройство предметной области.
    Дефектом становится незаявленный уровень: объекты одной группы зависимы, и
    модель, видевшая часть её, оценивается на остальной мягче.
    """
    rng = np.random.default_rng(seed)
    frame = world.main
    rows = frame.height
    groups = max(1, rows // per_group)
    names = [f"chain{int(i):05d}" for i in rng.integers(0, groups, rows)]
    return World(
        frames={**world.frames, "main": frame.with_columns(pl.Series("chain", names))},
        schema=Schema(columns=[*world.schema.columns, ColumnSpec(name="chain", role=Role.IGNORED)]),
    )


def declared_group(world: World) -> World:
    """Объявить колонку сети групповым уровнем."""
    columns = [
        c.model_copy(update={"role": Role.GROUP_ID}) if c.name == "chain" else c
        for c in world.schema.columns
    ]
    return World(frames=world.frames, schema=Schema(columns=columns))


def degenerate_outcome(world: World, past_deadline_days: int = 1) -> World:
    """Сдвинуть ранние события за срок: почти всякий исход становится поздним.

    Воспроизводит первую постановку шестого кейса, где доля одного класса
    вышла 95.3% и ни одна проверка не возразила. Дефект не в данных и не в
    разметке — они верны; дефект в ВОПРОСЕ: предсказывать нечего, потому что
    ответ известен заранее.

    Двигается СОБЫТИЕ, а не срок. Первая версия отодвигала срок в прошлое и
    попутно делала метку известной задолго до начала окна — строка попадала и
    в обучение, и в оценку, и кейс воспроизводил два дефекта вместо одного.
    Правило стенда «один инжектор — один дефект» держится на таких мелочах.
    """
    shifted = world.main.with_columns(
        pl.when(pl.col("event_at") <= pl.col("deadline_on"))
        .then(pl.col("deadline_on").dt.offset_by(f"{past_deadline_days}d"))
        .otherwise(pl.col("event_at"))
        .alias("event_at")
    )
    return world.replace_main(shifted)


def window_clock_from_the_event(
    world: World, lookback_days: float = 30.0, lag_days: int = 5, share: float = 1.0, seed: int = 53
) -> World:
    """Окно признака отсчитано по времени СОБЫТИЯ, а не по времени сведений о нём.

    Воспроизводит шестой кейс: у сводки о происшествии два времени — когда оно
    случилось и когда о нём сообщили, — и второе позже первого. Окно по первому
    захватывает записи, о которых на момент решения ещё не сообщили. В таблице
    обе колонки просто числа, и без объявления они неразличимы.

    Колонка добавляется, потому что в чистом мире её нет: событие там лежит
    ПОСЛЕ решения, а нужен случай, где запись описывает уже случившееся.

    `share` — у какой доли строк часы отстают. Единица портит все строки, и
    такой кейс о ГРАНИЦЕ проверки не говорит ничего: сто процентов больше любого
    порога, и мутация двигала порог вдвое в обе стороны незаметно. Доля около
    объявленных пяти процентов ставит случай по одну или другую сторону границы.
    """
    rng = np.random.default_rng(seed)
    moment = world.schema.decision_time.name
    struck = rng.random(world.main.height) < share
    frame = world.main.with_columns(pl.Series("_struck", struck)).with_columns(
        pl.when(pl.col("_struck"))
        .then(pl.col(moment) - pl.duration(days=lag_days))
        .otherwise(pl.col(moment))
        .alias("occurred_at")
    )
    frame = frame.drop("_struck")
    world = World(
        frames={**world.frames, "main": frame},
        schema=Schema(
            columns=[
                *world.schema.columns,
                ColumnSpec(name="occurred_at", role=Role.IGNORED, temporal=TemporalKind.INSTANT),
            ]
        ),
    )
    return declare_feature_windows(world, lookback_days, clock="occurred_at")


def deadline_revised_after_decision(
    world: World, lag_days: int = 120, share: float = 1.0, seed: int = 59
) -> World:
    """Срок зафиксирован ПОЗЖЕ момента решения: значение переписано задним числом.

    Воспроизводит девятый кейс. Там срок клинического исследования, обещанный
    при регистрации, пересматривался у 73.9% записей, а у завершившихся
    переписывался в дату самого завершения: 100% выполненных обещаний вместо
    36.5%.

    Колонка момента фиксации добавляется, потому что в чистом мире её нет:
    там срок назначается в момент решения и не меняется.

    `share` — у какой доли строк срок переписан задним числом. Смысл тот же,
    что и у соседнего инжектора: кейс, портящий все строки, доказывает
    срабатывание и молчит о границе.
    """
    rng = np.random.default_rng(seed)
    moment = world.schema.decision_time.name
    struck = rng.random(world.main.height) < share
    frame = (
        world.main.with_columns(pl.Series("_struck", struck))
        .with_columns(
            pl.when(pl.col("_struck"))
            .then(pl.col(moment) + pl.duration(days=lag_days))
            .otherwise(pl.col(moment))
            .alias("deadline_fixed_at")
        )
        .drop("_struck")
    )
    columns = [
        c.model_copy(update={"value_as_of": "deadline_fixed_at"}) if c.role is Role.DEADLINE else c
        for c in world.schema.columns
    ]
    columns.append(
        ColumnSpec(name="deadline_fixed_at", role=Role.IGNORED, temporal=TemporalKind.INSTANT)
    )
    return World(frames={**world.frames, "main": frame}, schema=Schema(columns=columns))


def unobservability_tied_to_feature(
    world: World, feature: str = "region", ahead: int = 400
) -> World:
    """Исход не наблюдается там, где признак принимает определённое значение.

    Воспроизводит девятый кейс. Там среди исследований с наблюдаемым исходом
    57.2% спонсированы индустрией, а среди замолчавших — 18.6%: наблюдаемость
    зависела от того, что известно в момент решения, и метрика описывала не ту
    популяцию, о которой делался вывод.

    Отличается от `late_maturing_labels` ровно одним: там ненаблюдаемые
    выбираются жребием, здесь — ПО ПРИЗНАКУ. Разница между случайным выпадением
    и выпадением по признаку и есть ось информативности наблюдения.
    """
    values = world.main[feature].unique().sort().to_list()
    silent = values[: max(1, len(values) // 3)]
    picked = pl.col(feature).is_in(silent)
    frame = world.main.with_columns(
        pl.when(picked)
        .then(pl.col("deadline_on").dt.offset_by(f"{ahead}d"))
        .otherwise(pl.col("deadline_on"))
        .alias("deadline_on"),
        pl.when(picked).then(None).otherwise(pl.col("event_at")).alias("event_at"),
        # Тот же статус, что у честной незрелости: иначе один статус имел бы
        # событие то есть, то нет, и это был бы уже другой дефект (A5).
        pl.when(picked).then(pl.lit("pending")).otherwise(pl.col("status")).alias("status"),
    )
    return world.replace_main(frame)


def categorical_feature_from_the_outcome(world: World) -> World:
    """Признак-СТРОКА, вычисленный из исхода, объявлен доступным при решении.

    Отличается от `outcome_component_as_feature` типом: там колонка числовая,
    здесь строковая. Десятый кейс показал, что проверка N6 смотрела только
    числовые признаки и строковую утечку пропускала целиком — связь +0.43 при
    типичной 0.05 проходила молча. В седьмом и восьмом кейсах подложенные
    утечки были числами, и слепота не проявлялась.
    """
    # Сравнение ПО ДАТЕ, как в контракте исхода этого мира. Первая версия
    # сравнивала моменты и совпадала с меткой лишь на 71.8%: событие вечером
    # назначенного дня она звала опозданием, а метка — нет. Сила разделения
    # выходила 0.335 при пороге 0.35, и кейс молчал не оттого, что проверка
    # слепа, а оттого, что утечка была неточной. Стенд обязан воспроизводить
    # дефект, а не его половину.
    frame = world.main.with_columns(
        pl.when(pl.col("event_at").is_null())
        .then(pl.lit("no_event"))
        .when(pl.col("event_at").dt.date() > pl.col("deadline_on").dt.date())
        .then(pl.lit("late"))
        .otherwise(pl.lit("on_time"))
        .alias("outcome_word")
    )
    schema = _add_column(
        world.schema,
        ColumnSpec(
            name="outcome_word",
            role=Role.FEATURE,
            value_as_of="decided_at",
            availability=Availability.AT_DECISION,
            window=FeatureWindow(lookback_days=0.0, source_of_claim=SOURCE),
            source_of_claim=SOURCE,
        ),
    )
    return World(frames={**world.frames, "main": frame}, schema=schema)


def undeclared_availability(world: World, feature: str = "lead_days") -> World:
    """Признак не объявил, доступен ли он в момент решения.

    Не ошибка данных, а молчание в объявлении: `Availability.UNKNOWN` означает
    «не ответил». Проверка C2 жила без кейса стенда десять кейсов — ровно в том
    положении, в каком N6 провела их слепой к строковым признакам.
    """
    columns = [
        c.model_copy(update={"availability": Availability.UNKNOWN}) if c.name == feature else c
        for c in world.schema.columns
    ]
    return World(frames=world.frames, schema=Schema(columns=columns))


def undeclared_column(world: World, name: str = "extra_measurement") -> World:
    """В таблице решений есть колонка, которой нет в схеме.

    Ядро видит только объявленное: незаявленная колонка не попадает ни в одну
    проверку. Роль `ignored` существует ровно затем, чтобы сказать «колонка
    есть и она не нужна»; промолчать — не ответ.
    """
    frame = world.main.with_columns(pl.lit(1.0).alias(name))
    return World(frames={**world.frames, "main": frame}, schema=world.schema)


def premise_contradicted_by_data(world: World) -> World:
    """Объявлено отсутствие колонки статуса, а она в данных есть.

    Предпосылка выключает проверки. Объявленная наугад, она выключает их ради
    тишины, и доказать обратное можно только сверкой с данными.

    Мир не меняется: меняется объявление. Кейс задаёт `has_process: false` при
    живой колонке статуса, и расхождение обязано быть названо.
    """
    return world


def outcome_depends_on_the_reason(world: World, seed: int = 47, shift_days: int = 5) -> World:
    """Доля исхода различается у наблюдений с разной ПРИЧИНОЙ появления.

    Воспроизводит десятый кейс: у проверок по сигналу нарушение находилось в
    15.0% случаев, у плановых — в 35.5%, и модель, обученная на одном поводе,
    теряла на другом 0.215 разрешающей способности.

    Повод назначается жребием, а затем одной его половине срок удлиняется:
    события у неё чаще успевают в срок, и доля исхода расходится. Жребий важен —
    иначе повод оказался бы связан ещё и с признаками, и кейс воспроизводил бы
    два дефекта вместо одного.

    Сдвиг подобран так, чтобы расхождение было СОПОСТАВИМО с настоящим, а не
    вырожденным: пять дней дают 14.3% против 5.3% — отношение 2.70, близко к
    2.4 на данных десятого кейса. Восемь дней обнуляли долю одного повода
    вовсе, и кейс воспроизводил бы крайность вместо дефекта.
    """
    rng = np.random.default_rng(seed)
    by_signal = pl.Series(rng.random(world.main.height) < 0.4)
    frame = world.main.with_columns(
        pl.when(by_signal).then(pl.lit("по поводу")).otherwise(pl.lit("плановая")).alias("reason"),
        pl.when(by_signal)
        .then(pl.col("deadline_on").dt.offset_by(f"{shift_days}d"))
        .otherwise(pl.col("deadline_on"))
        .alias("deadline_on"),
    )
    schema = _add_column(world.schema, ColumnSpec(name="reason", role=Role.OBSERVATION_REASON))
    return World(frames={**world.frames, "main": frame}, schema=schema)


def training_part_is_empty(world: World, ahead: int = 400) -> World:
    """Исход всех решений созревает после конца наблюдения.

    Обучаться окну становится не на чем: обучение — это объекты, чей исход был
    известен ДО начала окна, а здесь он не известен ни до одного из них.

    Воспроизводит одиннадцатый кейс, где первое окно начиналось нулевым днём
    периода и обучение выходило пустым. Прогон напечатал «обучение 0» и не
    возразил ни одним сигналом: строка в сводке — не возражение.

    Отличается от `late_maturing_labels` тем, что затрагивает ВСЕ строки, а не
    долю: там дефект в составе окна, здесь — в самой его пригодности.

    Событие сдвигается вместе со сроком, а не обнуляется: мир без единого
    события ломает построение снимка, и кейс проверял бы стенд вместо ядра.
    """
    frame = world.main.with_columns(
        pl.col("deadline_on").dt.offset_by(f"{ahead}d"),
        pl.col("event_at").dt.offset_by(f"{ahead}d"),
    )
    return world.replace_main(frame)


def event_recording_stops(
    world: World, tail: float = 0.25, share: float = 1.0, seed: int = 47
) -> World:
    """Перестать записывать событие на последней части периода.

    Восемнадцатый кейс: с июля 2023 город не проставил дату закрытия ни одному
    наряду — 69 600 подряд, — при том что 34 550 из них числятся закрытыми, а
    закрытия в наборе видны ещё полтора года вперёд. Наряды закрываться не
    переставали; перестала заполняться колонка.

    Дефект коварен тем, что средняя заполненность его не показывает: по всей
    популяции она осталась 48.5% и объявленный порог наблюдаемости в 20%
    прошла вдвое.

    Хуже низкого уровня он тем, что портит МЕТКУ, а не только полноту. Решение,
    чей срок истёк и чьё событие не записано, читается как «не наступило», и на
    отрезке обрыва отрицательными становятся все подряд.

    Статус НЕ трогается намеренно: в кейсе он остался «Closed», и именно
    расхождение статуса с пустым событием выдавало порчу тому, кто смотрел.

    `share` — какую долю хвоста лишить события. Единица воспроизводит кейс 18:
    запись прекратилась совсем. Меньшая доля даёт УМЕРЕННОЕ падение, и она
    здесь не для полноты картины: без неё объявленная чувствительность
    проверки ничем не обеспечена. Мутация показала, что при полном обрыве порог
    можно поднять с 0.25 до 0.9, и ни один тест этого не заметит — падение со
    100% до нуля ловится любым порогом.
    """
    rng = np.random.default_rng(seed)
    frame = world.main
    moment = "decided_at"
    lo, hi = frame[moment].min(), frame[moment].max()
    cut = lo + (hi - lo) * (1.0 - tail)
    struck = (frame[moment] >= cut).to_numpy() & (rng.random(frame.height) < share)
    return world.replace_main(
        frame.with_columns(pl.Series("_struck", struck))
        .with_columns(
            pl.when(pl.col("_struck")).then(None).otherwise(pl.col("event_at")).alias("event_at")
        )
        .drop("_struck")
    )
