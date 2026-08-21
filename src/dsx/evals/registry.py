"""Реестр проверочных кейсов.

Каждый кейс воспроизводит трение этапа 0 либо проверяет предпосылку, при
которой требование осмысленно. Negative controls обязательны: без них проверка,
поднимающая тревогу всегда, выглядит идеальной.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from dsx.evals import injectors as inj
from dsx.evals.case import Case, Expectation, Finding
from dsx.evals.world import World, build_child_frame, build_world
from dsx.outcome import ComparisonMode, MissingEventMeaning, OutcomeDefinition


def _with_children() -> World:
    world = build_world()
    return World(
        frames={**world.frames, "children": build_child_frame(world)},
        schema=world.schema,
    )


@dataclass(frozen=True)
class Bundle:
    """Кейс вместе со способом собрать его данные."""

    case: Case
    build: Callable[[], World]
    outcome: OutcomeDefinition

    @property
    def id(self) -> str:
        return self.case.id


def _outcome(comparison: ComparisonMode = ComparisonMode.BY_DATE) -> OutcomeDefinition:
    return OutcomeDefinition(
        event_column="event_at",
        deadline_column="deadline_on",
        comparison=comparison,
        missing_event={
            "completed": MissingEventMeaning.NOT_OCCURRED,
            "aborted": MissingEventMeaning.EXCLUDED,
        },
        estimand="событие произошло позже назначенного срока среди объектов, "
        "которые предполагалось обработать",
    )


def _bundle(
    case_id: str,
    title: str,
    rationale: str,
    findings: set[Finding],
    build: Callable[[], World],
    comparison: ComparisonMode = ComparisonMode.BY_DATE,
) -> Bundle:
    return Bundle(
        case=Case(
            id=case_id,
            title=title,
            rationale=rationale,
            expectation=Expectation(findings=frozenset(findings)),
        ),
        build=build,
        outcome=_outcome(comparison),
    )


ALL: tuple[Bundle, ...] = (
    # --- negative controls -------------------------------------------------
    _bundle(
        "clean-baseline",
        "Чистый мир",
        "Тревога здесь означает ложное срабатывание любой проверки.",
        set(),
        build_world,
    ),
    _bundle(
        "clean-other-seed",
        "Чистый мир, другое зерно",
        "Проверка не должна зависеть от случайности генерации.",
        set(),
        lambda: build_world(seed=101),
    ),
    _bundle(
        "clean-small",
        "Чистый мир, малый объём",
        "На малой выборке проверки не должны выдавать дефект из-за шума.",
        set(),
        lambda: build_world(rows=600, seed=23),
    ),
    _bundle(
        "clean-short-period",
        "Чистый мир, короткий период",
        "Короткий период — не то же самое, что обрыв сбора данных.",
        set(),
        lambda: build_world(rows=1200, days=70, seed=31),
    ),
    _bundle(
        "clean-with-children",
        "Чистый мир с дочерней таблицей",
        "Наличие связи один-ко-многим само по себе дефектом не является: "
        "дефект возникает при соединении без объявленной грануляции.",
        set(),
        _with_children,
    ),
    # --- контракт и лик ----------------------------------------------------
    _bundle(
        "mixed-temporal-comparison",
        "Дата сравнивается с моментом времени",
        "Трение 04: 1292 события в назначенный день помечены как поздние — "
        "16.5% положительного класса.",
        {Finding.MIXED_TEMPORAL_COMPARISON},
        build_world,
        comparison=ComparisonMode.DIRECT,
    ),
    _bundle(
        "undeclared-temporal-kind",
        "Временная колонка без объявленной грануляции",
        "Ядро не должно угадывать грануляцию эвристикой.",
        {Finding.UNDECLARED_TEMPORAL_KIND},
        lambda: inj.drop_temporal_declaration(build_world()),
    ),
    _bundle(
        "feature-after-decision",
        "Признак вычислен после момента решения",
        "Трение 05: объявленная доступность — единственная защита от такого лика.",
        {Finding.FEATURE_AFTER_DECISION},
        lambda: inj.feature_from_the_future(build_world()),
    ),
    _bundle(
        "outcome-component-as-feature",
        "Компонент исхода объявлен признаком",
        "Трение 05: связь возникает в формуле метки, проверки данных её не видят.",
        {Finding.OUTCOME_COMPONENT_AS_FEATURE},
        lambda: inj.outcome_component_as_feature(build_world()),
    ),
    # --- структура данных --------------------------------------------------
    _bundle(
        "surrogate-key-as-entity",
        "Суррогатный ключ выдан за идентификатор сущности",
        "Трение 01: признаки по истории объекта дали бы историю длиной в одну строку.",
        {Finding.SURROGATE_KEY_AS_ENTITY},
        lambda: inj.surrogate_key_as_entity(build_world()),
    ),
    _bundle(
        "duplicate-rows",
        "Полные дубликаты строк",
        "Трение 01: 26% дубликатов нашлись только потому, что проверка была дописана вручную.",
        {Finding.DUPLICATE_ROWS},
        lambda: inj.duplicate_rows(build_world()),
    ),
    _bundle(
        "entity-overlap",
        "Одна сущность в разных концах периода",
        "Пересечение сущностей между обучением и оценкой при временном сплите.",
        {Finding.ENTITY_OVERLAP_ACROSS_SPLITS},
        lambda: inj.entity_overlap(build_world()),
    ),
    # --- полнота периода ---------------------------------------------------
    _bundle(
        "missing-period",
        "Пропущенный месяц в середине периода",
        "Трение 01: ноябрь отсутствовал целиком, обнаружено вручную.",
        {Finding.MISSING_PERIOD},
        lambda: inj.missing_period(build_world()),
    ),
    _bundle(
        "truncated-tail",
        "Обрыв сбора данных в конце периода",
        "Трение 04: объём падал с 245 в день до 4, что выглядело как цензурирование.",
        {Finding.TRUNCATED_TAIL},
        lambda: inj.truncated_tail(build_world()),
    ),
    # --- семантика пропусков ----------------------------------------------
    _bundle(
        "status-timestamp-conflict",
        "Статус завершения без метки события",
        "Трение 02: 8 объектов со статусом завершения не имели даты события.",
        {Finding.STATUS_TIMESTAMP_CONFLICT},
        lambda: inj.status_timestamp_conflict(build_world()),
    ),
    _bundle(
        "post-treatment-missingness",
        "Пропуск признака объясняется исходом",
        "Трение 05: из 775 объектов без позиций 767 были отменены.",
        {Finding.POST_TREATMENT_MISSINGNESS},
        lambda: inj.post_treatment_missingness(build_world()),
    ),
    # --- дрейф -------------------------------------------------------------
    _bundle(
        "non-stationary-target",
        "Связь признака с исходом меняется во времени",
        "Трения 07 и 12: именно это трижды переворачивало вывод этапа 0.",
        {Finding.NON_STATIONARY_TARGET},
        lambda: inj.non_stationary_target(build_world()),
    ),
)

BY_ID: dict[str, Bundle] = {b.id: b for b in ALL}
NEGATIVE_CONTROLS = tuple(b for b in ALL if b.case.expectation.is_negative_control)
