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
from dsx.outcome import (
    ComparisonMode,
    MissingEventCause,
    MissingEventMeaning,
    OutcomeDefinition,
    PositiveClass,
)
from dsx.roles import Direction
from dsx.task import ObjectLifetime, TargetKind


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

    target_kind: TargetKind | None = None
    """Тип задачи, если кейс требует не бинарного."""

    reserve: bool = True
    """Резервировать ли измерительную выборку при построении сплита.

    Правильный протокол резервирует, поэтому умолчание такое. Кейс, где
    резерва нет, объявляет это явно.
    """

    lifetime: ObjectLifetime | None = None
    """Жизненный цикл объекта, объявленный автором кейса.

    Из данных он не выводится — на этом стоит F-2. Стенд, выводящий его сам,
    повторил бы ошибку F-4: объявление без свидетельства.
    """

    @property
    def id(self) -> str:
        return self.case.id


def _outcome(
    comparison: ComparisonMode = ComparisonMode.BY_DATE,
    expected_positive_rate: float | None = None,
) -> OutcomeDefinition:
    return OutcomeDefinition(
        event_column="event_at",
        deadline_column="deadline_on",
        comparison=comparison,
        positive_class=PositiveClass.EVENT_AFTER_DEADLINE,
        missing_causes=[
            MissingEventCause(
                name="событие не произошло",
                meaning=MissingEventMeaning.NOT_OCCURRED,
                status_value="completed",
            ),
            MissingEventCause(
                name="объект исключён из обработки",
                meaning=MissingEventMeaning.EXCLUDED,
                status_value="aborted",
            ),
            MissingEventCause(
                name="событие ещё не наступило",
                meaning=MissingEventMeaning.NOT_OCCURRED,
                status_value="pending",
            ),
        ],
        estimand="событие произошло позже назначенного срока среди объектов, "
        "которые предполагалось обработать",
        expected_positive_rate=expected_positive_rate,
    )


def _bundle(
    case_id: str,
    title: str,
    rationale: str,
    findings: set[Finding],
    build: Callable[[], World],
    comparison: ComparisonMode = ComparisonMode.BY_DATE,
    caught_by: set[str] | None = None,
    lifetime: ObjectLifetime | None = None,
    reserve: bool = True,
    target_kind: TargetKind | None = None,
    expected_positive_rate: float | None = None,
) -> Bundle:
    return Bundle(
        case=Case(
            id=case_id,
            title=title,
            rationale=rationale,
            expectation=Expectation(
                findings=frozenset(findings), caught_by=frozenset(caught_by or ())
            ),
        ),
        build=build,
        outcome=_outcome(comparison, expected_positive_rate),
        lifetime=lifetime,
        reserve=reserve,
        target_kind=target_kind,
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
        caught_by={"A10"},
    ),
    _bundle(
        "undeclared-temporal-kind",
        "Временная колонка без объявленной грануляции",
        "Ядро не должно угадывать грануляцию эвристикой.",
        {Finding.UNDECLARED_TEMPORAL_KIND},
        lambda: inj.drop_temporal_declaration(build_world()),
        caught_by={"A10"},
    ),
    _bundle(
        "feature-declared-after-decision",
        "Признак честно объявлен появляющимся после решения",
        "Трение 05: декларация говорит правду, проверки контракта достаточно.",
        {Finding.FEATURE_AFTER_DECISION},
        lambda: inj.feature_declared_after_decision(build_world()),
        caught_by={"C1"},
    ),
    _bundle(
        "feature-falsely-declared-available",
        "Признак ложно объявлен доступным в момент решения",
        "Трение 05: декларация лжёт, и проверка деклараций бессильна — "
        "нужна эмпирическая сверка силы связи с исходом (N6).",
        {Finding.FEATURE_AFTER_DECISION},
        lambda: inj.feature_from_the_future(build_world()),
        caught_by={"N6"},
    ),
    _bundle(
        "outcome-component-as-feature",
        "Компонент исхода объявлен признаком",
        "Трение 05: связь возникает в формуле метки, проверки данных её не видят.",
        {Finding.OUTCOME_COMPONENT_AS_FEATURE},
        lambda: inj.outcome_component_as_feature(build_world()),
        caught_by={"C6"},
    ),
    # --- структура данных --------------------------------------------------
    _bundle(
        "surrogate-key-as-entity",
        "Суррогатный ключ выдан за идентификатор сущности",
        "Трение 01: признаки по истории объекта дали бы историю длиной в одну строку. "
        "Пересечение сущностей здесь причинно связано: объект, встречающийся в "
        "нескольких строках, при временном сплите неизбежно попадает и в обучение, "
        "и в оценку.",
        {Finding.SURROGATE_KEY_AS_ENTITY, Finding.ENTITY_OVERLAP_ACROSS_SPLITS},
        lambda: inj.surrogate_key_as_entity(build_world()),
        caught_by={"A2", "N2"},
    ),
    _bundle(
        "duplicate-rows",
        "Полные дубликаты строк",
        "Трение 01: 26% дубликатов нашлись только потому, что проверка была дописана вручную.",
        {Finding.DUPLICATE_ROWS},
        lambda: inj.duplicate_rows(build_world()),
        caught_by={"A3"},
    ),
    _bundle(
        "entity-overlap",
        "Одна сущность в разных концах периода",
        "Пересечение сущностей между обучением и оценкой при временном сплите.",
        {Finding.ENTITY_OVERLAP_ACROSS_SPLITS},
        lambda: inj.entity_overlap(build_world()),
        caught_by={"N2"},
    ),
    _bundle(
        "stale-measurement-under-zero-window",
        "Признак объявлен мгновенным, а получен тремя днями раньше",
        "F-11: окно нулевой длины утверждает одновременность, которой у данных не "
        "бывает. Внешнее ревью назвало это местом, где объявление окна остаётся "
        "необеспеченным.",
        {Finding.FEATURE_WINDOW_MISMATCH},
        lambda: inj.stale_measurements(build_world()),
        caught_by={"S7"},
    ),
    _bundle(
        "clean-measured-within-window",
        "Время измерения укладывается в объявленное окно",
        "Отрицательный контроль к S7: проверка, срабатывающая при любом объявленном "
        "времени измерения, сделала бы роль бесполезной.",
        set(),
        lambda: inj.stale_measurements(build_world(), age_days=3, lookback_days=7),
    ),
    _bundle(
        "clean-recurring-narrow-windows",
        "Долгоживущий объект с узкими окнами признаков",
        "Отрицательный контроль к N2i: проверка, срабатывающая на любом долгоживущем "
        "объекте, выглядела бы идеальной и была бы бесполезна.",
        set(),
        lambda: inj.declare_feature_windows(inj.repeated_object(build_world()), 1),
        lifetime=ObjectLifetime.RECURRING,
    ),
    _bundle(
        "feature-window-undeclared",
        "Признаки не объявили окно, по которому посчитаны",
        "F-8: среднее за неделю и мгновенное показание выглядят одинаково — колонка "
        "с числом. У долгоживущего объекта разница решающая, и молчание проверки "
        "здесь означало бы, что утечку не искали.",
        {Finding.UNDECLARED_FEATURE_WINDOW},
        lambda: inj.repeated_object(build_world()),
        caught_by={"N2i"},
        lifetime=ObjectLifetime.RECURRING,
    ),
    _bundle(
        "feature-window-overlap",
        "Окна признаков двух решений одного объекта пересекаются",
        "F-5: у долгоживущего объекта присутствие по обе стороны сплита нормально, "
        "и N2 молчит. Утечка идёт через измерения, попавшие и в обучающие признаки, "
        "и в оценочные.",
        {Finding.FEATURE_WINDOW_OVERLAP},
        lambda: inj.declare_feature_windows(inj.repeated_object(build_world()), 120),
        caught_by={"N2i"},
        lifetime=ObjectLifetime.RECURRING,
    ),
    # --- полнота периода ---------------------------------------------------
    _bundle(
        "missing-period",
        "Пропущенный месяц в середине периода",
        "Трение 01: ноябрь отсутствовал целиком, обнаружено вручную.",
        {Finding.MISSING_PERIOD},
        lambda: inj.missing_period(build_world()),
        caught_by={"A7"},
    ),
    _bundle(
        "truncated-tail",
        "Обрыв сбора данных в конце периода",
        "Трение 04: объём падал с 245 в день до 4, что выглядело как цензурирование.",
        {Finding.TRUNCATED_TAIL},
        lambda: inj.truncated_tail(build_world()),
        caught_by={"A6"},
    ),
    # --- семантика пропусков ----------------------------------------------
    _bundle(
        "status-timestamp-conflict",
        "Статус завершения без метки события",
        "Трение 02: 8 объектов со статусом завершения не имели даты события.",
        {Finding.STATUS_TIMESTAMP_CONFLICT},
        lambda: inj.status_timestamp_conflict(build_world()),
        caught_by={"A5"},
    ),
    _bundle(
        "post-treatment-missingness",
        "Пропуск признака объясняется исходом",
        "Трение 05: из 775 объектов без позиций 767 были отменены.",
        {Finding.POST_TREATMENT_MISSINGNESS},
        lambda: inj.post_treatment_missingness(build_world()),
        caught_by={"A11"},
    ),
    # --- дрейф -------------------------------------------------------------
    _bundle(
        "label-immaturity",
        "Исход недавних объектов созревает после конца наблюдения",
        "Трение 06: отбросив незрелые, получим окно из объектов с короткими сроками.",
        {Finding.LABEL_IMMATURITY},
        lambda: inj.late_maturing_labels(build_world()),
        caught_by={"A12"},
    ),
    _bundle(
        "non-stationary-target",
        "Связь признака с исходом меняется во времени",
        "Трения 07 и 12: именно это трижды переворачивало вывод этапа 0.\n"
        "Перелом приходится между вторым и третьим окном намеренно. Первая версия "
        "ломала связь в середине периода, тогда как все оценочные окна лежат в его "
        "поздней части: окна оказывались по одну сторону перелома и различались только "
        "шумом, а проверка объявляла этот шум дрейфом.",
        {Finding.NON_STATIONARY_TARGET},
        lambda: inj.non_stationary_target(build_world()),
        caught_by={"N3", "N4"},
    ),
    _bundle(
        "flipped-feature-relation",
        "Признак меняет знак связи с исходом, доля класса та же",
        "Проверяет N4 отдельно от N3: зеркалирование значений сохраняет и распределение "
        "признака, и долю положительного класса. Меняется только знак связи.",
        {Finding.UNSTABLE_FEATURE_RELATION},
        lambda: inj.flipped_feature_relation(build_world()),
        caught_by={"N4"},
    ),
    _bundle(
        "disjoint-feature-support",
        "Значения признака в окнах не пересекаются",
        "Условие осмысленности N4: пока поддержка не пересекается, смена знака означает "
        "сравнение разных участков шкалы, а не разных времён.",
        {Finding.INCOMPARABLE_SUPPORT},
        lambda: inj.disjoint_support(build_world()),
        caught_by={"N5"},
    ),
    _bundle(
        "direction-contradicts-domain",
        "Объявленное доменное направление противоречит данным",
        "Расхождение ожидания с наблюдением означает либо неверное понимание процесса, "
        "либо испорченный признак. Оба случая дороже ошибки в модели.",
        {Finding.DIRECTION_CONTRADICTS_DOMAIN},
        lambda: inj.declared_direction(build_world(), "lead_days", Direction.INCREASES),
        caught_by={"N8"},
    ),
    _bundle(
        "competing-kinds-collapsed",
        "Четыре вида события сведены в один исход молча",
        "Остаток F-8: во втором кейсе отказывал один из четырёх компонентов, и я свёл "
        "их в исход «отказало хоть что-то». Ядро не возразило, потому что о видах "
        "событий не знало.",
        {Finding.COMPETING_KINDS_COLLAPSED},
        lambda: inj.competing_event_kinds(build_world()),
        caught_by={"N12"},
    ),
    _bundle(
        "simultaneity-undeclared",
        "Конкурирующие исходы объявлены, одновременность — нет",
        "F-12: допущение «побеждает ровно один» на втором кейсе оказалось нарушено в "
        "42 моментах из 719. По таблице решений это не проверить, поэтому объявление "
        "обязательно.",
        {Finding.SIMULTANEITY_UNDECLARED},
        lambda: inj.competing_event_kinds(build_world()),
        caught_by={"N12"},
        target_kind=TargetKind.COMPETING,
    ),
    _bundle(
        "rate-contradicts-expectation",
        "Наблюдаемая доля класса вдесятеро расходится с объявленным ожиданием",
        "Внешнее ревью формы: без объявленного ожидания отличить честный дисбаланс от "
        "ошибки фильтрации или разметки нечем — доля в один процент выглядит одинаково "
        "и там, и там.",
        {Finding.RATE_CONTRADICTS_EXPECTATION},
        build_world,
        caught_by={"N13"},
        expected_positive_rate=0.01,
    ),
    _bundle(
        "undeclared-group",
        "Данные содержат уровень группы, а он не объявлен",
        "Третий и пятый кейсы: топ-10 страховщиков покрывали 44.4% строк, 37.9% инспекций "
        "приходилось на сети. Ни то ни другое ядро не видело, потому что знало единицу "
        "решения и объект, а третьего уровня в нём не было.",
        {Finding.UNDECLARED_GROUP},
        lambda: inj.grouped_objects(build_world()),
        caught_by={"N14"},
    ),
    _bundle(
        "group-overlap-across-splits",
        "Объекты одной сети по обе стороны сплита",
        "Объявленный уровень группы делает зависимость видимой: модель, видевшая часть "
        "сети, оценивается на остальной мягче, чем на действительно новой.",
        {Finding.GROUP_OVERLAP_ACROSS_SPLITS},
        lambda: inj.declared_group(inj.grouped_objects(build_world())),
        caught_by={"N14"},
    ),
    _bundle(
        "sentinel-as-value",
        "Отсутствие записано строкой «NA» и прочитано значением",
        "Четвёртый кейс: дата отмены подписки хранила «NA», читатель CSV принял её "
        "значением, и доля отмен вышла 100% вместо 22.1%. Ошибка возникает раньше всех "
        "проверок — при чтении файла — и потому портит их разом.",
        {Finding.SENTINEL_AS_VALUE},
        lambda: inj.sentinel_as_value(build_world()),
        caught_by={"A14"},
    ),
    _bundle(
        "event-before-decision",
        "Событие датировано раньше момента решения",
        "Третий кейс: две претензии со слушанием до сборки дела молча становились "
        "положительными. Правило сравнения со сроком проверяло только верхнюю границу.",
        {Finding.EVENT_BEFORE_DECISION},
        lambda: inj.event_before_decision(build_world()),
        caught_by={"A13"},
    ),
    _bundle(
        "clean-excluded-from-population",
        "Часть объектов исключена из популяции",
        "Исключение — не дефект, а законное состояние. Кейс существует потому, что до "
        "него ветвь разметки «исключён из популяции» не выполнялась ни разу, и код, "
        "отличающий исключение от незрелости, оставался непроверенным.",
        set(),
        lambda: inj.excluded_from_population(build_world()),
    ),
    _bundle(
        "clean-rate-matches-expectation",
        "Объявленное ожидание доли класса совпадает с наблюдаемым",
        "Отрицательный контроль к N13: проверка, срабатывающая при любом объявленном "
        "ожидании, сделала бы объявление бессмысленным.",
        set(),
        build_world,
        expected_positive_rate=0.12,
    ),
    _bundle(
        "clean-direction-matches-domain",
        "Объявленное направление совпадает с наблюдаемым",
        "Отрицательный контроль к N8: проверка, срабатывающая при любом объявленном "
        "направлении, сделала бы объявление бессмысленным.",
        set(),
        lambda: inj.declared_direction(build_world(), "lead_days", Direction.DECREASES),
    ),
)

BY_ID: dict[str, Bundle] = {b.id: b for b in ALL}
NEGATIVE_CONTROLS = tuple(b for b in ALL if b.case.expectation.is_negative_control)
