"""Временной сплит по моменту узнавания исхода.

Обучение видит не всё, что случилось до отсечки, а только то, чей исход был
ИЗВЕСТЕН к ней. Разница существенна: на этапе 0 между выборками выпало 3342
объекта, купленных до отсечки, но узнанных после. Использовать их в обучении
значило бы знать будущее.

Момент узнавания выводится из определения исхода, а не объявляется отдельно.
Для исхода вида «событие не произошло к сроку» ответ известен в конце срока —
ждать фактического наступления не нужно. На этапе 0 это сократило зазор с 29
дней до 26, а максимум — с 209 до 145.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import polars as pl

from dsx.evals.world import World
from dsx.join import Cardinality, guarded_join
from dsx.label import LABEL, REASON, OutcomeReason, compute, observable
from dsx.outcome import ComparisonMode, OutcomeDefinition
from dsx.roles import Role
from dsx.samples import Extent, Training

KNOWN_AT = "__label_known_at"


def with_label_known_at(
    world: World, definition: OutcomeDefinition, snapshot: dt.datetime | None = None
) -> pl.DataFrame:
    """Добавить момент узнавания исхода к размеченной таблице.

    Исход известен в момент события, если оно произошло в срок; иначе — когда
    срок истёк и стало ясно, что событие не успело.

    Когда именно срок истёк, зависит от объявленного способа сравнения. При
    посуточном сравнении ответ появляется с началом следующей календарной даты,
    при прямом — сразу за моментом срока. Первая версия добавляла ровно сутки
    в обоих случаях и теряла лишние часы обучающих строк.
    """
    frame = compute(world, definition, snapshot)
    event = pl.col(definition.event_column)
    deadline = pl.col(definition.deadline_column)

    if definition.comparison is ComparisonMode.BY_DATE:
        in_time = event.dt.date() <= deadline.dt.date()
        expired = deadline.dt.date().cast(pl.Datetime).dt.offset_by("1d")
    else:
        in_time = event <= deadline
        expired = deadline

    return frame.with_columns(
        pl.when(event.is_not_null() & in_time).then(event).otherwise(expired).alias(KNOWN_AT)
    )


@dataclass(frozen=True)
class Window:
    """Оценочное окно."""

    name: str
    start: dt.datetime
    stop: dt.datetime


@dataclass
class Part:
    """Одна часть сплита: чему учимся и на чём оцениваемся."""

    window: Window
    train: pl.DataFrame | None
    """Обучающая часть. После прогона — `None`: она передана журналу выборок и
    выдаётся только через `SampleLedger.training` вместе с записью обучения
    (DS-010, класс 19 журнала повторов). Число строк остаётся в
    `SplitResult.train_rows`."""

    evaluate: pl.DataFrame

    @property
    def name(self) -> str:
        return self.window.name


@dataclass
class SplitResult:
    parts: list[Part] = field(default_factory=list)
    dropped_not_yet_known: int = 0
    """Объекты, выпавшие перед ПЕРВЫМ окном. Сохранено ради совместимости
    отчётов; полная картина — в `dropped_by_window`."""

    dropped_by_window: dict[str, int] = field(default_factory=dict)
    """Выпавшие перед каждым окном по отдельности.

    Первая версия считала зазор только перед первым окном, и строка отчёта
    «выпало между выборками» была неполной: у последующих окон зазоры свои и
    другой величины.
    """

    immature: dict[str, int] = field(default_factory=dict)
    """Объекты оценочного окна, чей исход не наблюдаем к концу наблюдения."""

    outside_everything: int = 0
    """Строки за верхней границей резерва: ни в обучении, ни в оценке, ни в
    измерении."""

    outside_immature: int = 0
    """Из них незрелых."""

    reserved_immature: int = 0
    """Строки резерва, чей исход ещё не наблюдаем."""

    purged_by_window: dict[str, int] = field(default_factory=dict)
    """Обучающие решения, вычищенные из-за пересечения окон признаков.

    Проверка N2i возражает верно, но ответить на неё объявлением нельзя:
    механизм, удовлетворяемый обещанием, защищает хуже своего отсутствия.
    Поэтому пересечение снимается делом — удалением обучающих решений,
    чьи окна достают до оценочных, — и цена этого записывается здесь.
    """

    reserved_rows: int = 0
    """Сколько строк в измерительной выборке.

    Сами данные после регистрации переезжают в журнал расхода и с результата
    сплита исчезают: держать их публичным полем значит оставить обход учёта
    открытым. Здесь остаётся только число для отчёта.
    """

    train_rows: dict[str, int] = field(default_factory=dict)
    """Строк в обучающей части каждого окна — для отчёта, когда сами части уже
    переданы журналу."""

    shared_objects: dict[str, int] = field(default_factory=dict)
    """Объекты по обе стороны сплита каждого окна, посчитанные до передачи
    обучающих частей журналу."""

    reserved: pl.DataFrame | None = None
    """Измерительная выборка, отрезанная до начала работы и не входящая ни в
    одно окно.

    Поле обнуляется сразу после передачи данных журналу. Читать его вне
    построения сплита нельзя: единственный законный доступ — `checkout`.

    Второй кейс показал, что скользящие окна строятся вложенно и потому
    пересекаются по составу: 42% на одном протоколе, 61% на другом. Ни одно из
    них не годится в измерительный инструмент после того, как хоть одно
    использовалось для выбора, а выбор неизбежен. Резерв разводит два вопроса:
    окна отвечают, устойчива ли связь во времени, резерв — сколько это стоит.
    """

    def part(self, name: str) -> Part:
        return next(p for p in self.parts if p.name == name)


def split_by_windows(
    world: World,
    definition: OutcomeDefinition,
    windows: list[Window],
    snapshot: dt.datetime,
    reserve_from: dt.datetime | None = None,
    reserve_until: dt.datetime | None = None,
) -> SplitResult:
    """Построить обучающие и оценочные части по временным окнам.

    Для каждого окна обучение — объекты, чей исход был известен до его начала.
    Оценка — объекты, решение по которым принято внутри окна.

    `reserve_from` отрезает измерительную выборку: решения с этого момента не
    попадают ни в одно окно. Момент задаётся до начала работы, иначе резерв
    выбирается по уже увиденным метрикам и перестаёт быть инструментом.
    """
    labelled = with_label_known_at(world, definition, snapshot)
    decision = world.schema.decision_time.name
    result = SplitResult()

    if reserve_from is not None:
        late = [w.name for w in windows if w.stop > reserve_from]
        if late:
            raise ValueError(
                f"окна {late!r} заходят за границу резерва {reserve_from:%Y-%m-%d}: "
                "измерительная выборка перестала бы быть независимой"
            )
        # Резерв НЕ фильтруется по известности исхода: отбор по исходу — это
        # отбор полных случаев, который приукрашивает итоговую метрику ровно
        # там, где она должна быть честной. Незрелость внутри резерва видна
        # отдельно и попадает в отчёт.
        reserved = labelled.filter(pl.col(decision) >= reserve_from)
        if reserve_until is not None:
            # Хвост за верхней границей не принадлежит ничему. Он остаётся
            # видимым отдельным числом: строки, которые нельзя ни обучить, ни
            # оценить, ни измерить, — это не пустое место, а цена горизонта.
            beyond = reserved.filter(pl.col(decision) >= reserve_until)
            result.outside_everything = beyond.height
            result.outside_immature = beyond.filter(
                pl.col(REASON) == OutcomeReason.IMMATURE.value
            ).height
            reserved = reserved.filter(pl.col(decision) < reserve_until)
        result.reserved = reserved
        result.reserved_immature = reserved.filter(
            pl.col(REASON) == OutcomeReason.IMMATURE.value
        ).height
        labelled = labelled.filter(pl.col(decision) < reserve_from)

    for window in windows:
        train = labelled.filter(pl.col(LABEL).is_not_null() & (pl.col(KNOWN_AT) < window.start))
        evaluate = labelled.filter(
            (pl.col(decision) >= window.start) & (pl.col(decision) < window.stop)
        )
        result.parts.append(Part(window=window, train=train, evaluate=evaluate))

        # Незрелость — не всякая пустая метка. Исключённый из популяции и
        # цензурированный объект тоже без метки, но период наблюдения тут ни
        # при чём, и считать их незрелыми значит диагностировать не то.
        immature = evaluate.filter(pl.col(REASON) == OutcomeReason.IMMATURE.value).height
        if immature:
            result.immature[window.name] = immature

    for window in windows:
        before_cutoff = labelled.filter(pl.col(decision) < window.start)
        result.dropped_by_window[window.name] = before_cutoff.filter(
            pl.col(KNOWN_AT) >= window.start
        ).height
    if windows:
        result.dropped_not_yet_known = result.dropped_by_window[windows[0].name]

    return result


def entity_overlap(parts: list[Part], world: World, role: Role | None = None) -> dict[str, int]:
    """Значения ключа, встречающиеся и в обучении, и в оценке одного окна.

    Без указания роли берётся естественный ключ, то есть объект реального мира.
    Роль указывают, когда объект долгоживущий: тогда его присутствие по обе
    стороны нормально, а утечкой является повтор самой ЕДИНИЦЫ РЕШЕНИЯ.
    """
    if role is not None:
        keys = world.schema.by_role(role)
    else:
        keys = world.schema.by_role(Role.NATURAL_KEY) or world.schema.by_role(Role.ENTITY_ID)
    if not keys:
        return {}
    key = keys[0].name

    overlaps: dict[str, int] = {}
    for part in parts:
        if key not in part.train.columns or key not in part.evaluate.columns:
            continue
        shared = set(part.train[key].unique().to_list()) & set(
            part.evaluate[key].unique().to_list()
        )
        if shared:
            overlaps[part.name] = len(shared)
    return overlaps


def reserved_extent(result: SplitResult, world: World) -> Extent | None:
    """Состав зарезервированной измерительной выборки."""
    if result.reserved is None or result.reserved.height == 0:
        return None
    return _extent_of_frame(result.reserved, world)


def group_candidates(
    world: World, min_groups: int, min_share: float
) -> dict[str, tuple[int, float]]:
    """Колонки, похожие на уровень ГРУППЫ над объектом.

    Группа отличается от категории числом значений. Сеть — это тысячи названий
    по два-три заведения; тип заведения — два десятка значений на тридцать
    тысяч. Поэтому кандидатом считается колонка, у которой различных значений
    сопоставимо много с числом объектов.

    ПОРОГИ ЗДЕСЬ НЕ ЖИВУТ, и это не мелочь. Они объявлены полями проверки N14
    (`checks/split_checks.py`), потому что суждение о данных выносит она, а
    здесь только счёт. Пока `MIN_GROUPS` лежал константой в этом модуле,
    измеритель порогов его не видел вовсе: он мутирует `checks/` и `measure.py`
    и отчитывался «47 из 50», не касаясь порога, из-за которого девятнадцатый
    кейс потерял положительный контроль.

    Умолчаний у параметров нет намеренно. Умолчание позволило бы вызвать
    подсказку, не назвав порога, — и порог снова стал бы невидимым.

    Возвращает имя колонки, число многообъектных групп и долю строк в них.
    """
    keys = world.schema.by_role(Role.NATURAL_KEY) or world.schema.by_role(Role.ENTITY_ID)
    if not keys:
        return {}
    key = keys[0].name
    frame = world.main
    if key not in frame.columns:
        return {}

    objects = frame[key].n_unique()
    declared = {c.name for c in world.schema.by_role(Role.GROUP_ID)}
    found: dict[str, tuple[int, float]] = {}

    for column in world.schema.columns:
        name = column.name
        if name in declared or name == key or name not in frame.columns:
            continue
        if column.role in (Role.ENTITY_ID, Role.NATURAL_KEY, Role.GROUP_ID):
            continue
        if frame[name].dtype != pl.String:
            continue

        values = frame[name].n_unique()
        # Категория — горстка значений; группа — их столько, что по ним можно
        # разделить выборку. Прежнее условие мерило долю от числа объектов и
        # тем самым отсекало группы крупнее десяти объектов.
        if values < min_groups or values >= objects:
            continue

        sized = frame.group_by(name).agg(pl.col(key).n_unique().alias("objects"))
        multi = sized.filter(pl.col("objects") > 1)
        if multi.is_empty():
            continue
        rows = frame.filter(pl.col(name).is_in(multi[name].implode())).height
        share = rows / frame.height
        if share >= min_share:
            found[name] = (multi.height, share)
    return found


def group_overlap(parts: list[Part], world: World) -> dict[str, tuple[int, float]]:
    """Группы, встречающиеся и в обучении, и в оценке одного окна.

    Для объекта пересечение при долгом жизненном цикле нормально; для ГРУППЫ
    оно означает, что модель видела другие объекты той же сети и потому
    оценивается мягче, чем на действительно новой.
    """
    groups = world.schema.by_role(Role.GROUP_ID)
    if not groups:
        return {}
    column = groups[0].name

    overlaps: dict[str, tuple[int, float]] = {}
    for part in parts:
        if column not in part.train.columns or column not in part.evaluate.columns:
            continue
        shared = set(part.train[column].drop_nulls().unique().to_list()) & set(
            part.evaluate[column].drop_nulls().unique().to_list()
        )
        if not shared:
            continue
        touched = part.evaluate.filter(pl.col(column).is_in(list(shared))).height
        overlaps[part.name] = (len(shared), touched / max(part.evaluate.height, 1))
    return overlaps


def extent_of(part: Part, world: World) -> Extent:
    """Состав части сплита: единицы решения обучения И оценки вместе.

    Обе половины входят намеренно. Загрязнение идёт не только через оценочные
    строки: на втором кейсе выборка w2 обучалась на днях, которые в w0 были
    оценочными, и именно по ним принимался выбор.
    """
    frame = pl.concat([part.train, part.evaluate], how="vertical_relaxed")
    return _extent_of_frame(frame, world)


def _extent_of_frame(frame: pl.DataFrame, world: World) -> Extent:
    moment = world.schema.decision_time.name
    return Extent(
        units=units_of(frame, world),
        since=frame[moment].min(),
        until=frame[moment].max(),
    )


def units_of(frame: pl.DataFrame, world: World) -> frozenset[str]:
    """Единицы решения кадра — значения колонки с ролью `ENTITY_ID`."""
    keys = world.schema.by_role(Role.ENTITY_ID)
    if not keys:
        raise ValueError("состав выборки требует объявленной единицы решения")
    # Идентификаторы приводятся к строке здесь и только здесь: целочисленные
    # ключи встречаются чаще строковых, и падение на регистрации выборок
    # означало бы, что ядро работает лишь на данных с текстовыми ключами.
    return frozenset(str(value) for value in frame[keys[0].name].to_list())


def training_of(part: Part, world: World) -> Training | None:
    """Обучающая часть окна в том виде, в каком её принимает журнал выборок.

    Момент знания меток — максимум `KNOWN_AT` по строкам обучения: столько же
    строго, сколько `split_by_windows` отбирает их по началу окна. `None` —
    обучающая часть пуста.
    """
    if part.train is None:
        raise ValueError(f"обучающая часть окна {part.name!r} уже передана журналу")
    if part.train.height == 0:
        return None
    return Training(
        frame=part.train,
        rows=part.train.height,
        units=units_of(part.train, world),
        labels_known_until=part.train[KNOWN_AT].max(),
    )


def feature_window_overlap(parts: list[Part], world: World, lookback_days: float) -> dict[str, int]:
    """Оценочные решения, чьё окно признаков задевает обучающие решения.

    У долгоживущего объекта присутствие по обе стороны сплита нормально.
    Утечка возникает иначе: признак оценочного решения посчитан по интервалу,
    который захватывает измерения, вошедшие в признаки обучающих решений того
    же объекта.

    Окно решения в момент t достаёт назад на lookback. Значит окна двух
    решений одного объекта пересекаются, когда между решениями меньше
    lookback: оценочное решение в момент e задевает обучающее в момент t
    при e <= t + lookback.
    """
    keys = world.schema.by_role(Role.NATURAL_KEY) or world.schema.by_role(Role.ENTITY_ID)
    if not keys:
        return {}
    key = keys[0].name
    moment = world.schema.decision_time.name
    reach = pl.lit(dt.timedelta(days=lookback_days))

    overlaps: dict[str, int] = {}
    for part in parts:
        if key not in part.train.columns or key not in part.evaluate.columns:
            continue
        last_train = part.train.group_by(key).agg(pl.col(moment).max().alias("__last_train"))
        # Грануляция объявляется даже там, где она очевидна по построению:
        # group_by даёт одну строку на ключ. Необъявленное соединение в ядре —
        # то же самое, что необъявленное в проекте.
        touching = (
            guarded_join(
                part.evaluate,
                last_train,
                on=[key],
                expect=Cardinality.MANY_TO_ONE,
                how="inner",
            )
            .filter(pl.col(moment) <= pl.col("__last_train") + reach)
            .height
        )
        if touching:
            overlaps[part.name] = touching
    return overlaps


def purge_overlapping_train(
    parts: list[Part], world: World, lookback_days: float
) -> dict[str, int]:
    """Убрать из обучения решения, чьи окна признаков достают до оценочных.

    Обратная сторона `feature_window_overlap`: та считает нарушение, эта его
    снимает. Чистится ОБУЧЕНИЕ, а не оценка. Оценочная выборка — измерительный
    инструмент, и подрезать её под удобный ответ значит менять вопрос; обучающие
    же решения расходуемы, и их потеря — честная цена независимости.

    Оценочное решение в момент e задевает обучающее в момент t при
    e <= t + lookback. Значит опасны обучающие решения объекта с
    t >= min(e) - lookback, и удаляются именно они.

    Возвращает, сколько решений удалено в каждом окне. Ноль удалённых
    возможен: если объект в оценке не встречается, чистить нечего.
    """
    keys = world.schema.by_role(Role.NATURAL_KEY) or world.schema.by_role(Role.ENTITY_ID)
    if not keys or lookback_days <= 0.0:
        return {}
    key = keys[0].name
    moment = world.schema.decision_time.name
    reach = pl.lit(dt.timedelta(days=lookback_days))

    purged: dict[str, int] = {}
    for part in parts:
        if key not in part.train.columns or key not in part.evaluate.columns:
            continue
        first_eval = part.evaluate.group_by(key).agg(pl.col(moment).min().alias("__first_eval"))
        # Грануляция объявляется и здесь: group_by даёт одну строку на ключ.
        marked = guarded_join(
            part.train,
            first_eval,
            on=[key],
            expect=Cardinality.MANY_TO_ONE,
            how="left",
        )
        # Объект, которого в оценке нет, пересечься не может: __first_eval
        # пуст, сравнение даёт null, и строка остаётся.
        kept = marked.filter(
            pl.col("__first_eval").is_null() | (pl.col(moment) < pl.col("__first_eval") - reach)
        ).drop("__first_eval")
        removed = part.train.height - kept.height
        if removed:
            part.train = kept
            purged[part.name] = removed
    return purged


def positive_rates(parts: list[Part]) -> dict[str, float]:
    """Доля положительного класса по окнам.

    На этапе 0 она менялась от 1.95% до 20.8%, и это ломало и калибровку, и
    выбор порога независимо от качества модели.
    """
    return {
        part.name: float(observable(part.evaluate)[LABEL].mean())
        for part in parts
        if observable(part.evaluate).height
    }
