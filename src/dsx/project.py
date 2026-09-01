"""Объявления проекта в виде заполняемой формы.

Два кейса добавляли поверхность объявлений по одному полю к живому коду:
роли и грануляция, причины отсутствия события, направление исхода, жизненный
цикл объекта, окна признаков, время измерения, состав выборок, граница
резерва. Каждое поле обосновано находкой, но заполнить их разом как форму
никто ни разу не пробовал.

Здесь они сведены в одно место. Граница проходит честно: форма описывает
ОБЪЯВЛЕНИЯ, код проекта строит таблицу решений. Соединить пять таблиц разного
уровня декларацией нельзя, и делать вид, что можно, значило бы получить
конфигурацию, в которой прячется программа.

Форма читается из YAML и превращается в те же объекты, что раньше писались
руками. Ошибка в форме — отказ при загрузке, а не неверный результат позже.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Annotated

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from dsx.assumptions import AssumptionRegistry, Basis
from dsx.outcome import (
    ComparisonMode,
    MissingEventCause,
    MissingEventMeaning,
    OutcomeDefinition,
    PositiveClass,
)
from dsx.roles import (
    Availability,
    ColumnSpec,
    Direction,
    Evidence,
    FeatureWindow,
    Role,
    Schema,
    TemporalKind,
)
from dsx.split import Window
from dsx.task import (
    ObjectLifetime,
    OutcomeTiming,
    Simultaneity,
    TargetKind,
    TaskSpec,
)


class ColumnForm(BaseModel):
    """Одна колонка таблицы решений."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: Evidence
    role: Role
    temporal: TemporalKind | None = None
    availability: Availability = Availability.UNKNOWN
    source_of_claim: str | None = None

    window_lookback_days: float | None = None
    window_lag_days: float = 0.0
    window_source: str | None = None
    window_clock: str | None = None
    """Колонка, по времени которой строка отбирается в окно признака."""

    measured_at: str | None = None
    value_as_of: str | None = None
    """Колонка с моментом, на который зафиксировано значение этой колонки."""

    direction: Direction | None = None
    """Куда признак двигает риск по доменному знанию, до просмотра данных."""

    def to_spec(self) -> ColumnSpec:
        window = None
        if self.window_lookback_days is not None:
            if not self.window_source:
                raise ValueError(
                    f"колонка {self.name!r} объявила окно признака без источника "
                    "утверждения: чем подтверждено, что окно именно такое"
                )
            window = FeatureWindow(
                lookback_days=self.window_lookback_days,
                lag_days=self.window_lag_days,
                source_of_claim=self.window_source,
                clock=self.window_clock,
            )
        return ColumnSpec(
            name=self.name,
            role=self.role,
            temporal=self.temporal,
            availability=self.availability,
            source_of_claim=self.source_of_claim,
            window=window,
            measured_at=self.measured_at,
            value_as_of=self.value_as_of,
            direction=self.direction,
        )


class CauseForm(BaseModel):
    """Одна причина, по которой события могло не быть."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: Annotated[str, Field(min_length=1)]
    meaning: MissingEventMeaning
    status_value: str | None = None
    assumption: str | None = None

    def to_cause(self) -> MissingEventCause:
        return MissingEventCause(
            name=self.name,
            meaning=self.meaning,
            status_value=self.status_value,
            assumption=self.assumption,
        )


class OutcomeForm(BaseModel):
    """Контракт исхода."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    event_column: Annotated[str, Field(min_length=1)]
    deadline_column: Annotated[str, Field(min_length=1)]
    comparison: ComparisonMode
    positive_class: PositiveClass
    primary_kind: str | None = None
    """Предсказываемый вид события. Обязателен при конкурирующей задаче."""

    event_name: Annotated[str, Field(min_length=1)]
    """Имя события. Существительное: «закрытие обращения», «отказ узла».

    Прежде здесь стоял `estimand` — свободное предложение, в котором автор
    объявлял направление вторично и мог разойтись с вычислением. Теперь
    предложение собирает ядро, а автор называет вещи."""

    deadline_name: Annotated[str, Field(min_length=1)]
    """Имя срока. Существительное: «сутки от приёма», «назначенная дата»."""
    missing_causes: Annotated[list[CauseForm], Field(min_length=1)]

    expected_positive_rate: Annotated[float, Field(gt=0.0, lt=1.0)] | None = None
    """Ожидаемая доля положительного класса по доменному знанию, до просмотра.

    Необязательно: ожидания может не быть, и выдумывать его хуже, чем не иметь.
    Но если оно есть, расхождение с наблюдаемым означает либо ошибку фильтрации
    и разметки, либо неверное понимание процесса — и то и другое дороже ошибки
    в модели. Тот же приём, что с доменным направлением признака.
    """

    degenerate_beyond: Annotated[float, Field(gt=0.0, lt=0.5)] | None = None
    """Порог невырожденности: доля класса ближе к нулю или единице этого
    объявляется вырожденной, и прогон блокируется.

    Введено шестым кейсом. Первая его постановка дала 95.3% одного класса, и
    ни одна проверка не возразила: N13 сверяет долю только с ОЖИДАЕМОЙ, а не
    объявив ожидания, автор выключал её ровно тогда, когда она нужнее всего.

    Не объявить порог тоже нельзя: отсутствие блокирует. Умолчания нет
    намеренно — умолчание, совпадающее с честным ответом, запрещено правилами
    проекта, потому что неотличимо от невнимательности."""

    def to_definition(self) -> OutcomeDefinition:
        return OutcomeDefinition(
            event_column=self.event_column,
            deadline_column=self.deadline_column,
            comparison=self.comparison,
            positive_class=self.positive_class,
            primary_kind=self.primary_kind,
            event_name=self.event_name,
            deadline_name=self.deadline_name,
            expected_positive_rate=self.expected_positive_rate,
            degenerate_beyond=self.degenerate_beyond,
            missing_causes=[c.to_cause() for c in self.missing_causes],
        )


class TaskForm(BaseModel):
    """Тип задачи и предпосылки."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    target_kind: TargetKind
    outcome_timing: OutcomeTiming
    has_process: bool
    is_stream: bool
    object_lifetime: ObjectLifetime
    kinds_collapsed: bool | None = None
    simultaneous_kinds: Simultaneity | None = None
    simultaneous_kind_value: str | None = None

    def to_spec(self) -> TaskSpec:
        return TaskSpec(
            target_kind=self.target_kind,
            outcome_timing=self.outcome_timing,
            has_process=self.has_process,
            is_stream=self.is_stream,
            object_lifetime=self.object_lifetime,
            kinds_collapsed=self.kinds_collapsed,
            simultaneous_kinds=self.simultaneous_kinds,
            simultaneous_kind_value=self.simultaneous_kind_value,
        )


class WindowForm(BaseModel):
    """Одно оценочное окно."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: Evidence
    start_day: int
    """Смещение от первого решения, в днях."""

    stop_day: int

    @model_validator(mode="after")
    def _ordered(self) -> WindowForm:
        if self.stop_day <= self.start_day:
            raise ValueError(f"окно {self.name!r}: конец не позже начала")
        return self


class SplitForm(BaseModel):
    """Как строится временной сплит."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    windows: Annotated[list[WindowForm], Field(min_length=1)]
    reserve_from_day: int
    """Смещение границы резерва от первого решения, в днях.

    Обязательно: измерительная выборка отрезается до начала работы, иначе она
    выбирается по уже увиденным метрикам (F-9).
    """

    reserve_until_day: int | None = None
    """Верхняя граница резерва, если она нужна.

    Необязательна: её отсутствие ловится проверкой незрелости, а не
    объявлением. Требовать ответ имеет смысл там, где проверить нечем; здесь
    есть чем.

    Нужна, когда наблюдение обрывается раньше, чем последнее решение успевает
    созреть: без неё в резерв попадает незрелый хвост, и измерение по
    размеченной части становится отбором полных случаев.
    """

    @model_validator(mode="after")
    def _reserve_bounds_are_ordered(self) -> SplitForm:
        if self.reserve_until_day is not None and self.reserve_until_day <= self.reserve_from_day:
            raise ValueError(
                f"конец резерва ({self.reserve_until_day}) не позже его начала "
                f"({self.reserve_from_day})"
            )
        return self

    @model_validator(mode="after")
    def _window_names_are_unique(self) -> SplitForm:
        """Одноимённые окна затирают друг друга в учёте выборок и в долях.

        Загрязнённое окно исчезало бы из проверки независимости только из-за
        повторённого имени.
        """
        names = [w.name for w in self.windows]
        if len(names) != len(set(names)):
            raise ValueError(f"имена окон должны быть различны, объявлено: {names}")
        return self

    @model_validator(mode="after")
    def _reserve_is_beyond_every_window(self) -> SplitForm:
        late = [w.name for w in self.windows if w.stop_day > self.reserve_from_day]
        if late:
            raise ValueError(
                f"окна {late!r} заходят за границу резерва: измерительная выборка "
                "перестала бы быть независимой"
            )
        return self

    def to_windows(self, origin: dt.datetime) -> list[Window]:
        return [
            Window(
                w.name,
                origin + dt.timedelta(days=w.start_day),
                origin + dt.timedelta(days=w.stop_day),
            )
            for w in self.windows
        ]


class AssumptionForm(BaseModel):
    """Одно допущение проекта."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    statement: Evidence
    basis: Basis
    author: Evidence
    consequence: Evidence
    evidence: str | None = None


class ProjectForm(BaseModel):
    """Проект целиком: всё, что объявляется, и ничего, что вычисляется."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    title: Evidence

    observed_until: dt.datetime
    """Конец наблюдения: до какого момента исходы регистрировались.

    Не выводится из данных. Максимум даты события — это последнее СЛУЧИВШЕЕСЯ
    событие, а не конец сбора: между ними лежит период, в котором событий не
    было, и он неотличим от периода, которого в выгрузке нет. На третьем кейсе
    разница составила десять дней, и незрелость занижалась на пять тысяч строк.
    """

    observed_until_source: Evidence
    """Чем подтверждён конец наблюдения: дата выгрузки, регламент, владелец."""
    columns: Annotated[list[ColumnForm], Field(min_length=1)]
    outcome: OutcomeForm
    task: TaskForm
    split: SplitForm
    assumptions: Annotated[list[AssumptionForm], Field(min_length=1)]
    """Хотя бы одно. Проект без единого записанного допущения означает, что
    допущения принимались молча, а не что их не было."""

    @model_validator(mode="after")
    def _competing_task_names_the_kind_it_predicts(self) -> ProjectForm:
        """Конкурирующая задача обязана назвать предсказываемый вид.

        Без него метка склеивается: положительным становится событие ЛЮБОГО
        вида, и проверки идут по величине, которой задача не предсказывает.
        Пятнадцатый кейс обошёлся дороже: там ядро вовсе выключало восемь
        проверок, лишь бы не считать по склейке. Требование объявления решает
        обе беды разом.
        """
        if self.task.target_kind is not TargetKind.COMPETING:
            return self
        if not self.outcome.primary_kind:
            raise ValueError(
                "задача объявлена конкурирующей, но предсказываемый вид не назван: "
                "укажите outcome.primary_kind. Без него метка склеивает виды, и "
                "проверки пойдут по величине, которую задача не предсказывает"
            )
        return self

    def unverifiable_by_kind(self) -> dict[str, tuple[str, ...]]:
        """Непроверяемые объявления, разложенные по РОДУ.

        Прежде мера была одна: доля непроверяемого среди всех объявлений, порог
        «не более половины». Тринадцатый кейс её опроверг. В выгрузке было
        двадцать значений статуса, каждое объявлено отдельной причиной со своим
        допущением, — и доля вышла 56.9%. Форма, свалившая бы статусы в три
        кучи, дала бы долю ниже порога и была бы ХУЖЕ.

        **Мера наказывала добросовестность.** Причина в том, что она складывала
        две несравнимые совокупности:

        * вопросы о ПРИЗНАКАХ — по одному на признак, доступный в момент
          решения. Их число зависит от того, сколько признаков взял автор, и
          доля среди колонок сравнима между отраслями;
        * вопросы о ПРИЧИНАХ отсутствия события — по одному на причину. Их
          число задаёт словарь источника, а не автор: где статусов двадцать,
          там и вопросов двадцать, и объявить их меньше значит соврать;
        * ПОСТОЯННЫЕ вопросы — об исходе, сроке, жизненном цикле объекта,
          границах окон и конце наблюдения. Их всегда примерно одинаково.

        Порогом мерится только первая доля. Остальные две называются числом.
        """
        return {
            "признаки": self._feature_questions(),
            "причины": self._cause_questions(),
            "постоянные": self._fixed_questions(),
        }

    def feature_uncertainty_share(self) -> float:
        """Доля колонок, чью доступность в момент решения ядро проверить не может.

        Мера, сравнимая между отраслями: и числитель, и знаменатель растут с
        числом колонок, а не со словарём статусов источника.
        """
        return len(self._feature_questions()) / max(len(self.columns), 1)

    def unverifiable_declarations(self) -> tuple[str, ...]:
        """Объявления, проверить которые ядро не может, — вопросами.

        Прежний список вопросов порождался из допущений, записанных вручную.
        Для человека без отраслевого знания это замкнутый круг: чтобы записать
        неизвестное, надо уже понимать, где требуется предметный ответ. А самые
        дорогие решения — что считать объектом, что исходом, каким взять
        горизонт, какие признаки доступны — принимаются раньше реестра и в него
        не попадали вовсе.

        Здесь наоборот: перечисляется всё, что ядро проверить бессильно,
        независимо от того, догадался ли кто-нибудь это записать.
        """
        return (*self._feature_questions(), *self._cause_questions(), *self._fixed_questions())

    def _feature_questions(self) -> tuple[str, ...]:
        available = [c for c in self.columns if c.availability is Availability.AT_DECISION]
        return tuple(
            f"Верно ли, что значение {column.name!r} было в системе в момент решения? "
            f"Объявлено со ссылкой на {column.source_of_claim!r}, но сверить это ядру "
            "нечем (иначе признак знает будущее, и оценка завышена)"
            for column in available
        )

    def _cause_questions(self) -> tuple[str, ...]:
        return tuple(
            f"Верно ли, что отсутствие события по причине {cause.name!r} означает "
            f"{cause.meaning.value!r}? (иначе строки молча попадают не в тот класс "
            "либо зря исключаются из популяции)"
            for cause in self.outcome.missing_causes
        )

    def _fixed_questions(self) -> tuple[str, ...]:
        questions: list[str] = []
        questions.append(
            f"Верно ли, что исходом является {self.outcome.event_column!r}, а сроком "
            f"{self.outcome.deadline_column!r}? (иначе отчёт считает не то, что заявлено "
            "словами)"
        )
        questions.append(
            f"Верно ли, что положительным исходом считается {self.outcome.positive_class.value!r}? "
            "(иначе доля класса означает обратное написанному)"
        )
        questions.append(
            f"Верно ли, что объект {self.task.object_lifetime.value!r}? (иначе повторные "
            "решения по одному объекту считаются разными объектами, и утечка между "
            "окнами невидима)"
        )
        questions.append(
            "Были ли границы окон и резерва выбраны ДО просмотра метрик? (иначе резерв "
            "независим лишь формально)"
        )
        questions.append(
            f"Верно ли, что исходы регистрировались до {self.observed_until:%Y-%m-%d}? "
            f"Объявлено со ссылкой на {self.observed_until_source!r} (иначе незрелость "
            "занижена, и часть строк размечена как наблюдение)"
        )
        if self.task.kinds_collapsed is not None:
            questions.append(
                "Верно ли, что вмешательство одно на все виды события? (иначе модель "
                "предсказывает не то, чем управляют)"
            )
        return tuple(questions)

    def declaration_count(self) -> int:
        """Сколько решений форма фиксирует.

        Считается по единицам выбора, а не по полям: колонка со всеми своими
        свойствами — одно решение, каждое поле контракта исхода — одно,
        каждое окно — одно. Нужен для доли непроверяемого: длина списка
        вопросов сама по себе ни о чём не говорит.
        """
        return (
            len(self.columns)
            + 6  # event/deadline column и имя, comparison, positive_class
            + len(self.outcome.missing_causes)
            + 5  # target_kind, outcome_timing, has_process, is_stream, object_lifetime
            + len(self.split.windows)
            + 2  # reserve_from_day, reserve_until_day
            + len(self.assumptions)
            + 1  # observed_until
        )

    def schema_spec(self) -> Schema:
        return Schema(columns=[c.to_spec() for c in self.columns])

    def registry(self) -> AssumptionRegistry:
        registry = AssumptionRegistry()

        # Допущения, принятые из-за неразличимости причин отсутствия события,
        # — такие же допущения проекта. Прежде они жили только в контракте
        # исхода и в реестр не попадали, то есть в отчёте их не было.
        for cause in self.outcome.missing_causes:
            if cause.assumption and cause.assumption.strip():
                registry.record(
                    cause.assumption,
                    basis=Basis.DOMAIN_KNOWLEDGE,
                    author="контракт исхода",
                    consequence=(
                        f"причина {cause.name!r} неотличима по данным, и её строки "
                        f"размечаются как {cause.meaning.value}"
                    ),
                )

        for item in self.assumptions:
            registry.record(
                item.statement,
                basis=item.basis,
                author=item.author,
                consequence=item.consequence,
                evidence=item.evidence,
            )
        return registry


def load(path: Path) -> ProjectForm:
    """Прочитать форму. Ошибка в объявлениях — отказ здесь, а не позже."""
    return ProjectForm(**yaml.safe_load(path.read_text(encoding="utf-8")))
