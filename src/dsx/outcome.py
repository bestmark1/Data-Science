"""Определение исхода.

Исход вычисляется по объявленному правилу, а не произвольным выражением.
Причина в том, что самая дорогая ошибка первого кейса сидела именно в способе
сравнения: момент времени сравнивался с датой, и 16.5% положительных меток
оказались ложными. Пока сравнение скрыто внутри кода анализа, проверить его
нечем.

Второй кейс показал вторую ошибку: трактовка отсутствия события была привязана
к колонке статуса. В обслуживании оборудования статуса нет вовсе, а причин
отсутствия отказа не меньше восьми. Требование удовлетворялось выдуманной
колонкой — механизм, который принимает фиктивное заполнение, защищает хуже
своего отсутствия.

Поэтому трактовка привязана к перечню ПРИЧИН, а не к колонке. Причина,
неразличимая по данным, не исчезает: она обязана назвать принимаемое допущение.
"""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dsx.roles import Schema, TemporalKind

RESERVED_DIRECTION_WORDS = frozenset(
    {
        "позже",
        "раньше",
        "ранее",
        "до",
        "после",
        "прежде",
        "менее",
        "более",
        "свыше",
        "течение",
        "пределах",
        "успеет",
        "уложится",
        "опоздает",
        "просрочка",
    }
)
"""Слова, которыми о сроке высказывается СУЖДЕНИЕ. Ядро оставляет их за собой:
в имени они означали бы, что направление объявлено дважды и из разных мест."""


def _contains_word(text: str, word: str) -> bool:
    """Есть ли слово в тексте как отдельное слово, а не как часть другого.

    Без этого «до» нашлось бы в «доставка», а «более» — в «наиболее».
    """
    return word in re.findall(r"[а-яёa-z]+", text.lower())


class ComparisonMode(StrEnum):
    """Как сравниваются две временные колонки."""

    DIRECT = "direct"
    """Как есть. Допустимо только при совпадающей грануляции."""

    BY_DATE = "by_date"
    """Обе стороны приводятся к календарной дате явно."""


class PositiveClass(StrEnum):
    """Какая сторона сравнения со сроком считается положительным исходом.

    Второй кейс показал, что направление было зашито в ядро: метка всегда
    означала «событие позже срока». Задача обслуживания оборудования требует
    обратного — положительным является отказ В ПРЕДЕЛАХ горизонта.

    Я объявил estimand словами «отказ в течение 30 дней», а вычислялось
    «отказ позже 30 дней». Ядро не возразило: estimand был свободным текстом
    и с вычислением не связан. Обе доли, названные в отчётах второго кейса,
    означали не то, что написано.

    Умолчания здесь нет намеренно. Направление объявляется всегда.
    """

    EVENT_AFTER_DEADLINE = "event_after_deadline"
    """Положительно, когда событие произошло позже срока или не произошло
    вовсе. Опоздание доставки, невозврат кредита."""

    EVENT_WITHIN_DEADLINE = "event_within_deadline"
    """Положительно, когда событие произошло в пределах срока. Отказ
    оборудования в горизонте, отток клиента за квартал."""


class MissingEventMeaning(StrEnum):
    """Что означает отсутствие события (C4)."""

    NOT_OCCURRED = "not_occurred"
    """Событие не произошло — это наблюдение, а не пропуск."""

    UNOBSERVED = "unobserved"
    """Исход ещё не наблюдаем; строка исключается."""

    EXCLUDED = "excluded"
    """Объект не предполагался к обработке; исключается из популяции."""


class MissingEventCause(BaseModel):
    """Одна причина, по которой события могло не быть.

    Причина, неразличимая по данным, не исчезает. Она обязана назвать
    принимаемое допущение, иначе растворится в умолчании.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: Annotated[str, Field(min_length=1)]
    meaning: MissingEventMeaning

    status_value: str | None = None
    """Значение статуса, по которому причина различима. None означает, что по
    данным она неотличима от прочих неразличимых причин."""

    assumption: str | None = None
    """Что принимается на веру, если причина неразличима. Обязательно."""

    @property
    def distinguishable(self) -> bool:
        return self.status_value is not None

    @model_validator(mode="after")
    def _indistinguishable_needs_an_assumption(self) -> MissingEventCause:
        if not self.distinguishable and not (self.assumption or "").strip():
            raise ValueError(
                f"причина {self.name!r} неразличима по данным и обязана назвать "
                "принимаемое допущение: неназванное допущение неотличимо от "
                "его отсутствия"
            )
        return self

    def __str__(self) -> str:
        mark = f"по статусу {self.status_value!r}" if self.distinguishable else "неразличима"
        return f"{self.name} → {self.meaning.value} ({mark})"


class OutcomeDefinition(BaseModel):
    """Исход как сравнение момента события с назначенным сроком."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    event_column: Annotated[str, Field(min_length=1)]
    deadline_column: Annotated[str, Field(min_length=1)]
    comparison: ComparisonMode
    positive_class: PositiveClass
    """Какая сторона считается положительным исходом. Умолчания нет: направление,
    выбранное молча, дало на втором кейсе долю, означавшую обратное."""

    missing_causes: Annotated[list[MissingEventCause], Field(min_length=1)]
    """Причины отсутствия события. Пустой список запрещён: склейка разных
    причин в один класс добавила 11.3% ложных положительных на первом кейсе."""

    expected_positive_rate: float | None = None
    """Ожидаемая доля положительного класса, объявленная до просмотра данных."""

    degenerate_beyond: float | None = None
    """Порог невырожденности: доля класса ближе к нулю или единице этого
    объявляется вырожденной.

    Объявляется автором, а не ядром, потому что общего числа тут нет: доля в
    один процент вырождена для просрочки доставки и совершенно законна для
    мошенничества. Порог, назначенный ядром, кричал бы на вторых и молчал бы
    там, где автор считает вырожденным и десять процентов.

    Объявление проверяется данными — тем и отличается от обещания."""

    event_name: Annotated[str, Field(min_length=1)]
    """Как называется событие ИМЕНЕМ: «закрытие обращения», «отказ узла».

    Существительное, а не утверждение. Всё, что несёт логическую силу —
    направление, сравнение, обращение с ненаблюдаемым, — дописывает ядро."""

    deadline_name: Annotated[str, Field(min_length=1)]
    """Как называется срок ИМЕНЕМ: «сутки от приёма», «назначенная дата»."""

    @model_validator(mode="after")
    def _names_do_not_state_the_logic(self) -> OutcomeDefinition:
        """Автор называет вещи, ядро высказывает о них суждение.

        Второй кейс: estimand был свободным текстом, я написал «отказ в течение
        30 дней», а вычислялось «отказ позже 30 дней». Ядро не возразило,
        потому что сверять свободный текст с вычислением нечем.

        Отсюда разделение источников. Автор объявляет ИМЕНА, направление
        объявляет `positive_class`, и предложение собирает `estimand`. Разойтись
        словам с вычислением негде: источник один.

        Проверка ниже стережёт границу. Написав в имени «в течение», автор
        вернул бы направление в текст — и вернул бы вместе с ним ровно ту
        ошибку, ради которой поле разбиралось.
        """
        for field, value in (
            ("event_name", self.event_name),
            ("deadline_name", self.deadline_name),
        ):
            said = sorted(w for w in RESERVED_DIRECTION_WORDS if _contains_word(value, w))
            if said:
                raise ValueError(
                    f"{field} содержит слова направления {said}: "
                    f"{value!r}. Имя называет вещь, а направление объявляет "
                    "positive_class и высказывает ядро. Слово в имени вернуло бы "
                    "расхождение между написанным и вычисляемым"
                )
        return self

    @property
    def estimand(self) -> str:
        """Что оценивается — предложение, СОБРАННОЕ из вычисляемого.

        Не поле. Автор его не пишет и переписать не может: каждое слово,
        несущее логику, взято из величины, по которой ядро считает.
        """
        side = "не позже" if self.positive_class is PositiveClass.EVENT_WITHIN_DEADLINE else "позже"
        head = (
            f"положительным считается: {self.event_name} наступает {side}, чем {self.deadline_name}"
        )
        if self.positive_class is PositiveClass.EVENT_AFTER_DEADLINE:
            head += ", либо не наступает вовсе"
        scale = (
            " Сравнение приведено к календарной дате."
            if self.comparison is ComparisonMode.BY_DATE
            else " Сравнение без приведения грануляции."
        )
        by_meaning: dict[str, list[str]] = {}
        for cause in self.missing_causes:
            by_meaning.setdefault(cause.meaning.value, []).append(cause.name)
        causes = " ".join(
            f"Отсутствие события по причине {', '.join(repr(n) for n in sorted(names))} "
            f"означает {meaning!r}."
            for meaning, names in sorted(by_meaning.items())
        )
        return f"{head}.{scale} {causes}".strip()

    @model_validator(mode="after")
    def _status_values_are_unique(self) -> OutcomeDefinition:
        values = [c.status_value for c in self.missing_causes if c.distinguishable]
        if len(values) != len(set(values)):
            raise ValueError("одно значение статуса объявлено для нескольких причин")
        return self

    def components(self) -> frozenset[str]:
        """Колонки, участвующие в вычислении исхода."""
        return frozenset({self.event_column, self.deadline_column})

    @property
    def distinguishable(self) -> tuple[MissingEventCause, ...]:
        return tuple(c for c in self.missing_causes if c.distinguishable)

    @property
    def indistinguishable(self) -> tuple[MissingEventCause, ...]:
        return tuple(c for c in self.missing_causes if not c.distinguishable)

    @property
    def conflated(self) -> bool:
        """Смешаны ли неразличимые причины с РАЗНЫМ смыслом.

        Если да, строку, попавшую в эту группу, разметить нельзя: неизвестно,
        наблюдение это, цензура или исключение из популяции.
        """
        return len({c.meaning for c in self.indistinguishable}) > 1

    def status_meaning(self) -> dict[str, MissingEventMeaning]:
        return {c.status_value: c.meaning for c in self.distinguishable if c.status_value}

    def fallback_meaning(self) -> MissingEventMeaning | None:
        """Смысл для строк, не отнесённых ни к одной различимой причине.

        None, когда неразличимые причины имеют разный смысл: такие строки
        размечать нельзя, и это отдельная находка.
        """
        meanings = {c.meaning for c in self.indistinguishable}
        return meanings.pop() if len(meanings) == 1 else None

    def assumptions(self) -> tuple[str, ...]:
        """Допущения, принятые из-за неразличимости причин."""
        return tuple(c.assumption for c in self.indistinguishable if c.assumption)


class OutcomeContractError(Exception):
    """Определение исхода несовместимо со схемой."""


def validate_outcome(definition: OutcomeDefinition, schema: Schema) -> None:
    """Проверить определение исхода против объявленной схемы."""
    left = schema.get(definition.event_column)
    right = schema.get(definition.deadline_column)

    for name, column in (
        (definition.event_column, left),
        (definition.deadline_column, right),
    ):
        if column is None:
            raise OutcomeContractError(f"колонка {name!r} отсутствует в схеме")
        if column.temporal is None:
            raise OutcomeContractError(
                f"колонка {name!r} участвует в вычислении исхода "
                "и обязана объявить временную грануляцию"
            )

    assert left is not None and right is not None
    if definition.comparison is ComparisonMode.DIRECT and left.temporal is not right.temporal:
        date_side = left if left.temporal is TemporalKind.DATE else right
        instant_side = right if date_side is left else left
        raise OutcomeContractError(
            f"{date_side.name!r} хранит дату, {instant_side.name!r} — момент времени. "
            "Прямое сравнение пометит событие в тот же день как произошедшее позже. "
            "Используйте comparison=by_date."
        )
