"""Проверки, требующие построенного временного сплита."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from dsx.checks.base import Context, NotApplicable, Signal
from dsx.evals.case import Finding
from dsx.label import LABEL
from dsx.roles import Role
from dsx.split import (
    SplitResult,
    entity_overlap,
    feature_window_overlap,
    group_candidates,
    group_overlap,
)
from dsx.task import ObjectLifetime, Premise


def _require_split(context: Context) -> SplitResult:
    if not isinstance(context.split, SplitResult):
        raise NotApplicable("временной сплит не построен")
    return context.split


@dataclass(frozen=True)
class EntityOverlapAcrossSplits:
    """N2. Один и тот же объект и в обучении, и в оценке.

    Проверка выведена на заказах, где объект одноразов: его появление по обе
    стороны сплита означало утечку. Второй кейс показал скрытое допущение —
    сотня машин, наблюдаемых полтора года, даёт полное пересечение при любом
    сплите, и это повторные измерения, а не утечка.

    Поэтому предмет проверки зависит от жизненного цикла. У одноразового
    объекта утечка — присутствие объекта. У долгоживущего — повтор самой
    единицы решения; присутствие объекта нормально.
    """

    requirement: str = "N2"
    premises: frozenset[Premise] = frozenset({Premise.UNIVERSAL})
    detects: frozenset[Finding] = frozenset({Finding.ENTITY_OVERLAP_ACROSS_SPLITS})

    def run(self, context: Context) -> list[Signal]:
        split = _require_split(context)
        recurring = context.task.object_lifetime is ObjectLifetime.RECURRING

        if not recurring:
            # Жизненный цикл не объявлен или объект одноразов. Неизвестность
            # трактуется строго: объявить её обязывает A2.
            overlaps = entity_overlap(split.parts, context.world)
            return [
                Signal(
                    Finding.ENTITY_OVERLAP_ACROSS_SPLITS,
                    f"в окне {name!r} {count:,} объектов присутствуют и в обучении, "
                    "и в оценке: модель оценивается на том, что видела",
                    blocking=True,
                )
                for name, count in sorted(overlaps.items())
            ]

        overlaps = entity_overlap(split.parts, context.world, role=Role.ENTITY_ID)
        return [
            Signal(
                Finding.ENTITY_OVERLAP_ACROSS_SPLITS,
                f"в окне {name!r} {count:,} ЕДИНИЦ РЕШЕНИЯ повторяются в обучении "
                "и в оценке. Объект долгоживущий, и его присутствие по обе стороны "
                "нормально, но одно и то же решение оценивается на себе же",
                blocking=True,
            )
            for name, count in sorted(overlaps.items())
        ]


@dataclass(frozen=True)
class FeatureWindowOverlap:
    """N2i. Окна признаков двух решений одного объекта пересекаются.

    Для долгоживущего объекта присутствие по обе стороны сплита нормально, и
    N2 на нём молчит. Утечка возникает иначе: признак оценочного решения
    посчитан по интервалу, захватывающему измерения обучающих решений того же
    объекта. Модель видела эти измерения и оценивается на них же.

    Проверка требует, чтобы окно каждого признака было объявлено. Без
    объявления она не молчит, а блокирует: ноль сигналов иначе читался бы как
    отсутствие утечки, хотя означал бы лишь, что её не искали.
    """

    requirement: str = "N2i"
    premises: frozenset[Premise] = frozenset({Premise.UNIVERSAL})
    detects: frozenset[Finding] = frozenset(
        {Finding.UNDECLARED_FEATURE_WINDOW, Finding.FEATURE_WINDOW_OVERLAP}
    )

    def run(self, context: Context) -> list[Signal]:
        split = _require_split(context)
        if context.task.object_lifetime is not ObjectLifetime.RECURRING:
            raise NotApplicable(
                "объект одноразов: два решения по одному объекту невозможны, "
                "и окна признаков пересечься не могут"
            )

        features = context.world.schema.usable_features()
        undeclared = [c.name for c in features if c.window is None]
        if undeclared:
            return [
                Signal(
                    Finding.UNDECLARED_FEATURE_WINDOW,
                    f"признаки {sorted(undeclared)!r} не объявили окно, по которому "
                    "посчитаны. Объект долгоживущий, и без окна нельзя сказать, "
                    "захватывает ли признак оценочного решения измерения обучающих",
                    blocking=True,
                )
            ]

        # Пересечение зависит только от ПРОТЯЖЁННОСТИ окна: отступ сдвигает оба
        # интервала одинаково и при сравнении сокращается. Первая версия
        # складывала его с протяжённостью и объявляла пересекающимися два
        # мгновенных измерения, разнесённые во времени.
        reach = max((c.window.lookback_days for c in features), default=0.0)
        if reach == 0.0:
            return []  # все признаки мгновенные: пересекаться нечему

        overlaps = feature_window_overlap(split.parts, context.world, reach)
        widest = max(features, key=lambda c: c.window.lookback_days)
        return [
            Signal(
                Finding.FEATURE_WINDOW_OVERLAP,
                f"в окне {name!r} у {count:,} оценочных решений окно признаков "
                f"(до {reach:g} дн назад, шире всех у {widest.name!r}) достаёт до "
                "обучающих решений того же объекта: модель оценивается на "
                "измерениях, которые видела",
                blocking=True,
            )
            for name, count in sorted(overlaps.items())
        ]


@dataclass(frozen=True)
class TrainingPartIsUsable:
    """P8. Обучающая часть окна пуста или ничтожна.

    Окно, которому не на чем учиться, модели не даёт. Всякая метрика по нему —
    описание пустоты, а не качества, и вывод из неё нельзя ни подтвердить, ни
    опровергнуть.

    Одиннадцатый кейс: первое окно начиналось нулевым днём периода, и обучаться
    ему было не на чем — до него данных нет. Прогон напечатал «обучение 0» и не
    возразил НИ ОДНИМ сигналом. Строка в сводке — не возражение: её пропускают
    глазами, а блокирующий сигнал требует ответа.

    Порог не в доле, а в числе строк: обучение из десятка строк негодно
    независимо от того, велика ли оценочная часть.
    """

    requirement: str = "P8"
    premises: frozenset[Premise] = frozenset({Premise.UNIVERSAL})
    detects: frozenset[Finding] = frozenset({Finding.UNUSABLE_TRAINING_PART})

    minimum: int = 100
    """Ниже этого числа обучающих строк окно объявляется негодным."""

    def run(self, context: Context) -> list[Signal]:
        split = context.split
        if not isinstance(split, SplitResult) or not split.parts:
            raise NotApplicable("временной сплит не построен")

        return [
            Signal(
                Finding.UNUSABLE_TRAINING_PART,
                f"в окне {part.name!r} обучающих решений {part.train.height:,} "
                f"при пороге {self.minimum}: учиться не на чем, и всякая метрика "
                "по этому окну описывает пустоту, а не качество",
                blocking=True,
            )
            for part in split.parts
            if part.train.height < self.minimum
        ]


@dataclass(frozen=True)
class MeasuredPartsAreUsable:
    """P9. Часть, на которой меряют, слишком мала, чтобы что-то показать.

    Пустая обучающая часть была найдена одиннадцатым кейсом и закрыта P8. Но
    закрыт был экземпляр, а не класс: проба четырьмя постановками показала, что
    молча проходят ещё три — пустая ОЦЕНОЧНАЯ часть, окно шириной в один день
    и резерв в один день. Все три дают отчёт, выглядящий нормальным.

    Судится число строк С НАБЛЮДАЕМЫМ ИСХОДОМ, а не всего: строки без метки в
    измерении не участвуют, и часть из тысячи незрелых строк так же пуста, как
    часть из нуля.

    Порог тот же, что у проверок дрейфа: ниже сотни наблюдений доля класса
    неотличима от любой другой, и всякое сравнение с ней — описание шума.
    """

    requirement: str = "P9"
    premises: frozenset[Premise] = frozenset({Premise.UNIVERSAL})
    detects: frozenset[Finding] = frozenset({Finding.UNUSABLE_MEASURED_PART})

    minimum: int = 100
    """Ниже этого числа наблюдаемых исходов часть объявляется негодной."""

    def _observable(self, frame: pl.DataFrame) -> int:
        if frame is None or LABEL not in frame.columns:
            return 0 if frame is None else frame.height
        return frame.filter(pl.col(LABEL).is_not_null()).height

    def run(self, context: Context) -> list[Signal]:
        split = context.split
        if not isinstance(split, SplitResult) or not split.parts:
            raise NotApplicable("временной сплит не построен")

        signals = [
            Signal(
                Finding.UNUSABLE_MEASURED_PART,
                f"в окне {part.name!r} наблюдаемых исходов в оценочной части "
                f"{self._observable(part.evaluate):,} при пороге {self.minimum}: "
                "измерять нечего, и всякое число по этому окну описывает шум",
                blocking=True,
            )
            for part in split.parts
            if self._observable(part.evaluate) < self.minimum
        ]

        # Резерв уже мог переехать в журнал расхода; тогда судить его здесь
        # нечем, и молчание честнее выдуманного числа.
        if split.reserved is not None:
            observed = self._observable(split.reserved)
            if observed < self.minimum:
                signals.append(
                    Signal(
                        Finding.UNUSABLE_MEASURED_PART,
                        f"в резерве наблюдаемых исходов {observed:,} при пороге "
                        f"{self.minimum}: измерительный инструмент короче собственной "
                        "погрешности",
                        blocking=True,
                    )
                )
        return signals


@dataclass(frozen=True)
class WindowsDoNotOverlap:
    """P10. Окна перекрываются между собой во времени.

    Скользящие окна вложены ПО СОСТАВУ ОБУЧЕНИЯ — это устройство протокола, и
    ради него существует резерв. Но оценочные периоды обязаны быть
    непересекающимися: решение, попавшее в оценку двух окон, учитывается дважды,
    и устойчивость связи «по окнам» меряется частично по одним и тем же строкам.

    Проба показала, что перекрытие проходило молча: окна 300–400, 350–450 и
    400–500 дали отчёт, ничем не отличимый от правильного.
    """

    requirement: str = "P10"
    premises: frozenset[Premise] = frozenset({Premise.UNIVERSAL})
    detects: frozenset[Finding] = frozenset({Finding.WINDOWS_OVERLAP})

    def run(self, context: Context) -> list[Signal]:
        split = context.split
        if not isinstance(split, SplitResult) or len(split.parts) < 2:
            raise NotApplicable("окон меньше двух: перекрываться нечему")

        ordered = sorted(split.parts, key=lambda p: p.window.start)
        return [
            Signal(
                Finding.WINDOWS_OVERLAP,
                f"оценочные периоды окон {earlier.name!r} и {later.name!r} "
                f"перекрываются: первое кончается {earlier.window.stop:%Y-%m-%d}, "
                f"второе начинается {later.window.start:%Y-%m-%d}. Решения из общего "
                "куска попадают в оценку дважды, и устойчивость связи по окнам "
                "меряется частично по одним и тем же строкам",
                blocking=True,
            )
            for earlier, later in zip(ordered, ordered[1:], strict=False)
            if later.window.start < earlier.window.stop
        ]


@dataclass(frozen=True)
class GroupDependence:
    """N14. Объекты одной группы попали и в обучение, и в оценку.

    Уровень выше объекта: сеть, работодатель, филиал. Объекты одной группы
    зависимы — общее руководство, общие поставки, общая практика, — и модель,
    видевшая часть группы, оценивается на остальной мягче, чем на действительно
    новой.

    Дыра названа третьим кейсом (топ-10 страховщиков покрывали 44.4% строк) и
    подтверждена пятым (37.9% инспекций у сетей, у SUBWAY 294 точки). Ни то ни
    другое ядро не видело.

    Если группа НЕ объявлена, проверка не молчит: она ищет колонку, похожую на
    групповую, и требует ответа. Молчание здесь читалось бы как отсутствие
    зависимости, а означало бы лишь, что уровень не назван.
    """

    requirement: str = "N14"
    premises: frozenset[Premise] = frozenset({Premise.UNIVERSAL})
    detects: frozenset[Finding] = frozenset(
        {Finding.GROUP_OVERLAP_ACROSS_SPLITS, Finding.UNDECLARED_GROUP}
    )

    def run(self, context: Context) -> list[Signal]:
        split = _require_split(context)
        declared = context.world.schema.by_role(Role.GROUP_ID)

        if not declared:
            candidates = group_candidates(context.world)
            if not candidates:
                raise NotApplicable(
                    "групповой уровень не объявлен, и колонок, похожих на него, в данных не найдено"
                )
            shown = ", ".join(
                f"{name!r}: {count:,} групп, {share:.0%} строк"
                for name, (count, share) in sorted(candidates.items())
            )
            return [
                Signal(
                    Finding.UNDECLARED_GROUP,
                    f"уровень группы не объявлен, но данные его содержат ({shown}). "
                    "Объекты одной группы зависимы, и модель, видевшая часть её, "
                    "оценивается на остальной мягче. Объявите роль group_id либо "
                    "зафиксируйте обход с причиной",
                    blocking=True,
                )
            ]

        overlaps = group_overlap(split.parts, context.world)
        return [
            Signal(
                Finding.GROUP_OVERLAP_ACROSS_SPLITS,
                f"в окне {name!r} {count:,} групп присутствуют и в обучении, и в оценке; "
                f"их доля в оценочном окне {share:.0%}. Модель видела другие объекты тех "
                "же групп: оценка мягче, чем на действительно новых",
                blocking=True,
            )
            for name, (count, share) in sorted(overlaps.items())
        ]


@dataclass(frozen=True)
class ReservedMeasurementSample:
    """P5. Измерительная выборка не зарезервирована.

    Скользящие окна строятся вложенно: обучение каждого следующего включает
    оценочные строки предыдущих. Поэтому окна пересекаются по составу — на
    втором кейсе 42% и 61%, — и ни одно не годится в измерительный инструмент
    после того, как хоть одно использовалось для выбора. Выбор же неизбежен:
    горизонт, окно признаков, порог.

    Резерв разводит два вопроса. Окна отвечают, устойчива ли связь во времени,
    и для этого нужны все окна. Резерв отвечает, сколько это стоит, и для
    этого нужна выборка, не участвовавшая ни в чём.
    """

    requirement: str = "P5"
    premises: frozenset[Premise] = frozenset({Premise.UNIVERSAL})
    detects: frozenset[Finding] = frozenset({Finding.NO_RESERVED_MEASUREMENT_SAMPLE})

    def run(self, context: Context) -> list[Signal]:
        split = _require_split(context)
        if split.reserved is not None and split.reserved.height:
            return []
        if not split.parts:
            raise NotApplicable("сплит пуст: резервировать нечего")

        moment = context.world.schema.decision_time.name
        last_stop = max(p.window.stop for p in split.parts)
        beyond = context.world.main.filter(pl.col(moment) >= last_stop).height
        if not beyond:
            raise NotApplicable(
                "за пределами последнего окна решений нет: период слишком короток, "
                "чтобы что-то резервировать"
            )

        return [
            Signal(
                Finding.NO_RESERVED_MEASUREMENT_SAMPLE,
                f"измерительная выборка не зарезервирована. Окна "
                f"{[p.name for p in split.parts]!r} строятся вложенно и потому "
                "пересекаются по составу: измерение на любом из них после выбора "
                "на другом смещено. Отрежьте период, не входящий ни в одно окно, "
                "до начала работы",
                blocking=True,
            )
        ]


@dataclass(frozen=True)
class LabelImmaturity:
    """A12. Исход части объектов оценочного окна ещё не наблюдаем.

    Если такие объекты просто отбросить, окно сместится в сторону тех, чей срок
    короче, и оценка перестанет описывать популяцию.
    """

    requirement: str = "A12"
    premises: frozenset[Premise] = frozenset({Premise.DELAYED_OUTCOME})
    detects: frozenset[Finding] = frozenset({Finding.LABEL_IMMATURITY})
    tolerance: float = 0.01
    """Какую долю незрелых строк в оценочной части терпеть молча.

    Доля от объёма части, а не число строк: одна незрелая строка на сто тысяч
    вывода не меняет, а один процент — уже меняет. Значение вне (0, 1)
    означало бы «терпеть всегда» или «не терпеть никогда».

    Смысл был дописан при объявлении области: порог стоял без пояснения.
    """

    def run(self, context: Context) -> list[Signal]:
        split = _require_split(context)
        signals = []
        for part in split.parts:
            immature = split.immature.get(part.name, 0)
            if not immature or not part.evaluate.height:
                continue
            share = immature / part.evaluate.height
            if share < self.tolerance:
                continue
            signals.append(
                Signal(
                    Finding.LABEL_IMMATURITY,
                    f"в окне {part.name!r} исход {immature:,} объектов "
                    f"({share:.1%}) не наблюдаем к концу наблюдения: отбросив их, "
                    "получим окно из объектов с короткими сроками",
                    blocking=True,
                )
            )

        # Резерв проверяется наравне с окнами. Прежде он не проверялся вовсе, и
        # на третьем кейсе половина измерительной выборки оказалась незрелой:
        # измеряя по размеченной части, получили бы долю 11.95% вместо 9.2%.
        reserved = split.reserved
        if reserved is not None and reserved.height and split.reserved_immature:
            share = split.reserved_immature / reserved.height
            if share >= self.tolerance:
                signals.append(
                    Signal(
                        Finding.LABEL_IMMATURITY,
                        f"в резерве исход {split.reserved_immature:,} объектов "
                        f"({share:.1%}) не наблюдаем. Измерение по размеченной части — "
                        "отбор полных случаев: остаются объекты с короткими сроками, и "
                        "метрика завышается",
                        blocking=True,
                    )
                )
        return signals


SPLIT_CHECKS = [
    TrainingPartIsUsable(),
    MeasuredPartsAreUsable(),
    WindowsDoNotOverlap(),
    EntityOverlapAcrossSplits(),
    FeatureWindowOverlap(),
    GroupDependence(),
    ReservedMeasurementSample(),
    LabelImmaturity(),
]
