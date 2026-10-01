"""Запуск проекта по заполненной форме.

Скрипт проекта сводится к двум действиям: построить таблицу решений и вызвать
`run`. Всё остальное — сплит, проверки, учёт выборок, отчёт — общее и живёт
здесь, а не переписывается в каждом проекте заново.

Порядок шагов не настраивается. На этапе 0 заключение переворачивалось трижды
именно из-за перестановок в протоколе, и возможность их сделать — не гибкость,
а незакрытая дыра.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from dsx.assumptions import Basis
from dsx.checks import ALL_CHECKS, Context, run_checks
from dsx.checks.base import Report as CheckReport
from dsx.evals.world import World
from dsx.policy import OverrideLedger
from dsx.project import ProjectForm
from dsx.report import Study
from dsx.roles import Availability, Role
from dsx.samples import SampleLedger
from dsx.split import (
    SplitResult,
    entity_overlap,
    extent_of,
    positive_rates,
    purge_overlapping_train,
    reserved_extent,
    split_by_windows,
    training_of,
    units_of,
)
from dsx.task import require_supported
from dsx.windows import unverified

RESERVE = "резерв"
"""Имя измерительной выборки. Одно на все проекты: разные имена для одной
роли — способ незаметно измерить дважды."""


class LeakageError(Exception):
    """Модель получила на вход то, что признаком момента решения не объявлено.

    Обхода у этого требования нет намеренно. Всякий обход возвращал бы ровно ту
    дыру, ради которой требование введено: утечку достаточно было бы объявить
    ролью `ignored` и приложить причину. Если колонка законно доступна в момент
    решения — объявите её признаком, и ядро проверит это объявление наравне с
    остальными.
    """


@dataclass
class Result:
    """Что получилось: данные протокола вместе с отчётом."""

    world: World
    split: SplitResult
    checks: CheckReport
    samples: SampleLedger
    study: Study

    def uses(self, names: Sequence[str]) -> None:
        """Объявить колонки, которые модель получила на вход, и сверить со схемой.

        Каждая обязана быть признаком, доступным в момент решения. Иное
        означает одно из двух, и оба хуже ошибки в модели: либо в модель попало
        то, чего решавший не знал, либо схема описывает не ту таблицу, по
        которой считается результат.

        Пятнадцатый кейс показал цену молчания опытом: утечка, объявленная в
        схеме ролью `ignored` и скормленная модели, подняла разрешающую
        способность с 0.7178 до 0.8009 при дословно совпавших сигналах.
        """
        schema = self.world.schema
        known = {column.name: column for column in schema.columns}
        wrong: list[str] = []
        for name in names:
            column = known.get(name)
            if column is None:
                wrong.append(f"{name!r} — в схеме не объявлена вовсе")
            elif column.role is not Role.FEATURE:
                wrong.append(f"{name!r} — объявлена ролью {column.role.value!r}, а не признаком")
            elif column.availability is not Availability.AT_DECISION:
                wrong.append(
                    f"{name!r} — доступность объявлена {column.availability.value!r}, "
                    "то есть решавшему она известна не была"
                )
        if wrong:
            raise LeakageError(
                "модель получила на вход то, что признаком момента решения не объявлено: "
                + "; ".join(wrong)
            )
        self.samples.declare_features(names)

    def summary(self) -> str:
        lines = [f"решений: {self.world.main.height:,}"]
        for part in self.split.parts:
            lines.append(
                f"  {part.name}: обучение {self.split.train_rows[part.name]:6,}  "
                f"оценка {part.evaluate.height:5,}"
            )
        immature = self.split.reserved_immature
        lines.append(
            f"  {RESERVE}: {self.split.reserved_rows:,}"
            + (f" (из них незрелых {immature:,})" if immature else "")
        )
        if self.split.purged_by_window:
            lines.append(
                "вычищено из обучения (пересечение окон признаков): "
                + str(self.split.purged_by_window)
            )
        lines.append(
            "выпало между выборками: "
            + str(self.split.dropped_by_window or self.split.dropped_not_yet_known)
        )
        if self.split.outside_everything:
            lines.append(
                f"вне всех выборок: {self.split.outside_everything:,} "
                f"(незрелых {self.split.outside_immature:,})"
            )
        lines.append("незрелых: " + (str(self.split.immature) if self.split.immature else "нет"))
        lines.append(
            "доли положительных: "
            + str({k: f"{v:.1%}" for k, v in positive_rates(self.split.parts).items()})
        )
        lines.append("")

        lines.append("СИГНАЛЫ")
        lines += [f"  {s}" for s in self.checks.signals] or ["  нет"]
        lines.append("")
        lines.append("ПРОПУЩЕННЫЕ ПРОВЕРКИ")
        lines += [f"  {s}" for s in self.checks.skipped] or ["  нет"]
        lines.append("")
        lines.append(
            f"блокирующих: {len(self.checks.blocking)}, "
            f"пропущено проверок: {len(self.checks.skipped)}"
        )
        lines.append(
            "окна признаков, сверить которые не с чем: "
            + (", ".join(unverified(self.world)) or "нет")
        )
        return "\n".join(lines)


def run(
    form: ProjectForm,
    frame: pl.DataFrame,
    report_dir: Path | None = None,
    overrides: OverrideLedger | None = None,
) -> Result:
    """Прогнать проект: сплит, проверки, учёт выборок, отчёт.

    `overrides` — журнал осознанных обходов. Без него блокирующий сигнал
    остаётся блокирующим, и это верно: обход обязан быть записан с причиной и
    автором, иначе он не отличается от невнимательности.

    Прежде параметра не было вовсе: `run_checks` журнал уважал, но рабочий путь
    его не передавал, и механизм был недостижим из проекта.
    """
    world = World(frames={"main": frame}, schema=form.schema_spec())
    definition = form.outcome.to_definition()
    task = form.task.to_spec()
    # Защита существовала, но не вызывалась: можно было объявить регрессию или
    # конкурирующие исходы и получить бинарную метку без единого возражения.
    require_supported(task)

    moment = world.schema.decision_time.name
    origin = frame[moment].min()
    # Конец наблюдения объявляется формой. Выводить его из максимума даты
    # события нельзя: это последнее СЛУЧИВШЕЕСЯ событие, а не конец сбора.
    snapshot = form.observed_until
    reserve_from = origin + dt.timedelta(days=form.split.reserve_from_day)
    reserve_until = (
        origin + dt.timedelta(days=form.split.reserve_until_day)
        if form.split.reserve_until_day is not None
        else None
    )

    split = split_by_windows(
        world, definition, form.split.to_windows(origin), snapshot, reserve_from, reserve_until
    )

    # Пересечение окон признаков снимается ДЕЛОМ, до проверок: обучающие
    # решения, чьи окна достают до оценочных, удаляются. Проверка N2i после
    # этого молчит не потому, что ей пообещали, а потому, что пересечения
    # больше нет. Цена записывается и попадает в отчёт.
    reach = max(
        (c.window.lookback_days for c in world.schema.usable_features() if c.window is not None),
        default=0.0,
    )
    split.purged_by_window = purge_overlapping_train(split.parts, world, reach)

    overrides = overrides or OverrideLedger()
    checks = run_checks(
        list(ALL_CHECKS), Context(world, definition, task, split, snapshot), overrides
    )

    samples = SampleLedger(overrides)
    split.shared_objects = entity_overlap(split.parts, world)
    for part in split.parts:
        # Оценочная часть окна отдаётся журналу: смотреть пооконные метрики и
        # решать по ним — это выбор, и он обязан расходовать выборку. Обучающая
        # часть — тоже, и снимается с результата, как резерв: публичное поле
        # позволяло учить модель последнего окна и мерить ею все (класс 19).
        samples.register_window(
            part.name,
            start=part.window.start,
            extent=extent_of(part, world),
            evaluation=units_of(part.evaluate, world),
            frame=part.evaluate,
            training=training_of(part, world),
        )
        split.train_rows[part.name] = part.train.height if part.train is not None else 0
        part.train = None

    # Резерв передаётся журналу ВМЕСТЕ С ДАННЫМИ и снимается с результата
    # сплита. После этого получить его можно только через checkout, который
    # записывает расход тем же действием. Публичное поле оставляло обход
    # открытым: посмотреть метрику, подкрутить порог, посмотреть снова.
    samples.register(RESERVE, reserved_extent(split, world), frame=split.reserved)
    split.reserved_rows = split.reserved.height if split.reserved is not None else 0
    split.reserved = None

    study = Study(title=form.title)
    study.overrides = overrides
    study.declarations = form.model_dump_json()
    study.open_questions = form.unverifiable_declarations()
    study.feature_uncertainty = form.feature_uncertainty_share()
    study.cause_questions = len(form.unverifiable_by_kind()["причины"])
    study.estimand = form.outcome.to_definition().estimand
    study.checks = checks
    study.samples = samples
    study.assumptions = form.registry()

    # Объявления, которые ядро проверить не может, обязаны хотя бы попасть в
    # отчёт: иначе они украшение. Одновременность видов по таблице решений не
    # проверяется — нарушение происходит раньше, при её сборке.
    if task.simultaneous_kinds is not None:
        study.assumptions.record(
            f"одновременное наступление нескольких видов события: {task.simultaneous_kinds.value}",
            basis=Basis.DOMAIN_KNOWLEDGE,
            author="форма проекта",
            consequence="если допущение неверно, часть строк отнесена к одному виду "
            "произвольно, и исход этого вида смещён",
        )

    result = Result(world=world, split=split, checks=checks, samples=samples, study=study)

    if report_dir is not None:
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / "report.md").write_text(study.render(), encoding="utf-8")
    return result


def objects_across_splits(result: Result) -> dict[str, int]:
    """Объекты по обе стороны сплита. Для долгоживущих это норма, не находка.

    Посчитаны в прогоне до передачи обучающих частей журналу.
    """
    return dict(result.split.shared_objects)
