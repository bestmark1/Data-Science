"""Переносимый отчёт: заключение вместе с протоколом, при котором получено.

На этапе 0 заключение менялось трижды, и каждый раз из-за исправления
протокола, а не новых данных: «модель хуже правила», затем «никто не лучше»,
затем «модель устойчиво лучше». Читатель первой версии не имел ни малейшей
возможности узнать, что она держится на протоколе, в котором тест использовался
для выбора четырёх решений.

Поэтому заключение здесь не строка, а строка вместе с отпечатком протокола.
Заключение, полученное при другом протоколе, отличается от предыдущего видимо,
а не по памяти автора. И история не затирается: вывод, менявшийся трижды,
заслуживает иного доверия, чем полученный один раз, и это должно быть видно.
"""

from __future__ import annotations

import datetime as dt
import hashlib
from dataclasses import dataclass, field
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from dsx.assumptions import AssumptionRegistry
from dsx.checks.base import Report as CheckReport
from dsx.policy import OverrideLedger
from dsx.samples import SampleLedger


def _digest(*parts: str) -> str:
    payload = "\n".join(parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


class Conclusion(BaseModel):
    """Утверждение вместе с отпечатком протокола, при котором получено."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    statement: Annotated[str, Field(min_length=1)]
    protocol: Annotated[str, Field(min_length=1)]
    """Отпечаток состояния протокола: расход выборок, обходы, находки."""

    at: dt.datetime = Field(default_factory=lambda: dt.datetime.now(dt.UTC))

    def __str__(self) -> str:
        return f"{self.statement}  [протокол {self.protocol}]"


@dataclass
class Study:
    """Исследование целиком: находки, протокол, допущения, заключения."""

    title: str
    checks: CheckReport | None = None
    samples: SampleLedger = field(default_factory=SampleLedger)
    assumptions: AssumptionRegistry = field(default_factory=AssumptionRegistry)
    overrides: OverrideLedger = field(default_factory=OverrideLedger)
    conclusions: list[Conclusion] = field(default_factory=list)

    declarations: str = ""
    """Объявления проекта целиком. Входят в отпечаток протокола: горизонт,
    направление исхода и доступность признаков меняют заключение не меньше, чем
    расход выборок, а прежде их изменение не делало вывод устаревшим."""

    open_questions: tuple[str, ...] = ()
    """Объявления, которые ядро проверить не может, — вопросы владельцу данных."""

    # --- протокол ---------------------------------------------------------

    def protocol_digest(self) -> str:
        """Отпечаток текущего состояния протокола.

        Меняется, когда меняется что-либо, способное изменить заключение:
        расход выборок, зафиксированные обходы, найденные дефекты.
        """
        return _digest(
            self.declarations,
            *(str(a) for a in self.samples.accesses),
            *(str(o) for o in self.overrides.entries),
            *sorted(f.value for f in (self.checks.findings if self.checks else frozenset())),
        )

    def conclude(self, statement: str) -> Conclusion:
        """Записать заключение, привязав его к текущему протоколу.

        Прежние заключения не затираются: история — часть ответа.
        """
        conclusion = Conclusion(statement=statement, protocol=self.protocol_digest())
        self.conclusions.append(conclusion)
        return conclusion

    @property
    def conclusion_changed_under_a_different_protocol(self) -> bool:
        """Менялось ли заключение вместе с протоколом.

        Если да, читателю нужно знать: заключение зависит не только от данных.
        """
        if len(self.conclusions) < 2:
            return False
        return any(
            a.statement != b.statement and a.protocol != b.protocol
            for a, b in zip(self.conclusions, self.conclusions[1:], strict=False)
        )

    @property
    def is_stale(self) -> bool:
        """Протокол изменился после последнего заключения."""
        if not self.conclusions:
            return False
        return self.conclusions[-1].protocol != self.protocol_digest()

    # --- рендер -----------------------------------------------------------

    def render(self) -> str:
        parts = [f"# {self.title}", "", self._conclusion_section()]
        if self.checks is not None:
            parts += ["", self._findings_section(), "", self._skipped_section()]
        parts += [
            "",
            self._questions_section(),
            "",
            self.samples.report_section(),
            "",
            self.assumptions.report_section(),
            "",
            self.overrides.report_section(),
        ]
        return "\n".join(parts)

    def _questions_section(self) -> str:
        """Объявления, проверить которые ядро не может.

        Порождаются из формы, а не из догадливости заполняющего. Чтобы записать
        неизвестное вручную, надо уже понимать, где требуется предметный ответ,
        — а человек без отраслевого знания этого как раз и не понимает.
        """
        lines = ["## Вопросы владельцу данных", ""]
        if not self.open_questions:
            lines.append("Форма не разбиралась: список порождается из объявлений проекта.")
            return "\n".join(lines)

        lines.append("Ядро проверить их не может. Зелёный прогон не означает, что ответы верны.")
        lines.append("")
        lines += [f"- {q}" for q in self.open_questions]
        return "\n".join(lines)

    def _conclusion_section(self) -> str:
        lines = ["## Заключение", ""]
        if not self.conclusions:
            lines.append("Не сформулировано.")
            return "\n".join(lines)

        lines.append(self.conclusions[-1].statement)
        lines.append("")
        lines.append(f"Протокол: `{self.conclusions[-1].protocol}`")

        if self.is_stale:
            lines += [
                "",
                "**Протокол изменился после того, как заключение было записано.** "
                "Заключение относится к прежнему состоянию и требует пересмотра.",
            ]

        if len(self.conclusions) > 1:
            lines += ["", "### История заключений", ""]
            for index, item in enumerate(self.conclusions, 1):
                lines.append(f"{index}. {item.statement} — протокол `{item.protocol}`")
            if self.conclusion_changed_under_a_different_protocol:
                lines += [
                    "",
                    "Заключение менялось вместе с протоколом, а не только с данными. "
                    "Вывод, менявшийся несколько раз, заслуживает иного доверия, "
                    "чем полученный один раз.",
                ]
        return "\n".join(lines)

    def _findings_section(self) -> str:
        assert self.checks is not None
        lines = ["## Находки", ""]
        if not self.checks.signals:
            lines.append("Проверки не нашли дефектов.")
            return "\n".join(lines)

        blocking = self.checks.blocking
        if blocking:
            lines += ["### Блокирующие", ""]
            lines += [f"- {signal.detail}" for signal in blocking]
            lines.append("")

        other = [s for s in self.checks.signals if not s.blocking]
        if other:
            lines += ["### Требуют внимания", ""]
            lines += [f"- {signal.detail}" for signal in other]
        return "\n".join(lines)

    def _skipped_section(self) -> str:
        assert self.checks is not None
        lines = ["## Непроведённые проверки", ""]
        if not self.checks.skipped:
            lines.append("Все применимые проверки выполнены.")
            return "\n".join(lines)

        lines.append(
            "Молча выключенная проверка неотличима от проверки, которая ничего "
            "не нашла, поэтому пропуски перечислены."
        )
        lines.append("")
        lines += [f"- {item}" for item in self.checks.skipped]
        return "\n".join(lines)
