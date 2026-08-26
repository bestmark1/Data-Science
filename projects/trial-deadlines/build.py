"""Сборка таблицы решений девятого кейса: обещанный срок против переписанного.

Единица решения — клиническое исследование. Момент решения — публикация
регистрации. Срок — дата первичного завершения, ОБЪЯВЛЕННАЯ ПРИ РЕГИСТРАЦИИ,
то есть взятая из версии 0 записи. Событие — фактическое первичное завершение.

Две колонки срока строятся намеренно и различаются только моментом взгляда:

  due_at_promised  — из версии 0: то, что обещали, когда решение принималось;
  due_at_current   — из текущей записи: то, что стоит в поле сейчас.

У завершившегося исследования вторая величина принимает тип ACTUAL и
становится записью о случившемся. Срок совпадает с событием, и исход
выполняется тождественно. Это и есть ретроспективная подгонка.

ТОЧНОСТЬ ДАТ. Реестр допускает месячную точность («2015-10») наравне с
дневной. Ядро различает только `date` и `instant`, месяца в нём нет, поэтому
правило объявляется здесь и единообразно: месячная дата читается как
ПОСЛЕДНИЙ день месяца — и у срока, и у события. Разные правила для двух
колонок, взятых из одного поля, были бы подгонкой.

ВНИМАНИЕ. В первый прогон намеренно внесены положительные контроли §5
пре-регистрации; они помечены словом КОНТРОЛЬ.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import polars as pl

from dsx.join import Cardinality, guarded_join

PROJECT = Path(__file__).resolve().parent
CURRENT = PROJECT / "data" / "raw" / "studies_current.jsonl"
ORIGINAL = PROJECT / "data" / "raw" / "studies_version0.jsonl"


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _month_end(year: int, month: int) -> dt.date:
    return dt.date(year + month // 12, month % 12 + 1, 1) - dt.timedelta(days=1)


def _parse(value: str | None) -> dt.date | None:
    """Разобрать дату реестра: «2015-10-27» либо «2015-10».

    Формат объявлен явно и обе его разновидности названы. Пятый кейс: угадывание
    обнулило 94 625 строк из 153 810 и переставило месяц с днём ещё в 54 927.
    """
    if not value:
        return None
    parts = value.split("-")
    if len(parts) == 3:
        return dt.date(int(parts[0]), int(parts[1]), int(parts[2]))
    if len(parts) == 2:
        return _month_end(int(parts[0]), int(parts[1]))
    return None


def _status_dates(record: dict) -> dict:
    status = record.get("protocolSection", {}).get("statusModule", {})
    primary = status.get("primaryCompletionDateStruct", {})
    return {
        "primary_date": primary.get("date"),
        "primary_type": primary.get("type"),
        "start_date": status.get("startDateStruct", {}).get("date"),
        "posted": status.get("studyFirstPostDateStruct", {}).get("date"),
        "overall_status": status.get("overallStatus"),
    }


def _current_rows() -> list[dict]:
    rows = []
    for record in _read_jsonl(CURRENT):
        protocol = record.get("protocolSection", {})
        dates = _status_dates(record)
        design = protocol.get("designModule", {})
        sponsor = protocol.get("sponsorCollaboratorsModule", {}).get("leadSponsor", {})
        enrollment = design.get("enrollmentInfo", {})
        rows.append(
            {
                "nct_id": protocol.get("identificationModule", {}).get("nctId"),
                "registered_at": _parse(dates["posted"]),
                "current_primary_date": _parse(dates["primary_date"]),
                "current_primary_type": dates["primary_type"],
                "started_at": _parse(dates["start_date"]),
                "status": dates["overall_status"],
                "sponsor": sponsor.get("name"),
                "sponsor_class": sponsor.get("class"),
                "enrollment": enrollment.get("count"),
                "enrollment_type": enrollment.get("type"),
                "allocation": design.get("designInfo", {}).get("allocation"),
                "intervention_model": design.get("designInfo", {}).get("interventionModel"),
                "primary_purpose": design.get("designInfo", {}).get("primaryPurpose"),
                "masking": design.get("designInfo", {}).get("maskingInfo", {}).get("masking"),
                "condition_count": len(protocol.get("conditionsModule", {}).get("conditions", [])),
                "has_results": record.get("hasResults"),
            }
        )
    return rows


def _original_rows() -> list[dict]:
    rows = []
    for record in _read_jsonl(ORIGINAL):
        if record.get("error"):
            continue
        dates = _status_dates(record.get("study", {}))
        rows.append(
            {
                "nct_id": record["nctId"],
                "promised_primary_date": _parse(dates["primary_date"]),
                "promised_primary_type": dates["primary_type"],
                "version_date": _parse(record.get("versionDate")),
            }
        )
    return rows


def build() -> pl.DataFrame:
    """Одна строка на исследование, с двумя сроками и фактическим завершением."""
    current = pl.DataFrame(_current_rows(), strict=False)
    original = pl.DataFrame(_original_rows(), strict=False)
    read = current.height
    print(f"  текущих записей {read:,}, первых версий {original.height:,}")

    # Грануляция объявляется: nct_id уникален с обеих сторон.
    frame = guarded_join(
        current, original, on=["nct_id"], expect=Cardinality.ONE_TO_ONE, how="inner"
    )

    # Непригодна строка без момента решения, без обещанного срока либо без
    # номера: без них решение нельзя ни поставить во времени, ни судить.
    #
    # Отдельно исключается РЕТРОСПЕКТИВНАЯ РЕГИСТРАЦИЯ: 1 035 исследований
    # (10.0%) внесены в реестр уже после первичного завершения, медиана
    # отставания 368 дней, худший случай 6 448. У 1 027 из них обещанный срок
    # уже был в прошлом в день регистрации.
    #
    # Это не порча данных, а свойство реестра, и потому решение здесь о
    # ПОПУЛЯЦИИ, а не о чистке: постановка спрашивает, сдержано ли обещание,
    # данное ДО исхода. Кто зарегистрировался после, обещания не давал, и
    # судить его этой меркой нельзя.
    #
    # Строки были оставлены в первом прогоне намеренно (контроль К-2) и
    # отброшены после того, как ядро их назвало.
    broken = (
        pl.col("registered_at").is_null()
        | pl.col("promised_primary_date").is_null()
        | pl.col("nct_id").is_null()
        | (pl.col("current_primary_date") < pl.col("registered_at")).fill_null(False)
        # Обещание, уже нарушенное в день, когда его дали: 20 строк (0.2%),
        # обещанный срок в прошлом на медианные 124 дня. Исход у них определён
        # в момент решения — ни одна из пятнадцати наблюдаемых срок не
        # соблюла, — и предсказывать там нечего.
        | (pl.col("promised_primary_date") < pl.col("registered_at")).fill_null(False)
    )
    unusable = frame.filter(broken).height
    kept = frame.filter(~broken)

    lost = read - kept.height
    if lost != unusable + (read - frame.height):
        raise AssertionError(
            f"баланс строк не сошёлся: прочитано {read:,}, после соединения "
            f"{frame.height:,}, осталось {kept.height:,}, непригодных {unusable:,}"
        )
    print(
        f"  прочитано {read:,}, потеряно {lost:,} (нет первой версии, момента "
        f"решения или обещанного срока, регистрация после завершения либо "
        f"обещание уже нарушено в день регистрации), "
        f"осталось {kept.height:,}"
    )

    # Часовые значения НЕ вычищаются намеренно и после снятия контроля К-3:
    # 'UNKNOWN' у статуса — законный статус реестра, означающий, что спонсор
    # перестал обновлять запись; 'NA' у схемы распределения — законное
    # «неприменимо» у одногрупповых исследований. Проверка A14 требует одного
    # из двух: исправить при сборке ИЛИ записать обход с причиной. Здесь верно
    # второе, и обход записан в model.py.

    return kept.with_columns(
        pl.col("promised_primary_date").alias("due_at_promised"),
        pl.col("current_primary_date").alias("due_at_current"),
        # Событие: фактическое первичное завершение. Тип ACTUAL — это утверждение
        # источника о том, что дата уже не обещание. Пока тип ESTIMATED, события
        # не наблюдалось, и метка обязана остаться незрелой, а не отрицательной.
        pl.when(pl.col("current_primary_type") == "ACTUAL")
        .then(pl.col("current_primary_date"))
        .otherwise(None)
        .alias("completed_at"),
        (pl.col("current_primary_date") - pl.col("promised_primary_date"))
        .dt.total_days()
        .alias("deadline_shift_days"),
        # УМЕРЕННЫЙ пересмотр: обещанный срок, сдвинутый на медианную величину
        # правки (184 дня). Не крайний случай, где срок становится записью о
        # случившемся, а обычный — такой, каким пересмотр бывает чаще всего.
        # Нужен, чтобы проверить, ловит ли ядро пересмотр КАК ТАКОВОЙ или
        # только его вырожденное следствие.
        (pl.col("promised_primary_date") + pl.duration(days=184)).alias("due_at_mild"),
    )


if __name__ == "__main__":
    table = build()
    print(f"строк: {table.height:,}, колонок: {len(table.columns)}")
    print("период регистрации:", table["registered_at"].min(), "..", table["registered_at"].max())
