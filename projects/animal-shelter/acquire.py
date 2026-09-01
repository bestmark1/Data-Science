"""Сбор данных пятнадцатого кейса: приют для животных Остина.

Наборов два — приёмы и исходы, — и это первый кейс, где единица решения
собирается из ДВУХ источников. Соединять их здесь нельзя: сборка есть работа
построителя, а сбор обязан сохранить то, что отдал источник.

Полнота сверяется СЧЁТОМ строк: числа прочитаны до опечатывания и записаны в
пре-регистрации — 116 697 приёмов и 128 446 исходов.

Страницы упорядочены по внутреннему ключу источника `:id`, а не по `animal_id`.
Причина в устройстве отрасли: животное поступает в приют много раз, и
`animal_id` не уникален. Упорядочивание по неуникальному ключу даёт страницы,
которые перекрываются и теряют строки, — и это ровно тот класс дефекта, который
проект уже дважды ловил как «порядок строк не определён».
"""

from __future__ import annotations

import http.client
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from dsx.io.manifest import build_manifest, write_manifest

PROJECT = Path(__file__).resolve().parent
RAW = PROJECT / "data" / "raw"
MANIFEST = PROJECT / "manifest.yaml"

HOST = "data.austintexas.gov"
LICENSE = "открытые данные города Остина (объявлено источником)"
PAGE_SIZE = 20_000


@dataclass(frozen=True)
class Source:
    """Один набор источника вместе с объявленным до опечатывания объёмом."""

    dataset: str
    target: str
    fields: tuple[str, ...]
    where: str
    expected: int
    title: str


INTAKES = Source(
    dataset="wter-evkm",
    target="intakes_2016_2023.csv",
    fields=(
        "animal_id",
        "datetime",
        "found_location",
        "intake_type",
        "intake_condition",
        "animal_type",
        "sex_upon_intake",
        "age_upon_intake",
        "breed",
        "color",
    ),
    where="datetime >= '2016-01-01T00:00:00' AND datetime < '2024-01-01T00:00:00'",
    expected=116_697,
    title="приёмы животных, Austin Animal Center",
)

OUTCOMES = Source(
    dataset="9t4d-g238",
    target="outcomes_2016_2024.csv",
    fields=(
        "animal_id",
        "datetime",
        "outcome_type",
        "outcome_subtype",
        "date_of_birth",
        "sex_upon_outcome",
        "age_upon_outcome",
    ),
    # Исходы берутся на год дальше приёмов: пребывание, начатое в декабре 2023,
    # обязано иметь возможность завершиться. Без этого запаса незрелость
    # притворилась бы отсутствием события.
    where="datetime >= '2016-01-01T00:00:00' AND datetime < '2025-01-01T00:00:00'",
    expected=128_446,
    title="исходы животных, Austin Animal Center",
)

SOURCES = (INTAKES, OUTCOMES)


def _fetch(source: Source, offset: int, attempts: int = 4) -> str:
    query = (
        f"SELECT {', '.join(source.fields)} WHERE {source.where} "
        f"ORDER BY :id LIMIT {PAGE_SIZE} OFFSET {offset}"
    )
    url = f"https://{HOST}/resource/{source.dataset}.csv?$query=" + urllib.parse.quote(query)
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "dsx-case15/1.0"})
            with urllib.request.urlopen(request, timeout=600) as response:
                return response.read().decode("utf-8")
        # Оборванная передача, выглядящая готовой, — класс, давший блокирующие
        # находки шестого и десятого кейсов. IncompleteRead ловится наравне с
        # остальными: в четырнадцатом кейсе его отсутствие в списке роняло сбор.
        except (
            urllib.error.URLError,
            TimeoutError,
            json.JSONDecodeError,
            http.client.IncompleteRead,
            ConnectionError,
        ):
            if attempt == attempts - 1:
                raise
            time.sleep(5 * 2**attempt)
    raise RuntimeError("недостижимо")


def _download(source: Source) -> bool:
    target = RAW / source.target
    if target.is_file():
        print(f"  {source.target}: уже получен")
        return True

    partial = target.with_suffix(".csv.part")
    rows = 0
    with partial.open("w", encoding="utf-8") as out:
        offset = 0
        while True:
            page = _fetch(source, offset)
            lines = page.splitlines()
            if offset == 0:
                out.write(lines[0] + "\n")
            body = lines[1:]
            if not body:
                break
            out.write("\n".join(body) + "\n")
            rows += len(body)
            offset += PAGE_SIZE
            print(f"  {source.target}: получено {rows:,}")

    if rows != source.expected:
        partial.unlink()
        print(
            f"Строк получено {rows:,}, а объявлено до опечатывания "
            f"{source.expected:,}: популяция изменилась, сбор остановлен",
            file=sys.stderr,
        )
        return False
    partial.rename(target)
    print(f"  {source.target}: {target.stat().st_size / 1048576:.1f} МБ")
    return True


def main() -> int:
    if MANIFEST.is_file():
        print(f"Манифест уже зафиксирован: {MANIFEST}. Замена данных требует поправки.")
        return 0
    RAW.mkdir(parents=True, exist_ok=True)

    for source in SOURCES:
        if not _download(source):
            return 1

    manifest = build_manifest(
        RAW,
        source="Austin Animal Center: приёмы и исходы",
        license=LICENSE,
        url=f"https://{HOST}/",
    )
    write_manifest(manifest, MANIFEST)
    print(f"Манифест зафиксирован: {MANIFEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
