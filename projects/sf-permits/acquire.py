"""Сбор данных тринадцатого кейса: нарушения жилищного кодекса Нью-Йорка.

Источник отдаёт страницами через открытый интерфейс. Берутся только объявленные
колонки: в наборе их сорок одна, а в постановке участвует двадцать одна, и
тянуть остальные значило бы платить за то, что всё равно будет объявлено ролью
`ignored`.

Полнота сверяется не размером файла, а СЧЁТОМ строк: интерфейс размера не
объявляет, зато объявляет число записей, и оно прочитано до опечатывания —
2 735 564 за 2022–2024.
"""

from __future__ import annotations

import http.client
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from dsx.io.manifest import build_manifest, write_manifest

PROJECT = Path(__file__).resolve().parent
RAW = PROJECT / "data" / "raw"
MANIFEST = PROJECT / "manifest.yaml"
TARGET = RAW / "sf_permits_2015_2023.csv"

DATASET = "i98e-djp9"
PAGE = f"https://data.sfgov.org/d/{DATASET}"
LICENSE = "открытые данные города Сан-Франциско (объявлено источником)"

WHERE = "filed_date >= '2015-01-01' AND filed_date < '2024-01-01'"
EXPECTED = 306_091
"""Число записей, прочитанное ДО опечатывания. Расхождение означает, что
популяция изменилась под руками, и сбор останавливается."""

FIELDS = [
    "permit_number",
    "permit_type",
    "permit_type_definition",
    "filed_date",
    "issued_date",
    "approved_date",
    "completed_date",
    "status",
    "status_date",
    "block",
    "lot",
    "estimated_cost",
    "revised_cost",
    "existing_use",
    "proposed_use",
    "number_of_existing_stories",
    "number_of_proposed_stories",
    "plansets",
    "site_permit",
    "fire_only_permit",
    "application_submission_method",
    "supervisor_district",
    "neighborhoods_analysis_boundaries",
    "zipcode",
    "existing_units",
    "proposed_units",
    "adu",
    "data_as_of",
]

PAGE_SIZE = 50_000


def _fetch(offset: int, attempts: int = 4) -> str:
    query = (
        f"SELECT {', '.join(FIELDS)} WHERE {WHERE} "
        f"ORDER BY permit_number LIMIT {PAGE_SIZE} OFFSET {offset}"
    )
    url = f"https://data.sfgov.org/resource/{DATASET}.csv?$query=" + urllib.parse.quote(query)
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "dsx-case13/1.0"})
            with urllib.request.urlopen(request, timeout=600) as response:
                return response.read().decode("utf-8")
        # IncompleteRead ловится наравне с остальными: сервер обрывает выдачу
        # посреди страницы, и это ровно тот класс, что дал блокирующие находки
        # шестого и десятого кейсов — оборванная передача, выглядящая готовой.
        # Первая версия списка его не покрывала, и сбор падал целиком.
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


def main() -> int:
    if MANIFEST.is_file():
        print(f"Манифест уже зафиксирован: {MANIFEST}. Замена данных требует поправки.")
        return 0
    RAW.mkdir(parents=True, exist_ok=True)

    if TARGET.is_file():
        print(f"Файл уже получен: {TARGET.name}")
    else:
        partial = TARGET.with_suffix(".csv.part")
        rows = 0
        with partial.open("w", encoding="utf-8") as out:
            offset = 0
            while True:
                page = _fetch(offset)
                lines = page.splitlines()
                if offset == 0:
                    out.write(lines[0] + "\n")
                body = lines[1:]
                if not body:
                    break
                out.write("\n".join(body) + "\n")
                rows += len(body)
                offset += PAGE_SIZE
                print(f"  получено {rows:,}")
        if rows != EXPECTED:
            partial.unlink()
            print(
                f"Строк получено {rows:,}, а объявлено до опечатывания {EXPECTED:,}: "
                "популяция изменилась, сбор остановлен",
                file=sys.stderr,
            )
            return 1
        partial.rename(TARGET)

    print(f"  {TARGET.name}: {TARGET.stat().st_size / 1048576:.1f} МБ")
    manifest = build_manifest(
        RAW,
        source="San Francisco Department of Building Inspection",
        license=LICENSE,
        url=PAGE,
    )
    write_manifest(manifest, MANIFEST)
    print(f"Манифест зафиксирован: {MANIFEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
