"""Сбор данных четырнадцатого кейса: нарушения жилищного кодекса Нью-Йорка.

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
TARGET = RAW / "dot_requests_2020_2023.csv"

DATASET = "erm2-nwe9"
PAGE = f"https://data.cityofnewyork.us/d/{DATASET}"
LICENSE = "открытые данные города Нью-Йорка (объявлено источником)"

WHERE = "agency = 'DOT' AND created_date >= '2020-01-01' AND created_date < '2024-01-01'"
EXPECTED = 871_117
"""Число записей, прочитанное ДО опечатывания. Расхождение означает, что
популяция изменилась под руками, и сбор останавливается."""

FIELDS = [
    "unique_key",
    "created_date",
    "closed_date",
    "agency",
    "complaint_type",
    "descriptor",
    "location_type",
    "incident_zip",
    "street_name",
    "address_type",
    "city",
    "landmark",
    "status",
    "resolution_action_updated_date",
    "community_board",
    "council_district",
    "borough",
    "open_data_channel_type",
]

PAGE_SIZE = 50_000


def _fetch(offset: int, attempts: int = 4) -> str:
    query = (
        f"SELECT {', '.join(FIELDS)} WHERE {WHERE} "
        f"ORDER BY unique_key LIMIT {PAGE_SIZE} OFFSET {offset}"
    )
    url = f"https://data.cityofnewyork.us/resource/{DATASET}.csv?$query=" + urllib.parse.quote(
        query
    )
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "dsx-case14/1.0"})
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
        source="NYC 311, обращения к Department of Transportation",
        license=LICENSE,
        url=PAGE,
    )
    write_manifest(manifest, MANIFEST)
    print(f"Манифест зафиксирован: {MANIFEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
