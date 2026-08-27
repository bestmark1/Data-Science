"""Сбор данных двенадцатого кейса: нарушения жилищного кодекса Нью-Йорка.

Источник отдаёт страницами через открытый интерфейс. Берутся только объявленные
колонки: в наборе их сорок одна, а в постановке участвует двадцать одна, и
тянуть остальные значило бы платить за то, что всё равно будет объявлено ролью
`ignored`.

Полнота сверяется не размером файла, а СЧЁТОМ строк: интерфейс размера не
объявляет, зато объявляет число записей, и оно прочитано до опечатывания —
2 735 564 за 2022–2024.
"""

from __future__ import annotations

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
TARGET = RAW / "housing_violations_2022_2024.csv"

DATASET = "wvxf-dwi5"
PAGE = f"https://data.cityofnewyork.us/d/{DATASET}"
LICENSE = "открытые данные города Нью-Йорка (объявлено источником)"

WHERE = "inspectiondate >= '2022-01-01' AND inspectiondate < '2025-01-01'"
EXPECTED = 2_735_564
"""Число записей, прочитанное ДО опечатывания. Расхождение означает, что
популяция изменилась под руками, и сбор останавливается."""

FIELDS = [
    "violationid",
    "buildingid",
    "registrationid",
    "boro",
    "class",
    "inspectiondate",
    "approveddate",
    "originalcertifybydate",
    "originalcorrectbydate",
    "newcertifybydate",
    "newcorrectbydate",
    "certifieddate",
    "novissueddate",
    "currentstatus",
    "currentstatusdate",
    "violationstatus",
    "novtype",
    "rentimpairing",
    "apartment",
    "story",
    "communityboard",
]

PAGE_SIZE = 50_000


def _fetch(offset: int, attempts: int = 4) -> str:
    query = (
        f"SELECT {', '.join(FIELDS)} WHERE {WHERE} "
        f"ORDER BY violationid LIMIT {PAGE_SIZE} OFFSET {offset}"
    )
    url = f"https://data.cityofnewyork.us/resource/{DATASET}.csv?$query=" + urllib.parse.quote(
        query
    )
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "dsx-case12/1.0"})
            with urllib.request.urlopen(request, timeout=600) as response:
                return response.read().decode("utf-8")
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
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
        source="NYC Department of Housing Preservation and Development",
        license=LICENSE,
        url=PAGE,
    )
    write_manifest(manifest, MANIFEST)
    print(f"Манифест зафиксирован: {MANIFEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
