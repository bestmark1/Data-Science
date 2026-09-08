"""Сбор данных двадцатого кейса: разрешения на бурение скважин, Колорадо.

Полнота сверяется счётом строк, объявленным до опечатывания: 33 852.

Популяция сужена разрешениями, у которых проставлена дата истечения: контракт
исхода строится на сроке, а брать те, у кого срока нет, значило бы назначать
срок автором — ровно то, от чего кейс уходит. Цена сужения сосчитана и записана
в §3 пре-регистрации.

ПЕРСОНАЛЬНЫЕ СВЕДЕНИЯ НЕ ЗАПРАШИВАЮТСЯ. Разрешения на частные скважины выдаются
физическим лицам, и `contact_name` с `address` и `parcel_name` образуют
персональные данные; точные координаты (`latitude`, `longitude`, `utm_*`,
`coords*`) указывают на участок конкретного человека. Задаче они не нужны:
география берётся округом и водным районом.

Не запрашиваются и кадастровые привязки (`township`, `range`, `section`, `q10`,
`q40`, `q160`): это та же точная локация, записанная иначе.

Страницы упорядочены по внутреннему ключу `:id`.
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
TARGET = RAW / "well_permits_2018_2022.csv"

HOST = "data.colorado.gov"
DATASET = "wumm-7awb"
LICENSE = "открытые данные штата Колорадо (объявлено источником)"

WHERE = (
    "permit_issued >= '2018-01-01' AND permit_issued < '2023-01-01' "
    "AND permit_expires IS NOT NULL"
)
EXPECTED = 33_852
"""Число строк, прочитанное ДО опечатывания. Расхождение означает, что
популяция изменилась под руками, и сбор останавливается."""

FIELDS = (
    "receipt",
    "permit",
    "current_status",
    "permit_category",
    "location_type",
    "county",
    "div",
    "wd",
    "designated_basin",
    "management_district",
    "denver_basin_aquifer",
    "associated_aquifers",
    "associated_uses",
    "permit_issued",
    "permit_expires",
    "well_constructed",
    "pump_installed",
    "well_plugged",
    "_1st_beneficial_use",
    "elev",
    "well_depth",
    "top_perforated_casing",
    "bottom_perforated_casing",
    "yield",
    "static_water_level",
    "static_water_level_date",
    "wdid",
    "modified",
)

PAGE_SIZE = 25_000


def _fetch(offset: int, attempts: int = 4) -> str:
    query = (
        f"SELECT {', '.join(FIELDS)} WHERE {WHERE} ORDER BY :id LIMIT {PAGE_SIZE} OFFSET {offset}"
    )
    url = f"https://{HOST}/resource/{DATASET}.csv?$query=" + urllib.parse.quote(query)
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "dsx-case20/1.0"})
            with urllib.request.urlopen(request, timeout=600) as response:
                return response.read().decode("utf-8")
        # Оборванная передача, выглядящая готовой, — класс, давший блокирующие
        # находки шестого и десятого кейсов.
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
        source="Колорадо: разрешения на бурение скважин (DWR)",
        license=LICENSE,
        url=f"https://{HOST}/d/{DATASET}",
    )
    write_manifest(manifest, MANIFEST)
    print(f"Манифест зафиксирован: {MANIFEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
