"""Сбор данных шестнадцатого кейса: вызовы пожарно-спасательной службы.

Строка источника — выезд ОДНОЙ машины. На вызов их приходится несколько, и
свёртка к вызову есть работа построителя, а не сбора: сбор обязан сохранить то,
что отдал источник.

Полнота сверяется счётом строк, объявленным до опечатывания: 695 428.

Страницы упорядочены по внутреннему ключу `:id`. Ни `call_number`, ни
`unit_id` строку не определяют — определяет их пара, — а упорядочивание по
неуникальному ключу даёт перекрывающиеся страницы. Класс «порядок строк не
определён» проект ловил трижды.
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
TARGET = RAW / "fire_calls_2022_2023.csv"

HOST = "data.sfgov.org"
DATASET = "nuek-vuh3"
LICENSE = "открытые данные города Сан-Франциско (объявлено источником)"

WHERE = "received_dttm >= '2022-01-01T00:00:00' AND received_dttm < '2024-01-01T00:00:00'"
EXPECTED = 695_428
"""Число строк, прочитанное ДО опечатывания. Расхождение означает, что
популяция изменилась под руками, и сбор останавливается."""

FIELDS = (
    "call_number",
    "unit_id",
    "received_dttm",
    "dispatch_dttm",
    "on_scene_dttm",
    "available_dttm",
    "call_type",
    "call_type_group",
    "original_priority",
    "priority",
    "final_priority",
    "als_unit",
    "number_of_alarms",
    "unit_type",
    "unit_sequence_in_call_dispatch",
    "battalion",
    "station_area",
    "zipcode_of_incident",
    "neighborhoods_analysis_boundaries",
    "call_final_disposition",
)

PAGE_SIZE = 25_000


def _fetch(offset: int, attempts: int = 4) -> str:
    query = (
        f"SELECT {', '.join(FIELDS)} WHERE {WHERE} ORDER BY :id LIMIT {PAGE_SIZE} OFFSET {offset}"
    )
    url = f"https://{HOST}/resource/{DATASET}.csv?$query=" + urllib.parse.quote(query)
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "dsx-case16/1.0"})
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
        source="SFFD: вызовы пожарно-спасательной службы Сан-Франциско",
        license=LICENSE,
        url=f"https://{HOST}/d/{DATASET}",
    )
    write_manifest(manifest, MANIFEST)
    print(f"Манифест зафиксирован: {MANIFEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
