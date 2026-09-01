"""Сбор данных семнадцатого кейса: заявки на лицензию бизнеса в Чикаго.

Полнота сверяется счётом строк, объявленным до опечатывания: 74 751.

Названия и адреса заявителей НЕ запрашиваются. Они в наборе есть и открыты
городом, но задаче не нужны: предсказывается срок выполнения требований, а не
кто именно их выполняет. Не скачивать лишнее дешевле, чем скачать и объявить
ролью `ignored`.

Страницы упорядочены по внутреннему ключу `:id`: `license_id` повторяется у
продлений одной лицензии, и упорядочивание по нему дало бы перекрывающиеся
страницы.
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
TARGET = RAW / "applications_2015_2023.csv"

HOST = "data.cityofchicago.org"
DATASET = "r5kz-chrr"
LICENSE = "открытые данные города Чикаго (объявлено источником)"

WHERE = (
    "application_created_date >= '2015-01-01T00:00:00' "
    "AND application_created_date < '2024-01-01T00:00:00'"
)
EXPECTED = 74_751
"""Число строк, прочитанное ДО опечатывания. Расхождение означает, что
популяция изменилась под руками, и сбор останавливается."""

FIELDS = (
    "id",
    "license_id",
    "application_created_date",
    "application_requirements_complete",
    "payment_date",
    "date_issued",
    "expiration_date",
    "license_status",
    "license_status_change_date",
    "application_type",
    "license_code",
    "license_description",
    "business_activity",
    "conditional_approval",
    "ward",
    "precinct",
    "police_district",
    "community_area",
    "neighborhood",
)

PAGE_SIZE = 25_000


def _fetch(offset: int, attempts: int = 4) -> str:
    query = (
        f"SELECT {', '.join(FIELDS)} WHERE {WHERE} ORDER BY :id LIMIT {PAGE_SIZE} OFFSET {offset}"
    )
    url = f"https://{HOST}/resource/{DATASET}.csv?$query=" + urllib.parse.quote(query)
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "dsx-case17/1.0"})
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
        source="Чикаго: заявки на лицензию бизнеса",
        license=LICENSE,
        url=f"https://{HOST}/d/{DATASET}",
    )
    write_manifest(manifest, MANIFEST)
    print(f"Манифест зафиксирован: {MANIFEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
