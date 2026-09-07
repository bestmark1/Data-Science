"""Сбор данных девятнадцатого кейса: жалобы на страховые компании Техаса.

Полнота сверяется счётом строк, объявленным до опечатывания: 64 580.

Отрасль новая, а свойство набора, важнее прочих, установлено до опечатывания
счётом: незакрытых жалоб в нём НЕТ ВООБЩЕ — ноль из 37 070 за 2025–2026 годы.
Департамент публикует жалобу, когда закрывает её. Наблюдаемость исхода 100% есть
свойство отбора, а не процесса, и предсказание P-10 говорит именно об этом.

Персональных сведений о заявителе набор не содержит: `complainant_role` и
`complainant_type` — роль и вид обратившегося, не он сам. `respondent_name` —
наименование страховой компании, то есть юридического лица.

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
TARGET = RAW / "tdi_complaints_2023_2025.csv"

HOST = "data.texas.gov"
DATASET = "ubdr-4uff"
LICENSE = "открытые данные штата Техас (объявлено источником)"

WHERE = "received_date >= '2023-01-01' AND received_date < '2026-01-01'"
EXPECTED = 64_580
"""Число строк, прочитанное ДО опечатывания. Расхождение означает, что
популяция изменилась под руками, и сбор останавливается."""

FIELDS = (
    "complaint_number",
    "respondent_id",
    "respondent_name",
    "respondent_role",
    "respondent_type",
    "received_date",
    "closed_date",
    "complaint_type",
    "coverage_type",
    "coverage_level",
    "complainant_role",
    "complainant_type",
    "involved_party_type",
    "reason",
    "complaint_confirmed_code",
    "disposition",
    "keyword",
)

PAGE_SIZE = 25_000


def _fetch(offset: int, attempts: int = 4) -> str:
    query = (
        f"SELECT {', '.join(FIELDS)} WHERE {WHERE} ORDER BY :id LIMIT {PAGE_SIZE} OFFSET {offset}"
    )
    url = f"https://{HOST}/resource/{DATASET}.csv?$query=" + urllib.parse.quote(query)
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "dsx-case19/1.0"})
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
        source="Техас: жалобы потребителей на страховые компании (TDI)",
        license=LICENSE,
        url=f"https://{HOST}/d/{DATASET}",
    )
    write_manifest(manifest, MANIFEST)
    print(f"Манифест зафиксирован: {MANIFEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
