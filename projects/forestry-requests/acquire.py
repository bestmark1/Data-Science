"""Сбор данных восемнадцатого кейса: наряды на работы с деревьями.

Полнота сверяется счётом строк, объявленным до опечатывания: 373 116.

Набор заменён до первого прогона: в первоначальном `mu46-p9is` колонка
`closeddate` оказалась пуста у ВСЕХ 291 320 строк, и события не существовало
вовсе. Запись о замене — в §3б пре-регистрации. Заполненность колонки события у
нового набора прочитана заранее: 181 049 из 373 116.

Сведения о заявителе — почтовый индекс, город, штат, вид обратившегося — НЕ
запрашиваются. Они в наборе есть и открыты городом, но задаче не нужны:
предсказывается срок работ с деревом, а не кто о нём сообщил. Не скачивать
лишнее дешевле, чем скачать и объявить ролью `ignored`.

Не запрашиваются и свободные тексты — подробности жалобы, заметки для
обратившегося: они могут нести имена и обстоятельства, а задаче не нужны.

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
TARGET = RAW / "tree_work_orders_2021_2023.csv"

HOST = "data.cityofnewyork.us"
DATASET = "bdjm-n7q4"
LICENSE = "открытые данные города Нью-Йорка (объявлено источником)"

WHERE = "createddate >= '2021-01-01T00:00:00' AND createddate < '2024-01-01T00:00:00'"
EXPECTED = 373_116
"""Число строк, прочитанное ДО опечатывания. Расхождение означает, что
популяция изменилась под руками, и сбор останавливается."""

FIELDS = (
    "objectid",
    "createddate",
    "closeddate",
    "canceldate",
    "actualfinishdate",
    "projstartdate",
    "updateddate",
    "wotype",
    "wocategory",
    "wopriority",
    "wostatus",
    "woentity",
    "wocontract",
    "cancelreason",
    "sidewalkdamage",
    "wowireconflict",
    "wowoodremains",
    "boroughcode",
    "communityboard",
    "parkzone",
    "zipcode",
    "citycouncil",
    "nta",
)

PAGE_SIZE = 25_000


def _fetch(offset: int, attempts: int = 4) -> str:
    query = (
        f"SELECT {', '.join(FIELDS)} WHERE {WHERE} ORDER BY :id LIMIT {PAGE_SIZE} OFFSET {offset}"
    )
    url = f"https://{HOST}/resource/{DATASET}.csv?$query=" + urllib.parse.quote(query)
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "dsx-case18/1.0"})
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
        source="Нью-Йорк: наряды на работы с городскими деревьями",
        license=LICENSE,
        url=f"https://{HOST}/d/{DATASET}",
    )
    write_manifest(manifest, MANIFEST)
    print(f"Манифест зафиксирован: {MANIFEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
