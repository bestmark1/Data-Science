"""Получение датасета восьмого кейса и фиксация манифеста.

Источник не Kaggle, и загрузка написана здесь, в проекте: это обвязка, а не
метод. Оговорено в пре-регистрации §2 до печати, чтобы потом не пришлось
переопределять, что считать правкой ядра.

Манифест фиксируется сразу после получения, до чтения значений. Перечень
колонок, число строк и дата среза прочитаны раньше и записаны в
пре-регистрации: подогнать по ним постановку под результат нельзя, а объявить
ось, которой в выгрузке нет, — можно.
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

from dsx.io.manifest import build_manifest, write_manifest

PROJECT = Path(__file__).resolve().parent
DATASET = "3x8r-34rs"
URL = f"https://data.ny.gov/api/views/{DATASET}/rows.csv?accessType=DOWNLOAD"
PAGE = f"https://data.ny.gov/d/{DATASET}"
LICENSE = "лицензия источником не объявлена; атрибуция NYSERDA"

RAW = PROJECT / "data" / "raw"
MANIFEST = PROJECT / "manifest.yaml"
TARGET = RAW / "ny_solar_projects.csv"

MIN_BYTES = 10 * 1024 * 1024
"""Нижняя граница правдоподобия. 189 335 строк на 45 колонок меньше десяти
мегабайт быть не может, и оборванная загрузка на этом ловится."""


def main() -> int:
    if MANIFEST.is_file():
        print(f"Манифест уже зафиксирован: {MANIFEST}. Замена данных требует поправки.")
        return 0

    RAW.mkdir(parents=True, exist_ok=True)
    if TARGET.is_file():
        print(f"Файл уже получен: {TARGET.name}")
    else:
        print(f"Загружаю {DATASET} ...")
        partial = TARGET.with_suffix(".csv.part")
        request = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=900) as response:
            partial.write_bytes(response.read())
        size = partial.stat().st_size
        if size < MIN_BYTES:
            partial.unlink()
            print(f"Получено {size:,} байт — это меньше правдоподобного минимума", file=sys.stderr)
            return 1
        # Переименование целиком: оборванная попытка не должна оставить файл,
        # который выглядит готовым. Тот же урок, что дал шестой кейс.
        partial.rename(TARGET)

    print(f"  {TARGET.name}: {TARGET.stat().st_size / 1048576:.1f} МБ")

    manifest = build_manifest(
        RAW,
        source="NYSERDA, портал открытых данных штата Нью-Йорк",
        license=LICENSE,
        url=PAGE,
    )
    write_manifest(manifest, MANIFEST)
    print(f"\nМанифест зафиксирован: {MANIFEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
