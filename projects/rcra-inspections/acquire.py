"""Получение данных десятого кейса и фиксация манифеста.

Источник — общий архив EPA ECHO по обращению с опасными отходами. Берутся
четыре файла из шести: проверки, нарушения, площадки и коды отрасли. Два
оставшихся (принудительные меры и история значимого несоблюдения) в постановку
не входят.

Полнота загрузки сверяется дважды: объявленный размер против полученного и
наличие оглавления архива. Черновая загрузка при подготовке кейса оборвалась
на 96 МБ из 114, и оборванный файл выглядел как готовый — ровно та находка,
что дала шестому кейсу первый блокирующий дефект.
"""

from __future__ import annotations

import sys
import urllib.request
import zipfile
from pathlib import Path

from dsx.io.manifest import build_manifest, write_manifest

PROJECT = Path(__file__).resolve().parent
RAW = PROJECT / "data" / "raw"
MANIFEST = PROJECT / "manifest.yaml"
ARCHIVE = RAW / "rcra_downloads.zip"

URL = "https://echo.epa.gov/files/echodownloads/rcra_downloads.zip"
PAGE = "https://echo.epa.gov/tools/data-downloads"
LICENSE = "общественное достояние правительства США (объявлено источником)"

WANTED = (
    "RCRA_EVALUATIONS.csv",
    "RCRA_VIOLATIONS.csv",
    "RCRA_FACILITIES.csv",
    "RCRA_NAICS.csv",
)


def main() -> int:
    if MANIFEST.is_file():
        print(f"Манифест уже зафиксирован: {MANIFEST}. Замена данных требует поправки.")
        return 0
    RAW.mkdir(parents=True, exist_ok=True)

    if not ARCHIVE.is_file():
        print(f"Загружаю {URL} ...")
        partial = ARCHIVE.with_suffix(".zip.part")
        request = urllib.request.Request(URL, headers={"User-Agent": "dsx-case10/1.0"})
        with urllib.request.urlopen(request, timeout=3600) as response:
            declared = int(response.headers.get("Content-Length") or 0)
            with partial.open("wb") as out:
                while chunk := response.read(1 << 20):
                    out.write(chunk)
        got = partial.stat().st_size
        if declared and declared != got:
            partial.unlink()
            print(f"Объявлено {declared:,} байт, получено {got:,}", file=sys.stderr)
            return 1
        if not zipfile.is_zipfile(partial):
            partial.unlink()
            print("В хвосте архива нет оглавления: загрузка оборвана", file=sys.stderr)
            return 1
        partial.rename(ARCHIVE)

    print(f"  архив: {ARCHIVE.stat().st_size / 1048576:.1f} МБ")
    with zipfile.ZipFile(ARCHIVE) as archive:
        for name in WANTED:
            target = RAW / name
            if not target.is_file():
                target.write_bytes(archive.read(name))
            print(f"  {name}: {target.stat().st_size / 1048576:.1f} МБ")

    manifest = build_manifest(
        RAW,
        source="U.S. EPA, ECHO: данные RCRA по обращению с опасными отходами",
        license=LICENSE,
        url=PAGE,
    )
    write_manifest(manifest, MANIFEST)
    print(f"\nМанифест зафиксирован: {MANIFEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
