"""Сбор данных одиннадцатого кейса: отчётность о соблюдении расписания.

Источник отдаёт помесячные архивы. Полнота каждого сверяется дважды:
объявленный размер против полученного и наличие оглавления архива. Шестой кейс
дал первый блокирующий дефект именно на этом — оборванный файл выглядел как
готовый; десятый повторил ту же беду при черновой загрузке мимо штатного пути.

Загрузка идёт во временный файл и переименовывается целиком: оборванная
попытка не должна оставить файл, который выглядит готовым.
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

YEAR = 2024
MONTHS = range(1, 13)
BASE = (
    "https://transtats.bts.gov/PREZIP/On_Time_Reporting_Carrier_On_Time_Performance_1987_present_"
)
PAGE = "https://www.transtats.bts.gov/Fields.asp?gnoyr_VQ=FGJ"
LICENSE = "общественное достояние правительства США (объявлено источником)"


def fetch(year: int, month: int) -> Path:
    target = RAW / f"bts_{year}_{month:02d}.zip"
    if target.is_file():
        return target

    partial = target.with_suffix(".zip.part")
    request = urllib.request.Request(
        f"{BASE}{year}_{month}.zip", headers={"User-Agent": "dsx-case11/1.0"}
    )
    with urllib.request.urlopen(request, timeout=1800) as response:
        declared = int(response.headers.get("Content-Length") or 0)
        with partial.open("wb") as out:
            while chunk := response.read(1 << 20):
                out.write(chunk)

    got = partial.stat().st_size
    if declared and declared != got:
        partial.unlink()
        raise RuntimeError(f"{year}-{month:02d}: объявлено {declared:,} байт, получено {got:,}")
    if not zipfile.is_zipfile(partial):
        partial.unlink()
        raise RuntimeError(f"{year}-{month:02d}: в хвосте архива нет оглавления")

    partial.rename(target)
    return target


def main() -> int:
    if MANIFEST.is_file():
        print(f"Манифест уже зафиксирован: {MANIFEST}. Замена данных требует поправки.")
        return 0
    RAW.mkdir(parents=True, exist_ok=True)

    total = 0
    for month in MONTHS:
        try:
            path = fetch(YEAR, month)
        except Exception as exc:  # noqa: BLE001 — причина называется, сбор не продолжается вслепую
            print(f"Сбор прерван: {exc}", file=sys.stderr)
            return 1
        size = path.stat().st_size
        total += size
        print(f"  {path.name}: {size / 1048576:.1f} МБ")

    print(f"\nвсего {total / 1048576:.0f} МБ за {len(list(MONTHS))} месяцев")
    manifest = build_manifest(
        RAW,
        source="U.S. Bureau of Transportation Statistics, On-Time Reporting Carrier Performance",
        license=LICENSE,
        url=PAGE,
    )
    write_manifest(manifest, MANIFEST)
    print(f"Манифест зафиксирован: {MANIFEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
