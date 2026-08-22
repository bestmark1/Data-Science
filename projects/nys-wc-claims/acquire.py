"""Получение датасета третьего кейса и фиксация манифеста.

Манифест фиксируется СРАЗУ после получения, до всякого анализа: переход на
другую версию датасета — способ обойти пре-регистрацию незаметно.

Скрипт намеренно не печатает ничего о содержимом: только имена, размеры и
контрольные суммы. Схема и строки считаются контактом с данными.
"""

from __future__ import annotations

import sys
from pathlib import Path

from dsx.io.acquire import (
    KaggleCredentialsMissing,
    download_dataset,
    extract_archive,
    read_credentials,
)
from dsx.io.manifest import build_manifest, write_manifest

PROJECT = Path(__file__).resolve().parent
SLUG = "new-york-state/nys-assembled-workers'-compensation-claims"
URL = f"https://www.kaggle.com/datasets/{SLUG}"
LICENSE = "CC0: Public Domain (объявлена источником)"

MAX_UNCOMPRESSED = 24 << 30
"""Лимит распакованного размера, 24 ГиБ.

Умолчание ядра — 4 ГиБ. Источник объявляет 1.59 ГиБ архива, а CSV сжимается
в пять-десять раз, поэтому лимит поднят вызывающей стороной осознанно, а не
подогнан правкой умолчания.
"""

RAW = PROJECT / "data" / "raw"
MANIFEST = PROJECT / "manifest.yaml"


def main() -> int:
    if MANIFEST.is_file():
        print(f"Манифест уже зафиксирован: {MANIFEST}. Замена данных требует поправки.")
        return 0

    try:
        credentials = read_credentials()
    except KaggleCredentialsMissing as exc:
        print(f"Нет доступа к Kaggle: {exc}", file=sys.stderr)
        return 1

    RAW.mkdir(parents=True, exist_ok=True)
    existing = sorted(RAW.glob("*.zip"))
    if existing:
        # Повторная загрузка полутора гигабайт ради упавшего последнего шага —
        # цена, которую платить незачем. Целостность обеспечивает манифест.
        archive = existing[0]
        print(f"Архив уже получен: {archive.name}, {archive.stat().st_size / 1048576:.1f} МБ")
    else:
        print(f"Загружаю {SLUG} ...")
        archive = download_dataset(SLUG, RAW, credentials=credentials)
        print(f"Архив получен: {archive.stat().st_size / 1048576:.1f} МБ")

    files = extract_archive(archive, RAW, max_uncompressed_bytes=MAX_UNCOMPRESSED)
    for path in sorted(files):
        print(f"  {path.name}: {path.stat().st_size / 1048576:.1f} МБ")

    manifest = build_manifest(
        RAW, source="Workers' Compensation Board, штат Нью-Йорк", license=LICENSE, url=URL
    )
    write_manifest(manifest, MANIFEST)
    print(f"\nМанифест зафиксирован: {MANIFEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
