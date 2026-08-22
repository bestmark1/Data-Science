"""Шаг 00 — получение датасета второго кейса и фиксация манифеста.

Манифест фиксируется СРАЗУ после получения, до всякого анализа: переход на
другую версию датасета незаметно назван в ревью одним из способов обойти
предрегистрацию.

Скрипт намеренно не печатает ничего о содержимом файлов — только имена, размеры
и контрольные суммы. По §2 предрегистрации схема и содержимое считаются
контактом с данными, а он допустим только после фиксации манифеста.
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

PROJECT = Path(__file__).resolve().parent.parent
SLUG = "arnabbiswas1/microsoft-azure-predictive-maintenance"
URL = f"https://www.kaggle.com/datasets/{SLUG}"
LICENSE = "не объявлена источником; данные не перераспределяются"

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

    print(f"Загружаю {SLUG}...")
    archive = download_dataset(SLUG, PROJECT / "data", credentials=credentials)
    extract_archive(archive, RAW)
    archive.unlink()

    manifest = build_manifest(RAW, source=f"kaggle:{SLUG}", license=LICENSE, url=URL)
    write_manifest(manifest, MANIFEST)

    print(f"Манифест зафиксирован: {MANIFEST}")
    for entry in manifest.files:
        print(f"  {entry.name:32} {entry.bytes / 1e6:8.2f} МБ  {entry.sha256[:16]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
