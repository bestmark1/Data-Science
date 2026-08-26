"""Получение датасета четвёртого кейса и фиксация манифеста.

Манифест фиксируется сразу после получения, до чтения значений. Перечни колонок
прочитаны раньше и записаны в пре-регистрации: по именам подогнать постановку
под результат нельзя, а объявить ось, которой в выгрузке нет, — можно.
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
SLUG = "mirbektoktogaraev/should-this-loan-be-approved-or-denied"
URL = f"https://www.kaggle.com/datasets/{SLUG}"
LICENSE = "CC BY-SA 4.0 (объявлена источником)"

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
        archive = existing[0]
        print(f"Архив уже получен: {archive.name}")
    else:
        print(f"Загружаю {SLUG} ...")
        archive = download_dataset(SLUG, RAW, credentials=credentials)

    files = extract_archive(archive, RAW)
    for path in sorted(files):
        print(f"  {path.name}: {path.stat().st_size / 1048576:.1f} МБ")

    manifest = build_manifest(
        RAW,
        source="U.S. Small Business Administration, выгрузка по FOIA",
        license=LICENSE,
        url=URL,
    )
    write_manifest(manifest, MANIFEST)
    print(f"\nМанифест зафиксирован: {MANIFEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
