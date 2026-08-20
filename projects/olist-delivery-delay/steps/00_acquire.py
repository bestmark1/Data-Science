"""Шаг 00 — получение датасета Olist и фиксация происхождения.

Требует токен Kaggle в ~/.kaggle/kaggle.json.
Повторный запуск при совпадающих контрольных суммах ничего не перекачивает.

    python projects/olist-delivery-delay/steps/00_acquire.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from dsx.io.acquire import (
    KaggleCredentialsMissing,
    csv_to_parquet,
    download_dataset,
    extract_archive,
    read_credentials,
)
from dsx.io.manifest import build_manifest, matches, read_manifest, write_manifest

PROJECT = Path(__file__).resolve().parent.parent
SLUG = "olistbr/brazilian-ecommerce"
URL = f"https://www.kaggle.com/datasets/{SLUG}"
LICENSE = "CC BY-NC-SA 4.0"

RAW = PROJECT / "data" / "raw"
PARQUET = PROJECT / "data" / "parquet"
MANIFEST = PROJECT / "manifest.yaml"


def main() -> int:
    if MANIFEST.is_file() and matches(read_manifest(MANIFEST), RAW):
        print(f"Данные на месте и совпадают с {MANIFEST.name}. Загрузка не нужна.")
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

    csv_files = sorted(RAW.glob("*.csv"))
    print(f"Получено файлов: {len(csv_files)}. Конвертирую в Parquet...")
    for csv_file in csv_files:
        csv_to_parquet(csv_file, PARQUET / f"{csv_file.stem}.parquet")

    manifest = build_manifest(RAW, source=f"kaggle:{SLUG}", license=LICENSE, url=URL)
    write_manifest(manifest, MANIFEST)
    print(f"Манифест записан: {MANIFEST} ({len(manifest.files)} файлов)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
