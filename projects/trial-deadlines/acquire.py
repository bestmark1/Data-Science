"""Сбор данных девятого кейса: текущие записи и ПЕРВЫЕ версии тех же записей.

Пересмотр значения виден только при сличении двух моментов. Публичный API
отдаёт запись «как сейчас»; архив версий — «как было при регистрации». Кейс
строится на разнице между ними, поэтому собираются оба.

Риск источника записан в пре-регистрации §6 до сбора: история отдаётся
недокументированным эндпоинтом, и он может исчезнуть. Поэтому собранное
складывается в файлы и фиксируется манифестом: воспроизводимость обеспечивает
файл, а не доступность сайта.

Популяция объявлена до сбора и отбирается только признаками, известными в
момент решения: интервенционные, третья фаза, регистрация 2012-2016.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dsx.io.manifest import build_manifest, write_manifest

PROJECT = Path(__file__).resolve().parent
RAW = PROJECT / "data" / "raw"
MANIFEST = PROJECT / "manifest.yaml"
CURRENT = RAW / "studies_current.jsonl"
ORIGINAL = RAW / "studies_version0.jsonl"

POPULATION = (
    "AREA[StudyFirstPostDate]RANGE[2012-01-01,2016-12-31] "
    "AND AREA[StudyType]INTERVENTIONAL AND AREA[Phase]PHASE3"
)

FIELDS = [
    "NCTId",
    "BriefTitle",
    "OverallStatus",
    "StudyType",
    "Phase",
    "LeadSponsorName",
    "LeadSponsorClass",
    "EnrollmentCount",
    "EnrollmentType",
    "StudyFirstPostDate",
    "StartDate",
    "StartDateType",
    "PrimaryCompletionDate",
    "PrimaryCompletionDateType",
    "CompletionDate",
    "CompletionDateType",
    "LastUpdatePostDate",
    "Condition",
    "InterventionType",
    "LocationCountry",
    "DesignAllocation",
    "DesignInterventionModel",
    "DesignMasking",
    "DesignPrimaryPurpose",
    "HasResults",
]

WORKERS = 12
"""Умеренно. Источник государственный и бесплатный, и выжимать из него всё,
что он отдаст, — плохая манера, а не оптимизация. Двенадцать — столько же,
сколько открывает обычный браузер на пару хостов."""

PAUSE = 0.05


def _get(url: str, attempts: int = 4) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "dsx-case9/1.0"})
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.load(response)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            if attempt == attempts - 1:
                raise
            time.sleep(2**attempt)
    raise RuntimeError("недостижимо")


def fetch_current() -> list[dict]:
    """Текущие записи всей популяции, страницами по тысяче."""
    studies: list[dict] = []
    token = None
    while True:
        query = {
            "filter.advanced": POPULATION,
            "fields": "|".join(FIELDS),
            "pageSize": "1000",
            "format": "json",
        }
        if token:
            query["pageToken"] = token
        page = _get("https://clinicaltrials.gov/api/v2/studies?" + urllib.parse.urlencode(query))
        studies += page.get("studies", [])
        token = page.get("nextPageToken")
        print(f"  получено {len(studies):,}")
        if not token:
            return studies


def fetch_original(nct: str) -> dict | None:
    """Первая версия записи: то, что было объявлено при регистрации.

    Пересмотр измеряется сличением этого значения с текущим, поэтому число
    версий здесь не нужно: важна разница величин, а не количество правок.
    """
    try:
        # Версия 0 берётся НАПРЯМУЮ. Первая версия скрипта сначала запрашивала
        # оглавление истории и лишь потом саму версию — два обращения на
        # запись вместо одного, и сбор растягивался на два часа. Номер версии
        # угадывать не пришлось: нумерация начинается с нуля, проверено на трёх
        # записях до перехода.
        version = _get(f"https://clinicaltrials.gov/api/int/studies/{nct}/history/0")
        time.sleep(PAUSE)
        study = version.get("study", {})
        posted = (
            study.get("protocolSection", {}).get("statusModule", {}).get("studyFirstSubmitDate")
        )
        return {"nctId": nct, "versionDate": posted, "study": study}
    except Exception as exc:  # noqa: BLE001 — причина записывается, строка не теряется
        return {"nctId": nct, "error": f"{type(exc).__name__}: {exc}"}


def main() -> int:
    if MANIFEST.is_file():
        print(f"Манифест уже зафиксирован: {MANIFEST}. Замена данных требует поправки.")
        return 0
    RAW.mkdir(parents=True, exist_ok=True)

    if CURRENT.is_file():
        current = [json.loads(line) for line in CURRENT.read_text(encoding="utf-8").splitlines()]
        print(f"Текущие записи уже собраны: {len(current):,}")
    else:
        print("Собираю текущие записи ...")
        current = fetch_current()
        CURRENT.write_text(
            "\n".join(json.dumps(s, ensure_ascii=False) for s in current) + "\n", encoding="utf-8"
        )

    ncts = [s["protocolSection"]["identificationModule"]["nctId"] for s in current]
    print(f"\nСобираю первые версии для {len(ncts):,} записей ...")

    done = 0
    with ORIGINAL.open("w", encoding="utf-8") as out, ThreadPoolExecutor(WORKERS) as pool:
        for record in pool.map(fetch_original, ncts):
            if record is not None:
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
            done += 1
            if done % 500 == 0:
                print(f"  {done:,} из {len(ncts):,}")

    failed = sum(
        1 for line in ORIGINAL.read_text(encoding="utf-8").splitlines() if '"error"' in line
    )
    print(f"\nпервых версий получено: {len(ncts) - failed:,}, с ошибкой: {failed:,}")
    if failed > len(ncts) * 0.05:
        print("Слишком много ошибок сбора: манифест не фиксируется", file=sys.stderr)
        return 1

    manifest = build_manifest(
        RAW,
        source="ClinicalTrials.gov, публичный API v2 и архив версий",
        license="общественное достояние правительства США (объявлено источником)",
        url="https://clinicaltrials.gov/",
    )
    write_manifest(manifest, MANIFEST)
    print(f"\nМанифест зафиксирован: {MANIFEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
