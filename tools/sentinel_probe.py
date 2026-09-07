"""Отсутствие, записанное значением, — счётом по ВСЕМ колонкам, до опечатывания.

Правило применимости контроля, введённое перед девятнадцатым кейсом, требует:
контроль, опирающийся на свойство данных, объявляется только когда свойство
сосчитано до опечатывания. Иначе объявленный контроль оказывается пустой
тратой — это случалось в шести кейсах из одиннадцати.

В девятнадцатом кейсе счёт был сделан и всё равно соврал. Автор перечислил
колонки, которые счёл текстовыми, — двенадцать из семнадцати, — и не проверил
`respondent_name`, где заглушка и лежала. Контроль на `sentinel_as_value` был
снят как неприменимый, будучи применимым: ядро нашло 'UNKNOWN' в четырёх
строках уже после сбора.

Механизм отработал верно на тех данных, что ему дали. Ошибка была в том, что
список колонок составлялся СУЖДЕНИЕМ. Эта программа не спрашивает, какие
колонки текстовые: она берёт все, какие объявляет источник.

ЧТО ОНА ЧИТАЕТ. Только счёт: сколько строк равно каждому значению из списка
заглушек ЯДРА (`dsx.checks.data.SENTINELS`). Список задан кодом и от данных не
зависит; наружу выходит число. Ни одно значение колонки не запрашивается.

Список берётся у ядра намеренно. Контроль обязан ждать того, что ядро умеет
находить: заглушка, которой нет в списке ядра, сигнала не даст, и объявлять
контроль на неё значило бы ждать молчания.

Пример:

    python tools/sentinel_probe.py data.texas.gov ubdr-4uff \\
        received_date --from 2023-01-01 --to 2026-01-01
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dsx.checks.data import SENTINELS  # noqa: E402

TIMEOUT = 180
ATTEMPTS = 3
IDENT = "abcdefghijklmnopqrstuvwxyz0123456789_"


def _valid(name: str) -> str:
    """Имя колонки без кавычек и скобок: запрос собирается строкой."""
    if not name or not set(name.lower()) <= set(IDENT):
        raise SystemExit(f"недопустимое имя колонки: {name!r}")
    return name


def _get(url: str):
    for attempt in range(ATTEMPTS):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "dsx-sentinel-probe/1.0"})
            with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
                return json.load(response)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            if attempt == ATTEMPTS - 1:
                raise
    raise RuntimeError("недостижимо")


def columns(host: str, dataset: str) -> list[str]:
    """Все колонки набора — по объявлению источника, а не по выбору автора."""
    meta = _get(f"https://{host}/api/views/{dataset}.json")
    return [column["fieldName"] for column in meta["columns"]]


def counts(host: str, dataset: str, where: str, names: list[str]) -> dict[str, int]:
    """Сколько строк в каждой колонке равно заглушке из списка ядра."""
    values = sorted(value for value in SENTINELS if value.strip())
    listed = ", ".join("'" + value.replace("'", "''") + "'" for value in values)
    found: dict[str, int] = {}
    for name in names:
        query = f"select count(*) as n where ({where}) and {_valid(name)} in ({listed})"
        url = f"https://{host}/resource/{dataset}.json?$query=" + urllib.parse.quote(query)
        try:
            rows = _get(url)
        except urllib.error.HTTPError:
            # Колонка не строковая — источник отказывает на сравнении типов.
            # Это ответ, а не отказ инструмента: заглушки в ней нет.
            continue
        number = int(rows[0]["n"]) if rows else 0
        if number:
            found[name] = number
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description="счёт заглушек по всем колонкам набора")
    parser.add_argument("host")
    parser.add_argument("dataset")
    parser.add_argument("decided", help="колонка момента решения — ею ограничивается период")
    parser.add_argument("--from", dest="since", required=True)
    parser.add_argument("--to", dest="until", required=True)
    args = parser.parse_args()

    decided = _valid(args.decided)
    where = f"{decided} >= '{args.since}' and {decided} < '{args.until}'"

    names = columns(args.host, args.dataset)
    print(f"колонок в наборе: {len(names)}")
    print(f"значений в списке заглушек ядра: {len(SENTINELS)}")

    found = counts(args.host, args.dataset, where, names)
    if not found:
        print("\nЗаглушек из списка ядра нет ни в одной колонке.")
        print("Контроль на `sentinel_as_value` НЕПРИМЕНИМ — объявлять его нельзя.")
        return 0

    print("\nЗаглушки найдены:")
    for name, number in sorted(found.items(), key=lambda item: -item[1]):
        print(f"  {name:30s} {number:8,}")
    print("\nКонтроль на `sentinel_as_value` применим. В пре-регистрации:")
    best = max(found.items(), key=lambda item: item[1])
    print(f'  basis: "counted: заглушка в {best[0]} — {best[1]:,} строк, счёт до опечатывания"')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
