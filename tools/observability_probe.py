"""Заполненность колонки события — РАЗРЕЗОМ ПО ВРЕМЕНИ, до опечатывания.

Восемнадцатый кейс споткнулся об это дважды подряд.

Сначала по имени `closeddate` было заключено, что колонка несёт значения; она
оказалась пуста у 100% строк, и набор пришлось менять уже после сбора. Отсюда
правило: имя не говорит о заполненности, и число непустых читается ДО
опечатывания — это такой же счёт, как объём популяции.

Потом счёт непустых был прочитан — 48.5% по всей популяции, вдвое выше
объявленного порога в 20%, — и набор всё равно оказался негодным: с июля 2023
дата закрытия не проставлена НИ У ОДНОГО наряда. Средняя заполненность обрыва
не показывает.

Проверка N18 такой обрыв находит, но ПОСЛЕ сбора, когда пре-регистрация уже
опечатана. Ядро спасает от неверных выводов, а не от испорченного кейса. Эта
программа читает то же самое ДО.

ЧТО ОНА ЧИТАЕТ. Только счёт: число строк и число непустых значений по месяцам.
Значения колонок не запрашиваются ни одной строкой — до опечатывания их читать
нельзя, и запрос устроен так, что прочитать их нечем.

Пример:

    python tools/observability_probe.py data.cityofnewyork.us bdjm-n7q4 \\
        createddate closeddate --from 2021-01-01 --to 2024-01-01
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.parse
import urllib.request

TIMEOUT = 180
"""Источник считает группировку медленно: пять месяцев набора `bdjm-n7q4`
занимают около пятидесяти секунд, все три года — заметно дольше. Число взято
по замеру, а не по привычке."""

ATTEMPTS = 3
IDENT = "abcdefghijklmnopqrstuvwxyz0123456789_"


def _valid(name: str) -> str:
    """Имя колонки без кавычек и скобок: запрос собирается строкой."""
    if not name or not set(name.lower()) <= set(IDENT):
        raise SystemExit(f"недопустимое имя колонки: {name!r}")
    return name


def monthly(
    host: str, dataset: str, decided: str, event: str, since: str, until: str
) -> list[dict]:
    """Число строк и непустых значений события по месяцам решения."""
    decided, event = _valid(decided), _valid(event)
    where = f"{decided} >= '{since}T00:00:00' AND {decided} < '{until}T00:00:00'"
    query = (
        f"SELECT date_trunc_ym({decided}) AS month, count(*) AS rows, "
        f"count({event}) AS filled WHERE {where} GROUP BY month ORDER BY month LIMIT 5000"
    )
    url = f"https://{host}/resource/{dataset}.json?$query=" + urllib.parse.quote(query)
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    for attempt in range(1, ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            body = error.read()[:200].decode(errors="replace")
            raise SystemExit(f"источник отказал: {error.code} {body}") from error
        except (TimeoutError, urllib.error.URLError, OSError) as error:
            # Молчаливый повтор скрыл бы, что источник еле отвечает, а это
            # сведение о нём: тринадцатый кейс упал на неучтённом обрыве чтения.
            print(f"  попытка {attempt} из {ATTEMPTS} не удалась: {error}", file=sys.stderr)
            if attempt == ATTEMPTS:
                raise SystemExit("источник не ответил за отведённое время") from error
    raise SystemExit("недостижимо")


def report(rows: list[dict], floor: float) -> int:
    """Напечатать разрез и назвать провалы. Возврат — код выхода."""
    if not rows:
        print("источник вернул пусто: проверьте имена колонок и границы периода")
        return 1

    total = sum(int(r["rows"]) for r in rows)
    filled = sum(int(r["filled"]) for r in rows)
    print(f"популяция: {total:,}, с непустым событием: {filled:,} ({filled / total:.1%})\n")

    weak = []
    for row in rows:
        n, f = int(row["rows"]), int(row["filled"])
        share = f / n if n else 0.0
        mark = "  ← НИЖЕ ПОРОГА" if share < floor else ""
        if share < floor:
            weak.append((row["month"][:7], n, share))
        print(f"  {row['month'][:7]}  строк {n:>8,}  непустых {f:>8,}  {share:6.1%}{mark}")

    if not weak:
        print(f"\nвсе месяцы выше {floor:.0%}: обрыва записи нет")
        return 0

    # Молчание здесь было бы тем же умолчанием, которое стоило кейса: средняя
    # заполненность порог прошла, а месяцы под ним съели треть периода.
    lost = sum(n for _, n, _ in weak)
    print(
        f"\nМЕСЯЦЕВ НИЖЕ {floor:.0%}: {len(weak)}, в них {lost:,} строк "
        f"({lost / total:.1%} популяции)."
    )
    print("Средняя заполненность этого НЕ показывает. Прежде чем опечатывать")
    print("пре-регистрацию, решите: сузить период, сменить набор или объявить")
    print("обрыв частью кейса.")
    return 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("host", help="например data.cityofnewyork.us")
    parser.add_argument("dataset", help="идентификатор набора, например bdjm-n7q4")
    parser.add_argument("decided", help="колонка момента решения")
    parser.add_argument("event", help="колонка события, чью заполненность проверяем")
    parser.add_argument("--from", dest="since", required=True, help="ГГГГ-ММ-ДД включительно")
    parser.add_argument("--to", dest="until", required=True, help="ГГГГ-ММ-ДД исключительно")
    parser.add_argument(
        "--floor",
        type=float,
        default=0.2,
        help="порог заполненности месяца; по умолчанию тот же 0.2, что объявлен в "
        "пре-регистрации восемнадцатого кейса",
    )
    args = parser.parse_args()
    rows = monthly(args.host, args.dataset, args.decided, args.event, args.since, args.until)
    return report(rows, args.floor)


if __name__ == "__main__":
    sys.exit(main())
