"""Прибиты ли числа, которыми пользуются проверки.

Стенд доказывает, КАКИЕ проверки существуют: выключение любой из них он
замечает. О числах он не говорит ничего. Порог, который можно сдвинуть вдвое и
не увидеть ни одного красного теста, защищён ровно тем же, чем защищено
необеспеченное объявление, — ничем.

Программа берёт каждое числовое поле каждой проверки, меняет его вдвое в обе
стороны прямо в исходнике, прогоняет ВЕСЬ набор тестов и смотрит, покраснел ли
он. Мутант, переживший прогон, означает: это число не проверяется, и его дрейф
пройдёт незамеченным.

Исходник восстанавливается из git после каждого мутанта. Незакоммиченных правок
в рабочем дереве быть не должно — программа отказывается работать иначе, чтобы
не потерять чужую работу.

Что входит в знаменатель. Первая версия брала только числовые поля КЛАССОВ
проверок и объявила долю прибитых в 18%. Доля была завышена на неизвестную
величину: самые весомые числа ядра объявлены модульными константами и в замер
не попали вовсе. `SIGMA` меряет шум во ВСЕХ пороговых проверках разом, и её
сдвиг двигает границу каждой из них.

Что в знаменатель НЕ входит и почему:

* Зёрна (`SEED`). Зерно — не порог: его сдвиг меняет числа законно, и тесты,
  сверяющие точные значения, убили бы такого мутанта, ничего не проверив. Доля
  прибитых выросла бы даром.
* Параметры фикстур стенда (`src/dsx/evals`). Мутируя стенд, мы проверяем не
  ядро, а сам стенд его же средствами.
* Операционные величины ввода-вывода (размер буфера, таймаут). Это не суждения
  о данных, и границы у них нет.
* Значения ВНЕ области смысла (`dsx.checks.domains`). Доля выше единицы, порог
  силы выше 0.5, кратность ниже единицы — всё это не «ослабленный порог», а
  проверка, ставшая всегда-включённой или всегда-выключенной. Стенд убивает
  такого мутанта неизбежно, и это убийство уже посчитано мутантом выключения.
  Считая его закреплением границы, замер завышал долю: объявленные 22% включали
  такие убийства.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

sys.path.insert(0, "src")

from dsx.checks.domains import DOMAINS  # noqa: E402


class Threshold(NamedTuple):
    """Числовое место в исходнике и ключ его объявленной области."""

    lineno: int
    name: str
    value: float | int
    key: str


TARGETS = (Path("src/dsx/checks"), Path("src/dsx/measure.py"))
"""Модули, выносящие суждение о данных. Ввод-вывод сюда не входит."""

SEED_NAMES = ("SEED", "seed")


def _paths() -> list[Path]:
    found: list[Path] = []
    for target in TARGETS:
        found.extend(sorted(target.glob("*.py")) if target.is_dir() else [target])
    return found


def _requirement_of(node: ast.ClassDef) -> str | None:
    """Требование, которое несёт класс проверки: `requirement: str = "N6"`.

    Нужно, чтобы сопоставить поле с объявленной областью: ключи областей
    именуются требованием, а не именем класса.
    """
    for item in node.body:
        if not isinstance(item, ast.AnnAssign) or not isinstance(item.target, ast.Name):
            continue
        if item.target.id != "requirement":
            continue
        if isinstance(item.value, ast.Constant) and isinstance(item.value.value, str):
            return item.value.value
    return None


def _module_constants(tree: ast.Module, stem: str) -> list[Threshold]:
    """Числовые константы уровня модуля.

    Именно здесь лежат числа, общие для нескольких проверок сразу, — и именно
    их первая версия замера не видела.
    """
    found: list[Threshold] = []
    for item in tree.body:
        if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
            name, value = item.target.id, item.value
        elif (
            isinstance(item, ast.Assign)
            and len(item.targets) == 1
            and isinstance(item.targets[0], ast.Name)
        ):
            name, value = item.targets[0].id, item.value
        else:
            continue
        if value is None or not isinstance(value, ast.Constant):
            continue
        if not isinstance(value.value, int | float) or isinstance(value.value, bool):
            continue
        if any(part in name for part in SEED_NAMES):
            continue
        found.append(Threshold(value.lineno, name, value.value, f"{stem}.{name}"))
    return found


def _numeric_fields(path: Path) -> list[Threshold]:
    """Числовые поля классов проверок и константы модуля."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: list[Threshold] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        requirement = _requirement_of(node)
        for item in node.body:
            if not isinstance(item, ast.AnnAssign) or item.value is None:
                continue
            if not isinstance(item.target, ast.Name):
                continue
            value = item.value
            if isinstance(value, ast.Constant) and isinstance(value.value, int | float):
                if isinstance(value.value, bool):
                    continue
                key = f"{requirement}.{item.target.id}" if requirement else ""
                found.append(Threshold(value.lineno, item.target.id, value.value, key))
    return _module_constants(tree, path.stem) + found


def _mutate(path: Path, lineno: int, old: float | int, new: float | int) -> bool:
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    line = lines[lineno - 1]
    text = repr(old)
    if text not in line:
        return False
    lines[lineno - 1] = line.replace(text, repr(new), 1)
    path.write_text("".join(lines), encoding="utf-8")
    return True


def _suite_is_green() -> bool:
    result = subprocess.run(
        [".venv/bin/python", "-m", "pytest", "tests", "-q", "-x", "--no-header"],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def main() -> int:
    dirty = subprocess.run(
        ["git", "status", "--porcelain", *(str(t) for t in TARGETS)],
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        print("ОТКАЗ: в мутируемых модулях есть незакоммиченные правки")
        return 1

    survivors: list[str] = []
    killed: list[str] = []
    outside: list[str] = []
    for path in _paths():
        for lineno, name, value, key in _numeric_fields(path):
            domain = DOMAINS.get(key)
            for factor in (0.5, 2.0):
                new = value * factor
                new = max(1, int(new)) if isinstance(value, int) else new
                if new == value:
                    continue
                label = f"{path.stem}:{name}={value}→{new}"
                if domain is not None and not domain.admits(new):
                    outside.append(f"{label} ({domain.reason})")
                    continue
                if not _mutate(path, lineno, value, new):
                    continue
                try:
                    survived = _suite_is_green()
                finally:
                    subprocess.run(["git", "checkout", "--", str(path)], check=True)
                # Списки разные, а не один с проверкой вхождения: одинаковые
                # имена полей встречаются в разных классах, и проверка
                # вхождения печатала «выжил» там, где мутант убит.
                (survivors if survived else killed).append(label)
                print(f"  {label:52s} {'ВЫЖИЛ' if survived else 'убит'}", flush=True)

    total = len(survivors) + len(killed)
    if not total:
        print("мутантов не найдено")
        return 1
    print(
        f"\nчисловых мутантов: {total}, убито набором: {len(killed)}, "
        f"доля прибитых: {len(killed) / total:.0%}"
    )
    if survivors:
        print("\nвыжившие — эти числа можно двигать вдвое незаметно:")
        for item in survivors:
            print(f"  {item}")
    if outside:
        print(
            f"\nне мутировали ({len(outside)}): значение вышло бы за область смысла, "
            "и убийство такого мутанта закреплением границы не является"
        )
        for item in outside:
            print(f"  {item}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
