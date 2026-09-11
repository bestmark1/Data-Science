"""Прикладной пример сверяется с ядром, а не с самим собой.

Файл не становится актуальным оттого, что его можно запустить: он расходится с
кодом молча — ровно так же, как разошлись бы числа в документах без сторожа.

Проверяется НАСТОЯЩАЯ команда примера, целиком, отдельным процессом и во
временном каталоге: результаты исследований не перезаписываются, мусора в
репозитории не остаётся.
"""

from __future__ import annotations

import math
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

EXAMPLE = ROOT / "examples" / "synthetic-delay"
WORLD = "feature-falsely-declared-available"


def _снимок(каталог: Path) -> dict[str, bytes]:
    """Содержимое каталога результатов, каким оно было до запуска."""
    if not каталог.is_dir():
        return {}
    return {f.name: f.read_bytes() for f in sorted(каталог.iterdir()) if f.is_file()}


@pytest.fixture(scope="module")
def прогон(tmp_path_factory):
    """Один запуск на все проверки: он занимает секунды, а не миллисекунды."""
    out = tmp_path_factory.mktemp("пример")
    было = _снимок(EXAMPLE / "report")
    done = subprocess.run(
        [sys.executable, str(EXAMPLE / "run.py"), "--out", str(out)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    return out, done.stdout, было


def test_the_example_produces_both_reports(прогон) -> None:
    out = прогон[0]

    for имя in ("report.md", "measured.md", "findings.txt"):
        файл = out / имя
        assert файл.is_file(), f"{имя} не создан"
        assert файл.read_text(encoding="utf-8").strip(), f"{имя} пуст"


def test_the_example_finds_exactly_what_the_world_hides(прогон) -> None:
    """Множество находок сверяется с ядром на том же мире — точным равенством.

    Совпадение заголовков или наличие файла доказали бы только, что что-то
    напечаталось. Мир стенда знает свой ответ по построению, и сравнивать надо
    множества, как это делает `judge`.
    """
    from dsx.checks import ALL_CHECKS, run_checks
    from dsx.evals.registry import BY_ID
    from harness import context_for

    out = прогон[0]
    ожидаемые = {f.value for f in run_checks(list(ALL_CHECKS), context_for(BY_ID[WORLD])).findings}
    assert ожидаемые == {"feature_after_decision"}, f"мир изменился: {sorted(ожидаемые)}"

    # Сверяется машиночитаемый итог, а не проза отчёта: `report.md` печатает
    # `signal.detail`, и имён находок в нём нет ни одного. Проверено счётом.
    отчёт = (out / "report.md").read_text(encoding="utf-8")
    assert not any(имя in отчёт for имя in ожидаемые), (
        "report.md вдруг стал печатать имена находок — сверку можно упростить, "
        "но сначала убедиться, что это не совпадение"
    )

    названы = set((out / "findings.txt").read_text(encoding="utf-8").split())
    assert названы == ожидаемые, f"пример и ядро разошлись: {названы} против {ожидаемые}"


def _числа_измерения(измерение: str, имя: str) -> tuple[float, float, float, float, float]:
    """Числа одной метрики из строки `Comparison.__str__`, без привязки к значениям."""
    число = r"[+-]?(?:\d+(?:\.\d+)?|\.\d+|nan|inf(?:inity)?)"
    шаблон = re.compile(
        rf"{re.escape(имя)}:\s*модель\s*({число})\s*,\s*"
        rf"правило\s*({число})\s*,\s*разница\s*({число})\s*"
        rf"\[({число})\s*;\s*({число})\]",
        re.IGNORECASE,
    )
    совпадение = шаблон.search(измерение)
    assert совпадение is not None, f"у метрики {имя!r} нет числовых значений и границ"
    модель, baseline, разница, низ, верх = (float(группа) for группа in совпадение.groups())
    for значение in (модель, baseline, разница, низ, верх):
        assert math.isfinite(значение), f"у метрики {имя!r} неконечное значение"
    assert низ <= верх, f"у метрики {имя!r} нижняя граница больше верхней"
    assert abs((модель - baseline) - разница) <= 0.0002, (
        f"у метрики {имя!r} разница не согласуется с моделью минус baseline"
    )
    return модель, baseline, разница, низ, верх


def test_the_example_measures_against_a_baseline_with_bounds(прогон) -> None:
    """Измерение доведено до конца: разница с базовым правилом и границы.

    Обрыв на середине — «модель обучена» без сравнения — выглядел бы успехом.
    Одних заголовков мало: отчёт без чисел прежде проходил эту проверку.
    """
    out = прогон[0]
    измерение = (out / "measured.md").read_text(encoding="utf-8")

    assert "Базовое правило" in измерение
    assert "разрешающая способность" in измерение
    assert "ошибка калибровки" in измерение
    assert "разница" in измерение
    # Доверительные границы печатаются в виде [+0.1140; +0.2615].
    assert измерение.count("[") >= 2 and ";" in измерение, "границ разницы нет"
    assert "резерв" in измерение

    for имя in ("разрешающая способность", "ошибка калибровки"):
        _числа_измерения(измерение, имя)

    выборка = re.search(r"Выборка:\s*[^\n]*строк\s+([\d,\s\u00a0]+)", измерение)
    assert выборка is not None, "число строк измерения не найдено"
    строк = int(выборка.group(1).replace(",", "").replace(" ", "").replace("\u00a0", ""))
    assert строк > 0, "число строк измерения неположительно"


def test_the_example_prints_what_it_does_not_prove(прогон) -> None:
    """Ограничения печатаются РЯДОМ с результатом, а не прячутся в примечание.

    Инструмент, принятый за универсальный, вреднее отсутствующего.
    """
    вывод = прогон[1]
    измерение = (прогон[0] / "measured.md").read_text(encoding="utf-8")

    for место in (вывод, измерение):
        assert "НЕ подтверждает" in место
        assert "не отличает утечку" in место or "НЕ отличает" in место
        assert "ПРОЦЕДУРУ, а не качество" in место


def test_the_example_leaves_nothing_behind(прогон) -> None:
    """Запуск с `--out` не трогает каталог результатов в репозитории.

    Требовать его ОТСУТСТВИЯ нельзя: документированная команда без `--out`
    законно создаёт `examples/synthetic-delay/report/`, и такая проверка падала
    бы ровно у того, кто пример запускал. Тот же дефект уже был найден в
    проверке диагностического лога. Сверяется содержимое до и после.
    """
    было = прогон[2]

    assert _снимок(EXAMPLE / "report") == было, "запуск с --out написал в каталог примера"
