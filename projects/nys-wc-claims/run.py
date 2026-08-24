"""Прогон третьего кейса по заполненной форме."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from dsx.policy import OverrideLedger
from dsx.project import load
from dsx.runner import RESERVE, objects_across_splits, run

PROJECT = Path(__file__).resolve().parent

_spec = importlib.util.spec_from_file_location("build", PROJECT / "build.py")
_build = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_build)

SINCE = "2015-01-01"
"""Отсечка популяции. Решение о том, какие годы берутся в работу, — ваше;
здесь оно записано явно, а не спрятано в умолчании."""


def main() -> int:
    overrides = OverrideLedger()
    # A13 нашла две претензии, у которых слушание датировано раньше сборки дела:
    # несчастные случаи 1941 года, слушания 1999 и 2001, а даты сборки — ровно
    # первое января 2015 и 2017. Это заглушки миграции: старые дела перенесены в
    # систему с фиктивной датой. Ядро уже разметило их пустой меткой, из анализа
    # они выпали сами. Обход записан, чтобы блокировка не скрывала прочие
    # находки, а причина осталась видимой.
    overrides.override(
        "A13",
        reason="две строки из 1 804 676 — заглушки миграции старых дел: "
        "травмы 1941 года, слушания 1999 и 2001, дата сборки 1 января. "
        "Размечены пустой меткой и в анализ не входят",
        author="автор",
    )

    form = load(PROJECT / "project.yaml")
    result = run(form, _build.build(since=SINCE), PROJECT / "report", overrides=overrides)

    print(result.summary())
    print("объектов по обе стороны:", objects_across_splits(result) or "нет")

    # Измерение НЕ проводится: моделей здесь не строится, и вызывать measure()
    # ради красивой строки в выводе значило бы записать расход выборки, которого
    # не было. Резерв остаётся нерасходованным до настоящего измерения.
    print("резерв зарегистрирован и не израсходован:", result.samples.extent(RESERVE))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
