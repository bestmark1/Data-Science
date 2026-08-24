"""Прогон третьего кейса по заполненной форме."""

from __future__ import annotations

import importlib.util
from pathlib import Path

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
    form = load(PROJECT / "project.yaml")
    result = run(form, _build.build(since=SINCE), PROJECT / "report")

    print(result.summary())
    print("объектов по обе стороны:", objects_across_splits(result) or "нет")

    # Измерение НЕ проводится: моделей здесь не строится, и вызывать measure()
    # ради красивой строки в выводе значило бы записать расход выборки, которого
    # не было. Резерв остаётся нерасходованным до настоящего измерения.
    print("резерв зарегистрирован и не израсходован:", result.samples.extent(RESERVE))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
