"""Прогон девятого кейса с ПЕРЕПИСАННЫМ сроком (контроль К-1).

Сроком объявлено поле, которое источник переписывает после исхода. Проверяется
не реестр, а ядро: возразит ли оно, не получая подсказки.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from dsx.project import load
from dsx.runner import RESERVE, objects_across_splits, run

PROJECT = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("build", PROJECT / "build.py")
_build = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_build)


def main() -> int:
    form = load(PROJECT / "project-leak.yaml")
    result = run(form, _build.build(), PROJECT / "report-leak")

    print(result.summary())
    print("исследований по обе стороны:", len(objects_across_splits(result)) or "нет")
    print("резерв не израсходован:", result.samples.extent(RESERVE))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
