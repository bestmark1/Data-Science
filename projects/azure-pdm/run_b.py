"""Протокол B через форму проекта.

Скрипт строит скользящую сетку решений и агрегаты телеметрии. Объявления —
в protocol-b.yaml.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from dsx.project import load
from dsx.runner import objects_across_splits, run

PROJECT = Path(__file__).resolve().parent

_spec = importlib.util.spec_from_file_location("build_b", PROJECT / "build_b.py")
_build = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_build)


def main() -> int:
    form = load(PROJECT / "protocol-b.yaml")
    result = run(form, _build.build_case().main, PROJECT / "report" / "b")

    print(result.summary())
    print("машин по обе стороны:", objects_across_splits(result) or "нет", "(ожидаемо)")

    result.samples.select("w0", "выбор окна признаков и горизонта")
    result.samples.measure("резерв")
    result.study.conclude("протокол B: пересечение окон признаков остаётся, измерение на резерве")
    print("измерение на резерве:", result.samples.extent("резерв"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
