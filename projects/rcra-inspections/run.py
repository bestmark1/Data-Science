"""Прогон десятого кейса по заполненной форме."""

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
    form = load(PROJECT / "project.yaml")
    result = run(form, _build.build(), PROJECT / "report")

    print(result.summary())
    print("площадок по обе стороны:", len(objects_across_splits(result)) or "нет")
    print("резерв не израсходован:", result.samples.extent(RESERVE))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
