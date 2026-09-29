"""Проба стенда на чужих seed сдвигает ТОЛЬКО мир и инжекторы.

Иначе частоты мерили бы внутренний seed проверок ядра, а не данные, — ошибку,
которую пришлось исключать вручную при разборе 29 сентября.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

import dsx.evals.injectors as injectors
import dsx.evals.world as world

TOOL = Path(__file__).resolve().parents[1] / "tools" / "bench_seeds.py"
_spec = importlib.util.spec_from_file_location("bench_seeds", TOOL)
bench_seeds = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench_seeds)


def test_the_shift_moves_the_world_by_exactly_its_amount() -> None:
    with bench_seeds.shifted(3000):
        moved = world.build_world(seed=7).main
    assert moved.equals(world.build_world(seed=3007).main)


def test_the_shift_leaves_numpy_of_the_core_alone() -> None:
    original = np.random.default_rng
    with bench_seeds.shifted(3000):
        assert np.random.default_rng is original, "подменён numpy всего процесса"
        assert world.np is not np and injectors.np is not np
    assert world.np is np and injectors.np is np
