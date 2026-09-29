"""Стенд на чужих seed: частоты вместо исхода на одном мире.

Стенд проходит 75 из 75 на своих seed, и это одно наблюдение на кейс. Разбор
29 сентября 2026 (класс 18 журнала повторов) показал, что часть кейсов зелёная
по удаче seed. Эта программа повторяет тот расчёт, чтобы его мог проверить
любой, а не только автор: прежде скрипты лежали во временной папке сессии.

КАК СДВИГАЕТСЯ SEED. Только в модулях мира и инжекторов (`dsx.evals.world`,
`dsx.evals.injectors`): их имя `np` подменяется посредником, чей
`random.default_rng(seed)` получает `seed + сдвиг`. Генератор ядра
(`measure.py`) не трогается — иначе сдвиг мерил бы внутренний seed проверок, а
не данные.

ПРЕЖДЕ ЧЕМ ВЕРИТЬ. Каждая команда сначала прогоняет стенд со сдвигом 0 и
отказывается работать, если провалов не ноль: это известный ответ.

Команды:

    python tools/bench_seeds.py sweep 1000 2000 3000
    python tools/bench_seeds.py rates --worlds 100 flipped-feature-relation
    python tools/bench_seeds.py n4 --worlds 100
    python tools/bench_seeds.py n18 --worlds 100

Seed, использованные при разведке 29 сентября (сдвиги 0–199, 1000, 2000, 3000),
для отложенного набора DS-006 уже не годятся: их результаты видел автор.
"""

from __future__ import annotations

import argparse
import sys
import types
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

import dsx.evals.injectors as injectors  # noqa: E402
import dsx.evals.world as world  # noqa: E402
from dsx.checks.drift import (  # noqa: E402
    SIGMA,
    _association_error,
    _windows_with_labels,
    association,
)
from dsx.evals.case import judge  # noqa: E402
from dsx.evals.injectors import _moment_at  # noqa: E402
from dsx.evals.registry import ALL  # noqa: E402
from dsx.label import LABEL  # noqa: E402
from harness import context_for, report_for  # noqa: E402

BY_ID = {bundle.id: bundle for bundle in ALL}


def _numpy_with_shift(shift: int) -> types.ModuleType:
    """Посредник numpy, у которого `random.default_rng(seed)` сдвинут на `shift`."""
    rnd = types.SimpleNamespace(
        **{name: getattr(np.random, name) for name in dir(np.random) if not name.startswith("__")}
    )
    rnd.default_rng = lambda seed=None, *a, **k: np.random.default_rng(
        None if seed is None else seed + shift, *a, **k
    )

    class Shifted(types.ModuleType):
        def __getattr__(self, name):
            return getattr(np, name)

    module = Shifted("numpy_shifted")
    module.random = rnd
    return module


@contextmanager
def shifted(shift: int) -> Iterator[None]:
    """Сдвинуть seed мира и инжекторов на время блока; numpy ядра не трогается."""
    world.np, injectors.np = _numpy_with_shift(shift), _numpy_with_shift(shift)
    try:
        yield
    finally:
        world.np, injectors.np = np, np


def failures(bundles, shift: int) -> list[tuple[str, str, list[str], list[str]]]:
    """Кейсы, где ядро разошлось с заложенным, при данном сдвиге."""
    out = []
    with shifted(shift):
        for bundle in bundles:
            outcome = judge(bundle.case, frozenset(s.finding for s in report_for(bundle).signals))
            if not outcome.ok:
                out.append(
                    (
                        bundle.id,
                        outcome.verdict.value,
                        sorted(f.value for f in outcome.missed),
                        sorted(f.value for f in outcome.unexpected),
                    )
                )
    return out


def known_answer() -> None:
    """Сдвиг 0 обязан дать ноль провалов: иначе измеритель сломан, а не стенд."""
    broken = failures(ALL, 0)
    if broken:
        raise SystemExit(f"ОТКАЗ: при сдвиге 0 провалов {len(broken)} — измеритель не годен")
    print(f"известный ответ: сдвиг 0 → 0 провалов из {len(ALL)}")


def sweep(shifts: list[int]) -> None:
    for shift in shifts:
        broken = failures(ALL, shift)
        print(f"сдвиг {shift}: провалов {len(broken)} из {len(ALL)}")
        for row in broken:
            print("   ", row)


def rates(ids: list[str], worlds: int) -> None:
    bundles = [BY_ID[i] for i in ids]
    count = dict.fromkeys(ids, 0)
    for shift in range(worlds):
        for row in failures(bundles, shift):
            count[row[0]] += 1
    for i in ids:
        print(f"{i}: провалов {count[i]} из {worlds}")


def _strength(frame: pl.DataFrame) -> float:
    return abs(association(frame["lead_days"], frame[LABEL]))


def n4(worlds: int) -> None:
    """Сила связи lead_days с исходом против порога 3σ N4 на кейсе смены знака.

    Момент разворота берётся из мира, построенного ВНУТРИ того же сдвига с seed по
    умолчанию: `build_world(seed=7 + shift)` внутри `shifted(shift)` сдвинул бы
    seed дважды — ошибка, найденная повторным ревью PR #2.
    """
    bundle = BY_ID["flipped-feature-relation"]
    clean = BY_ID["clean-baseline"]
    caught, thresholds, whole, early, late, flipped_late = [], [], [], [], [], []
    where, share = [], []
    for shift in range(worlds):
        with shifted(shift):
            windows = _windows_with_labels(context_for(bundle))
            caught.append(
                any(
                    s.finding.value == "unstable_feature_relation"
                    for s in report_for(bundle).signals
                )
            )
            flip = _moment_at(world.build_world().main, 0.72)
            clean_windows = _windows_with_labels(context_for(clean))
        thresholds.append(np.median([SIGMA * _association_error(f[LABEL]) for _, f in windows]))
        flipped_late.append(_strength(windows[-1][1]))
        whole.append(_strength(pl.concat([f for _, f in clean_windows])))
        early.append(_strength(clean_windows[0][1]))
        late.append(_strength(clean_windows[-1][1]))
        spot = "до окон"
        for name, frame in windows:
            if frame["decided_at"].min() <= flip:
                spot = name if flip <= frame["decided_at"].max() else f"после {name}"
        where.append(spot)
        share.append(float((windows[-1][1]["decided_at"] >= flip).mean()))
    caught, share = np.array(caught), np.array(share)
    print(f"N4 поймала смену знака: {caught.sum()} из {worlds}")
    print(f"порог 3σ по окнам, медиана: {np.median(thresholds):.4f}")
    print(
        f"чистый мир (|association|), медианы: все окна вместе {np.median(whole):.4f}, "
        f"первое окно {np.median(early):.4f}, последнее {np.median(late):.4f}"
    )
    print(f"мир с разворотом, последнее окно: медиана {np.median(flipped_late):.4f}")
    print(
        f"отношение силы (чистый, все окна) к порогу: "
        f"{np.median(whole) / np.median(thresholds):.2f}"
    )
    for spot in sorted(set(where)):
        mask = np.array([w == spot for w in where])
        print(f"разворот {spot}: миров {mask.sum()}, поймано {caught[mask].sum()}")
    print(f"доля последнего окна после разворота: {share.min():.2f}–{share.max():.2f}")
    low = share < np.median(share)
    print(
        f"поймано при доле ниже медианы: {caught[low].mean():.2f} ({low.sum()} миров); "
        f"не ниже: {caught[~low].mean():.2f} ({(~low).sum()} миров)"
    )


def n18(worlds: int) -> None:
    """Размах помесячной заполненности события в `unobservability-tied-to-feature`."""
    bundle = BY_ID["unobservability-tied-to-feature"]
    drops = []
    for shift in range(worlds):
        with shifted(shift):
            ctx = context_for(bundle)
        mature = ctx.world.main.filter(pl.col("deadline_on") < pl.lit(ctx.observed_until))
        monthly = (
            mature.with_columns(pl.col("decided_at").dt.truncate("1mo").alias("month"))
            .group_by("month")
            .agg(pl.len().alias("rows"), pl.col("event_at").is_not_null().mean().alias("filled"))
            .filter(pl.col("rows") >= 100)
        )
        drops.append(monthly["filled"].max() - monthly["filled"].min())
    d = np.array(drops)
    print(
        f"размах заполненности: медиана {np.median(d):.5f}; "
        f"не меньше 0.25 в {(d >= 0.25).sum()} из {worlds}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("sweep").add_argument("shifts", nargs="+", type=int)
    r = sub.add_parser("rates")
    r.add_argument("--worlds", type=int, default=100)
    r.add_argument("ids", nargs="+", choices=sorted(BY_ID))
    for name in ("n4", "n18"):
        sub.add_parser(name).add_argument("--worlds", type=int, default=100)
    args = parser.parse_args()

    known_answer()
    if args.command == "sweep":
        sweep(args.shifts)
    elif args.command == "rates":
        rates(args.ids, args.worlds)
    elif args.command == "n4":
        n4(args.worlds)
    else:
        n18(args.worlds)


if __name__ == "__main__":
    main()
