import datetime as dt

from dsx.checks import ALL_CHECKS, Context, run_checks
from dsx.split import Window, split_by_windows
from dsx.task import OutcomeTiming, TargetKind, TaskSpec

TASK = TaskSpec(
    target_kind=TargetKind.BINARY,
    outcome_timing=OutcomeTiming.DELAYED,
    has_process=True,
    is_stream=True,
)


def context_for(bundle):
    """Стенд: снимок — конец НАБЛЮДЕНИЯ, окна отстоят от него.

    Снимок, равный последнему решению, гарантирует незрелость меток: они
    созревают позже. Окна, упирающиеся в конец наблюдения, гарантируют её тоже.
    """
    world = bundle.build()
    frame = world.main
    snapshot = max(frame["decided_at"].max(), frame["event_at"].max())
    lo = frame["decided_at"].min()
    windows = [
        Window(f"w{i}", lo + dt.timedelta(days=300 + i * 45), lo + dt.timedelta(days=345 + i * 45))
        for i in range(3)
    ]
    try:
        split = split_by_windows(world, bundle.outcome, windows, snapshot)
    except Exception:
        split = None
    return Context(world, bundle.outcome, TASK, split)


def report_for(bundle, checks=None):
    return run_checks(list(checks or ALL_CHECKS), context_for(bundle))
