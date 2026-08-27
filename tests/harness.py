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


def task_for(world, lifetime=None, target_kind=None, declared_process=None) -> TaskSpec:
    """Объявить предпосылки так, как они есть в данных.

    Объявлять их наугад — ровно та ошибка, которую ловит проверка S6: стенд
    сам объявлял наличие процесса там, где статус принимает одно значение.
    """
    from dsx.premises import _process_observed, _stream_observed

    has_process, _ = _process_observed(world)
    # Объявление автора кейса главнее наблюдения: без этого расхождение
    # объявленного с данными на стенде не воспроизвести, и проверять S6 нечем.
    if declared_process is not None:
        has_process = declared_process
    is_stream, _ = _stream_observed(world)
    return TaskSpec(
        target_kind=target_kind or TargetKind.BINARY,
        outcome_timing=OutcomeTiming.DELAYED,
        has_process=has_process,
        is_stream=is_stream,
        object_lifetime=lifetime,
    )


def context_for(bundle):
    """Стенд: снимок — конец НАБЛЮДЕНИЯ, окна отстоят от него.

    Снимок, равный последнему решению, гарантирует незрелость меток: они
    созревают позже. Окна, упирающиеся в конец наблюдения, гарантируют её тоже.
    """
    world = bundle.build()
    frame = world.main
    snapshot = max(frame["decided_at"].max(), frame["event_at"].max())
    lo, hi = frame["decided_at"].min(), frame["decided_at"].max()

    # Окна ставятся ДОЛЕЙ периода мира, а не числом дней от его начала.
    #
    # Прежде они стояли на 300-м дне жёстко, и это молча предполагало, что все
    # миры стенда длиной с базовый — 539 дней. Мир `clean-short-period` длится
    # 69 дней: все три окна оказывались ЗА его пределами, оценочные части были
    # пусты, резерв пуст. Отрицательный контроль, на котором нечему сработать,
    # не доказывает молчания проверок — он его имитирует.
    #
    # Нашла это проверка P9, введённая после одиннадцатого кейса; до неё
    # пустота была невидима, потому что пустоту никто не проверял.
    span = max((hi - lo).days, 4)
    # Шаг меньше ширины окна означает перекрытие: так воспроизводится дефект
    # протокола, до которого инжектор данных не дотягивается.
    step = 0.04 if bundle.overlapping_windows else 0.08
    windows = [
        Window(
            f"w{i}",
            lo + dt.timedelta(days=int(span * (0.55 + step * i))),
            lo + dt.timedelta(days=int(span * (0.63 + step * i))),
        )
        for i in range(3)
    ]
    reserve_from = lo + dt.timedelta(days=int(span * 0.88)) if bundle.reserve else None
    try:
        split = split_by_windows(world, bundle.outcome, windows, snapshot, reserve_from)
    except Exception:
        split = None
    return Context(
        world,
        bundle.outcome,
        task_for(world, bundle.lifetime, bundle.target_kind, bundle.declared_process),
        split,
    )


def report_for(bundle, checks=None):
    return run_checks(list(checks or ALL_CHECKS), context_for(bundle))
