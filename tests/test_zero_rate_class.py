"""Нулевая доля — крайний случай, а не повод к раннему возврату.

Класс повторялся дважды за три кейса:

* седьмой кейс, N3 — при доле ровно 100% разброс равен нулю, и строгое
  `0 < 0` ложно: проверка сообщала о расхождении между 100% и 100%;
* десятый кейс, N17 — `if low["rate"] <= 0: return []` заставляло проверку
  молчать на расхождении бесконечной кратности.

Оба раза чинился экземпляр. Здесь проверяется класс: каждая проверка,
сравнивающая доли, обязана считать ноль крайним значением, а не отсутствием
данных.
"""

from __future__ import annotations

import pytest

from dsx.evals import injectors as inj
from dsx.evals.registry import BY_ID, Bundle
from dsx.evals.world import build_world
from harness import context_for


def test_zero_rate_for_one_reason_is_an_infinite_divergence() -> None:
    """N17: у одного повода исход не наступает ни разу, у другого наступает.

    Отношение долей бесконечно, и молчать тут нельзя. Первая версия проверки
    возвращалась рано именно на этом.
    """
    from dsx.checks.drift import ReasonForObservationTransfers

    base = BY_ID["outcome-depends-on-the-reason"]
    bundle = Bundle(
        case=base.case,
        build=lambda: inj.outcome_depends_on_the_reason(build_world(), shift_days=8),
        outcome=base.outcome,
    )
    context = context_for(bundle)

    signals = ReasonForObservationTransfers().run(context)

    assert signals, "нулевая доля у одного повода — крайнее расхождение, а не тишина"
    assert "0.0%" in signals[0].detail


def test_identical_extreme_rates_are_not_a_divergence() -> None:
    """N3: доля 100% с обеих сторон — это совпадение, а не расхождение.

    Обратная ошибка того же класса: при доле ровно единица разброс равен нулю,
    и строгое сравнение с шумовым порогом объявляло различие там, где его нет.
    """
    from dsx.checks.drift import TargetRateStationarity

    base = BY_ID["clean-baseline"]
    bundle = Bundle(
        case=base.case,
        build=lambda: inj.degenerate_outcome(build_world()),
        outcome=base.outcome,
    )

    assert TargetRateStationarity().run(context_for(bundle)) == []


def test_degenerate_check_sees_a_zero_rate() -> None:
    """N15: доля ноль при объявленном пороге — вырожденность, а не «нет данных»."""
    from dsx.checks.drift import NonDegenerateOutcome

    base = BY_ID["clean-baseline"]
    outcome = base.outcome.model_copy(update={"degenerate_beyond": 0.30})
    bundle = Bundle(case=base.case, build=build_world, outcome=outcome)
    context = context_for(bundle)

    signals = NonDegenerateOutcome().run(context)

    assert signals, "доля 14% при пороге 30% обязана быть названа вырожденной"


@pytest.mark.parametrize("rate", [0.0, 1.0])
def test_rate_error_is_zero_at_the_extremes(rate: float) -> None:
    """Основание класса: у вырожденной доли разброс равен нулю.

    Отсюда и берутся обе ошибки. Строгое сравнение с нулевым порогом даёт
    ложь при нулевой разнице, нестрогое — истину, и разница между ними решает,
    сообщит проверка о несуществующем расхождении или промолчит о настоящем.
    """
    from dsx.checks.drift import _rate_error

    assert _rate_error(rate, 1000) == 0.0


def test_a_single_positive_row_still_has_spread() -> None:
    """А у почти вырожденной — уже нет: порог обязан работать непрерывно."""
    from dsx.checks.drift import _rate_error

    assert _rate_error(0.001, 1000) > 0.0
