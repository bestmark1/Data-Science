"""Соединения рабочего пути идут через объявленную грануляцию.

За второй кейс механизм `guarded_join` был обойдён трижды: я писал
`frame.join(...)` напрямую и замечал беду по числу строк, а не по отказу.
Механизм, который надо не забыть позвать, защищает настолько, насколько
хороша память зовущего, — поэтому его зов проверяется тестом.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dsx.quality import find_unguarded_joins

REPO_ROOT = Path(__file__).resolve().parent.parent

WORKING_PATH = sorted(
    path
    for path in (REPO_ROOT / "projects").rglob("*.py")
    if "steps" not in path.parts and "__pycache__" not in path.parts
)
"""Рабочий путь проектов: сборка таблицы решений и запуск по форме.

Папки `steps/` исключены намеренно. Это записи расследования — ручные прогоны,
которыми находки и были получены. Переписать их значило бы задним числом
изобразить, что работа шла иначе, чем шла.
"""


def test_working_path_has_no_unguarded_joins() -> None:
    findings = find_unguarded_joins(*WORKING_PATH)

    assert not findings, "\n".join(str(f) for f in findings)


def test_core_has_no_unguarded_joins() -> None:
    """Ядро тем более: guarded_join живёт в нём самом."""
    findings = [
        f for f in find_unguarded_joins(REPO_ROOT / "src" / "dsx") if f.path.name != "join.py"
    ]

    assert not findings, "\n".join(str(f) for f in findings)


def test_detector_finds_a_dataframe_join(tmp_path: Path) -> None:
    source = tmp_path / "sample.py"
    source.write_text("result = left.join(right, on='key', how='left')\n", encoding="utf-8")

    assert len(find_unguarded_joins(source)) == 1


def test_detector_ignores_string_join(tmp_path: Path) -> None:
    """', '.join(items) — не соединение таблиц, и тревога здесь была бы обрядом."""
    source = tmp_path / "sample.py"
    source.write_text("text = ', '.join(items)\n", encoding="utf-8")

    assert find_unguarded_joins(source) == []


def test_detector_ignores_guarded_join(tmp_path: Path) -> None:
    source = tmp_path / "sample.py"
    source.write_text(
        "result = guarded_join(left, right, on=['key'], expect=Cardinality.MANY_TO_ONE)\n",
        encoding="utf-8",
    )

    assert find_unguarded_joins(source) == []


def test_detector_survives_unparseable_files(tmp_path: Path) -> None:
    (tmp_path / "broken.py").write_text("def (\n", encoding="utf-8")

    assert find_unguarded_joins(tmp_path) == []


@pytest.mark.parametrize("name", ["run_a.py", "run_b.py", "build_b.py"])
def test_named_working_files_are_covered(name: str) -> None:
    """Файл, выпавший из выборки, проверяться перестанет незаметно."""
    assert any(path.name == name for path in WORKING_PATH)


# --- соединение по времени (join_asof) --------------------------------------


def _timeline() -> tuple:
    """Два участка, решения и события на них.

    У участка 'a' событие лежит ПОСЛЕ решения, у 'b' — ДО. Так одна и та же
    пара таблиц годится и для проверки направления вперёд, и назад.
    """
    import datetime as dt

    import polars as pl

    day = dt.datetime(2020, 1, 1)
    left = pl.DataFrame(
        {
            "segment": ["a", "b"],
            "at": [day + dt.timedelta(days=10), day + dt.timedelta(days=10)],
        }
    ).sort("at")
    right = pl.DataFrame(
        {
            "segment": ["a", "b"],
            "event_at": [day + dt.timedelta(days=20), day + dt.timedelta(days=1)],
        }
    ).sort("event_at")
    return left, right


def test_asof_join_preserves_the_unit_of_decision() -> None:
    from dsx.join import AsofDirection, guarded_asof_join

    left, right = _timeline()

    joined = guarded_asof_join(
        left,
        right,
        left_on="at",
        right_on="event_at",
        by=["segment"],
        direction=AsofDirection.FORWARD,
    )

    assert joined.height == left.height
    # Вперёд от решения лежит только событие участка 'a'.
    assert joined.filter(joined["event_at"].is_not_null())["segment"].to_list() == ["a"]


def test_asof_join_refuses_keys_with_the_same_name() -> None:
    """Совпавшие имена polars схлопывает, и сверять направление станет нечем."""
    import polars as pl

    from dsx.join import AsofDirection, JoinExpectationViolated, guarded_asof_join

    left, right = _timeline()
    right = right.rename({"event_at": "at"}).sort("at")

    with pytest.raises(JoinExpectationViolated, match="схлопнет"):
        guarded_asof_join(
            left,
            right,
            left_on="at",
            right_on="at",
            by=["segment"],
            direction=AsofDirection.FORWARD,
        )
    assert isinstance(left, pl.DataFrame)


def test_asof_join_refuses_an_unsorted_frame() -> None:
    """Polars не проверяет сортировку, когда задан `by`, и лишь предупреждает."""
    from dsx.join import AsofDirection, JoinExpectationViolated, guarded_asof_join

    left, right = _timeline()

    with pytest.raises(JoinExpectationViolated, match="не отсортирован"):
        guarded_asof_join(
            left,
            right.reverse(),
            left_on="at",
            right_on="event_at",
            by=["segment"],
            direction=AsofDirection.FORWARD,
        )


def test_asof_join_without_a_declared_direction_is_blocked() -> None:
    """Направление — то самое, чего число строк не покажет."""
    from dsx.join import Blocked, guarded_asof_join
    from dsx.policy import OverrideLedger

    left, right = _timeline()

    with pytest.raises(Blocked):
        guarded_asof_join(
            left,
            right,
            left_on="at",
            right_on="event_at",
            by=["segment"],
            ledger=OverrideLedger(),
        )


def test_asof_join_refuses_a_blank_left_timestamp() -> None:
    """Пустой момент решения дал бы «ничего не случилось» вместо «неизвестно»."""
    import polars as pl

    from dsx.join import AsofDirection, JoinExpectationViolated, guarded_asof_join

    left, right = _timeline()
    left = pl.concat([left, pl.DataFrame({"segment": ["c"], "at": [None]}, schema=left.schema)])

    with pytest.raises(JoinExpectationViolated, match="пуста"):
        guarded_asof_join(
            left,
            right,
            left_on="at",
            right_on="event_at",
            by=["segment"],
            direction=AsofDirection.FORWARD,
        )
