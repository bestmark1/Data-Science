"""Отбор строк, чей порядок не определён.

Тринадцатый кейс потерял на этом предсказание P-8: точечные оценки совпадали,
а доверительные интервалы расходились между прогонами, потому что пересчёт
берёт НОМЕРА строк, и при другом порядке это другие строки.

Класс был известен с седьмого кейса, где порядок был не определён после
`group_by` в ядре. Там его закрыли; в коде проектов не закрыли, и он вернулся.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dsx.quality import find_unordered_selections

REPO_ROOT = Path(__file__).resolve().parent.parent


def _written(tmp_path: Path, code: str) -> Path:
    source = tmp_path / "sample.py"
    source.write_text(code, encoding="utf-8")
    return source


def test_repository_is_clean() -> None:
    """Весь код репозитория обязан проходить без единой находки."""
    findings = find_unordered_selections(
        REPO_ROOT / "src" / "dsx", REPO_ROOT / "projects", REPO_ROOT / "tests"
    )

    assert not findings, "\n".join(str(f) for f in findings)


def test_catches_row_deduplication_without_order(tmp_path: Path) -> None:
    """Тринадцатый кейс: сведение к одной строке на заявку без сохранения порядка."""
    source = _written(tmp_path, 'frame = source.unique(subset=["permit_number"], keep="first")\n')

    assert len(find_unordered_selections(source)) == 1


@pytest.mark.parametrize(
    "code",
    [
        pytest.param(
            'frame = source.unique(subset=["id"], keep="first", maintain_order=True)\n',
            id="порядок объявлен",
        ),
        pytest.param(
            'names = sorted(table["sector"].unique().to_list())\n',
            id="перечень значений, а не отбор строк",
        ),
        pytest.param(
            "count = frame.select(keys).unique().height\n",
            id="счёт различных, порядок безразличен",
        ),
        pytest.param(
            "values = np.unique(scores, return_counts=True)\n",
            id="unique из numpy — другая функция",
        ),
    ],
)
def test_stays_silent_on_safe_forms(tmp_path: Path, code: str) -> None:
    """Отрицательные контроли: проверка, кричащая на законном, запрещена.

    Первая версия ловила всякий `unique` и дала двадцать девять находок, ни
    одна из которых не была дефектом. Сужена до принятия.
    """
    assert find_unordered_selections(_written(tmp_path, code)) == []
