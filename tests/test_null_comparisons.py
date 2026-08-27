"""Сравнения, у которых пустое значение даёт пустоту вместо лжи.

Класс повторялся трижды за десять кейсов, и каждый раз чинился экземпляр:
пятый кейс — `== None` в проверке A11; седьмой и восьмой — отрицание сравнения
дат в сборке. Правило баланса строк ловило второй случай ПОСЛЕ запуска, в ядре
не ловило ничто.

Здесь проверяется сам детектор: он обязан молчать на верном коде и возражать
на каждом из трёх исторических случаев. Проверка, кричащая на законном, хуже
её отсутствия; проверка, молчащая всегда, — обряд.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dsx.quality import find_null_comparisons

REPO_ROOT = Path(__file__).resolve().parent.parent


def _written(tmp_path: Path, code: str) -> Path:
    source = tmp_path / "sample.py"
    source.write_text(code, encoding="utf-8")
    return source


def test_repository_is_clean() -> None:
    """Весь код репозитория обязан проходить без единой находки."""
    findings = find_null_comparisons(
        REPO_ROOT / "src" / "dsx", REPO_ROOT / "projects", REPO_ROOT / "tests"
    )

    assert not findings, "\n".join(str(f) for f in findings)


def test_catches_comparison_with_none(tmp_path: Path) -> None:
    """Пятый кейс: `pl.col(status) == None` даёт null, а не истину.

    Доля считалась по пустой выборке и выходила нулевой, после чего порог
    «больше нуля в полтора раза» выполнялся всегда. Проверка сообщала
    невозможное «доля статуса в данных 0%» при настоящей 21%.
    """
    source = _written(tmp_path, 'share = frame.filter(pl.col("status") == None).height\n')

    findings = find_null_comparisons(source)

    assert len(findings) == 1
    assert "None" in findings[0].kind


def test_catches_negated_comparison_through_a_variable(tmp_path: Path) -> None:
    """Седьмой и восьмой кейсы: отрицание сравнения выбрасывает пустые молча.

    737 715 потерянных строк вместо объявленных 3 994. Поймало правило баланса,
    и только потому, что оно было написано.
    """
    source = _written(
        tmp_path,
        'broken = pl.col("a").is_null() | (pl.col("b") < pl.col("c"))\n'
        "kept = frame.filter(~broken)\n",
    )

    findings = find_null_comparisons(source)

    assert len(findings) == 1
    assert "fill_null" in findings[0].kind


def test_catches_negated_comparison_written_inline(tmp_path: Path) -> None:
    source = _written(tmp_path, 'kept = frame.filter(~(pl.col("b") < pl.col("c")))\n')

    findings = find_null_comparisons(source)

    assert len(findings) == 1


@pytest.mark.parametrize(
    "code",
    [
        pytest.param(
            'broken = (pl.col("b") < pl.col("c")).fill_null(False)\nkept = frame.filter(~broken)\n',
            id="пустое названо через fill_null",
        ),
        pytest.param(
            'broken = pl.col("b").is_null() | (pl.col("b") <= 0)\nkept = frame.filter(~broken)\n',
            id="пустое отсеяно соседним is_null",
        ),
        pytest.param(
            'kept = frame.filter(pl.col("b") < pl.col("c"))\n',
            id="сравнение без отрицания: пустое просто не пройдёт",
        ),
        pytest.param(
            "if value is None:\n    pass\n",
            id="обычная сверка с None вне polars",
        ),
    ],
)
def test_stays_silent_on_correct_code(tmp_path: Path, code: str) -> None:
    """Отрицательные контроли: проверка, кричащая на законном, запрещена."""
    assert find_null_comparisons(_written(tmp_path, code)) == []
