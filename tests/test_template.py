"""Шаблон формы не должен отставать от ядра.

Форма росла девять кейсов, а шаблон и руководство обновлялись не всегда.
К девятому кейсу в шаблоне не хватало семи полей, включая три обязательных:
срез окна, момент фиксации значения и порог невырожденности. Заполнявший по
такому шаблону получал отказ прогона и лез в исходники — то есть рабочий путь
был недостижим ровно в том месте, где проект девять кейсов искал недостижимое.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from dsx.project import (
    AssumptionForm,
    CauseForm,
    ColumnForm,
    OutcomeForm,
    ProjectForm,
    SplitForm,
    TaskForm,
)

TEMPLATE = Path(__file__).resolve().parent.parent / "templates" / "project.yaml"
GUIDE = Path(__file__).resolve().parent.parent / "docs" / "filling-the-form.md"

FORMS = {
    "колонка": ColumnForm,
    "исход": OutcomeForm,
    "задача": TaskForm,
    "проект": ProjectForm,
    "причина": CauseForm,
    "сплит": SplitForm,
    "допущение": AssumptionForm,
}


def test_template_is_a_valid_form() -> None:
    """Шаблон обязан разбираться формой, а не только выглядеть как она."""
    form = ProjectForm.model_validate(yaml.safe_load(TEMPLATE.read_text(encoding="utf-8")))

    assert form.columns, "шаблон без колонок не образец"
    assert form.assumptions, "проект без записанного допущения принимает их молча"


@pytest.mark.parametrize("label", sorted(FORMS))
def test_every_field_is_named_in_the_template(label: str) -> None:
    """Поле, которого нет в шаблоне, заполнявший не найдёт."""
    text = TEMPLATE.read_text(encoding="utf-8")
    missing = sorted(name for name in FORMS[label].model_fields if name not in text)

    assert not missing, f"нет в шаблоне: {missing}"


def test_mandatory_declarations_are_explained_in_the_guide() -> None:
    """Объявление, которое ядро ТРЕБУЕТ, обязано быть объяснено словами.

    Проверяются только обязательные: заполнявший упрётся в отказ прогона именно
    на них, и отсылать его в исходники значило бы признать руководство
    украшением.
    """
    text = GUIDE.read_text(encoding="utf-8")
    mandatory = ("window_clock", "value_as_of", "degenerate_beyond")
    missing = [name for name in mandatory if name not in text]

    assert not missing, f"не объяснено в руководстве: {missing}"
