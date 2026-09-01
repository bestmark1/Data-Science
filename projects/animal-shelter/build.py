"""Сборка единицы решения пятнадцатого кейса: пребывание в приюте.

Единица решения — НЕ животное. Животное поступает в приют много раз: из 116 697
приёмов 17.4% приходится на животных, поступавших повторно. Единица —
пребывание, то есть пара «приём и первый следующий за ним исход».

Отсюда соединение по времени, а не по ключу. `animal_id` не уникален ни слева,
ни справа, и обычное соединение по нему дало бы каждому приёму все исходы этого
животного — включая исходы прошлых пребываний, то есть прошлое, выданное за
будущее, и будущее, выданное за настоящее.

Направление объявляется явно: FORWARD. Исход пребывания лежит ПОСЛЕ приёма, и
обратное направление подставило бы исход предыдущего пребывания, не изменив
числа строк.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from dsx.join import AsofDirection, guarded_asof_join

PROJECT = Path(__file__).resolve().parent
RAW = PROJECT / "data" / "raw"
INTAKES = RAW / "intakes_2016_2023.csv"
OUTCOMES = RAW / "outcomes_2016_2024.csv"

FORMAT = "%Y-%m-%dT%H:%M:%S%.f"

TOLERANCE = 7
"""Допуск, выбранный правилом `choose_tolerance` на объявленном порядке
кандидатов. Три суток дали 5.0% — вырожденно; семь дали 18.8%."""

TOLERANCES = (3, 7, 14, 30)
"""Сроки-кандидаты в сутках. Срока в данных нет, и допуск выбирается правилом:
берётся первый, дающий невырожденную долю. Порядок объявлен до просмотра долей.
"""


def _read(path: Path, columns: tuple[str, ...]) -> pl.DataFrame:
    frame = pl.read_csv(path, infer_schema_length=0)
    missing = [name for name in columns if name not in frame.columns]
    if missing:
        raise AssertionError(f"в {path.name} нет объявленных колонок: {missing}")
    return frame.select(columns).with_columns(
        pl.col("datetime").str.to_datetime(FORMAT, strict=False)
    )


def build() -> pl.DataFrame:
    """Собрать таблицу пребываний."""
    intakes = _read(
        INTAKES,
        (
            "animal_id",
            "datetime",
            "found_location",
            "intake_type",
            "intake_condition",
            "animal_type",
            "sex_upon_intake",
            "age_upon_intake",
            "breed",
            "color",
        ),
    ).rename({"datetime": "intake_at"})
    outcomes = _read(
        OUTCOMES,
        (
            "animal_id",
            "datetime",
            "outcome_type",
            "outcome_subtype",
            "date_of_birth",
            "sex_upon_outcome",
            "age_upon_outcome",
        ),
    ).rename({"datetime": "outcome_at"})

    read = intakes.height
    # Порядок задан явно и полностью: соединение по времени требует
    # отсортированных таблиц, а `animal_id` ключом строки не является.
    intakes = intakes.sort(["intake_at", "animal_id"], maintain_order=True)
    outcomes = outcomes.sort(["outcome_at", "animal_id"], maintain_order=True)

    stays = guarded_asof_join(
        intakes,
        outcomes,
        left_on="intake_at",
        right_on="outcome_at",
        by=["animal_id"],
        direction=AsofDirection.FORWARD,
    )

    # КОНТРОЛЬ К-2: строки, где исход датирован раньше приёма, намеренно
    # оставлены в первом прогоне.
    # КОНТРОЛЬ К-3: отсутствие, записанное значением, намеренно не вычищено.

    broken = pl.col("animal_id").is_null() | pl.col("intake_at").is_null()
    kept = stays.filter(~broken)
    lost = read - kept.height
    print(f"  приёмов прочитано {read:,}, потеряно {lost:,}, пребываний {kept.height:,}")

    return kept.with_columns(
        (pl.col("outcome_at") - pl.col("intake_at")).dt.total_days().alias("stay_days"),
        # Ключ пребывания собирается, а не берётся: в источнике его нет.
        # `animal_id` есть ключ ЖИВОТНОГО, и объявить его ключом строки значило
        # бы объявить единицей решения объект, а не пребывание.
        (pl.col("animal_id") + "@" + pl.col("intake_at").dt.strftime("%Y%m%dT%H%M%S")).alias(
            "stay_id"
        ),
        # Срок — вычисленный горизонт от приёма, а не величина из данных:
        # колонки со сроком в источнике нет. Значение выбрано правилом
        # `choose_tolerance` и подставлено сюда.
        (pl.col("intake_at") + pl.duration(days=TOLERANCE)).alias("due_at"),
        # Состояние пребывания. Нужно, чтобы отличить «событие не наступило» от
        # «наступило, но вид не записан»: 28 пребываний имеют момент выбытия и
        # пустой вид, и молча свести их с ещё не завершёнными значило бы
        # объявить наблюдением то, чего не наблюдали.
        pl.when(pl.col("outcome_at").is_null())
        .then(pl.lit("в приюте"))
        .when(pl.col("outcome_type").is_null())
        .then(pl.lit("вид не записан"))
        .otherwise(pl.lit("выбыло"))
        .alias("stay_status"),
    )


def choose_tolerance(stays: pl.DataFrame) -> tuple[int, float]:
    """Первый допуск, дающий невырожденную долю усыновлений.

    Правило объявлено до просмотра долей: перебираются сроки в объявленном
    порядке, берётся первый попавший в [10%, 90%]. Подбор срока под желаемую
    долю был бы выбором порога по результату.
    """
    observed = stays.filter(pl.col("outcome_type").is_not_null())
    for days in TOLERANCES:
        adopted = ((pl.col("outcome_type") == "Adoption") & (pl.col("stay_days") <= days)).mean()
        share = float(observed.select(adopted).item())
        print(f"  допуск {days:>2} сут: усыновлено к сроку {share:.1%}")
        if 0.10 <= share <= 0.90:
            return days, share
    raise AssertionError("ни один объявленный допуск не дал невырожденной доли")


if __name__ == "__main__":
    frame = build()
    days, share = choose_tolerance(frame)
    print(f"выбран допуск {days} суток, доля {share:.1%}")
    print(f"строк: {frame.height:,}, колонок: {len(frame.columns)}")
