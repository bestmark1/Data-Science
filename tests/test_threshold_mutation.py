"""Прерванный прогон мутации не оставляет ослабленную проверку в дереве.

Инструмент правит исходник НА ДИСКЕ, а восстанавливал его только `finally`
работающего процесса. Этого хватало ровно до первого убийства извне: прогон,
снятый по таймауту, оставил в рабочем дереве мутацию `A11.min_missing` 30→15 —
ослабленную проверку, которую никто не вносил.

Вред не в самой мутации, а в её незаметности. Она молча уехала бы в следующий
коммит, и ядро стало бы возражать реже, чем объявляет.

`finally` не исполняется ни при SIGKILL, ни при SIGTERM по умолчанию.
Восстанавливать после смерти процесса некому — если не осталось следа на диске.
Проверяется здесь именно след: убийство настоящим сигналом, а не имитация.
"""

from __future__ import annotations

import importlib.util
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "threshold_mutation.py"


def _tool():
    spec = importlib.util.spec_from_file_location("threshold_mutation", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _repo(tmp_path: Path) -> Path:
    """Крошечный репозиторий с одним закоммиченным порогом."""
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "thresholds.py").write_text("floor = 0.25\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init"],
        cwd=tmp_path,
        check=True,
    )
    return tmp_path


def test_a_killed_run_is_recovered_by_the_marker(tmp_path, monkeypatch) -> None:
    """След на диске переживает смерть процесса — в этом весь его смысл."""
    repo = _repo(tmp_path)
    module = _tool()
    target = repo / "thresholds.py"

    # Так выглядит дерево после убитого прогона: исходник мутирован, след цел.
    target.write_text("floor = 0.9\n", encoding="utf-8")
    (repo / module.MARKER.name).write_text("thresholds.py", encoding="utf-8")

    monkeypatch.chdir(repo)
    recovered = module._recover()

    assert recovered == "thresholds.py"
    assert target.read_text(encoding="utf-8") == "floor = 0.25\n"
    assert not (repo / module.MARKER.name).exists(), "след обязан сниматься после возврата"


def test_recovery_is_silent_when_there_was_nothing_to_recover(tmp_path, monkeypatch) -> None:
    """Отсутствие следа означает, что прошлый прогон закончился сам."""
    monkeypatch.chdir(_repo(tmp_path))

    assert _tool()._recover() is None


def test_a_marker_left_before_the_edit_is_harmless(tmp_path, monkeypatch) -> None:
    """След пишется ДО правки, и по нему может достаться нетронутый файл.

    Возврат нетронутого файла ничего не портит, и это дешевле, чем правка,
    оставшаяся без следа.
    """
    repo = _repo(tmp_path)
    module = _tool()
    (repo / module.MARKER.name).write_text("thresholds.py", encoding="utf-8")

    monkeypatch.chdir(repo)
    module._recover()

    assert (repo / "thresholds.py").read_text(encoding="utf-8") == "floor = 0.25\n"


def test_sigterm_restores_the_source_before_dying(tmp_path) -> None:
    """Убийство НАСТОЯЩИМ сигналом, а не его имитацией.

    Обработчик возвращает исходник немедленно. След спасает и без него, но
    только к следующему прогону, а закоммитить чужую мутацию можно раньше.
    """
    repo = _repo(tmp_path)
    module = _tool()
    target = repo / "thresholds.py"

    script = repo / "run.py"
    script.write_text(
        "import signal, sys, time\n"
        # Инструмент кладёт в путь относительный 'src' и потому рассчитан на
        # запуск из корня проекта. Здесь он запускается из чужого каталога, и
        # путь передаётся явно.
        f"sys.path.insert(0, {str(ROOT / 'src')!r})\n"
        "import importlib.util\n"
        f"spec = importlib.util.spec_from_file_location('tm', {str(TOOL)!r})\n"
        "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n"
        "signal.signal(signal.SIGTERM, m._on_signal)\n"
        "m._mutate(__import__('pathlib').Path('thresholds.py'), 1, 0.25, 0.9)\n"
        "print('мутировано', flush=True)\n"
        "time.sleep(30)\n",
        encoding="utf-8",
    )
    process = subprocess.Popen(
        [sys.executable, "run.py"], cwd=repo, stdout=subprocess.PIPE, text=True
    )
    assert process.stdout.readline().strip() == "мутировано"
    assert target.read_text(encoding="utf-8") == "floor = 0.9\n", "мутация не легла на диск"

    os.kill(process.pid, signal.SIGTERM)
    process.wait(timeout=10)

    assert target.read_text(encoding="utf-8") == "floor = 0.25\n", "SIGTERM оставил мутацию"
    assert not (repo / module.MARKER.name).exists()


def test_sigkill_leaves_the_marker_for_the_next_run(tmp_path, monkeypatch) -> None:
    """SIGKILL перехватить нельзя — и именно он случился на деле.

    Здесь проверяется единственное, что тогда работает: след остаётся, и
    следующий запуск возвращает по нему исходник.
    """
    repo = _repo(tmp_path)
    module = _tool()
    target = repo / "thresholds.py"

    script = repo / "run.py"
    script.write_text(
        "import sys, time, importlib.util, pathlib\n"
        f"sys.path.insert(0, {str(ROOT / 'src')!r})\n"
        f"spec = importlib.util.spec_from_file_location('tm', {str(TOOL)!r})\n"
        "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n"
        "m._mutate(pathlib.Path('thresholds.py'), 1, 0.25, 0.9)\n"
        "print('мутировано', flush=True)\n"
        "time.sleep(30)\n",
        encoding="utf-8",
    )
    process = subprocess.Popen(
        [sys.executable, "run.py"], cwd=repo, stdout=subprocess.PIPE, text=True
    )
    assert process.stdout.readline().strip() == "мутировано"

    os.kill(process.pid, signal.SIGKILL)
    process.wait(timeout=10)
    time.sleep(0.1)

    assert target.read_text(encoding="utf-8") == "floor = 0.9\n", "SIGKILL не должен ничего чинить"
    assert (repo / module.MARKER.name).exists(), "без следа восстанавливать будет нечем"

    monkeypatch.chdir(repo)
    assert module._recover() == "thresholds.py"
    assert target.read_text(encoding="utf-8") == "floor = 0.25\n"
