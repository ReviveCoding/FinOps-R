from __future__ import annotations

import importlib.util
import signal
import subprocess
from pathlib import Path
from types import ModuleType


def _load_api_smoke() -> ModuleType:
    path = Path(__file__).parents[1] / "scripts" / "api_smoke.py"
    spec = importlib.util.spec_from_file_location("finops_r_api_smoke", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Process:
    def __init__(self) -> None:
        self.signals: list[object] = []

    def send_signal(self, value: object) -> None:
        self.signals.append(value)


def test_api_smoke_uses_sigint_on_posix(monkeypatch) -> None:
    module = _load_api_smoke()
    monkeypatch.setattr(module.os, "name", "posix")
    process = _Process()

    assert module._process_group_creation_flags() == 0
    module._request_graceful_shutdown(process)

    assert process.signals == [signal.SIGINT]


def test_api_smoke_uses_new_process_group_and_ctrl_break_on_windows(monkeypatch) -> None:
    module = _load_api_smoke()
    monkeypatch.setattr(module.os, "name", "nt")
    monkeypatch.setattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 512, raising=False)
    monkeypatch.setattr(signal, "CTRL_BREAK_EVENT", 1, raising=False)
    process = _Process()

    assert module._process_group_creation_flags() == 512
    module._request_graceful_shutdown(process)

    assert process.signals == [1]
