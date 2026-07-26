from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any


def _free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _process_group_creation_flags() -> int:
    if os.name != "nt":
        return 0
    return int(getattr(subprocess, "CREATE_NEW_PROCESS_GROUP"))


def _request_graceful_shutdown(process: subprocess.Popen[bytes]) -> None:
    if os.name == "nt":
        process.send_signal(getattr(signal, "CTRL_BREAK_EVENT"))
    else:
        process.send_signal(signal.SIGINT)


def _request(
    port: int,
    path: str,
    payload: dict[str, Any] | bytes | None = None,
    content_type: str = "application/json",
) -> tuple[int, dict[str, Any]]:
    data: bytes | None
    if isinstance(payload, dict):
        data = json.dumps(payload).encode("utf-8")
    else:
        data = payload
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=data,
        headers={"Content-Type": content_type} if data is not None else {},
        method="POST" if data is not None else "GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def _wait_ready(port: int, process: subprocess.Popen[bytes]) -> dict[str, Any]:
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"API exited before readiness with code {process.returncode}")
        try:
            status, body = _request(port, "/health")
            if status == 200:
                return body
        except (OSError, ValueError):
            pass
        time.sleep(0.1)
    raise TimeoutError("API did not become ready within 20 seconds")


def _cycle(run_dir: Path, fixture: dict[str, Any], cycle: int) -> dict[str, Any]:
    port = _free_port()
    command = [
        sys.executable,
        "-m",
        "finops_r.cli",
        "serve",
        "--run-dir",
        str(run_dir),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    started = time.perf_counter()
    process = subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=_process_group_creation_flags(),
    )
    try:
        health = _wait_ready(port, process)
        startup_seconds = time.perf_counter() - started
        status, assessment = _request(port, "/v1/cases/assess", fixture)
        if status != 200 or assessment.get("case_id") != fixture["case_id"]:
            raise AssertionError("valid assessment request failed")

        missing = dict(fixture)
        missing.pop("case_id")
        unknown = dict(fixture)
        unknown["unexpected"] = True
        checks = {
            "missing_field": _request(port, "/v1/cases/assess", missing)[0],
            "unknown_field": _request(port, "/v1/cases/assess", unknown)[0],
            "invalid_content_type": _request(port, "/v1/cases/assess", b"not-json", "text/plain")[
                0
            ],
            "nonexistent_resource": _request(port, "/missing")[0],
        }
        if checks != {
            "missing_field": 422,
            "unknown_field": 422,
            "invalid_content_type": 400,
            "nonexistent_resource": 404,
        }:
            raise AssertionError(f"unexpected API error contract: {checks}")

        with ThreadPoolExecutor(max_workers=8) as pool:
            statuses = list(pool.map(lambda _: _request(port, "/health")[0], range(32)))
        if set(statuses) != {200}:
            raise AssertionError(f"concurrent health checks failed: {statuses}")

        conflicting = subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=20,
            check=False,
        )
        if conflicting.returncode == 0:
            raise AssertionError("a second service unexpectedly acquired an occupied port")
        return {
            "cycle": cycle,
            "startup_seconds": startup_seconds,
            "health": health,
            "assessment_status": status,
            "error_contract": checks,
            "concurrent_requests": len(statuses),
            "concurrent_errors": sum(value != 200 for value in statuses),
            "port_conflict_exit_code": conflicting.returncode,
        }
    finally:
        shutdown_requested = process.poll() is None
        if shutdown_requested:
            _request_graceful_shutdown(process)
        try:
            return_code = process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
            raise RuntimeError("API failed graceful shutdown")
        expected_windows_ctrl_break_exit = (
            os.name == "nt" and shutdown_requested and return_code == 3
        )
        if return_code != 0 and not expected_windows_ctrl_break_exit:
            raise RuntimeError(f"API exited with code {return_code}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Exercise the installed FinOps-R API over TCP")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--cycles", type=int, default=3)
    args = parser.parse_args()
    if args.cycles < 1:
        parser.error("--cycles must be positive")
    fixture = json.loads(args.fixture.read_text(encoding="utf-8"))
    results = [_cycle(args.run_dir.resolve(), fixture, cycle + 1) for cycle in range(args.cycles)]
    print(json.dumps({"status": "PASS", "cycles": results}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
