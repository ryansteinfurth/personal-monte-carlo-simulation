"""Launch the simulation in its own desktop window.

Starts Streamlit headless on a free local port, then points a native window
(WKWebView on macOS) at it. Closing the window shuts the server down.

Run with:  .venv/bin/python run.py
"""

import atexit
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import webview

APP = Path(__file__).parent / "app.py"
STARTUP_TIMEOUT = 30  # seconds to wait for the server before giving up


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start_server(port: int) -> subprocess.Popen:
    # app.py refuses to render without this, so the browser path stays closed.
    env = {**os.environ, "MONTE_CARLO_DESKTOP": "1"}
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(APP),
            "--server.port",
            str(port),
            "--server.address",
            "127.0.0.1",
            "--server.headless",
            "true",
        ],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def wait_until_ready(port: int, server: subprocess.Popen) -> None:
    health = f"http://127.0.0.1:{port}/_stcore/health"
    deadline = time.monotonic() + STARTUP_TIMEOUT
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError(f"Streamlit exited with code {server.returncode}")
        try:
            with urllib.request.urlopen(health, timeout=1) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            time.sleep(0.2)
    raise RuntimeError(f"Streamlit did not come up within {STARTUP_TIMEOUT}s")


def main() -> None:
    port = free_port()
    server = start_server(port)

    def shutdown() -> None:
        if server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server.kill()

    atexit.register(shutdown)

    try:
        wait_until_ready(port, server)
    except RuntimeError as exc:
        shutdown()
        sys.exit(f"Could not start the app: {exc}")

    webview.create_window(
        "Monte Carlo Simulation",
        f"http://127.0.0.1:{port}",
        width=1440,
        height=900,
        min_size=(1024, 700),
    )
    webview.start()  # blocks until the window is closed
    shutdown()


if __name__ == "__main__":
    main()
