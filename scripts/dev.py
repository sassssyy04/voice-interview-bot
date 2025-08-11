import os
import sys
import time
import webbrowser
import subprocess

HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8000"))
URL = f"http://{HOST}:{PORT}/"


def run_server() -> subprocess.Popen:
    """Run uvicorn dev server with reload.
    Returns:
        subprocess.Popen: running server process
    """
    env = os.environ.copy()
    # Prefer poetry's python env already active via `poetry run`
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "app.main:app",
        "--host",
        HOST,
        "--port",
        str(PORT),
        "--reload",
    ]
    return subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def main() -> None:
    proc = run_server()
    # Give server a moment to start
    time.sleep(1.5)
    try:
        webbrowser.open(URL)
    except Exception:
        pass

    # Stream logs to stdout; exit with server
    try:
        if proc.stdout is not None:
            for line in iter(proc.stdout.readline, b""):
                if not line:
                    break
                try:
                    sys.stdout.write(line.decode(errors="ignore"))
                except Exception:
                    pass
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except Exception:
            proc.kill()


if __name__ == "__main__":
    main() 