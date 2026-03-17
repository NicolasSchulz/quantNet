from __future__ import annotations

import os
import socket
import shutil
import signal
import subprocess
import sys
import time
import venv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
API_DIR = PROJECT_ROOT / "api"
FRONTEND_DIR = PROJECT_ROOT / "frontend"
REQUIRED_BACKEND_MODULES = ("fastapi", "uvicorn", "pydantic", "dotenv")
BACKEND_BOOTSTRAP_PACKAGES = ("fastapi", "uvicorn", "pydantic", "python-dotenv")


def _require_command(command: str) -> None:
    if shutil.which(command) is None:
        raise SystemExit(f"Missing required command: {command}")


def _python_candidates() -> list[str]:
    candidates = [
        sys.executable,
        str(PROJECT_ROOT / ".venv" / "bin" / "python"),
        str(PROJECT_ROOT / "venv" / "bin" / "python"),
        shutil.which("python3") or "",
        shutil.which("python") or "",
    ]
    unique: list[str] = []
    for candidate in candidates:
        if candidate and candidate not in unique and Path(candidate).exists():
            unique.append(candidate)
    return unique


def _has_modules(python_executable: str, modules: tuple[str, ...]) -> bool:
    code = (
        "import importlib.util as u, sys;"
        f"mods={modules!r};"
        "missing=[m for m in mods if u.find_spec(m) is None];"
        "sys.exit(0 if not missing else 1)"
    )
    check = subprocess.run(
        [python_executable, "-c", code],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=PROJECT_ROOT,
    )
    return check.returncode == 0


def _ensure_virtualenv(path: Path) -> str:
    python_executable = path / "bin" / "python"
    if python_executable.exists():
        return str(python_executable)
    print(f"[setup] Creating virtualenv at {path}")
    venv.create(path, with_pip=True)
    return str(python_executable)


def _bootstrap_backend_python() -> str:
    preferred_candidates = [
        sys.executable,
        str(PROJECT_ROOT / "venv" / "bin" / "python"),
        str(PROJECT_ROOT / ".venv" / "bin" / "python"),
    ]
    target = ""
    for candidate in preferred_candidates:
        if candidate and Path(candidate).exists():
            target = candidate
            break
    if not target:
        target = _ensure_virtualenv(PROJECT_ROOT / "venv")

    print(f"[setup] Installing backend dependencies into {target}")
    install_cmd = [target, "-m", "pip", "install", *BACKEND_BOOTSTRAP_PACKAGES]
    result = subprocess.run(install_cmd, cwd=PROJECT_ROOT)
    if result.returncode != 0 or not _has_modules(target, REQUIRED_BACKEND_MODULES):
        raise SystemExit(
            "Backend dependency bootstrap failed. Run "
            f"`{target} -m pip install {' '.join(BACKEND_BOOTSTRAP_PACKAGES)}` manually."
        )
    return target


def _pick_backend_python() -> str:
    for candidate in _python_candidates():
        if _has_modules(candidate, REQUIRED_BACKEND_MODULES):
            return candidate
    return _bootstrap_backend_python()


def _start_process(name: str, cmd: list[str], cwd: Path, env: dict[str, str]) -> subprocess.Popen[bytes]:
    print(f"[start] {name}: {' '.join(cmd)}")
    return subprocess.Popen(cmd, cwd=cwd, env=env)


def _port_available(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def _pick_port(host: str, preferred: int, fallback_count: int = 20) -> int:
    for offset in range(fallback_count):
        port = preferred + offset
        if _port_available(host, port):
            return port
    raise SystemExit(f"No free port found starting from {preferred} on {host}")


def _write_frontend_runtime_env(host: str, backend_port: int) -> None:
    env_local = FRONTEND_DIR / ".env.local"
    env_local.write_text(f"VITE_API_URL=http://{host}:{backend_port}\n", encoding="utf-8")


def main() -> None:
    _require_command("npm")
    backend_python = _pick_backend_python()

    env = os.environ.copy()
    env.setdefault("LIVE_MODE", "false")
    env.setdefault("CORS_ORIGIN", "http://localhost:5173")
    env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    host = env.get("APP_HOST", "127.0.0.1")
    backend_port = _pick_port(host, int(env.get("BACKEND_PORT", "8000")))
    frontend_port = _pick_port(host, int(env.get("FRONTEND_PORT", "5173")))
    env["CORS_ORIGIN"] = f"http://{host}:{frontend_port}"
    _write_frontend_runtime_env(host, backend_port)

    backend_cmd = [backend_python, "-m", "uvicorn", "main:app", "--host", host, "--port", str(backend_port)]
    frontend_cmd = ["npm", "run", "dev", "--workspace", "frontend", "--", "--host", host, "--port", str(frontend_port)]

    processes: list[subprocess.Popen[bytes]] = []

    def shutdown(*_args: object) -> None:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        deadline = time.time() + 5
        for process in processes:
            while process.poll() is None and time.time() < deadline:
                time.sleep(0.1)
            if process.poll() is None:
                process.kill()
        raise SystemExit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    try:
        processes.append(_start_process("backend", backend_cmd, API_DIR, env))
        processes.append(_start_process("frontend", frontend_cmd, PROJECT_ROOT, env))
        print(f"[ready] Frontend: http://{host}:{frontend_port}")
        print(f"[ready] Backend:  http://{host}:{backend_port}")
        print("[stop] Press Ctrl+C to stop both processes.")

        while True:
            for process in processes:
                code = process.poll()
                if code is not None:
                    print(f"[exit] Process ended with code {code}")
                    shutdown()
            time.sleep(0.5)
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()


if __name__ == "__main__":
    main()
