"""Find a working PyQt6 Python while preserving the PyTorch interpreter."""
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    if importlib.util.find_spec("torch"):
        os.environ.setdefault("VTW_TORCH_PYTHON", sys.executable)
    candidates = [sys.executable, shutil.which("python3"), "/usr/bin/python3"]
    for executable in dict.fromkeys(candidates):
        if not executable or not Path(executable).is_file():
            continue
        try:
            result = subprocess.run(
                [executable, "-c", "from PyQt6.QtWebEngineWidgets import QWebEngineView"],
                capture_output=True, timeout=10,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        if result.returncode == 0:
            # Keep the caller's directory so relative workspace paths still work.
            app_path = str(Path(__file__).resolve().parent)
            existing_path = os.environ.get("PYTHONPATH")
            os.environ["PYTHONPATH"] = app_path + (os.pathsep + existing_path if existing_path else "")
            os.execv(executable, [executable, "-m", "app.desktop", *sys.argv[1:]])
    raise SystemExit("Install PyQt6 and PyQt6-WebEngine in a Python environment, then rerun this launcher. See README.md.")


if __name__ == "__main__":
    main()
