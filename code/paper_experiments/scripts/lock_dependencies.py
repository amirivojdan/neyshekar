"""Resolve a hash-checked dependency lock for the current training platform.

Run in a tools environment with requirements-dev.txt installed. The lock is
platform-specific; regenerate on each supported training architecture.
"""

import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def main():
    code = Path(__file__).resolve().parents[1]
    if sys.version_info[:2] != (3, 12) or platform.system() != "Linux":
        raise SystemExit("Resolve the training lock on Linux with Python 3.12")
    name = f"requirements-linux-{platform.machine()}-py312.lock"
    # An interrupted or failed resolution must not replace a working lock.
    with tempfile.TemporaryDirectory(prefix="neyshekar-lock-") as temporary:
        output = Path(temporary) / name
        if (code / name).exists():
            shutil.copyfile(code / name, output)  # Reuse verified hashes for unchanged pins.
        command = [
            sys.executable,
            "-m",
            "piptools",
            "compile",
            "--generate-hashes",
            "--allow-unsafe",
            "--no-emit-index-url",
            "--no-emit-trusted-host",
            "--no-header",
            "--output-file",
            str(output),
            "requirements.txt",
        ]
        subprocess.run(command, cwd=code, check=True)
        header = (
            f"# Linux {platform.machine()}, Python 3.12\n"
            "# Generated with pip-tools; see requirements-dev.txt for the tool version.\n"
            "# Regenerate: python scripts/lock_dependencies.py\n"
        )
        staged = code / (name + ".tmp")
        staged.write_text(header + output.read_text(), encoding="utf-8")
        staged.replace(code / name)
    print(code / name)


if __name__ == "__main__":
    main()
