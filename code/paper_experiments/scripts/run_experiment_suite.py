"""Run the authorized experiment suite sequentially, with durable status and logs."""

import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
ROOT = CODE.parent
sys.path.insert(0, str(CODE))


def now():
    return datetime.now(timezone.utc).isoformat()


def stages(external):
    queue = [("zero_shot_whisper_small", ["zero-shot"])]
    for family in (
        "matched",
        "mixture_updates",
        "mixture",
        "updates",
        "scaling",
        "scaling_updates",
    ):
        command = ["run", family]
        if family.startswith("scaling"):
            command += ["--seeds", "42"]
        queue.append((family, command))
    for model in ("openai/whisper-large-v3", "facebook/mms-1b-all"):
        queue.append(
            (
                "zero_shot_" + model.split("/")[-1],
                ["zero-shot", "--model", model, "--external", *external],
            )
        )
    if external:
        queue.append(
            (
                "zero_shot_whisper_small_external",
                ["zero-shot", "--external-only", "--external", *external],
            )
        )
        queue.append(("external_finetuned", ["evaluate-external", "--datasets", *external]))
    for command in ("stratify", "analyze", "entities", "tables", "verify-validation"):
        queue.append((command, [command]))
    return queue


def main():
    from neyshekar_experiments.protocol import file_digest, write_json
    from neyshekar_experiments.training import require_gpu

    require_gpu()
    logs = ROOT / "logs"
    logs.mkdir(exist_ok=True)
    lock = (logs / "suite.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("Another experiment suite is already running") from None
    external = ["psrb_sample", "youtube_timestamps"]
    queue = stages(external)
    frozen_source = {
        str(path.relative_to(CODE)): file_digest(path)
        for path in sorted((CODE / "neyshekar_experiments").glob("*.py"))
    }
    frozen_source.update(
        {path.name: file_digest(path) for path in CODE.glob("requirements*") if path.is_file()}
    )
    status = {
        "pid": os.getpid(),
        "started": now(),
        "status": "running",
        "python": sys.executable,
        "external_datasets": external,
        "source_sha256": frozen_source,
        "stages": [],
        "pending_external": [
            name
            for name in ("psrb_sample", "youtube_timestamps")
            if not (ROOT / f"data/external/manifests/{name}.json").exists()
        ],
    }
    state = logs / "suite_status.json"
    for name, arguments in queue:
        for relative, expected in frozen_source.items():
            if file_digest(CODE / relative) != expected:
                status.update(
                    status="failed", error="Source/environment specification changed during suite"
                )
                write_json(state, status)
                raise SystemExit(status["error"])
        if shutil.disk_usage(ROOT).free < 15 * 1024**3:
            status.update(
                status="failed", error="Less than 15 GiB free; stopping before next stage"
            )
            write_json(state, status)
            raise SystemExit(status["error"])
        command = [sys.executable, "-u", "-B", "-m", "neyshekar_experiments", *arguments]
        entry = {
            "name": name,
            "command": command,
            "started": now(),
            "status": "running",
            "log": str((logs / f"{name}.log").relative_to(ROOT)),
        }
        status["stages"].append(entry)
        status["current_stage"] = name
        with (ROOT / entry["log"]).open("a") as output:
            output.write(f"\n--- {now()} ---\n")
            output.flush()
            worker = subprocess.Popen(
                command,
                cwd=CODE,
                stdout=output,
                stderr=subprocess.STDOUT,
                env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1"},
            )
            entry["pid"] = worker.pid
            while worker.poll() is None:
                status["updated"] = now()
                write_json(state, status)
                time.sleep(10)
            entry.update(
                returncode=worker.returncode,
                ended=now(),
                status="complete" if worker.returncode == 0 else "failed",
            )
        if worker.returncode != 0:
            status.update(status="failed", updated=now())
            write_json(state, status)
            raise SystemExit(f"Stage {name} failed; see {entry['log']}")
        write_json(state, status)
    status["pending_external"] = [
        name for name in external if not (ROOT / f"data/external/manifests/{name}.json").exists()
    ]
    status.update(
        status="complete_with_pending_external" if status["pending_external"] else "complete",
        current_stage=None,
        updated=now(),
    )
    write_json(state, status)
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
