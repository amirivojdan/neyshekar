"""Run independent experiments on separate GPUs without changing training budgets.

Each task has exactly one writer and sees exactly one GPU. Analysis starts only
after all training and evaluation tasks succeed. Existing artifacts are validated
by the research package; the scheduler does not bypass provenance checks.
"""

import argparse
import fcntl
import hashlib
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
ROOT = CODE.parent
sys.path.insert(0, str(CODE))
EXTERNAL = ["psrb_sample", "youtube_timestamps"]


def now():
    return datetime.now(timezone.utc).isoformat()


def write_state(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def source_hashes():
    paths = list((CODE / "neyshekar_experiments").glob("*.py"))
    paths += list(CODE.glob("requirements*")) + [Path(__file__)]
    return {
        str(p.relative_to(CODE)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(paths)
        if p.is_file()
    }


def task_specs():
    from dataclasses import asdict

    from neyshekar_experiments.training import experiment_grid

    tasks = []
    for family in (
        "matched",
        "mixture_updates",
        "mixture",
        "updates",
        "scaling",
        "scaling_updates",
    ):
        seeds = (42,) if family.startswith("scaling") else (42, 43, 44)
        for run in experiment_grid(family, seeds):
            tasks.append({"name": run.name, "run": asdict(run), "family": family})
    for model in ("openai/whisper-small", "openai/whisper-large-v3", "facebook/mms-1b-all"):
        tasks.append({"name": "zero_shot_" + model.split("/")[-1], "model": model})
    names = [task["name"] for task in tasks]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate tasks would write the same artifacts")
    inventory = ROOT / "logs/remote/import_inventory.json"
    if inventory.exists():
        imported = json.loads(inventory.read_text())["runs"]
        # Start new experiments while saved local models are still uploading.
        tasks.sort(key=lambda task: task["name"] in imported or "model" in task)
        for task in tasks:
            if task["name"] in imported:
                task["import_required"] = True
    return tasks


def ready(task):
    return (
        not task.get("import_required")
        or (ROOT / "logs/remote/imports" / (task["name"] + ".json")).exists()
    )


def worker(task):
    import torch

    from neyshekar_experiments.training import Run, evaluate_run, evaluate_zero_shot, train_run

    if torch.cuda.device_count() != 1:
        raise RuntimeError("Each worker must see exactly one GPU")
    print(
        json.dumps({"started": now(), "task": task, "gpu": torch.cuda.get_device_name(0)}),
        flush=True,
    )
    if "model" in task:
        wait_for_external_audio()
        evaluate_zero_shot(task["model"], EXTERNAL)
    else:
        run = Run(**task["run"])
        directory = train_run(run)
        external = EXTERNAL if task["family"] in ("matched", "mixture_updates") else ()
        if external:
            wait_for_external_audio()
        evaluate_run(run, directory, external)


def wait_for_external_audio():
    marker = ROOT / "logs/remote/data_transfer_complete.json"
    while not marker.exists():
        print("Waiting for verified external audio transfer", flush=True)
        time.sleep(10)


def worker_environment(gpu):
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "OMP_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONUNBUFFERED": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE"):
        env.pop(key, None)
    return env


def main(gpus):
    import torch

    if len(gpus) != len(set(gpus)) or not gpus:
        raise ValueError("GPU IDs must be unique")
    if any(gpu < 0 or gpu >= torch.cuda.device_count() for gpu in gpus):
        raise ValueError("Requested GPU is unavailable")
    logs = ROOT / "logs/remote"
    logs.mkdir(parents=True, exist_ok=True)
    lock = (ROOT / "logs/suite.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    queue = task_specs()
    state = {
        "pid": os.getpid(),
        "started": now(),
        "status": "running",
        "gpus": gpus,
        "host": platform.node(),
        "python": sys.version,
        "platform": platform.platform(),
        "gpu_names": [torch.cuda.get_device_name(gpu) for gpu in gpus],
        "source_sha256": source_hashes(),
        "tasks": [],
        "analysis": [],
    }
    path = logs / "parallel_status.json"
    active = {}
    stopping = False

    def stop(signum, frame):
        nonlocal stopping
        stopping = True
        state["stop_signal"] = signum
        for process, _, _ in active.values():
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        while queue or active:
            for gpu, (process, output, entry) in list(active.items()):
                result = process.poll()
                if result is None:
                    continue
                output.close()
                entry.update(
                    returncode=result, ended=now(), status="complete" if result == 0 else "failed"
                )
                del active[gpu]
                if result:
                    state.update(status="failed", error=f"Task {entry['name']} failed; see its log")
            if source_hashes() != state["source_sha256"]:
                state.update(status="failed", error="Source changed during execution")
            if shutil.disk_usage(ROOT).free < 20 * 1024**3:
                state.update(status="failed", error="Less than 20 GiB free")
            if stopping:
                state["status"] = "stopped"
            if state["status"] == "running":
                for gpu in gpus:
                    if gpu in active or not queue:
                        continue
                    index = next((i for i, task in enumerate(queue) if ready(task)), None)
                    if index is None:
                        break
                    task = queue.pop(index)
                    entry = {**task, "gpu": gpu, "started": now(), "status": "running"}
                    log = logs / (task["name"] + ".log")
                    entry["log"] = str(log.relative_to(ROOT))
                    output = log.open("a")
                    process = subprocess.Popen(
                        [sys.executable, "-u", "-B", __file__, "--task", json.dumps(task)],
                        cwd=CODE,
                        env=worker_environment(gpu),
                        stdout=output,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    entry["pid"] = process.pid
                    state["tasks"].append(entry)
                    active[gpu] = (process, output, entry)
            state.update(updated=now(), pending=len(queue), active=len(active))
            write_state(path, state)
            if state["status"] != "running" and not active:
                break
            time.sleep(5)
        if state["status"] == "running":
            for command in ("stratify", "analyze", "entities", "tables", "verify-validation"):
                if stopping:
                    state["status"] = "stopped"
                    break
                entry = {"name": command, "status": "running", "started": now()}
                state["analysis"].append(entry)
                with (logs / (command + ".log")).open("a") as output:
                    process = subprocess.Popen(
                        [sys.executable, "-u", "-B", "-m", "neyshekar_experiments", command],
                        cwd=CODE,
                        env=worker_environment(gpus[0]),
                        stdout=output,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    active[gpus[0]] = (process, output, entry)
                    while process.poll() is None:
                        state["updated"] = now()
                        write_state(path, state)
                        time.sleep(5)
                    active.clear()
                entry.update(
                    returncode=process.returncode,
                    ended=now(),
                    status="complete" if process.returncode == 0 else "failed",
                )
                if process.returncode:
                    state.update(
                        status="stopped" if stopping else "failed",
                        error=f"Analysis {command} stopped",
                    )
                    break
            if state["status"] == "running":
                state["status"] = "complete"
    finally:
        if state["status"] == "running":
            state.update(status="failed", error="Scheduler exited unexpectedly")
        for process, output, entry in active.values():
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            output.close()
        state["updated"] = now()
        write_state(path, state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--task", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.task:
        worker(json.loads(args.task))
    else:
        main(args.gpus)
