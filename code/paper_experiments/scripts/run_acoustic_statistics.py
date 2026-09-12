"""Characterise the signal quality of every released clip.

Recordings were contributed from unrestricted personal devices with no
signal-to-noise threshold, so the delivered audio is described here directly:
clipping, silence, level, and an estimated SNR per clip.

Reads the pinned Neyshekar release from the local Hugging Face cache and writes
per-clip measures plus a summary to results/analysis/. Decoding dominates the
runtime, so clips are processed in parallel over the release's parquet shards.

    python scripts/run_acoustic_statistics.py [--workers N] [--limit N]
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
ROOT = CODE.parent
sys.path.insert(0, str(CODE))

from neyshekar_experiments.acoustics import measure, summarise  # noqa: E402
from neyshekar_experiments.protocol import HF_REPO, HF_REVISION, write_json  # noqa: E402

OUT = ROOT / "results/analysis"


def shards() -> list[Path]:
    """Parquet shards of the pinned release, from the local cache."""
    from huggingface_hub import snapshot_download

    local = Path(
        snapshot_download(
            repo_id=HF_REPO,
            revision=HF_REVISION,
            repo_type="dataset",
            allow_patterns=["data/*.parquet"],
        )
    )
    found = sorted(local.glob("data/*.parquet"))
    if not found:
        raise SystemExit(f"No parquet shards under {local}")
    return found


def split_of(path: Path) -> str:
    name = path.name.split("-")[0]
    return {"validation": "val"}.get(name, name)


def scan(path_str: str) -> list[dict]:
    """Measure every clip in one shard."""
    import numpy as np
    import pyarrow.parquet as pq
    import soundfile as sf

    path = Path(path_str)
    split = split_of(path)
    rows = []
    for batch in pq.ParquetFile(path).iter_batches(batch_size=64, columns=["id", "audio"]):
        for record in batch.to_pylist():
            payload = record["audio"]
            raw = payload["bytes"] if isinstance(payload, dict) else payload
            signal, rate = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
            if signal.ndim > 1:  # Release is mono; fold defensively rather than assume.
                signal = signal.mean(axis=1)
            result = measure(np.asarray(signal), rate)
            result["id"] = record["id"]
            result["split"] = split
            rows.append(result)
    return rows


MEASURES = [
    "duration_s",
    "rms_dbfs",
    "speech_rms_dbfs",
    "peak_dbfs",
    "noise_floor_dbfs",
    "snr_db",
    "silence_ratio",
    "leading_silence_s",
    "trailing_silence_s",
    "clipped_fraction",
]


def describe(rows: list[dict]) -> dict:
    total = len(rows)
    if not total:
        return {"clips": 0}
    gated = [r for r in rows if r.get("gated_silence")]
    room = [r for r in rows if r.get("gated_silence") is False]
    summary = {
        "clips": total,
        "hours": sum(r["duration_s"] for r in rows) / 3600.0,
        "measures": {name: summarise([r.get(name) for r in rows]) for name in MEASURES},
        "clipping": {
            "clips_with_flattened_run": sum(1 for r in rows if r.get("clipped_run")),
            "clips_with_any_full_scale_sample": sum(
                1 for r in rows if (r.get("clipped_fraction") or 0) > 0
            ),
            "clips_over_0.1pct_samples": sum(
                1 for r in rows if (r.get("clipped_fraction") or 0) > 1e-3
            ),
        },
        # A gated pause reports the capture chain's noise suppression, not the
        # room, so its SNR is summarised apart from the rest.
        "gated_silence": {
            "clips": len(gated),
            "snr_db": summarise([r.get("snr_db") for r in gated]),
        },
        "measured_noise_floor": {
            "clips": len(room),
            "snr_db": summarise([r.get("snr_db") for r in room]),
            "noise_floor_dbfs": summarise([r.get("noise_floor_dbfs") for r in room]),
        },
        "sample_rates": sorted({r["sample_rate"] for r in rows}),
        "max_abs_dc_offset": max(abs(r.get("dc_offset") or 0.0) for r in rows),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=min(os.cpu_count() or 4, 16))
    parser.add_argument("--limit", type=int, help="Only scan this many shards (for a smoke run)")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    files = shards()
    if args.limit:
        files = files[: args.limit]
    print(f"Scanning {len(files)} shards with {args.workers} workers", flush=True)

    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for done, batch in enumerate(pool.map(scan, [str(f) for f in files]), start=1):
            rows.extend(batch)
            print(f"  {done}/{len(files)} shards, {len(rows):,} clips", flush=True)

    rows.sort(key=lambda r: r["id"])
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "acoustic_quality_clips.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = {
        "source": {"repo": HF_REPO, "revision": HF_REVISION, "shards": len(files)},
        "method": {
            "frames": "25 ms window, 10 ms hop",
            "clipping": "|x| >= 32767/32768; a flattened run is >= 3 consecutive such samples",
            "silence": "frame > 30 dB below the clip's 95th-percentile frame level, or below -70 dBFS",
            "snr": "95th vs 5th percentile of frame power; recovers 0-30 dB white noise within ~1 dB",
            "gated_silence": "5th-percentile frame level below -80 dBFS, i.e. a suppressed pause",
        },
        "overall": describe(rows),
        "by_split": {
            split: describe([r for r in rows if r["split"] == split])
            for split in sorted({r["split"] for r in rows})
        },
    }
    write_json(args.out / "acoustic_quality.json", report)

    overall = report["overall"]
    print(f"\n{overall['clips']:,} clips, {overall['hours']:.2f} h")
    for name in ("snr_db", "rms_dbfs", "silence_ratio", "clipped_fraction"):
        s = overall["measures"][name]
        print(f"  {name:18s} median {s['median']:8.3f}  IQR [{s['q1']:.3f}, {s['q3']:.3f}]")
    print(f"  clips with a flattened run: {overall['clipping']['clips_with_flattened_run']:,}")
    print(f"  gated-pause clips:          {overall['gated_silence']['clips']:,}")
    print(f"\nWrote {args.out / 'acoustic_quality.json'}")


if __name__ == "__main__":
    main()
