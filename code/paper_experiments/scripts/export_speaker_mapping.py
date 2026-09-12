"""Export the clip-to-contributor mapping released with the corpus.

Version 6 of the release carries an opaque `speaker_id` on every clip. This
reads that column straight from the pinned revision and writes the flat
`clip_id,speaker_id,split` table the analysis scripts join against.

Only the identifier, split, and duration columns are fetched; the audio column
is never read, so this costs a few megabytes rather than the full release.

    python scripts/export_speaker_mapping.py [--out data/validation/speaker_mapping.csv]
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
ROOT = CODE.parent
sys.path.insert(0, str(CODE))

from neyshekar_experiments.protocol import DATA, HF_REPO, SPEAKER_REVISION  # noqa: E402

COLUMNS = ["id", "speaker_id", "duration"]


def split_of(name: str) -> str:
    return name.split("-")[0]


def read_release(repo: str, revision: str) -> list[dict]:
    """Every released clip with its contributor, read column-wise over the network."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    root = f"datasets/{repo}@{revision}/data"
    shards = sorted(fs.glob(f"{root}/*.parquet"))
    if not shards:
        raise SystemExit(f"No parquet shards under {root}")

    rows: list[dict] = []
    for shard in shards:
        split = split_of(Path(shard).name)
        with fs.open(shard, "rb") as handle:
            table = pq.ParquetFile(handle).read(columns=COLUMNS)
        for record in table.to_pylist():
            rows.append(
                {
                    "clip_id": record["id"],
                    "speaker_id": record["speaker_id"],
                    "split": split,
                    "duration": record["duration"],
                }
            )
        print(f"  {Path(shard).name}: {table.num_rows:,} clips", flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=HF_REPO)
    parser.add_argument("--revision", default=SPEAKER_REVISION)
    parser.add_argument("--out", type=Path, default=DATA / "validation/speaker_mapping.csv")
    args = parser.parse_args()

    print(f"Reading {args.repo}@{args.revision[:12]}", flush=True)
    rows = read_release(args.repo, args.revision)

    missing = [r for r in rows if not r["speaker_id"]]
    if missing:
        raise SystemExit(f"{len(missing):,} clips carry no speaker_id")
    if len({r["clip_id"] for r in rows}) != len(rows):
        raise SystemExit("Clip ids are not unique across shards")

    rows.sort(key=lambda r: r["clip_id"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["clip_id", "speaker_id", "split"])
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in ("clip_id", "speaker_id", "split")})

    speakers = {r["speaker_id"] for r in rows}
    hours = sum(r["duration"] for r in rows) / 3600.0
    print(f"\n{len(rows):,} clips, {len(speakers):,} contributors, {hours:.2f} h")
    for split in sorted({r["split"] for r in rows}):
        part = [r for r in rows if r["split"] == split]
        named = {r["speaker_id"] for r in part}
        print(f"  {split:<11} {len(part):>6,} clips  {len(named):>4} contributors")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
