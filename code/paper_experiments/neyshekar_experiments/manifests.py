"""Freeze data independently of model and training seed."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .protocol import (
    CV_DIR,
    DATA,
    DATA_SEED,
    HF_REVISION,
    MAX_TRAIN_SECONDS,
    ROOT,
    digest,
    file_digest,
    write_json,
)

MANIFESTS = ROOT / "data/manifests/v2"


def frames() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Cache parsing by content, rechecking file hashes and returning private copies."""
    paths = (
        DATA / "neyshekar_v6_meta.parquet",
        CV_DIR / "clip_durations.tsv",
        *(CV_DIR / f"{split}.tsv" for split in ("train", "dev", "test")),
    )
    signature = tuple((str(path.resolve()), file_digest(path)) for path in paths)
    ney, cv = _frames_cached(signature)
    return ney.copy(deep=True), {split: frame.copy(deep=True) for split, frame in cv.items()}


@lru_cache(maxsize=2)
def _frames_cached(signature):
    paths = [Path(path) for path, _ in signature]
    ney_path, duration_path, *split_paths = paths
    ney = pd.read_parquet(ney_path)
    ney = ney.assign(corpus="ney")
    durations = pd.read_csv(duration_path, sep="\t", quoting=3)
    durations.columns = ["path", "ms"]
    cv = {}
    for split, path in zip(("train", "dev", "test"), split_paths):
        f = pd.read_csv(path, sep="\t", quoting=3, low_memory=False)
        f["duration"] = f.path.map(durations.set_index("path").ms) / 1000
        f = f[f.sentence.notna() & f.duration.notna()].copy()
        cv[split] = f.assign(
            id=f.path, text=f.sentence, corpus="cv", split="validation" if split == "dev" else split
        )
    for split in ("dev", "test"):
        for column in ("path", "client_id"):
            if set(cv["train"][column]) & set(cv[split][column]):
                raise ValueError(f"CV train/{split} overlap in {column}")
    if tuple((str(path), file_digest(path)) for path in paths) != signature:
        raise ValueError("Source data changed while parsing; retry with stable input files")
    return ney, cv


def prefix(frame: pd.DataFrame, hours: float) -> pd.DataFrame:
    """Take a prefix of an already frozen order; never re-shuffle for each seed."""
    return frame.loc[frame.duration.cumsum() <= hours * 3600 + 1e-8].copy()


def shuffled_ney(frame: pd.DataFrame) -> pd.DataFrame:
    # Dataset.shuffle uses default_rng permutation; retain release order first.
    return frame.iloc[np.random.default_rng(DATA_SEED).permutation(len(frame))].copy()


def manifest_payload(name: str, frame: pd.DataFrame, **extra) -> dict:
    records = [
        {
            "corpus": r.corpus,
            "id": int(r.id) if r.corpus == "ney" else str(r.id),
            "duration": float(r.duration),
            "text_sha256": digest(r.text),
        }
        for r in frame.itertuples()
    ]
    body = {
        "name": name,
        "data_seed": DATA_SEED,
        "dataset_revision": HF_REVISION,
        "clips": len(records),
        "hours": float(frame.duration.sum() / 3600),
        "source_hours": {c: float(f.duration.sum() / 3600) for c, f in frame.groupby("corpus")},
        "records": records,
        **extra,
    }
    return {**body, "sha256": digest(body)}


def freeze(path: Path, payload: dict) -> None:
    if path.exists():
        if json.loads(path.read_text()) != payload:
            raise ValueError(f"Frozen manifest differs: {path}; create a new protocol version")
    else:
        write_json(path, payload)


def prepare() -> pd.DataFrame:
    ney, cv = frames()
    train = ney[ney.split == "train"]
    # Filter BEFORE selecting/matching. Both architectures read the same records.
    n = shuffled_ney(train[train.duration <= MAX_TRAIN_SECONDS])
    c = cv["train"][cv["train"].duration <= MAX_TRAIN_SECONDS].sample(
        frac=1, random_state=DATA_SEED
    )
    budget = float(c.duration.sum() / 3600)
    selected = {
        "ney_matched": prefix(n, budget),
        "cv_matched": c,
        "mixed_matched": pd.concat([prefix(n, budget / 2), prefix(c, budget / 2)]),
        "mixed_double": pd.concat([prefix(n, budget), c]),
        "ney_double": prefix(n, 2 * budget),
        **{f"ney_{h}h": prefix(n, h) for h in (5, 10, 20, 40)},
        "ney_full": n,
    }
    summary = []
    for name, frame in selected.items():
        p = manifest_payload(
            name,
            frame,
            max_train_seconds=MAX_TRAIN_SECONDS,
            eligibility="duration <= 20s before matching; common to both architectures",
        )
        freeze(MANIFESTS / f"{name}.json", p)
        summary.append({k: p[k] for k in ("name", "clips", "hours", "sha256")})
    return pd.DataFrame(summary)


def load_manifest(name: str) -> dict:
    p = json.loads((MANIFESTS / f"{name}.json").read_text())
    body = {k: v for k, v in p.items() if k != "sha256"}
    if digest(body) != p["sha256"]:
        raise ValueError(f"Manifest hash mismatch: {name}")
    return p
