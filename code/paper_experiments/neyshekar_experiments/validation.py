"""Export minimal pseudonymous evidence and recompute collection reliability.

Raw platform exports and the HMAC key stay outside the public artifact. Public
files contain only pseudonyms, release clip IDs, split labels and binary votes.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from .protocol import DATA, ROOT, file_digest, text_tools, write_json

PUBLIC = DATA / "validation"


def export_evidence(raw_path: Path, key: bytes, owners_path: Path | None = None):
    if len(key) < 32:
        raise ValueError("Use at least 32 random bytes as the private pseudonymization key")
    raw = json.loads(raw_path.read_text())
    if not isinstance(raw, list):
        raise ValueError("Expected a flat platform export")
    normalizer, _ = text_tools()
    accepted = [r for r in raw if r.get("is_correct", True)]
    random.Random(42).shuffle(accepted)  # Exact v6 preparation algorithm.
    released = pd.read_parquet(DATA / "neyshekar_v6_meta.parquet").set_index("id")
    release_id = {}
    if len(accepted) != len(released):
        raise ValueError(
            "Accepted export count differs from v6; supply the original release snapshot"
        )
    for cid, r in enumerate(accepted):
        actual = released.loc[cid]
        if normalizer(r["text_content"].strip()) != actual.text or r["split"] != actual.split:
            raise ValueError(
                f"Release-ID reconstruction failed at clip {cid}; cannot guess speaker mapping"
            )
        release_id[r["voice_id"]] = cid

    def pseudo(kind, value):
        return (
            kind + "_" + hmac.new(key, f"{kind}:{value}".encode(), hashlib.sha256).hexdigest()[:24]
        )

    labels = []
    for r in raw:
        for a in r["annotations"]:
            if type(a["is_correct"]) is not bool:
                raise ValueError("Annotations must be boolean accept/reject decisions")
            labels.append(
                {
                    "item_id": pseudo("item", r["voice_id"]),
                    "rater_id": pseudo("rater", a["annotator"]),
                    "accepted": int(a["is_correct"]),
                    "cohort": "shared" if r["is_shared_sample"] else "other",
                    "clip_id": release_id.get(r["voice_id"]),
                    "in_release": r["voice_id"] in release_id,
                }
            )
    PUBLIC.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(labels).to_csv(PUBLIC / "rater_labels.csv", index=False)
    status = {
        "raw_snapshot_sha256": file_digest(raw_path),
        "snapshot_items": len(raw),
        "release_records_matched": len(release_id),
        "labels": len(labels),
        "speaker_evidence": "missing: raw snapshot contains no recorder IDs",
    }
    if owners_path is not None:
        owners = json.loads(owners_path.read_text())
        records = []
        for r in accepted:
            if r["voice_id"] not in owners:
                raise ValueError("Owner export is missing an accepted recording")
            records.append(
                {
                    "clip_id": release_id[r["voice_id"]],
                    "speaker_id": pseudo("speaker", owners[r["voice_id"]]),
                    "split": r["split"],
                }
            )
        speakers = pd.DataFrame(records).sort_values("clip_id")
        verify_speakers(speakers)
        speakers.to_csv(PUBLIC / "speaker_mapping.csv", index=False)
        status["speaker_evidence"] = (
            "verified from owner projection joined to original release snapshot"
        )
    write_json(PUBLIC / "export_provenance.json", status)
    return status


def verify_speakers(mapping: pd.DataFrame, metadata: pd.DataFrame | None = None):
    metadata = pd.read_parquet(DATA / "neyshekar_v6_meta.parquet") if metadata is None else metadata
    if mapping[["clip_id", "speaker_id", "split"]].isna().any().any():
        raise ValueError("Missing speaker mapping fields")
    if mapping.clip_id.duplicated().any() or set(mapping.clip_id) != set(metadata.id):
        raise ValueError("Speaker mapping must cover every release ID exactly once")
    joined = mapping.merge(
        metadata[["id", "split"]],
        left_on="clip_id",
        right_on="id",
        suffixes=("", "_release"),
        validate="one_to_one",
    )
    if not joined.split.eq(joined.split_release).all():
        raise ValueError("Speaker mapping disagrees with release splits")
    if mapping.groupby("speaker_id").split.nunique().gt(1).any():
        raise ValueError("Speaker leakage: a recorder appears in multiple splits")
    return {
        "clips": len(mapping),
        "speakers": mapping.speaker_id.nunique(),
        "speakers_by_split": mapping.groupby("split").speaker_id.nunique().to_dict(),
        "pairwise_split_intersections": 0,
    }


def reliability(labels: pd.DataFrame, replicates=20000, seed=42):
    """Fleiss kappa and binary Gwet AC1; percentile CIs resample whole items."""
    if labels.duplicated(["item_id", "rater_id"]).any():
        raise ValueError("Duplicate item/rater decisions")
    if not labels.accepted.isin([0, 1]).all():
        raise ValueError("Labels must be binary")
    matrix = labels.pivot(index="item_id", columns="rater_id", values="accepted")
    if matrix.isna().any().any() or matrix.shape[1] < 2:
        raise ValueError("Primary reliability requires a complete, shared multi-rater sample")
    votes = matrix.to_numpy(dtype=float)
    n, k = votes.shape
    positive = votes.sum(1)
    agreement = (positive * (positive - 1) + (k - positive) * (k - positive - 1)) / (k * (k - 1))

    def estimates(po, p):
        pe = p * p + (1 - p) * (1 - p)
        ac_pe = 2 * p * (1 - p)
        return {
            "raw_agreement": po,
            "accept_share": p,
            "fleiss_kappa": np.divide(
                po - pe, 1 - pe, out=np.full_like(np.asarray(po), np.nan), where=(1 - pe) > 0
            ),
            "gwet_ac1": (po - ac_pe) / (1 - ac_pe),
        }

    point = estimates(agreement.mean(), positive.mean() / k)
    rng = np.random.default_rng(seed)
    samples = {key: [] for key in point}
    for start in range(0, replicates, 500):
        ix = rng.integers(0, n, size=(min(500, replicates - start), n))
        e = estimates(agreement[ix].mean(1), positive[ix].mean(1) / k)
        for key, values in e.items():
            samples[key].extend(np.asarray(values).tolist())
    same_accept = np.sum(positive * (positive - 1) / 2)
    same_reject = np.sum((k - positive) * (k - positive - 1) / 2)
    disagree = np.sum(positive * (k - positive))
    output = {
        "items": n,
        "raters": k,
        "labels": n * k,
        "replicates": replicates,
        "seed": seed,
        "ci_method": "percentile bootstrap of complete items, preserving all rater votes",
        "accept_agreement": float(2 * same_accept / (2 * same_accept + disagree)),
        "reject_agreement": float(2 * same_reject / (2 * same_reject + disagree)),
    }
    for key, value in point.items():
        finite = np.asarray(samples[key])
        finite = finite[np.isfinite(finite)]
        output[key] = {
            "value": float(value) if np.isfinite(value) else None,
            "ci95": np.quantile(finite, [0.025, 0.975]).tolist() if len(finite) else None,
        }
    return output


def verify_all(replicates=20000):
    status = {}
    labels_path = PUBLIC / "rater_labels.csv"
    if labels_path.exists():
        labels = pd.read_csv(labels_path)
        status["agreement"] = reliability(labels[labels.cohort == "shared"], replicates)
        status["snapshot_multiply_reviewed_items"] = int(
            labels.groupby("item_id").size().ge(2).sum()
        )
        status["rater_labels_sha256"] = file_digest(labels_path)
    else:
        status["agreement"] = {
            "status": "missing item-level labels; aggregate constants are not verification"
        }
    speakers_path = PUBLIC / "speaker_mapping.csv"
    if speakers_path.exists():
        status["speakers"] = verify_speakers(pd.read_csv(speakers_path))
        status["speaker_mapping_sha256"] = file_digest(speakers_path)
    else:
        status["speakers"] = {
            "status": "missing per-record recorder mapping; speaker disjointness unverified"
        }
    write_json(ROOT / "results/validation_verification.json", status)
    return status
