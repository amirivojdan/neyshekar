"""Immutable full-test predictions; every score and slice reads this same file."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .protocol import ROOT, SCORING, digest, file_digest, normalise_for_scoring, score, write_json


def saved_decode(path, dataset, decode, provenance):
    """Resume complete decodings, rejecting changes instead of silently replacing them."""
    path = Path(path)
    records = dataset.rows.to_dict("records")
    identity = {
        "model": provenance,
        "evaluation_records": records,
        "dataset": getattr(dataset, "provenance", None),
        "scoring": SCORING,
    }
    expected = digest(identity)
    meta = path.with_suffix(".meta.json")
    if path.exists() or meta.exists():
        if not path.exists() or not meta.exists():
            raise ValueError(
                f"Incomplete decoding cache; inspect and remove the incomplete pair: {path}"
            )
        info = json.loads(meta.read_text())
        if info["identity_sha256"] != expected or info["file_sha256"] != file_digest(path):
            raise ValueError(f"Changed or corrupt full-test decoding: {path}")
    else:
        hypotheses = decode()
        if len(hypotheses) != len(records):
            raise ValueError("Decoder did not return exactly one hypothesis per evaluation item")
        rows = [{**r, "reference": r["text"], "hypothesis": h} for r, h in zip(records, hypotheses)]
        if len({str(r["id"]) for r in rows}) != len(rows):
            raise ValueError("Duplicate evaluation IDs")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        temporary.replace(path)
        write_json(
            meta,
            {
                "identity_sha256": expected,
                "file_sha256": file_digest(path),
                "provenance": provenance,
                "n": len(rows),
            },
        )
    return pd.DataFrame(json.loads(line) for line in path.read_text().splitlines())


def select_subset(rows, ids):
    """Strict ID join: no decoding, reordering or silent missing-ID intersection."""
    ids = list(ids)
    if rows.id.duplicated().any() or len(ids) != len(set(ids)):
        raise ValueError("Duplicate subset/full-test IDs")
    if not set(ids).issubset(set(rows.id)):
        raise ValueError("Subset contains IDs absent from the full-test decoding")
    return rows.set_index("id").loc[ids].reset_index()


def subsets(name, rows, train_text=()):
    result = {name: rows}
    if name == "ney_test":
        result[name + "_disjoint"] = rows.loc[
            ~rows.reference.map(normalise_for_scoring).isin(train_text)
        ]
    if "duration" in rows:
        for label, mask in [
            ("short", rows.duration.lt(4)),
            ("medium", rows.duration.between(4, 10)),
            ("long", rows.duration.gt(10)),
        ]:
            result[name + "__duration_" + label] = rows.loc[mask]
    for column in ("formality", "spontaneous", "data_source"):
        if column in rows:
            for value in sorted(rows[column].dropna().astype(str).unique()):
                result[name + "__" + column + "=" + value] = rows.loc[
                    rows[column].astype(str).eq(value)
                ]
    return {key: value for key, value in result.items() if not value.empty}


def evaluate_sets(run_name, spec, provenance, sets, decoder, train_text=()):
    results, hashes, sources = {}, {}, {}
    for name, dataset in sets.items():
        path = ROOT / "results/v2/hyps" / f"{run_name}__{name}.jsonl"
        rows = saved_decode(path, dataset, lambda: decoder(dataset), provenance)
        relative = str(path.relative_to(ROOT))
        hashes[relative] = file_digest(path)
        for label, subset in subsets(name, rows, train_text).items():
            results[label] = score(subset.reference.tolist(), subset.hypothesis.tolist())
            sources[label] = {"path": relative, "ids": subset.id.tolist()}
    output = ROOT / "results/v2" / f"{run_name}.json"
    result = {
        "protocol": "v2",
        "run": spec,
        "provenance_sha256": digest(provenance),
        "hypothesis_sha256": hashes,
        "evaluation_sources": sources,
        "results": results,
    }
    # External evaluations can be added later without erasing the internal scores.
    if output.exists():
        previous = json.loads(output.read_text())
        if previous["provenance_sha256"] != result["provenance_sha256"]:
            raise ValueError("Cannot combine evaluations with different model provenance")
        for field in ("results", "hypothesis_sha256", "evaluation_sources"):
            result[field] = {**previous[field], **result[field]}
    write_json(output, result)
    return result


def result_subset(run_name, evaluation):
    """Canonical reader shared by bootstrap, reporting and downstream analyses."""
    result = json.loads((ROOT / "results/v2" / f"{run_name}.json").read_text())
    source = result["evaluation_sources"][evaluation]
    path = ROOT / source["path"]
    if file_digest(path) != result["hypothesis_sha256"][source["path"]]:
        raise ValueError("Saved hypotheses changed after scoring")
    rows = pd.DataFrame(json.loads(line) for line in path.read_text().splitlines())
    return select_subset(rows, source["ids"])
