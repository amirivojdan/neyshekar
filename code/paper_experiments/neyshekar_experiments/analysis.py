"""Paired WER/CER inference from immutable hypotheses, with optional clustering."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from jiwer import process_characters, process_words

from .protocol import ROOT, normalise_for_scoring, write_json


@lru_cache(maxsize=150000)
def normalized(text):
    return normalise_for_scoring(text)


def load_hypotheses(path: Path) -> pd.DataFrame:
    rows = pd.DataFrame(json.loads(line) for line in path.read_text().splitlines())
    if not {"id", "reference", "hypothesis"}.issubset(rows.columns):
        raise ValueError(f"Missing hypothesis fields: {path}")
    if rows.id.duplicated().any() or rows[["id", "reference", "hypothesis"]].isna().any().any():
        raise ValueError(f"Duplicate/missing evaluation records: {path}")
    return rows


def error_counts(rows: pd.DataFrame) -> pd.DataFrame:
    out = []
    for row in rows.itertuples(index=False):
        ref, hyp = normalized(row.reference), normalized(row.hypothesis)
        if not ref:
            out.append(
                {
                    "id": row.id,
                    "word_errors": 0,
                    "words": 0,
                    "char_errors": 0,
                    "characters": 0,
                    "scored": False,
                }
            )
            continue
        w, c = process_words(ref, hyp), process_characters(ref, hyp)
        out.append(
            {
                "id": row.id,
                "scored": True,
                "word_errors": w.substitutions + w.deletions + w.insertions,
                "words": w.hits + w.substitutions + w.deletions,
                "char_errors": c.substitutions + c.deletions + c.insertions,
                "characters": c.hits + c.substitutions + c.deletions,
            }
        )
    if not any(row["scored"] for row in out):
        raise ValueError("No nonempty scoring references")
    return pd.DataFrame(out)


def aggregate(counts: pd.DataFrame) -> dict:
    n_scored = int(counts.scored.sum()) if "scored" in counts else len(counts)
    return {
        "n": n_scored,
        "n_total": len(counts),
        "n_scored": n_scored,
        "n_empty_reference": len(counts) - n_scored,
        "wer": 100 * counts.word_errors.sum() / counts.words.sum(),
        "cer": 100 * counts.char_errors.sum() / counts.characters.sum(),
    }


def paired_bootstrap(
    a: pd.DataFrame,
    b: pd.DataFrame,
    *,
    clusters: pd.Series | None = None,
    replicates=20000,
    seed=42,
) -> dict:
    """Return B-minus-A error-rate differences; positive means A is better.

    Input IDs/references must agree. A cluster resample includes all recordings
    belonging to its sampled speaker/prompt; it does not resample labels or seeds.
    ``clusters`` must be a Series mapping evaluation IDs to cluster identifiers.
    """
    if replicates < 1:
        raise ValueError("replicates must be positive")
    if a.id.duplicated().any() or b.id.duplicated().any() or set(a.id) != set(b.id):
        raise ValueError("Paired systems require the same unique evaluation IDs")
    a = a.set_index("id").sort_index()
    b = b.set_index("id").loc[a.index]
    if not a.reference.eq(b.reference).all():
        raise ValueError("Paired references disagree")
    ca = error_counts(a.reset_index()).set_index("id")
    cb = error_counts(b.reset_index()).set_index("id").loc[ca.index]
    n_total = len(ca)
    ca = ca.loc[ca.scored]
    cb = cb.loc[ca.index]
    frame = pd.DataFrame(
        {
            "word_delta": cb.word_errors - ca.word_errors,
            "char_delta": cb.char_errors - ca.char_errors,
            "words": ca.words,
            "characters": ca.characters,
        }
    )
    unit = "utterance"
    if clusters is not None:
        if clusters.index.duplicated().any():
            raise ValueError("Cluster map contains duplicate IDs")
        mapped = clusters.reindex(frame.index)
        if mapped.isna().any():
            raise ValueError("Cluster map does not cover all scored IDs")
        frame = frame.groupby(mapped).sum()
        unit = "cluster"
    values = frame.to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    draws = []
    for start in range(0, replicates, 200):
        ix = rng.integers(0, len(values), (min(200, replicates - start), len(values)))
        sums = values[ix].sum(1)
        draws.append(100 * sums[:, :2] / sums[:, 2:])
    draws = np.concatenate(draws)
    totals = values.sum(0)
    result = {
        "n_total": n_total,
        "n_scored": len(ca),
        "n_empty_reference": n_total - len(ca),
        "n_utterances": len(ca),
        "n_resampling_units": len(values),
        "unit": unit,
        "replicates": replicates,
        "seed": seed,
        "direction": "B minus A",
        "uncertainty_scope": "test sampling conditional on fixed trained systems",
    }
    for j, metric in enumerate(("wer", "cer")):
        tails = min(np.count_nonzero(draws[:, j] <= 0), np.count_nonzero(draws[:, j] >= 0))
        result[metric] = {
            "difference": float(100 * totals[j] / totals[j + 2]),
            "ci95": np.quantile(draws[:, j], [0.025, 0.975]).tolist(),
            "two_sided_bootstrap_tail_p": min(1.0, 2 * (tails + 1) / (replicates + 1)),
        }
    return result


def analyze_fresh(replicates=20000):
    """Compare completed conditions at the same seed and optimization budget."""
    from .evaluation import result_subset
    from .reporting import load_scores

    scores = load_scores()
    if scores.empty:
        return {"status": "No fresh experiments completed; no analysis artifact written"}
    if replicates < 1:
        raise ValueError("replicates must be positive")
    output = {}
    speaker_path = ROOT / "data/validation/speaker_mapping.csv"
    speakers = (
        pd.read_csv(speaker_path).set_index("clip_id").speaker_id if speaker_path.exists() else None
    )
    pairs = [
        ("ney_matched", "cv_matched"),
        ("mixed_matched", "ney_matched"),
        ("mixed_matched", "cv_matched"),
        ("mixed_double", "ney_double"),
    ]
    for group_key, group in scores.groupby(["system", "seed", "budget", "max_steps"], dropna=False):
        for a_name, b_name in pairs:
            for evaluation in sorted(group.evaluation.unique()):
                a_info = group[group.condition.eq(a_name) & group.evaluation.eq(evaluation)]
                b_info = group[group.condition.eq(b_name) & group.evaluation.eq(evaluation)]
                if len(a_info) != 1 or len(b_info) != 1:
                    continue
                a = result_subset(a_info.iloc[0].run, evaluation)
                b = result_subset(b_info.iloc[0].run, evaluation)
                key = f"{group_key}: {a_name} vs {b_name} / {evaluation}"
                output[key] = {"utterance": paired_bootstrap(a, b, replicates=replicates)}
                if "recording_id" in a and a.recording_id.notna().all():
                    output[key]["recording"] = paired_bootstrap(
                        a, b, clusters=a.set_index("id").recording_id, replicates=replicates
                    )
                if evaluation.startswith("ney_test") and speakers is not None:
                    output[key]["speaker"] = paired_bootstrap(
                        a, b, clusters=speakers, replicates=replicates
                    )
                elif evaluation.startswith("ney_test"):
                    output[key]["speaker"] = {"status": "pending recorder-ID mapping"}
    write_json(ROOT / "results/analysis/paired_wer_cer.json", output)
    return output
