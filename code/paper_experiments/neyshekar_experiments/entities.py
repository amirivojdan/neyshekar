"""Reference-side entity attribution, explicitly separating SD rates from WER/CER."""

import pandas as pd
from jiwer import process_characters, process_words

from .analysis import aggregate, error_counts, normalized
from .protocol import ROOT, file_digest, write_json


def token_labels(tokens, entities):
    labels = [None] * len(tokens)
    unmatched = 0
    for text, label in entities:
        span = normalized(text).split()
        if not span:
            unmatched += 1
            continue
        for i in range(len(tokens) - len(span) + 1):
            if tokens[i : i + len(span)] == span and all(
                x is None for x in labels[i : i + len(span)]
            ):
                labels[i : i + len(span)] = [label] * len(span)
                break
        else:
            unmatched += 1
    return labels, unmatched


def attribution(rows, entities):
    counts = {
        unit: {group: {"n": 0, "errors": 0} for group in ("entity", "non_entity")}
        for unit in ("word", "character")
    }
    unmatched = 0
    for r in rows.itertuples(index=False):
        ref, hyp = normalized(r.reference), normalized(r.hypothesis)
        tokens = ref.split()
        labels, missed = token_labels(tokens, entities[r.id])
        unmatched += missed
        character_labels = []
        for i, (token, label) in enumerate(zip(tokens, labels)):
            if i:
                character_labels.append("SPACE")
            character_labels.extend([label] * len(token))
        for unit, alignment, position_labels in (
            ("word", process_words(ref, hyp), labels),
            ("character", process_characters(ref, hyp), character_labels),
        ):
            for chunk in alignment.alignments[0]:
                if chunk.type == "insert":
                    continue
                for i in range(chunk.ref_start_idx, chunk.ref_end_idx):
                    label = position_labels[i]
                    if unit == "character" and label == "SPACE":
                        continue
                    group = "entity" if label else "non_entity"
                    counts[unit][group]["n"] += 1
                    counts[unit][group]["errors"] += chunk.type != "equal"
    result = {
        "unmatched_mentions": unmatched,
        "overall": aggregate(error_counts(rows)),
        "counts": counts,
    }
    for unit, groups in counts.items():
        for group, c in groups.items():
            result[f"{group}_{unit}_sd"] = 100 * c["errors"] / c["n"] if c["n"] else None
    return result


def analyze_entities():
    """Tag and score the references of fresh runs, never old hypothesis caches."""
    from .evaluation import result_subset
    from .protocol import digest
    from .reporting import load_scores

    scores = load_scores()
    if scores.empty:
        return pd.DataFrame()
    from shekar import NER

    ner = NER()
    annotation_maps = {}
    results = []
    for run in scores.run.unique():
        available = scores.loc[scores.run.eq(run), "evaluation"]
        for evaluation in (name for name in ("ney_test", "cv_test") if name in set(available)):
            path = ROOT / f"results/v2/hyps/{run}__{evaluation}.jsonl"
            full = result_subset(run, evaluation)
            signature = digest(full[["id", "reference"]].to_dict("records"))
            if signature not in annotation_maps:
                # Reference-conditioned annotation uses normalized scoring text.
                annotation_maps[signature] = {
                    r.id: ner(normalized(r.reference)) for r in full.itertuples()
                }
            entities = annotation_maps[signature]
            sets = {evaluation: full}
            if evaluation == "ney_test":
                sets["ney_test_disjoint"] = result_subset(run, "ney_test_disjoint")
            for name, rows in sets.items():
                result = attribution(rows, entities)
                results.append(
                    {
                        "run": run,
                        "evaluation": name,
                        "hypotheses_sha256": file_digest(path),
                        **result,
                    }
                )
    payload = {
        "annotation_status": "Fresh automatic NER on scoring-normalized references; not human-validated",
        "metric_definition": "Reference-side word/character substitution+deletion rate; excludes insertions. Character categories exclude spaces. Unmatched mentions remain unlabeled/non-entity. Overall WER/CER include insertions.",
        "results": results,
    }
    write_json(ROOT / "results/analysis/entity_word_character_errors.json", payload)
    return pd.json_normalize(results)
