"""Automatic reference strata, scored by selecting saved full-test predictions."""

import importlib.metadata
import json

from .evaluation import result_subset
from .protocol import ROOT, digest, normalise_for_scoring, score, write_json
from .reporting import load_scores


def stratify_fresh():
    scores = load_scores()
    runs = scores.loc[scores.evaluation.eq("ney_test"), "run"].unique()
    if not len(runs):
        return {"status": "No fresh Neyshekar test decoding; no artifacts written"}
    from shekar import NER, InformalLanguageClassifier

    classifier, ner = InformalLanguageClassifier(), NER()
    maps = {}
    for run in runs:
        full = result_subset(run, "ney_test")
        signature = digest(
            {
                "references": full[["id", "reference"]].to_dict("records"),
                "shekar": importlib.metadata.version("shekar"),
                "input": "common scoring normalization",
            }
        )
        annotation_path = ROOT / "results/annotations" / f"ney_test_{signature}.json"
        if signature not in maps:
            if annotation_path.exists():
                annotation = json.loads(annotation_path.read_text())
                if digest(annotation["labels"]) != annotation["labels_sha256"]:
                    raise ValueError("Changed automatic reference labels")
            else:
                labels = []
                for row in full.itertuples():
                    text = normalise_for_scoring(row.reference)
                    labels.append(
                        {
                            "id": row.id,
                            "register": "informal" if classifier(text)[1] == 1 else "formal",
                            "entity": "present" if ner(text) else "absent",
                        }
                    )
                annotation = {
                    "reference_signature": signature,
                    "labels": labels,
                    "labels_sha256": digest(labels),
                    "status": "automatic; not human validated",
                }
                write_json(annotation_path, annotation)
            maps[signature] = annotation
        annotation = maps[signature]
        label_map = {row["id"]: row for row in annotation["labels"]}
        output = ROOT / "results/v2" / f"{run}.json"
        result = json.loads(output.read_text())
        for column in ("register", "entity"):
            for value in sorted({row[column] for row in annotation["labels"]}):
                subset = full.loc[full.id.map(lambda cid: label_map[cid][column] == value)]
                name = f"ney_test__{column}_{value}"
                result["results"][name] = score(
                    subset.reference.tolist(), subset.hypothesis.tolist()
                )
                result["evaluation_sources"][name] = {
                    "path": result["evaluation_sources"]["ney_test"]["path"],
                    "ids": subset.id.tolist(),
                    "annotation_signature": signature,
                    "labels_sha256": annotation["labels_sha256"],
                }
        write_json(output, result)
    return {"runs": len(runs), "annotation_sets": len(maps)}
