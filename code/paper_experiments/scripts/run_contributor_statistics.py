"""Describe how recordings and duration are spread across contributors.

A contributor count alone does not say whether 198 people carried 99 hours
evenly or whether a handful of them carried most of it. This reports the
per-contributor distribution and its concentration, for Neyshekar and for
Persian Common Voice under the same measures.

Common Voice publishes a per-clip `client_id`, so its side always runs. The
Neyshekar side needs the item-level contributor mapping exported by
`export_speaker_mapping.py` from the release's own `speaker_id` column. Only
aggregate statistics are written.

    python scripts/run_contributor_statistics.py \
        --mapping data/validation/speaker_mapping.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parents[1]
ROOT = CODE.parent
sys.path.insert(0, str(CODE))

from neyshekar_experiments.acoustics import summarise  # noqa: E402
from neyshekar_experiments.protocol import CV_DIR, DATA, write_json  # noqa: E402

OUT = ROOT / "results/analysis"


def gini(values) -> float:
    """0 when every contributor carries an equal share, approaching 1 when one carries all."""
    ordered = sorted(float(v) for v in values)
    n = len(ordered)
    total = sum(ordered)
    if n == 0 or total <= 0:
        return 0.0
    weighted = sum((index + 1) * value for index, value in enumerate(ordered))
    return (2.0 * weighted) / (n * total) - (n + 1.0) / n


def top_share(values, k: int) -> float:
    ordered = sorted((float(v) for v in values), reverse=True)
    total = sum(ordered)
    return sum(ordered[:k]) / total if total > 0 else 0.0


def effective_speakers(hours) -> float:
    """Inverse Simpson index of the hours distribution.

    The count of registered contributors overstates diversity when a few of them
    supply most of the audio. This is the number of equally contributing speakers
    that would produce the same concentration, so it is comparable across corpora
    whose nominal counts differ by an order of magnitude.
    """
    share = np.asarray([float(h) for h in hours], dtype=np.float64)
    total = share.sum()
    if total <= 0:
        return 0.0
    share = share / total
    return float(1.0 / np.sum(share**2))


def coverage(hours, fraction: float) -> int:
    """How many of the largest contributors together supply this share of audio."""
    ordered = np.sort(np.asarray([float(h) for h in hours], dtype=np.float64))[::-1]
    if ordered.sum() <= 0:
        return 0
    return int(np.searchsorted(np.cumsum(ordered / ordered.sum()), fraction) + 1)


def describe(clips_per_speaker, hours_per_speaker) -> dict:
    hours = list(hours_per_speaker)
    return {
        "contributors": len(hours),
        "effective_speakers": effective_speakers(hours),
        "clips_per_contributor": summarise(clips_per_speaker),
        "hours_per_contributor": summarise(hours),
        "concentration": {
            "gini_hours": gini(hours),
            "top1_share_hours": top_share(hours, 1),
            "top5_share_hours": top_share(hours, 5),
            "top10_share_hours": top_share(hours, 10),
            "contributors_for_50pct_hours": coverage(hours, 0.5),
            "contributors_for_80pct_hours": coverage(hours, 0.8),
            "contributors_for_90pct_hours": coverage(hours, 0.9),
            "share_under_1_minute": float(np.mean([h * 3600 < 60 for h in hours])) if hours else 0.0,
            "share_over_1_hour": float(np.mean([h > 1 for h in hours])) if hours else 0.0,
        },
    }


def load_mapping(path: Path) -> dict[int, str]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        missing = {"clip_id", "speaker_id"} - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"{path} is missing column(s): {', '.join(sorted(missing))}")
        return {int(row["clip_id"]): row["speaker_id"] for row in reader}


def neyshekar(metadata: Path, mapping_path: Path) -> dict:
    import pandas as pd

    frame = pd.read_parquet(metadata)
    frame["speaker_id"] = frame["id"].map(load_mapping(mapping_path))
    unmapped = int(frame["speaker_id"].isna().sum())
    if unmapped:
        raise SystemExit(
            f"{unmapped:,} of {len(frame):,} released clips have no contributor; "
            "the mapping must cover every clip exactly once."
        )

    # Speaker-disjointness is the property the evaluation rests on, so it is
    # checked here rather than assumed.
    per_split = {
        split: set(part["speaker_id"]) for split, part in frame.groupby("split", observed=True)
    }
    names = sorted(per_split)
    shared = {
        f"{a}|{b}": len(per_split[a] & per_split[b])
        for i, a in enumerate(names)
        for b in names[i + 1 :]
        if per_split[a] & per_split[b]
    }

    grouped = frame.groupby("speaker_id", observed=True)["duration"]
    report = {
        "clips": int(len(frame)),
        "hours": float(frame["duration"].sum() / 3600.0),
        "speaker_disjoint_splits": not shared,
        "shared_speakers": shared,
        **describe(list(grouped.size()), list(grouped.sum() / 3600.0)),
        "by_split": {},
    }
    for split, part in frame.groupby("split", observed=True):
        by_speaker = part.groupby("speaker_id", observed=True)["duration"]
        report["by_split"][str(split)] = {
            "clips": int(len(part)),
            "hours": float(part["duration"].sum() / 3600.0),
            **describe(list(by_speaker.size()), list(by_speaker.sum() / 3600.0)),
        }
    return report


def common_voice(directory: Path) -> dict:
    """Validated Persian Common Voice, keyed on the released client_id."""
    import pandas as pd

    validated = pd.read_csv(directory / "validated.tsv", sep="\t", quoting=3, low_memory=False)
    durations = pd.read_csv(directory / "clip_durations.tsv", sep="\t", quoting=3)
    durations.columns = ["path", "ms"]
    frame = validated.merge(durations, on="path", how="left", validate="one_to_one")
    frame = frame[frame.sentence.notna() & frame.ms.notna()].copy()
    frame["hours"] = frame["ms"] / 3.6e6

    grouped = frame.groupby("client_id")["hours"]
    return {
        "clips": int(len(frame)),
        "hours": float(frame["hours"].sum()),
        **describe(list(grouped.size()), list(grouped.sum())),
        # Demographic fields Neyshekar does not collect; coverage is partial.
        "self_reported_coverage": {
            column: float(frame[column].notna().mean())
            for column in ("age", "gender", "accents")
            if column in frame
        },
    }


def show(name: str, report: dict) -> None:
    clips, hours = report["clips_per_contributor"], report["hours_per_contributor"]
    concentration = report["concentration"]
    print(f"\n{name}: {report['clips']:,} clips, {report['hours']:.2f} h")
    print(
        f"  contributors {report['contributors']:,}"
        f"   effective (hours-weighted) {report['effective_speakers']:.1f}"
    )
    print(
        f"  clips/contributor  median {clips['median']:.0f}"
        f"  IQR [{clips['q1']:.0f}, {clips['q3']:.0f}]  range [{clips['min']:.0f}, {clips['max']:.0f}]"
    )
    print(
        f"  hours/contributor  median {hours['median']:.3f}"
        f"  IQR [{hours['q1']:.3f}, {hours['q3']:.3f}]  range [{hours['min']:.3f}, {hours['max']:.2f}]"
    )
    print(f"  Gini (hours) {concentration['gini_hours']:.3f}")
    print(f"  top 10 hold {100 * concentration['top10_share_hours']:.1f}% of hours")
    print(f"  {concentration['contributors_for_50pct_hours']:,} contributors cover 50% of hours")
    print(f"  under 1 minute total: {100 * concentration['share_under_1_minute']:.1f}%")
    if "speaker_disjoint_splits" in report:
        print(f"  speaker-disjoint splits: {report['speaker_disjoint_splits']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping", type=Path, help="Neyshekar clip_id,speaker_id,split CSV")
    parser.add_argument("--metadata", type=Path, default=DATA / "neyshekar_v6_meta.parquet")
    parser.add_argument("--cv-dir", type=Path, default=CV_DIR)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    report = {"common_voice": common_voice(args.cv_dir)}
    show("Common Voice 26.0 fa (validated)", report["common_voice"])

    if args.mapping:
        report["neyshekar"] = neyshekar(args.metadata, args.mapping)
        show("Neyshekar v6", report["neyshekar"])
    else:
        report["neyshekar"] = {
            "unavailable": "The item-level contributor mapping is not part of the release; "
            "pass --mapping to include Neyshekar."
        }
        print("\nNeyshekar v6: skipped, no --mapping given.")

    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / "contributor_statistics.json", report)
    print(f"\nWrote {args.out / 'contributor_statistics.json'}")


if __name__ == "__main__":
    main()
