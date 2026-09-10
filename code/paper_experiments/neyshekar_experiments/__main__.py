"""Command-line entry point. Training is explicit; reports never start GPU work."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare", help="Freeze reproducible training manifests")
    train = sub.add_parser("run", help="Execute an explicit corrected experiment grid")
    train.add_argument(
        "family",
        choices=["matched", "mixture", "scaling", "updates", "mixture_updates", "scaling_updates"],
    )
    train.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    train.add_argument(
        "--architectures", nargs="+", choices=["whisper", "ctc"], default=["whisper", "ctc"]
    )
    train.add_argument("--dry-run", action="store_true")
    psrb = sub.add_parser("prepare-psrb", help="Freeze the pinned public PSRB sample")
    psrb.add_argument("--download", action="store_true")
    youtube = sub.add_parser(
        "prepare-youtube", help="Freeze the reference-timestamp YouTube condition"
    )
    youtube.add_argument("--episodes", type=Path, required=True)
    youtube.add_argument("--audio-root", type=Path, required=True)
    youtube.add_argument("--revision", required=True)
    zero = sub.add_parser("zero-shot", help="Evaluate an unadapted pretrained model")
    zero.add_argument(
        "--model",
        choices=["openai/whisper-small", "openai/whisper-large-v3", "facebook/mms-1b-all"],
        default="openai/whisper-small",
    )
    zero.add_argument(
        "--external", nargs="*", choices=["psrb_sample", "youtube_timestamps"], default=[]
    )
    zero.add_argument("--external-only", action="store_true")
    external = sub.add_parser(
        "evaluate-external", help="Evaluate completed checkpoints; never train"
    )
    external.add_argument(
        "--datasets", nargs="+", choices=["psrb_sample", "youtube_timestamps"], required=True
    )
    external.add_argument(
        "--families",
        nargs="+",
        choices=["matched", "mixture_updates"],
        default=["matched", "mixture_updates"],
    )
    analysis = sub.add_parser("analyze", help="Compute fresh-run paired WER/CER uncertainty")
    analysis.add_argument("--replicates", type=int, default=20000)
    sub.add_parser(
        "stratify", help="Score automatic register/entity subsets from saved predictions"
    )
    sub.add_parser("tables", help="Generate paper tables from fresh results")
    sub.add_parser("entities", help="Recompute entity word/character SD rates and overall WER/CER")
    verify = sub.add_parser("verify-validation", help="Check public speaker and rater evidence")
    verify.add_argument("--replicates", type=int, default=20000)
    export = sub.add_parser(
        "export-validation", help="Export pseudonymous evidence from private platform records"
    )
    export.add_argument("--raw-export", type=Path, required=True)
    export.add_argument("--key-file", type=Path, required=True)
    export.add_argument("--owners-json", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        from .manifests import prepare

        print(prepare().to_string(index=False))
    elif args.command == "run":
        from .training import execute, experiment_grid, plan

        runs = experiment_grid(args.family, args.seeds, args.architectures)
        print(plan(runs).to_string(index=False))
        if not args.dry_run:
            execute(runs)
    elif args.command == "prepare-psrb":
        from .external import prepare_psrb

        result = prepare_psrb(args.download)
        print({k: result[k] for k in ("name", "clips", "hours", "sha256")})
    elif args.command == "prepare-youtube":
        from .external import prepare_youtube

        result = prepare_youtube(args.episodes, args.audio_root, args.revision)
        print({k: result[k] for k in ("name", "clips", "hours", "sha256")})
    elif args.command == "zero-shot":
        from .training import evaluate_zero_shot

        print(evaluate_zero_shot(args.model, args.external, not args.external_only)["results"])
    elif args.command == "evaluate-external":
        from .protocol import ROOT
        from .training import evaluate_run, experiment_grid

        runs = [run for family in args.families for run in experiment_grid(family)]
        missing = [
            run.name
            for run in runs
            if not (ROOT / "checkpoints/v2" / run.name / "complete.json").exists()
        ]
        if missing:
            raise ValueError("Complete the requested training grid first: " + ", ".join(missing))
        for run in runs:
            evaluate_run(run, ROOT / "checkpoints/v2" / run.name, args.datasets, internal=False)
    elif args.command == "analyze":
        from .analysis import analyze_fresh

        print(analyze_fresh(args.replicates))
    elif args.command == "stratify":
        from .stratification import stratify_fresh

        print(stratify_fresh())
    elif args.command == "tables":
        from .reporting import generate_paper_tables

        generate_paper_tables()
        print("Generated fresh WER/CER tables in acl/")
    elif args.command == "entities":
        from .entities import analyze_entities

        print(analyze_entities().to_string(index=False))
    elif args.command == "verify-validation":
        from .validation import verify_all

        print(verify_all(args.replicates))
    elif args.command == "export-validation":
        from .validation import export_evidence

        print(export_evidence(args.raw_export, args.key_file.read_bytes(), args.owners_json))


if __name__ == "__main__":
    main()
