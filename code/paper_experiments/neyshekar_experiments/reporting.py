"""Report only newly completed runs; never substitute old or inferred scores."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from .protocol import ROOT, file_digest

EVALS = ("ney_test", "ney_test_disjoint", "cv_test")
LABELS = {"ney_test": "Neyshekar", "ney_test_disjoint": "Text-disjoint", "cv_test": "Common Voice"}
COLUMNS = [
    "run",
    "system",
    "condition",
    "seed",
    "budget",
    "max_steps",
    "evaluation",
    "wer",
    "cer",
    "n",
    "n_total",
    "n_scored",
    "n_empty_reference",
]


def load_scores() -> pd.DataFrame:
    rows = []
    for path in sorted((ROOT / "results/v2").glob("*.json")):
        p = json.loads(path.read_text())
        if p.get("protocol") != "v2" or "results" not in p:
            continue
        for source, expected in p["hypothesis_sha256"].items():
            if file_digest(ROOT / source) != expected:
                raise ValueError(f"Changed predictions for {path.stem}")
        spec = p["run"]
        for evaluation, result in p["results"].items():
            rows.append(
                {
                    "run": path.stem,
                    "system": system_label(spec),
                    "condition": spec["condition"],
                    "seed": spec["optimization_seed"],
                    "budget": spec["budget"],
                    "max_steps": spec["max_steps"],
                    "evaluation": evaluation,
                    **{
                        key: result[key]
                        for key in ("wer", "cer", "n", "n_total", "n_scored", "n_empty_reference")
                    },
                }
            )
    return pd.DataFrame(rows, columns=COLUMNS)


MODEL_LABELS = {
    "openai/whisper-small": "Whisper small",
    "openai/whisper-large-v3": "Whisper large-v3",
    "facebook/mms-1b-all": "MMS-1B-all",
}
FAMILIES = {
    "matched": ("ney_matched", "cv_matched"),
    "mixture": ("ney_matched", "cv_matched", "mixed_matched", "ney_double", "mixed_double"),
    "scaling": ("ney_5h", "ney_10h", "ney_20h", "ney_40h", "ney_full"),
}
GROUP_COLUMNS = ["system", "condition", "budget", "max_steps"]


def system_label(spec):
    if spec.get("model") in MODEL_LABELS:
        return MODEL_LABELS[spec["model"]]
    return {"whisper": "Whisper small", "ctc": "XLS-R-300M"}[spec["architecture"]]


def corrected_results():
    return load_scores()


def seed_summary(scores=None):
    scores = load_scores() if scores is None else scores
    return scores.groupby(
        ["system", "condition", "budget", "max_steps", "evaluation"], dropna=False
    )[["wer", "cer"]].agg(["mean", "std", "count"])


def family_scores(family, scores=None):
    scores = load_scores() if scores is None else scores
    return scores.loc[scores.condition.isin(FAMILIES[family])].copy()


def metric_figure(frame, x="condition", output=None):
    import matplotlib.pyplot as plt
    import seaborn as sns

    if frame.empty:
        print("No completed runs yet; no figure generated.")
        return None
    # Do not average across different optimization budgets in a single curve.
    if frame[["budget", "max_steps"]].drop_duplicates().shape[0] > 1:
        raise ValueError("Choose one optimization budget per figure")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for metric, ax in zip(("wer", "cer"), axes):
        sns.lineplot(
            data=frame,
            x=x,
            y=metric,
            hue="evaluation",
            style="system",
            markers=True,
            dashes=False,
            errorbar=None,
            ax=ax,
        )
        ax.set_ylabel(metric.upper() + " (%)")
        ax.tick_params(axis="x", rotation=30)
    if output:
        fig.savefig(output, dpi=200, bbox_inches="tight")
    return fig


def latex_table(rows, caption, label, headers, colspec="llrrrrrr"):
    return "\n".join(
        [
            r"\begin{table*}[t]",
            r"\centering",
            rf"\caption{{{caption}}}",
            rf"\label{{{label}}}",
            r"\scriptsize",
            r"\setlength{\tabcolsep}{3pt}",
            rf"\begin{{tabular}}{{{colspec}}}",
            r"\toprule",
            *headers,
            r"\midrule",
            *[" & ".join(row) + r" \\" for row in rows],
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
            "",
        ]
    )


@dataclass(frozen=True)
class TableSpec:
    filename: str
    label: str
    family: str
    required_seeds: tuple[int, ...] = (42, 43, 44)
    include_zero_shot: bool = False
    per_seed: bool = False
    evaluation_prefixes: tuple[str, ...] = ()

    @property
    def detail(self):
        return bool(self.evaluation_prefixes)


TABLES = (
    TableSpec("table4_generated.tex", "tab:baselines", "matched", include_zero_shot=True),
    TableSpec("table5_generated.tex", "tab:scaling", "scaling", required_seeds=(42,)),
    TableSpec("table7_generated.tex", "tab:mixture", "mixture"),
    TableSpec("table8_generated.tex", "tab:seed-variability", "matched", per_seed=True),
    TableSpec(
        "external_generated.tex",
        "tab:external",
        "mixture",
        include_zero_shot=True,
        evaluation_prefixes=("psrb_", "youtube_"),
    ),
    TableSpec(
        "stratified_generated.tex", "tab:stratified", "matched", evaluation_prefixes=("ney_test__",)
    ),
)
CAPTION = (
    r"Fresh WER and CER (\%). Repeated runs are mean $\pm$ sample SD over "
    r"optimization seeds. Zero-shot references have no optimization seeds. "
    r"All subsets reuse full-test predictions. Saved scores include total, scored, "
    r"and empty-reference counts."
)


def escape_latex(value):
    replacements = {
        "_": r"\_",
        "&": r"\&",
        "%": r"\%",
        "#": r"\#",
        "{": r"\{",
        "}": r"\}",
        "$": r"\$",
        "\\": r"\textbackslash{}",
    }
    return "".join(replacements.get(char, char) for char in str(value))


def table_data(scores, spec):
    conditions = list(FAMILIES[spec.family])
    if spec.include_zero_shot:
        conditions.append("zero_shot")
    evaluations = (
        scores.evaluation.str.startswith(spec.evaluation_prefixes)
        if spec.detail
        else scores.evaluation.isin(EVALS)
    )
    return scores.loc[scores.condition.isin(conditions) & evaluations]


def validate_table(data, spec):
    for key, group in data.groupby(GROUP_COLUMNS, dropna=False):
        condition = key[1]
        if not spec.detail and set(group.evaluation) != set(EVALS):
            raise ValueError(f"Missing full-test evaluation in {spec.label}: {key}")
        for evaluation, subset in group.groupby("evaluation"):
            if condition == "zero_shot":
                valid = len(subset) == 1 and subset.seed.isna().all()
            else:
                valid = (
                    not subset.seed.isna().any()
                    and not subset.seed.duplicated().any()
                    and set(subset.seed) == set(spec.required_seeds)
                )
            if not valid:
                raise ValueError(
                    f"Incomplete or duplicate seeds in {spec.label}: {key}/{evaluation}"
                )
            for column in ("n", "n_total", "n_scored", "n_empty_reference"):
                if subset[column].nunique() != 1:
                    raise ValueError(f"Scoring populations differ across seeds: {key}/{evaluation}")


def metric_value(group, metric):
    if len(group) == 1:
        return f"{group[metric].iloc[0]:.2f}"
    return f"${group[metric].mean():.2f} \\pm {group[metric].std():.2f}$"


def training_label(condition, budget, steps, seed_count, seed=None):
    label = escape_latex(condition)
    if condition != "zero_shot":
        label += f", {int(steps)} updates" if budget == "updates" else ", 3 epochs"
        label += f", n={seed_count} seeds"
        if seed is not None:
            label += f", seed {int(seed)}"
    return label


def render_table(data, spec):
    columns = [*GROUP_COLUMNS]
    if spec.per_seed:
        columns.append("seed")
    if spec.detail:
        columns.append("evaluation")
    rows = []
    for keys, group in data.groupby(columns, dropna=False):
        system, condition, budget, steps = keys[:4]
        seed = keys[4] if spec.per_seed else None
        training = training_label(condition, budget, steps, group.seed.nunique(), seed)
        if spec.detail:
            values = [metric_value(group, metric) for metric in ("wer", "cer")]
            rows.append([system, training, escape_latex(keys[-1]), *values])
        else:
            values = []
            for evaluation in EVALS:
                subset = group[group.evaluation.eq(evaluation)]
                values.extend(metric_value(subset, metric) for metric in ("wer", "cer"))
            rows.append([system, training, *values])
    if spec.detail:
        headers = [r"System & Training / budget & Evaluation & WER & CER \\"]
        colspec = "lllrr"
    else:
        headers = [
            r"System & Training / budget & \multicolumn{2}{c}{Neyshekar test} "
            r"& \multicolumn{2}{c}{Text-disjoint} & \multicolumn{2}{c}{CV test} \\",
            r" & & WER & CER & WER & CER & WER & CER \\",
        ]
        colspec = "llrrrrrr"
    return latex_table(rows, CAPTION, spec.label, headers, colspec)


def render_duration_table():
    from .manifests import load_manifest

    names = (
        "ney_matched",
        "cv_matched",
        "mixed_matched",
        "ney_double",
        "mixed_double",
        "ney_5h",
        "ney_10h",
        "ney_20h",
        "ney_40h",
        "ney_full",
    )
    rows = []
    for name in names:
        manifest = load_manifest(name)
        rows.append([escape_latex(name), str(manifest["clips"]), f"{manifest['hours']:.6f}"])
    return latex_table(
        rows,
        "Actual post-filtering training durations from frozen manifests.",
        "tab:training-durations",
        [r"Condition & Clips & Hours \\"],
        "lrr",
    )


def publish_tables(rendered):
    """Stage complete output, then replace files; roll back on a write error.

    Validation/rendering precede this function. An abrupt process/filesystem crash
    during multi-file replacement is not a transaction; rerun generation then.
    """
    directory = ROOT / "acl"
    originals = {
        name: (directory / name).read_bytes() if (directory / name).exists() else None
        for name in rendered
    }
    with TemporaryDirectory(prefix=".tables-", dir=directory) as staging:
        for name, content in rendered.items():
            (Path(staging) / name).write_text(content, encoding="utf-8")
        replaced = []
        try:
            for name in rendered:
                (Path(staging) / name).replace(directory / name)
                replaced.append(name)
        except OSError:
            for name in replaced:
                if originals[name] is None:
                    (directory / name).unlink()
                else:
                    (directory / name).write_bytes(originals[name])
            raise


def generate_paper_tables():
    scores = load_scores()  # One verified snapshot for the whole report.
    if scores.empty:
        raise ValueError("No fresh results. Run the experiments before generating tables.")
    rendered = {}
    for spec in TABLES:
        data = table_data(scores, spec)
        if data.empty:
            continue
        validate_table(data, spec)
        rendered[spec.filename] = render_table(data, spec)
    rendered["training_durations_generated.tex"] = render_duration_table()
    summary = seed_summary(scores)
    publish_tables(rendered)
    return summary
