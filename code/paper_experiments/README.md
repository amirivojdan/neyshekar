# Paper experiments

Code backing the ASR experiments in *Neyshekar: A Dual-Register, Named-Entity-Rich
Persian Read-Speech Corpus*. It reproduces the training runs, the decodings, the
paired bootstrap intervals, and the corpus statistics reported there.

Everything here is the experiment pipeline itself. Manuscript typesetting, remote
job orchestration, and the private contributor mapping are deliberately not
included — see [Not included](#not-included).

## Layout

The package resolves its roots from its own location, not the working directory:

```
code/                       <- ROOT
├── data/                   <- DATA  (you create this; see below)
├── logs/                   <- suite logs
├── results/                <- run outputs, saved decodings, analysis JSON
├── checkpoints/            <- trained models
└── paper_experiments/      <- CODE  (this directory)
    ├── neyshekar_experiments/   shared package
    ├── scripts/                 suite runners and preflight checks
    ├── tests/
    └── *.ipynb                  analysis notebooks
```

Run the notebooks from this directory; they resolve imports from the directory
that contains `neyshekar_experiments/`. Outputs always land under `ROOT`
regardless of where a process is launched.

## Data

Create `code/data/` and populate it before running anything:

- `data/cv26/cv-corpus-26.0-2026-06-12/fa/` — Common Voice 26.0 Persian TSVs and
  extracted audio.
- `data/external/psrb/` — populated by `prepare-psrb --download`.
- `data/manifests/v2/` — written by `prepare`.

The Neyshekar audio and pinned pretrained checkpoints are pulled from Hugging
Face at the revisions pinned in `neyshekar_experiments/protocol.py`.

## Environment

Python 3.12, plus an NVIDIA driver and a CUDA-enabled PyTorch. Training fails
explicitly when CUDA is unavailable; no experiment starts merely by importing the
package or opening a notebook.

The checked-in lock resolves for **Linux aarch64, Python 3.12** — the platform the
published runs were trained on:

```bash
python3.12 -m venv .venv
.venv/bin/pip install --require-hashes -r requirements-linux-aarch64-py312.lock
.venv/bin/python -m unittest discover -s tests
```

On any other architecture, regenerate the lock first and install that file
instead:

```bash
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python scripts/lock_dependencies.py
```

Note that `requirements*` file digests are folded into each run's identity hash,
so a regenerated lock produces different run IDs than the published ones. The
scientific results do not depend on this; the identity check does.

## Reproducing the experiments

`prepare` freezes clip-ID manifests; every later grid reuses them, so seeds 42/43/44
differ only in optimisation. Both architectures share one 20-second training
eligibility filter applied before hour matching, and every CTC condition uses the
same a priori alphabet.

```bash
.venv/bin/python -m neyshekar_experiments prepare
.venv/bin/python -m neyshekar_experiments run matched --dry-run
.venv/bin/python -m neyshekar_experiments run matched
.venv/bin/python -m neyshekar_experiments run mixture
.venv/bin/python -m neyshekar_experiments run updates
.venv/bin/python -m neyshekar_experiments run mixture_updates
.venv/bin/python -m neyshekar_experiments run scaling --seeds 42
.venv/bin/python -m neyshekar_experiments run scaling_updates --seeds 42
```

Which grid backs which result:

| Grid | Paper |
| --- | --- |
| `matched` | 32h duration-matched comparison, three epochs |
| `updates` | the same comparison at 1,425 updates |
| `mixture` | 64h Neyshekar-only vs mixed, three epochs |
| `mixture_updates` | the 64h control at 2,382 updates |
| `scaling`, `scaling_updates` | 5h→91.52h curves under both budgets |

Equal updates are not equal FLOPs or equal audio exposure.

### Zero-shot references and PSRB

```bash
.venv/bin/python -m neyshekar_experiments zero-shot
.venv/bin/python -m neyshekar_experiments zero-shot --model openai/whisper-large-v3
.venv/bin/python -m neyshekar_experiments zero-shot --model facebook/mms-1b-all
.venv/bin/python -m neyshekar_experiments prepare-psrb --download
.venv/bin/python -m neyshekar_experiments zero-shot --external psrb_sample
.venv/bin/python -m neyshekar_experiments evaluate-external --datasets psrb_sample
```

`prepare-psrb` pins [PartAI's public sample](https://huggingface.co/datasets/PartAI/PSRB)
to revision `626745e790667d6cbf70bdb10a260d3af90ed2d2`, corrects that revision's
misordered CSV header, and freezes durations and hashes. Results describe this
344-clip sample, not the full benchmark. External data never enters a training
manifest or development set.

`evaluate-external` needs completed `matched` and `mixture_updates` checkpoints
for all three seeds and fails if any are missing; pass `--families matched` to
score only the matched grid.

### Analysis

```bash
.venv/bin/python -m neyshekar_experiments analyze     # paired bootstrap intervals
.venv/bin/python -m neyshekar_experiments entities    # entity word/character S-D rates
.venv/bin/python -m neyshekar_experiments tables      # LaTeX tables -> ROOT/acl/
.venv/bin/python scripts/run_corpus_statistics.py     # corpus statistics
```

Each system writes one full-test JSONL per corpus with a provenance/hash sidecar.
Every stratified and text-disjoint number in the paper is sliced from those saved
decodings rather than re-decoded, so subset results cannot drift from the totals.
Resuming reuses a saved decoding and refuses one whose references, metadata, model
provenance, or file contents have changed.

### Running the whole suite

```bash
.venv/bin/python scripts/preflight_experiments.py     # check before committing GPU time
.venv/bin/python scripts/run_experiment_suite.py      # sequential, single GPU
.venv/bin/python scripts/run_parallel_suite.py        # one task per GPU
```

The parallel runner gives each task exactly one writer and one visible GPU, and
starts analysis only after every training and evaluation task succeeds. It does
not bypass the package's provenance checks.

## Notebooks

| Notebook | Contents |
| --- | --- |
| `statistics.ipynb` | corpus statistics, automatic annotations, rater agreement |
| `baselines.ipynb` | duration-matched training and in-domain results |
| `seed_variability.ipynb` | per-seed repetitions and WER/CER uncertainty |
| `corpus_mixture.ipynb` | mixture and equal-update controls |
| `scaling.ipynb` | equal-epoch and equal-update learning curves |
| `external_evaluation.ipynb` | zero-shot and PSRB workflow |
| `entity_error_attribution.ipynb` | reference-side entity error attribution |

Outputs are cleared and training cells are disabled by default; set
`RUN_TRAINING = True`, or use the CLI above.

## Not included

- **Remote orchestration.** The published runs were executed on a private
  multi-GPU host. Those sync, upload, and monitoring scripts carry that host's
  address and key path and are of no use elsewhere. `run_parallel_suite.py`
  covers the same multi-GPU scheduling locally.
- **Manuscript typesetting.** The scripts that assemble and compile the paper
  are not experiment code. `tables` still emits the LaTeX result tables.
- **The item-level contributor mapping** and the MongoDB exporter that produces
  it. As stated in the paper, partitions are speaker-disjoint by construction but
  the mapping itself is not released.
- **The Persian YouTube condition.** It is out of scope for the paper, so its
  acquisition scripts are absent. The `prepare-youtube` and `youtube_timestamps`
  code paths remain in the package because run-identity digests cover package
  sources; the suite reports a missing YouTube manifest as pending rather than
  failing.

## Tests

```bash
.venv/bin/python -m unittest discover -s tests
```

51 tests covering scoring, gradient accumulation, evaluation resume semantics,
artifact hardening, and reproducibility. `test_parallel_suite.py` reads the frozen
manifests, so run `prepare` (or provide `data/manifests/v2/`) before expecting a
full pass.
