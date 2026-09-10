"""One training/evaluation implementation for baselines, seeds and controls.

Importing this module never starts training. New results live under v2; existing
checkpoints cannot silently satisfy the corrected protocol.
"""

from __future__ import annotations

import importlib.metadata
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from .manifests import frames, load_manifest
from .protocol import (
    DATA,
    DATA_SEED,
    EFFECTIVE_BATCH,
    EPOCHS,
    HF_REPO,
    HF_REVISION,
    MAX_TRAIN_SECONDS,
    MODEL_REVISIONS,
    ROOT,
    SAMPLE_RATE,
    SCORING,
    digest,
    environment_spec_hashes,
    file_digest,
    fixed_ctc_vocabulary,
    normalise_for_scoring,
    write_json,
)


@dataclass(frozen=True)
class Run:
    architecture: str
    condition: str
    optimization_seed: int = 42
    budget: str = "epochs"
    max_steps: int | None = None

    def __post_init__(self):
        if self.architecture not in ("whisper", "ctc"):
            raise ValueError("architecture must be whisper or ctc")
        if self.budget not in ("epochs", "updates"):
            raise ValueError("budget must be epochs or updates")
        if (self.budget == "updates") != (self.max_steps is not None):
            raise ValueError("Only update-budget runs require max_steps")
        if self.max_steps is not None and self.max_steps < 1:
            raise ValueError("max_steps must be positive")

    @property
    def name(self):
        suffix = f"steps{self.max_steps}" if self.budget == "updates" else f"epochs{EPOCHS}"
        return f"{self.architecture}_{self.condition}_seed{self.optimization_seed}_{suffix}"


def experiment_grid(family="matched", seeds=(42, 43, 44), architectures=("whisper", "ctc")):
    """Small, explicit grids; no implicit training or test-based selection."""
    conditions = {
        "matched": ("ney_matched", "cv_matched"),
        "mixture": ("ney_double", "mixed_double", "mixed_matched"),
        "scaling": ("ney_5h", "ney_10h", "ney_20h", "ney_40h", "ney_full"),
        "updates": ("ney_matched", "cv_matched", "mixed_matched"),
        "mixture_updates": ("ney_double", "mixed_double"),
        "scaling_updates": ("ney_5h", "ney_10h", "ney_20h", "ney_40h", "ney_full"),
    }
    if family not in conditions:
        raise ValueError(f"Unknown family: {family}")
    steps = None
    if "updates" in family:
        anchor = "mixed_double" if family == "mixture_updates" else "cv_matched"
        steps = EPOCHS * math.ceil(load_manifest(anchor)["clips"] / EFFECTIVE_BATCH)
    if family.startswith("scaling"):
        architectures = tuple(a for a in architectures if a == "whisper")
    return [
        Run(a, c, seed, "updates" if steps else "epochs", steps)
        for a in architectures
        for c in conditions[family]
        for seed in seeds
    ]


def plan(runs) -> pd.DataFrame:
    rows = []
    for run in runs:
        m = load_manifest(run.condition)
        steps = run.max_steps or EPOCHS * math.ceil(m["clips"] / EFFECTIVE_BATCH)
        complete = ROOT / "checkpoints/v2" / run.name / "complete.json"
        rows.append(
            {
                **asdict(run),
                "run": run.name,
                "clips": m["clips"],
                "hours": m["hours"],
                "planned_updates": steps,
                "status": "checkpoint marker present; validate before reuse"
                if complete.exists()
                else "pending training",
            }
        )
    return pd.DataFrame(rows)


def require_gpu():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Corrected runs are pending; use --dry-run to inspect the plan."
        )
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError(
            "This protocol uses one GPU per run. Do not launch it with distributed workers."
        )


def ctc_decode_ids(logits, lengths, processor):
    """Crop padded output frames before CTC collapse/decoding."""
    ids = logits.argmax(-1).detach().cpu().numpy()
    lengths = lengths.detach().cpu().tolist()
    return [processor.decode(row[: int(length)]) for row, length in zip(ids, lengths)]


class SpeechData:
    """Manifest records map to release IDs/paths, never to a fresh random subset."""

    def __init__(self, rows, audio_by_id=None):
        self.rows = rows[["id", "text", "duration", "corpus", "split"]].reset_index(drop=True)
        self.audio_by_id = audio_by_id

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        import soundfile as sf

        r = self.rows.iloc[index]
        if r.corpus == "cv":
            audio, sr = sf.read(DATA / "cv26/clips" / r.id, dtype="float32")
        else:
            source = self.audio_by_id[int(r.id)]
            if isinstance(source, dict):
                audio, sr = source["array"], source["sampling_rate"]
            else:
                samples = source.get_all_samples()
                audio, sr = samples.data.numpy(), samples.sample_rate
                if audio.ndim == 2:
                    audio = audio.mean(axis=0)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != SAMPLE_RATE:
            import librosa

            audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
        return {"audio": np.asarray(audio, dtype=np.float32), "text": r.text}


class AudioIndex:
    def __init__(self, splits):
        self.splits = splits
        self.index = {
            int(cid): (split, i) for split, ds in splits.items() for i, cid in enumerate(ds["id"])
        }

    def __getitem__(self, cid):
        split, i = self.index[cid]
        return self.splits[split][i]["audio"]


def datasets_for(manifest):
    from datasets import Audio, load_dataset

    ney, cv = frames()
    release = load_dataset(HF_REPO, revision=HF_REVISION)
    release = release.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))
    audio = AudioIndex(release)
    metadata = {
        (r.corpus, r.id): r._asdict()
        for r in pd.concat([ney, *cv.values()], ignore_index=True).itertuples(index=False)
    }
    selected = []
    for item in manifest["records"]:
        r = metadata[(item["corpus"], item["id"])]
        if digest(r["text"]) != item["text_sha256"] or abs(r["duration"] - item["duration"]) > 1e-8:
            raise ValueError("Training records differ from the frozen manifest")
        if r["split"] != "train":
            raise ValueError("Evaluation data found in training manifest")
        selected.append(r)
    train = SpeechData(
        pd.DataFrame(selected, columns=["id", "text", "duration", "corpus", "split"]), audio
    )
    evaluations = {
        "ney_test": SpeechData(ney[ney.split == "test"], audio),
        "cv_test": SpeechData(cv["test"], audio),
    }
    # Same development population for all conditions; never select on test.
    dev = pd.concat([ney[ney.split == "validation"], cv["dev"]], ignore_index=True)
    validation = SpeechData(dev[dev.duration <= MAX_TRAIN_SECONDS], audio)
    return train, validation, evaluations


@dataclass
class Collator:
    processor: object
    architecture: str
    decoder_start_token_id: int | None = None

    def __call__(self, features):
        kwargs = {"padding": True} if self.architecture == "ctc" else {}
        batch = self.processor.feature_extractor(
            [f["audio"] for f in features], sampling_rate=SAMPLE_RATE, return_tensors="pt", **kwargs
        )
        texts = [
            normalise_for_scoring(f["text"]) if self.architecture == "ctc" else f["text"]
            for f in features
        ]
        tokens = self.processor.tokenizer(texts, padding=True, return_tensors="pt")
        labels = tokens.input_ids.masked_fill(tokens.attention_mask.ne(1), -100)
        if self.architecture == "whisper":
            if (labels[:, 0] == self.decoder_start_token_id).all():
                labels = labels[:, 1:]
            if labels.shape[1] > 448:
                raise ValueError("Whisper target exceeds context; do not silently truncate labels")
        batch["labels"] = labels
        return batch


def make_ctc_processor(directory):
    from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor

    vocabulary = fixed_ctc_vocabulary()
    path = directory / "vocab.json"
    if path.exists() and json.loads(path.read_text()) != vocabulary:
        raise ValueError("Existing CTC vocabulary is incompatible; retrain in the v2 directory")
    write_json(path, vocabulary)
    tokenizer = Wav2Vec2CTCTokenizer(
        str(path), unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|"
    )
    extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=SAMPLE_RATE,
        padding_value=0,
        do_normalize=True,
        return_attention_mask=True,
    )
    return Wav2Vec2Processor(feature_extractor=extractor, tokenizer=tokenizer)


def fingerprint(run, manifest):
    return {
        "run": asdict(run),
        "manifest_sha256": manifest["sha256"],
        "data_seed": DATA_SEED,
        "model_revisions": MODEL_REVISIONS,
        "environment_spec_sha256": environment_spec_hashes(),
        "requirements_sha256": file_digest(
            Path(__file__).resolve().parents[1] / "requirements.txt"
        ),
        "vocabulary_sha256": digest(fixed_ctc_vocabulary()) if run.architecture == "ctc" else None,
        "source_sha256": {
            p.name: file_digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))
        },
        "environment": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "datasets", "shekar", "jiwer", "numpy")
        },
        "epochs": EPOCHS,
        "effective_batch": EFFECTIVE_BATCH,
        "loss_accumulation": "mean microbatch losses divided by actual accumulation steps",
        "scoring": SCORING,
        "decoding": "greedy; batch=1; CTC valid frames, unsegmented; Whisper long-form, max_new_tokens=440; full-test once",
    }


def configure_mean_loss_model(model):
    """These ASR forwards return mean losses and ignore num_items_in_batch.

    New Trainer versions infer support from **kwargs, which would otherwise skip
    division by gradient accumulation steps and multiply the effective gradient.
    """
    model.accepts_loss_kwargs = False
    return model


def train_run(run: Run):
    import torch
    from transformers import (
        AutoModelForCTC,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        Trainer,
        TrainingArguments,
        WhisperForConditionalGeneration,
        WhisperProcessor,
        set_seed,
    )

    require_gpu()
    manifest = load_manifest(run.condition)
    provenance = fingerprint(run, manifest)
    directory = ROOT / "checkpoints/v2" / run.name
    directory.mkdir(parents=True, exist_ok=True)
    provenance_path = directory / "provenance.json"
    if provenance_path.exists() and json.loads(provenance_path.read_text()) != provenance:
        raise ValueError(
            f"Run provenance changed: {directory}; version the run instead of reusing it"
        )
    write_json(provenance_path, provenance)
    marker = directory / "complete.json"
    if marker.exists():
        if json.loads(marker.read_text())["provenance_sha256"] != digest(provenance):
            raise ValueError("Invalid checkpoint completion marker")
        return directory
    train, validation, _ = datasets_for(manifest)
    # Reset immediately before creating the pretrained model and its random head.
    set_seed(run.optimization_seed)
    if run.architecture == "whisper":
        base = "openai/whisper-small"
        processor = WhisperProcessor.from_pretrained(
            base, revision=MODEL_REVISIONS[base], language="persian", task="transcribe"
        )
        model = WhisperForConditionalGeneration.from_pretrained(
            base, revision=MODEL_REVISIONS[base]
        )
        model.config.forced_decoder_ids = None
        model.generation_config.forced_decoder_ids = None
        model.generation_config.language = "fa"
        model.generation_config.task = "transcribe"
        batch_size, lr = 32, 1e-5
        trainer_class, arguments_class = Seq2SeqTrainer, Seq2SeqTrainingArguments
    else:
        base = "facebook/wav2vec2-xls-r-300m"
        processor = make_ctc_processor(directory)
        vocabulary = fixed_ctc_vocabulary()
        unsupported = sorted(
            {
                c
                for text in train.rows.text
                for c in normalise_for_scoring(text)
                if c != " " and c not in vocabulary
            }
        )
        write_json(
            directory / "alphabet_audit.json", {"unsupported_training_characters": unsupported}
        )
        if unsupported:
            raise ValueError(
                "Training contains unsupported characters; review the versioned alphabet policy"
            )
        set_seed(run.optimization_seed)
        model = AutoModelForCTC.from_pretrained(
            base,
            revision=MODEL_REVISIONS[base],
            vocab_size=len(vocabulary),
            pad_token_id=vocabulary["[PAD]"],
            ctc_loss_reduction="mean",
            ignore_mismatched_sizes=True,
        )
        model.freeze_feature_encoder()
        batch_size, lr = 16, 3e-4
        trainer_class, arguments_class = Trainer, TrainingArguments
    configure_mean_loss_model(model)
    total_steps = run.max_steps or EPOCHS * math.ceil(len(train) / EFFECTIVE_BATCH)
    bf16 = torch.cuda.is_bf16_supported()
    args = arguments_class(
        output_dir=str(directory),
        remove_unused_columns=False,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=EFFECTIVE_BATCH // batch_size,
        learning_rate=lr,
        warmup_steps=max(1, int(0.1 * total_steps)),
        num_train_epochs=EPOCHS,
        max_steps=run.max_steps or -1,
        bf16=bf16,
        fp16=not bf16,
        dataloader_num_workers=0,
        logging_steps=25,
        save_steps=500,
        save_total_limit=1,
        eval_strategy="steps",
        eval_steps=500,
        report_to=[],
        seed=run.optimization_seed,
        data_seed=run.optimization_seed,
    )
    trainer = trainer_class(
        model=model,
        args=args,
        train_dataset=train,
        eval_dataset=validation,
        data_collator=Collator(
            processor, run.architecture, getattr(model.config, "decoder_start_token_id", None)
        ),
    )
    checkpoints = sorted(directory.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    started = time.time()
    trainer.train(resume_from_checkpoint=str(checkpoints[-1]) if checkpoints else None)
    final_dev = trainer.evaluate()
    trainer.save_model(str(directory))
    processor.save_pretrained(str(directory))
    trainer.state.save_to_json(str(directory / "trainer_state.json"))
    write_json(
        marker,
        {
            "provenance_sha256": digest(provenance),
            "global_step": trainer.state.global_step,
            "elapsed_seconds_this_invocation": time.time() - started,
            "clips": manifest["clips"],
            "hours": manifest["hours"],
            "dev_loss": final_dev["eval_loss"],
        },
    )
    del trainer, model
    torch.cuda.empty_cache()
    return directory


def transcribe(model, processor, dataset, architecture, batch_size=1):
    import torch

    hypotheses = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(dataset), batch_size):
            batch = [dataset[i] for i in range(start, min(start + batch_size, len(dataset)))]
            if architecture == "ctc":
                inputs = processor.feature_extractor(
                    [b["audio"] for b in batch],
                    sampling_rate=SAMPLE_RATE,
                    padding=True,
                    return_tensors="pt",
                )
                mask = inputs.attention_mask.to(model.device)
                logits = model(
                    inputs.input_values.to(model.device, model.dtype), attention_mask=mask
                ).logits
                lengths = model._get_feat_extract_output_lengths(mask.sum(-1))
                hypotheses.extend(ctc_decode_ids(logits, lengths, processor))
            else:
                # Preserve >30s evaluation audio; use the model's long-form path.
                long_form = max(len(b["audio"]) for b in batch) > 30 * SAMPLE_RATE
                inputs = processor.feature_extractor(
                    [b["audio"] for b in batch],
                    sampling_rate=SAMPLE_RATE,
                    return_tensors="pt",
                    padding="longest" if long_form else "max_length",
                    truncation=False,
                    return_attention_mask=True,
                )
                feats = inputs.input_features.to(model.device, model.dtype)
                model.generation_config.max_length = None
                ids = model.generate(
                    feats,
                    attention_mask=inputs.attention_mask.to(model.device),
                    language="fa",
                    task="transcribe",
                    max_new_tokens=440,
                    return_timestamps=feats.shape[-1] > 3000,
                    num_beams=1,
                    do_sample=False,
                )
                hypotheses.extend(
                    processor.batch_decode(
                        ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )
                )
            print(f"Decoded {min(start + batch_size, len(dataset))}/{len(dataset)}", flush=True)
    return hypotheses


def evaluate_run(run: Run, directory: Path, external=(), internal=True):
    import torch
    from transformers import (
        AutoModelForCTC,
        AutoProcessor,
        WhisperForConditionalGeneration,
        WhisperProcessor,
    )

    require_gpu()
    manifest = load_manifest(run.condition)
    provenance = fingerprint(run, manifest)
    marker = json.loads((directory / "complete.json").read_text())
    if marker["provenance_sha256"] != digest(provenance):
        raise ValueError("Checkpoint and requested evaluation provenance differ")
    from .evaluation import evaluate_sets
    from .external import external_sets

    sets = datasets_for(manifest)[2] if internal else {}
    sets.update(external_sets(external))
    if not sets:
        raise ValueError("Select at least one evaluation corpus")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    if run.architecture == "whisper":
        model = WhisperForConditionalGeneration.from_pretrained(directory, dtype=dtype).to("cuda")
        processor = WhisperProcessor.from_pretrained(
            directory, language="persian", task="transcribe"
        )
    else:
        model = AutoModelForCTC.from_pretrained(directory, dtype=dtype).to("cuda")
        processor = AutoProcessor.from_pretrained(directory)
    ney, _ = frames()
    train_text = {normalise_for_scoring(t) for t in ney.loc[ney.split == "train", "text"]}
    result = evaluate_sets(
        run.name,
        asdict(run),
        provenance,
        sets,
        partial(transcribe, model, processor, architecture=run.architecture, batch_size=1),
        train_text,
    )
    del model
    torch.cuda.empty_cache()
    return result


def execute(runs):
    require_gpu()
    return [evaluate_run(run, train_run(run)) for run in runs]


ZERO_SHOT_MODELS = ("openai/whisper-small", "openai/whisper-large-v3", "facebook/mms-1b-all")


def evaluate_zero_shot(base="openai/whisper-small", external=(), internal=True):
    """No adaptation: same scoring and saved-test decoder as fine-tuned systems."""
    import torch
    from transformers import (
        AutoModelForCTC,
        AutoProcessor,
        WhisperForConditionalGeneration,
        WhisperProcessor,
        set_seed,
    )

    from .evaluation import evaluate_sets
    from .external import external_sets

    if base not in ZERO_SHOT_MODELS:
        raise ValueError("Unsupported zero-shot reference")
    require_gpu()
    set_seed(42)
    # Empty training selection only loads the release's evaluation splits.
    sets = datasets_for({"records": []})[2] if internal else {}
    sets.update(external_sets(external))
    if not sets:
        raise ValueError("Select at least one evaluation corpus")
    revision = MODEL_REVISIONS[base]
    architecture = "ctc" if base == "facebook/mms-1b-all" else "whisper"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    if architecture == "whisper":
        processor = WhisperProcessor.from_pretrained(
            base, revision=revision, language="persian", task="transcribe"
        )
        model = WhisperForConditionalGeneration.from_pretrained(
            base, revision=revision, dtype=dtype
        ).to("cuda")
    else:
        processor = AutoProcessor.from_pretrained(base, revision=revision)
        processor.tokenizer.set_target_lang("fas")
        model = AutoModelForCTC.from_pretrained(
            base, revision=revision, target_lang="fas", ignore_mismatched_sizes=True, dtype=dtype
        ).to("cuda")
    spec = {
        "architecture": architecture,
        "model": base,
        "condition": "zero_shot",
        "optimization_seed": None,
        "budget": "none",
        "max_steps": None,
    }
    provenance = {
        "model": base,
        "revision": revision,
        "scoring": SCORING,
        "environment_spec_sha256": environment_spec_hashes(),
        "requirements_sha256": file_digest(
            Path(__file__).resolve().parents[1] / "requirements.txt"
        ),
        "source_sha256": {
            p.name: file_digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))
        },
        "environment": {
            n: importlib.metadata.version(n)
            for n in ("torch", "transformers", "datasets", "shekar", "jiwer", "numpy")
        },
        "decoding": "greedy; batch=1; full-test once; unsegmented CTC; Whisper long-form",
    }
    ney, _ = frames()
    train_text = {normalise_for_scoring(t) for t in ney.loc[ney.split == "train", "text"]}
    result = evaluate_sets(
        "zero_shot_" + base.split("/")[-1],
        spec,
        provenance,
        sets,
        partial(transcribe, model, processor, architecture=architecture, batch_size=1),
        train_text,
    )
    del model
    torch.cuda.empty_cache()
    return result
