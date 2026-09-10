"""Exercise real GPU forward/backward and decoding before scheduling full runs.

Smoke-test weights are discarded. No smoke score is a research result.
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))


def main(trainer_check=False):
    import torch
    from transformers import (
        AutoModelForCTC,
        WhisperForConditionalGeneration,
        WhisperProcessor,
        set_seed,
    )

    from neyshekar_experiments.manifests import frames, load_manifest, prepare
    from neyshekar_experiments.protocol import (
        MODEL_REVISIONS,
        ROOT,
        fixed_ctc_vocabulary,
        normalise_for_scoring,
        write_json,
    )
    from neyshekar_experiments.training import (
        Collator,
        configure_mean_loss_model,
        datasets_for,
        make_ctc_processor,
        require_gpu,
        transcribe,
    )

    require_gpu()
    prepare()
    ney, cv = frames()
    vocabulary = fixed_ctc_vocabulary()
    texts = [
        *ney.loc[(ney.split == "train") & (ney.duration <= 20), "text"],
        *cv["train"].loc[cv["train"].duration <= 20, "text"],
    ]
    unsupported = sorted(
        {
            char
            for text in texts
            for char in normalise_for_scoring(text)
            if char != " " and char not in vocabulary
        }
    )
    if unsupported:
        raise ValueError(f"Unsupported CTC training characters: {unsupported}")
    train, _, _ = datasets_for(load_manifest("ney_matched"))
    samples = [train[i] for i in range(2)]
    report = {
        "kind": "GPU smoke test; not an experiment result",
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "architectures": {},
    }
    for architecture in ("whisper", "ctc"):
        set_seed(42)
        with tempfile.TemporaryDirectory(prefix="neyshekar-smoke-") as temporary:
            if architecture == "whisper":
                base = "openai/whisper-small"
                processor = WhisperProcessor.from_pretrained(
                    base, revision=MODEL_REVISIONS[base], language="persian", task="transcribe"
                )
                model = WhisperForConditionalGeneration.from_pretrained(
                    base, revision=MODEL_REVISIONS[base]
                )
                model.generation_config.forced_decoder_ids = None
            else:
                base = "facebook/wav2vec2-xls-r-300m"
                processor = make_ctc_processor(Path(temporary))
                model = AutoModelForCTC.from_pretrained(
                    base,
                    revision=MODEL_REVISIONS[base],
                    vocab_size=len(vocabulary),
                    pad_token_id=vocabulary["[PAD]"],
                    ctc_loss_reduction="mean",
                    ignore_mismatched_sizes=True,
                )
                model.freeze_feature_encoder()
            model.to("cuda").train()
            collator = Collator(
                processor, architecture, getattr(model.config, "decoder_start_token_id", None)
            )
            batch = {key: value.to("cuda") for key, value in collator(samples).items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise ValueError(f"Nonfinite {architecture} smoke loss")
            loss.backward()
            if any(
                not torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
                if parameter.grad is not None
            ):
                raise ValueError(f"Nonfinite {architecture} gradients")
            hypotheses = transcribe(model, processor, samples, architecture, batch_size=1)
            report["architectures"][architecture] = {
                "loss": float(loss.detach()),
                "decoded": len(hypotheses),
            }
            if trainer_check:
                from transformers import (
                    Seq2SeqTrainer,
                    Seq2SeqTrainingArguments,
                    Trainer,
                    TrainingArguments,
                )

                trainer_class = Seq2SeqTrainer if architecture == "whisper" else Trainer
                argument_class = (
                    Seq2SeqTrainingArguments if architecture == "whisper" else TrainingArguments
                )
                batch_size = 32 if architecture == "whisper" else 16
                configure_mean_loss_model(model)
                model.zero_grad(set_to_none=True)
                trainer = trainer_class(
                    model=model,
                    args=argument_class(
                        output_dir=str(Path(temporary) / "trainer"),
                        max_steps=1,
                        per_device_train_batch_size=batch_size,
                        per_device_eval_batch_size=batch_size,
                        gradient_accumulation_steps=64 // batch_size,
                        learning_rate=1e-5 if architecture == "whisper" else 3e-4,
                        bf16=True,
                        remove_unused_columns=False,
                        eval_strategy="steps",
                        eval_steps=1,
                        save_steps=1,
                        logging_steps=1,
                        report_to=[],
                        dataloader_num_workers=0,
                        seed=42,
                        data_seed=42,
                    ),
                    train_dataset=samples * 32,
                    eval_dataset=samples,
                    data_collator=collator,
                )
                assert trainer.model_accepts_loss_kwargs is False
                trainer.train()
                report["architectures"][architecture]["trainer_global_step"] = (
                    trainer.state.global_step
                )
                del trainer
            print(architecture, report["architectures"][architecture], flush=True)
            del model, loss, batch
            torch.cuda.empty_cache()
    write_json(ROOT / "results/preflight.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trainer",
        action="store_true",
        help="Also test one full-batch optimizer step, evaluation and checkpoint saving",
    )
    main(parser.parse_args().trainer)
