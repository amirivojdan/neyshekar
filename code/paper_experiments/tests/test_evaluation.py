"""Regression tests for canonical decoding and external-source preparation."""

import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import soundfile as sf

from neyshekar_experiments.evaluation import (
    evaluate_sets,
    result_subset,
    saved_decode,
    select_subset,
)
from neyshekar_experiments.external import external_sets, freeze_external, psrb_rows, timestamp_rows
from neyshekar_experiments.training import ZERO_SHOT_MODELS, experiment_grid


class EvaluationTests(unittest.TestCase):
    def dataset(self):
        return SimpleNamespace(
            rows=pd.DataFrame({"id": ["a", "b"], "text": ["a", "b c"], "duration": [2.0, 8.0]})
        )

    def test_resume_and_subset_never_redecode(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "full.jsonl"
            full = saved_decode(path, self.dataset(), lambda: ["a x", "b"], {"model": "test"})

            def fail():
                self.fail("Attempted to decode a completed full test again")

            resumed = saved_decode(path, self.dataset(), fail, {"model": "test"})
            self.assertEqual(select_subset(resumed, ["b"]).hypothesis.tolist(), ["b"])
            pd.testing.assert_frame_equal(full, resumed)
            with self.assertRaises(ValueError):
                select_subset(full, ["missing"])
            with self.assertRaises(ValueError):
                select_subset(full, ["b", "b"])

    def test_reference_model_and_cache_changes_are_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "full.jsonl"
            saved_decode(path, self.dataset(), lambda: ["a", "b"], {"model": "test"})
            changed = self.dataset()
            changed.rows.loc[0, "text"] = "changed"
            with self.assertRaises(ValueError):
                saved_decode(path, changed, None, {"model": "test"})
            with self.assertRaises(ValueError):
                saved_decode(path, self.dataset(), None, {"model": "other"})
            path.write_text(path.read_text().replace('"hypothesis": "a"', '"hypothesis": "x"'))
            with self.assertRaises(ValueError):
                saved_decode(path, self.dataset(), None, {"model": "test"})

    def test_new_corpus_preserves_existing_scores_and_saved_sources(self):
        with (
            tempfile.TemporaryDirectory() as d,
            patch("neyshekar_experiments.evaluation.ROOT", Path(d)),
        ):
            evaluate_sets("run", {}, {}, {"ney_test": self.dataset()}, lambda ds: ["a", "b"], {"a"})
            evaluate_sets("run", {}, {}, {"external_test": self.dataset()}, lambda ds: ["x", "b"])
            result = json.loads((Path(d) / "results/v2/run.json").read_text())
            self.assertIn("ney_test_disjoint", result["results"])
            self.assertIn("external_test", result["results"])
            self.assertEqual(result_subset("run", "ney_test_disjoint").id.tolist(), ["b"])
            self.assertEqual(result_subset("run", "ney_test_disjoint").hypothesis.tolist(), ["b"])

    def test_decoder_length_mismatch_cannot_create_cache(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "full.jsonl"
            with self.assertRaises(ValueError):
                saved_decode(path, self.dataset(), lambda: ["a"], {})
            self.assertFalse(path.exists())

    def test_primary_model_and_three_seed_equal_update_control(self):
        self.assertIn("openai/whisper-small", ZERO_SHOT_MODELS)
        with patch("neyshekar_experiments.training.load_manifest", return_value={"clips": 50814}):
            runs = experiment_grid("mixture_updates")
        self.assertEqual(len(runs), 12)
        self.assertEqual({r.optimization_seed for r in runs}, {42, 43, 44})
        self.assertEqual({r.max_steps for r in runs}, {2382})
        self.assertEqual({r.condition for r in runs}, {"ney_double", "mixed_double"})


class ExternalTests(unittest.TestCase):
    def test_psrb_known_header_permutation(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "Labels.csv"
            with path.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(
                    ["audio_path", "audio_duration", "number_of_speakers", "text", "formality"]
                )
                writer.writerow(["Files/audio_1.wav", "سلام دنیا", "1.25", "2", "informal"])
            row = psrb_rows(path)[0]
            self.assertEqual(row["text"], "سلام دنیا")
            self.assertEqual(row["audio_duration"], "1.25")
            self.assertEqual(row["number_of_speakers"], "2")

    def test_youtube_last_segment_and_invalid_boundaries(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "transcript.csv"
            path.write_text("0:05\nسلام\n0:23\nدنیا\n")
            rows = timestamp_rows(path, "episode.wav", "episode", 31)
            self.assertEqual(
                [(r["start_seconds"], r["end_seconds"]) for r in rows], [(5, 23), (23, 31)]
            )
            self.assertEqual({r["recording_id"] for r in rows}, {"episode"})
            with self.assertRaises(ValueError):
                timestamp_rows(path, "episode.wav", "episode", 20)
            path.write_text("0:05\nسلام\n0:23\n")
            with self.assertRaises(ValueError):
                timestamp_rows(path, "episode.wav", "episode", 31)

    def test_audio_hash_and_full_segment_are_verified(self):
        with (
            tempfile.TemporaryDirectory() as d,
            patch("neyshekar_experiments.external.DATA", Path(d)),
            patch("neyshekar_experiments.external.ROOT", Path(d)),
        ):
            root = Path(d)
            sf.write(root / "a.wav", np.zeros(16000), 16000)
            rows = [{"id": "a", "text": "سلام", "audio_path": "a.wav"}]
            manifest = freeze_external("psrb_sample", rows, root, {"segmentation": "released"})
            self.assertEqual(manifest["hours"], 1 / 3600)
            ds = external_sets(["psrb_sample"])["psrb_sample"]
            self.assertEqual(len(ds[0]["audio"]), 16000)
            sf.write(root / "a.wav", np.ones(16000), 16000)
            with self.assertRaises(ValueError):
                external_sets(["psrb_sample"])


class ReportingTests(unittest.TestCase):
    def test_tables_include_zero_shot_cer_seeds_external_and_actual_hours(self):
        from neyshekar_experiments.reporting import generate_paper_tables

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "acl").mkdir()
            with (
                patch("neyshekar_experiments.evaluation.ROOT", root),
                patch("neyshekar_experiments.reporting.ROOT", root),
                patch(
                    "neyshekar_experiments.manifests.load_manifest",
                    return_value={"clips": 2, "hours": 1.234567},
                ),
            ):
                ds = SimpleNamespace(
                    rows=pd.DataFrame(
                        {"id": ["a", "b"], "text": ["a", "b c"], "duration": [2.0, 8.0]}
                    )
                )
                for seed in (42, 43, 44):
                    for arch in ("whisper", "ctc"):
                        for condition in ("ney_matched", "cv_matched"):
                            spec = {
                                "architecture": arch,
                                "condition": condition,
                                "optimization_seed": seed,
                                "budget": "epochs",
                                "max_steps": None,
                            }
                            evaluate_sets(
                                f"{arch}_{condition}_{seed}",
                                spec,
                                {},
                                {"ney_test": ds, "cv_test": ds, "psrb_sample": ds},
                                lambda ds: ["a", "b"],
                                {"a"},
                            )
                spec = {
                    "architecture": "whisper",
                    "model": "openai/whisper-small",
                    "condition": "zero_shot",
                    "optimization_seed": None,
                    "budget": "none",
                    "max_steps": None,
                }
                evaluate_sets(
                    "zero_shot_whisper-small",
                    spec,
                    {},
                    {"ney_test": ds, "cv_test": ds, "psrb_sample": ds},
                    lambda ds: ["a", "b"],
                    {"a"},
                )
                generate_paper_tables()
                table = (root / "acl/table4_generated.tex").read_text()
                self.assertIn("zero\\_shot", table)
                self.assertIn("CER", table)
                self.assertIn("n=3 seeds", table)
                self.assertIn(
                    "1.234567", (root / "acl/training_durations_generated.tex").read_text()
                )
                self.assertIn("psrb\\_sample", (root / "acl/external_generated.tex").read_text())

    def test_register_entity_strata_use_canonical_predictions(self):
        from neyshekar_experiments.stratification import stratify_fresh

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            with (
                patch("neyshekar_experiments.evaluation.ROOT", root),
                patch("neyshekar_experiments.reporting.ROOT", root),
                patch("neyshekar_experiments.stratification.ROOT", root),
                patch(
                    "shekar.InformalLanguageClassifier",
                    return_value=lambda t: ("label", int(t == "a")),
                ),
                patch("shekar.NER", return_value=lambda t: [("a", "PER")] if t == "a" else []),
            ):
                ds = SimpleNamespace(
                    rows=pd.DataFrame(
                        {"id": ["a", "b"], "text": ["a", "b c"], "duration": [2.0, 8.0]}
                    )
                )
                spec = {
                    "architecture": "whisper",
                    "condition": "ney_matched",
                    "optimization_seed": 42,
                    "budget": "epochs",
                    "max_steps": None,
                }
                evaluate_sets("run", spec, {}, {"ney_test": ds}, lambda ds: ["a x", "b"], {"a"})
                stratify_fresh()
                self.assertEqual(
                    result_subset("run", "ney_test__register_informal").hypothesis.tolist(), ["a x"]
                )
                self.assertEqual(result_subset("run", "ney_test__entity_absent").id.tolist(), ["b"])


class ZeroShotDataTests(unittest.TestCase):
    def test_empty_training_manifest_still_loads_internal_evaluation_sets(self):
        from neyshekar_experiments.training import datasets_for

        class Release(dict):
            def cast_column(self, *args):
                return self

        ney = pd.DataFrame(
            {
                "id": [1, 2],
                "text": ["a", "b"],
                "duration": [1.0, 2.0],
                "corpus": ["ney", "ney"],
                "split": ["test", "validation"],
            }
        )
        cv = {
            split: pd.DataFrame(
                {
                    "id": [split],
                    "text": ["a"],
                    "duration": [1.0],
                    "corpus": ["cv"],
                    "split": [
                        "train" if split == "train" else "test" if split == "test" else "validation"
                    ],
                }
            )
            for split in ("train", "test", "dev")
        }
        with (
            patch("neyshekar_experiments.training.frames", return_value=(ney, cv)),
            patch("datasets.load_dataset", return_value=Release()),
        ):
            train, dev, tests = datasets_for({"records": []})
        self.assertEqual(len(train), 0)
        self.assertEqual(set(tests), {"ney_test", "cv_test"})
        self.assertEqual(len(dev), 2)


if __name__ == "__main__":
    unittest.main()
