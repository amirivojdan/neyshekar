"""Behavioral checks for the failure modes identified in the paper audit."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from neyshekar_experiments.analysis import aggregate, error_counts, paired_bootstrap
from neyshekar_experiments.entities import attribution, token_labels
from neyshekar_experiments.manifests import freeze, manifest_payload, prefix
from neyshekar_experiments.protocol import (
    fixed_ctc_vocabulary,
    normalise_for_scoring,
    score,
)
from neyshekar_experiments.training import (
    Collator,
    Run,
    ctc_decode_ids,
    experiment_grid,
    make_ctc_processor,
)
from neyshekar_experiments.validation import reliability, verify_speakers


class ScoringTests(unittest.TestCase):
    def test_directional_formatting_controls_do_not_enter_ctc_targets(self):
        expected = normalise_for_scoring("سلام ۱۲۳")
        actual = normalise_for_scoring("\u2066سلام\u2069 \u200e۱۲۳\u200f")
        self.assertEqual(actual, expected)
        self.assertEqual(score(["سلام"], ["\u2066سلام\u2069"])["wer"], 0)

    def test_digit_scripts_share_support_without_verbalizing(self):
        self.assertEqual(normalise_for_scoring("۱۲۳"), "123")
        self.assertEqual(normalise_for_scoring("١٢٣"), "123")
        self.assertNotEqual(normalise_for_scoring("۱۸"), normalise_for_scoring("هجده"))
        self.assertEqual(score(["۱۲۳"], ["١٢٣"])["cer"], 0)

    def test_word_and_character_metrics_both_count_insertions(self):
        rows = pd.DataFrame({"id": [1, 2], "reference": ["a", "b c"], "hypothesis": ["a x", "b c"]})
        result = aggregate(error_counts(rows))
        expected = score(rows.reference.tolist(), rows.hypothesis.tolist())
        self.assertAlmostEqual(result["wer"], 100 / 3)
        self.assertEqual(result["n"], expected["n"])
        self.assertAlmostEqual(result["wer"], expected["wer"])
        self.assertAlmostEqual(result["cer"], expected["cer"])

    def test_mismatched_length_is_rejected(self):
        with self.assertRaises(ValueError):
            score(["a"], [])

    def test_empty_scoring_set_is_rejected(self):
        with self.assertRaises(ValueError):
            score([""], [""])

    def test_fixed_alphabet_is_independent_and_complete_for_digits(self):
        a, b = fixed_ctc_vocabulary(), fixed_ctc_vocabulary()
        self.assertEqual(a, b)
        self.assertTrue(set("0123456789پچژکگی").issubset(a))
        self.assertEqual(sorted(a.values()), list(range(len(a))))

    def test_actual_ctc_tokenizer_preserves_digits_and_rejects_legacy_vocab(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            processor = make_ctc_processor(p)
            tokens = processor.tokenizer("123").input_ids
            self.assertNotIn(processor.tokenizer.unk_token_id, tokens)
            (p / "vocab.json").write_text(json.dumps({"a": 0}))
            with self.assertRaises(ValueError):
                make_ctc_processor(p)

    def test_ctc_padding_is_not_decoded(self):
        class Processor:
            def decode(self, ids):
                return list(ids)

        logits = torch.tensor(
            [[[0.0, 1.0], [1.0, 0.0], [0.0, 10.0]], [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]]
        )
        self.assertEqual(ctc_decode_ids(logits, torch.tensor([2, 1]), Processor()), [[1, 0], [0]])

    def test_whisper_eos_survives_padding_mask(self):
        class Tokens:
            input_ids = torch.tensor([[9, 3, 0, 0], [9, 4, 5, 0]])
            attention_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]])

        class Processor:
            def feature_extractor(self, *a, **k):
                return {}

            def tokenizer(self, *a, **k):
                return Tokens()

        result = Collator(Processor(), "whisper", 9)([{"audio": np.zeros(10), "text": "a"}] * 2)
        self.assertEqual(result["labels"].tolist(), [[3, 0, -100], [4, 5, 0]])


class ManifestTests(unittest.TestCase):
    def test_freeze_rejects_drift(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "m.json"
            freeze(p, {"ids": [1, 2]})
            freeze(p, {"ids": [1, 2]})
            with self.assertRaises(ValueError):
                freeze(p, {"ids": [2, 1]})

    def test_prefix_is_nested_and_never_exceeds_budget(self):
        f = pd.DataFrame({"id": [1, 2, 3], "duration": [4, 5, 6]})
        small, large = prefix(f, 9 / 3600), prefix(f, 15 / 3600)
        self.assertEqual(small.id.tolist(), [1, 2])
        self.assertEqual(large.id.tolist(), [1, 2, 3])

    def test_manifest_captures_text_and_order(self):
        f = pd.DataFrame(
            {"id": [1, 2], "duration": [1.0, 2.0], "corpus": ["ney", "ney"], "text": ["a", "b"]}
        )
        a = manifest_payload("test", f)
        b = manifest_payload("test", f.iloc[::-1])
        self.assertNotEqual(a["sha256"], b["sha256"])

    def test_seed_repetitions_only_change_optimization(self):
        runs = experiment_grid("matched", seeds=(42, 43), architectures=("ctc",))
        self.assertEqual(runs[0].condition, runs[1].condition)
        self.assertNotEqual(runs[0].optimization_seed, runs[1].optimization_seed)
        self.assertNotEqual(runs[0].name, runs[1].name)

    def test_run_budget_cannot_be_ambiguous(self):
        with self.assertRaises(ValueError):
            Run("ctc", "ney_matched", budget="updates")
        with self.assertRaises(ValueError):
            Run("ctc", "ney_matched", max_steps=100)


class InferenceTests(unittest.TestCase):
    def setUp(self):
        self.a = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "reference": ["a b", "c d", "e f"],
                "hypothesis": ["a b", "c d", "e f"],
            }
        )
        self.b = self.a.copy()
        self.b["hypothesis"] = ["a x", "c x", "e x"]

    def test_pairing_is_by_id_not_row_order(self):
        a = paired_bootstrap(self.a, self.b, replicates=100)
        b = paired_bootstrap(self.a, self.b.iloc[::-1], replicates=100)
        self.assertEqual(a, b)
        self.assertEqual(a["wer"]["difference"], 50)
        self.assertGreater(a["cer"]["difference"], 0)

    def test_identical_systems_have_zero_intervals(self):
        r = paired_bootstrap(self.a, self.a, replicates=100)
        self.assertEqual(r["wer"]["ci95"], [0, 0])
        self.assertEqual(r["cer"]["two_sided_bootstrap_tail_p"], 1)

    def test_pairing_requires_same_references(self):
        b = self.b.copy()
        b.loc[0, "reference"] = "changed"
        with self.assertRaises(ValueError):
            paired_bootstrap(self.a, b, replicates=10)

    def test_cluster_bootstrap_counts_speakers(self):
        clusters = pd.Series({1: "speaker_a", 2: "speaker_a", 3: "speaker_b"})
        r = paired_bootstrap(self.a, self.b, clusters=clusters, replicates=100)
        self.assertEqual(r["n_resampling_units"], 2)
        self.assertEqual(r["n_utterances"], 3)

    def test_missing_cluster_id_is_not_silently_dropped(self):
        with self.assertRaises(ValueError):
            paired_bootstrap(self.a, self.b, clusters=pd.Series({1: "a"}), replicates=10)


class ValidationTests(unittest.TestCase):
    def test_actual_cross_split_speaker_leak_is_rejected(self):
        meta = pd.DataFrame({"id": [0, 1], "split": ["train", "test"]})
        m = pd.DataFrame(
            {"clip_id": [0, 1], "split": ["train", "test"], "speaker_id": ["same", "same"]}
        )
        with self.assertRaises(ValueError):
            verify_speakers(m, meta)
        m.loc[1, "speaker_id"] = "different"
        self.assertEqual(verify_speakers(m, meta)["speakers"], 2)

    def test_missing_release_mapping_is_rejected(self):
        meta = pd.DataFrame({"id": [0, 1], "split": ["train", "test"]})
        m = pd.DataFrame({"clip_id": [0], "split": ["train"], "speaker_id": ["a"]})
        with self.assertRaises(ValueError):
            verify_speakers(m, meta)

    def test_reliability_uses_actual_labels(self):
        labels = pd.DataFrame(
            {"item_id": [1, 1, 2, 2], "rater_id": ["a", "b", "a", "b"], "accepted": [1, 1, 0, 0]}
        )
        r = reliability(labels, replicates=100)
        self.assertEqual(r["fleiss_kappa"]["value"], 1)
        self.assertEqual(r["gwet_ac1"]["value"], 1)
        self.assertEqual(r["labels"], 4)

    def test_incomplete_rater_matrix_is_rejected(self):
        labels = pd.DataFrame(
            {"item_id": [1, 1, 2], "rater_id": ["a", "b", "a"], "accepted": [1, 0, 0]}
        )
        with self.assertRaises(ValueError):
            reliability(labels, replicates=10)


class EntityTests(unittest.TestCase):
    def test_repeated_entities_get_distinct_spans(self):
        labels, missing = token_labels(["a", "a"], [("a", "PER"), ("a", "LOC")])
        self.assertEqual(labels, ["PER", "LOC"])
        self.assertEqual(missing, 0)

    def test_insertions_are_in_overall_metrics_but_not_reference_sd(self):
        rows = pd.DataFrame({"id": [0], "reference": ["abc def"], "hypothesis": ["abc x def"]})
        r = attribution(rows, {0: [("abc", "PER")]})
        self.assertEqual(r["entity_word_sd"], 0)
        self.assertEqual(r["entity_character_sd"], 0)
        self.assertGreater(r["overall"]["wer"], 0)
        self.assertGreater(r["overall"]["cer"], 0)


if __name__ == "__main__":
    unittest.main()
