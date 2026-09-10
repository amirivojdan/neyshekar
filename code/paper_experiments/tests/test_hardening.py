"""Portability, exclusion accounting, cache invalidation, and report publication."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import soundfile as sf

from neyshekar_experiments.analysis import aggregate, error_counts, paired_bootstrap
from neyshekar_experiments.external import external_sets, freeze_external
from neyshekar_experiments.manifests import _frames_cached, frames
from neyshekar_experiments.protocol import score
from neyshekar_experiments.reporting import COLUMNS, generate_paper_tables


class PortabilityTests(unittest.TestCase):
    def test_manifest_survives_repository_move_without_rehashing(self):
        with tempfile.TemporaryDirectory() as temporary:
            first = Path(temporary) / "first"
            moved = Path(temporary) / "moved"
            audio = first / "data/external/audio"
            audio.mkdir(parents=True)
            sf.write(audio / "clip.wav", np.zeros(16000), 16000)
            rows = [{"id": "clip", "text": "سلام", "audio_path": "clip.wav"}]
            with (
                patch("neyshekar_experiments.external.ROOT", first),
                patch("neyshekar_experiments.external.DATA", first / "data"),
            ):
                original = freeze_external("psrb_sample", rows, audio, {"segmentation": "released"})
            self.assertEqual(original["audio_root"], "data/external/audio")
            first.rename(moved)
            with (
                patch("neyshekar_experiments.external.ROOT", moved),
                patch("neyshekar_experiments.external.DATA", moved / "data"),
            ):
                reused = freeze_external(
                    "psrb_sample", rows, moved / "data/external/audio", {"segmentation": "released"}
                )
                self.assertEqual(reused, original)
                self.assertEqual(
                    len(external_sets(["psrb_sample"])["psrb_sample"][0]["audio"]), 16000
                )


class ExclusionTests(unittest.TestCase):
    def test_score_aggregation_and_bootstrap_agree_on_empty_references(self):
        rows = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "reference": ["", "!!!", "a b"],
                "hypothesis": ["ignored", "also ignored", "a"],
            }
        )
        direct = score(rows.reference.tolist(), rows.hypothesis.tolist())
        counts = error_counts(rows)
        self.assertEqual(len(counts), 3)  # Dropped references remain explicit in counts.
        aggregated = aggregate(counts)
        boot = paired_bootstrap(rows, rows, replicates=20)
        for result in (direct, aggregated, boot):
            self.assertEqual(result["n_total"], 3)
            self.assertEqual(result["n_scored"], 1)
            self.assertEqual(result["n_empty_reference"], 2)
        self.assertEqual(direct["wer"], aggregated["wer"])
        self.assertAlmostEqual(direct["cer"], aggregated["cer"])
        self.assertEqual(boot["wer"]["difference"], 0)


class FrameCacheTests(unittest.TestCase):
    def test_cache_returns_copies_and_detects_same_size_timestamp_preserving_edits(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            meta = root / "neyshekar_v6_meta.parquet"
            pd.DataFrame(
                {"id": [1], "text": ["a"], "duration": [1.0], "split": ["train"]}
            ).to_parquet(meta)
            pd.DataFrame({"path": ["train", "dev", "test"], "ms": [1000] * 3}).to_csv(
                root / "clip_durations.tsv", sep="\t", index=False
            )
            for split in ("train", "dev", "test"):
                pd.DataFrame({"path": [split], "sentence": ["a"], "client_id": [split]}).to_csv(
                    root / f"{split}.tsv", sep="\t", index=False
                )
            _frames_cached.cache_clear()
            with (
                patch("neyshekar_experiments.manifests.DATA", root),
                patch("neyshekar_experiments.manifests.CV_DIR", root),
                patch(
                    "neyshekar_experiments.manifests.pd.read_parquet", wraps=pd.read_parquet
                ) as reader,
            ):
                ney, cv = frames()
                ney.loc[0, "text"] = "mutated"
                cv["train"].loc[0, "text"] = "mutated"
                fresh_ney, fresh_cv = frames()
                self.assertEqual(reader.call_count, 1)
                self.assertEqual(fresh_ney.loc[0, "text"], "a")
                self.assertEqual(fresh_cv["train"].loc[0, "text"], "a")
                path = root / "train.tsv"
                old = path.stat()
                path.write_text(path.read_text().replace("\ta\t", "\tb\t"))
                os.utime(path, ns=(old.st_atime_ns, old.st_mtime_ns))
                _, changed = frames()
                self.assertEqual(reader.call_count, 2)
                self.assertEqual(changed["train"].loc[0, "text"], "b")
            _frames_cached.cache_clear()


class PublicationTests(unittest.TestCase):
    def rows(self):
        rows = []
        for seed in (42, 43, 44):
            for evaluation in ("ney_test", "ney_test_disjoint", "cv_test"):
                rows.append(
                    {
                        "run": str(seed),
                        "system": "Whisper small",
                        "condition": "ney_matched",
                        "seed": seed,
                        "budget": "epochs",
                        "max_steps": None,
                        "evaluation": evaluation,
                        "wer": 1.0,
                        "cer": 1.0,
                        "n": 2,
                        "n_total": 2,
                        "n_scored": 2,
                        "n_empty_reference": 0,
                    }
                )
        return pd.DataFrame(rows, columns=COLUMNS)

    def test_late_validation_failure_cannot_overwrite_any_table(self):
        data = self.rows()
        incomplete = data.iloc[[0]].copy()
        incomplete["evaluation"] = "psrb_sample"
        data = pd.concat([data, incomplete])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "acl").mkdir()
            table = root / "acl/table4_generated.tex"
            table.write_text("previous complete report")
            with (
                patch("neyshekar_experiments.reporting.ROOT", root),
                patch("neyshekar_experiments.reporting.load_scores", return_value=data) as loader,
            ):
                with self.assertRaisesRegex(ValueError, "seeds"):
                    generate_paper_tables()
                self.assertEqual(loader.call_count, 1)
            self.assertEqual(table.read_text(), "previous complete report")
            self.assertEqual(list((root / "acl").iterdir()), [table])

    def test_missing_evaluation_and_duplicate_seed_are_rejected(self):
        for data in (
            self.rows().query("evaluation != 'cv_test'"),
            pd.concat([self.rows(), self.rows().iloc[[0]]]),
        ):
            with patch("neyshekar_experiments.reporting.load_scores", return_value=data):
                with self.assertRaises(ValueError):
                    generate_paper_tables()

    def test_missing_duration_manifest_does_not_publish_partial_results(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "acl").mkdir()
            with (
                patch("neyshekar_experiments.reporting.ROOT", root),
                patch("neyshekar_experiments.reporting.load_scores", return_value=self.rows()),
                patch(
                    "neyshekar_experiments.manifests.load_manifest", side_effect=FileNotFoundError
                ),
            ):
                with self.assertRaises(FileNotFoundError):
                    generate_paper_tables()
            self.assertFalse(list((root / "acl").iterdir()))


class IntegrityTests(unittest.TestCase):
    def test_reporting_rechecks_hypothesis_contents_even_with_preserved_timestamp(self):
        from types import SimpleNamespace

        from neyshekar_experiments.evaluation import evaluate_sets
        from neyshekar_experiments.reporting import load_scores

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ds = SimpleNamespace(rows=pd.DataFrame({"id": ["a"], "text": ["a"]}))
            spec = {
                "architecture": "whisper",
                "condition": "ney_matched",
                "optimization_seed": 42,
                "budget": "epochs",
                "max_steps": None,
            }
            with (
                patch("neyshekar_experiments.evaluation.ROOT", root),
                patch("neyshekar_experiments.reporting.ROOT", root),
            ):
                evaluate_sets("run", spec, {}, {"cv_test": ds}, lambda ds: ["a"])
                self.assertEqual(len(load_scores()), 1)
                path = root / "results/v2/hyps/run__cv_test.jsonl"
                stamp = path.stat()
                path.write_text(path.read_text().replace('"hypothesis": "a"', '"hypothesis": "b"'))
                os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
                with self.assertRaisesRegex(ValueError, "Changed predictions"):
                    load_scores()

    def test_publish_rolls_back_files_on_replace_failure(self):
        from neyshekar_experiments.reporting import publish_tables

        original_replace = Path.replace
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "acl"
            directory.mkdir()
            (directory / "one.tex").write_text("old one")
            (directory / "two.tex").write_text("old two")

            def failing_replace(path, destination):
                if Path(destination).name == "two.tex":
                    raise OSError("simulated storage failure")
                return original_replace(path, destination)

            with (
                patch("neyshekar_experiments.reporting.ROOT", root),
                patch.object(Path, "replace", failing_replace),
            ):
                with self.assertRaises(OSError):
                    publish_tables({"one.tex": "new one", "two.tex": "new two"})
            self.assertEqual((directory / "one.tex").read_text(), "old one")
            self.assertEqual((directory / "two.tex").read_text(), "old two")
            self.assertEqual(len(list(directory.iterdir())), 2)


class DependencyLockTests(unittest.TestCase):
    def load_script(self):
        import importlib.util

        from neyshekar_experiments.protocol import CODE

        spec = importlib.util.spec_from_file_location(
            "lock_dependencies", CODE / "scripts/lock_dependencies.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_resolution_failure_preserves_existing_lock(self):
        import subprocess

        module = self.load_script()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lock = root / "requirements-linux-aarch64-py312.lock"
            lock.write_text("previous lock")
            with (
                patch.object(module, "__file__", str(root / "scripts/lock_dependencies.py")),
                patch.object(module.platform, "machine", return_value="aarch64"),
                patch.object(module.platform, "system", return_value="Linux"),
                patch.object(
                    module.subprocess,
                    "run",
                    side_effect=subprocess.CalledProcessError(1, "resolver"),
                ),
            ):
                with self.assertRaises(subprocess.CalledProcessError):
                    module.main()
            self.assertEqual(lock.read_text(), "previous lock")

    def test_successful_lock_has_portable_header_and_hashes(self):
        module = self.load_script()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            def resolve(command, **kwargs):
                self.assertEqual(kwargs["cwd"], root)
                output = Path(command[command.index("--output-file") + 1])
                output.write_text("example==1.0 --hash=sha256:" + "a" * 64 + "\n")

            with (
                patch.object(module, "__file__", str(root / "scripts/lock_dependencies.py")),
                patch.object(module.platform, "machine", return_value="aarch64"),
                patch.object(module.platform, "system", return_value="Linux"),
                patch.object(module.subprocess, "run", side_effect=resolve),
                patch("builtins.print"),
            ):
                module.main()
            text = (root / "requirements-linux-aarch64-py312.lock").read_text()
            self.assertIn("--hash=sha256:", text)
            self.assertNotIn(temporary, text)


class CheckedInLockTests(unittest.TestCase):
    def test_dependency_lock_covers_direct_pins_and_every_archive_has_valid_hash(self):
        import re

        from neyshekar_experiments.protocol import CODE, environment_spec_hashes

        name = "requirements-linux-aarch64-py312.lock"
        text = (CODE / name).read_text()
        pins = {
            line.strip()
            for line in (CODE / "requirements.txt").read_text().splitlines()
            if line.strip() and not line.startswith("#")
        }
        locked = set()
        for block in re.split(r"\n(?=[a-zA-Z0-9])", text):
            match = re.match(r"([-\w.]+==[^\s;]+)", block)
            if match:
                locked.add(match.group(1))
                hashes = re.findall(r"--hash=sha256:([^\s]+)", block)
                self.assertTrue(hashes, match.group(1))
                self.assertTrue(all(re.fullmatch(r"[0-9a-f]{64}", value) for value in hashes))
        self.assertTrue(pins.issubset(locked))
        self.assertIn(name, environment_spec_hashes())
        self.assertNotIn("/tmp/", text)
