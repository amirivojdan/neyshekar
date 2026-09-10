"""Parallel workers must preserve single-GPU budgets and unique artifact writers."""

import importlib.util
import os
import unittest
from pathlib import Path
from unittest.mock import patch

path = Path(__file__).resolve().parents[1] / "scripts/run_parallel_suite.py"
spec = importlib.util.spec_from_file_location("parallel_suite", path)
suite = importlib.util.module_from_spec(spec)
spec.loader.exec_module(suite)


class ParallelSuiteTests(unittest.TestCase):
    def test_each_worker_sees_one_gpu_and_no_distributed_rank(self):
        with patch.dict(os.environ, {"WORLD_SIZE": "4", "LOCAL_RANK": "3", "RANK": "3"}):
            environment = suite.worker_environment(2)
        self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "2")
        for key in ("WORLD_SIZE", "LOCAL_RANK", "RANK"):
            self.assertNotIn(key, environment)

    def test_grid_has_one_writer_for_each_of_70_runs_and_three_baselines(self):
        tasks = suite.task_specs()
        self.assertEqual(len(tasks), 73)
        self.assertEqual(len({task["name"] for task in tasks}), 73)
        runs = [task for task in tasks if "run" in task]
        self.assertEqual(len(runs), 70)
        for task in runs:
            if task["family"] == "mixture_updates":
                self.assertEqual(task["run"]["max_steps"], 2382)
            if task["family"].startswith("scaling"):
                self.assertEqual(task["run"]["optimization_seed"], 42)

    def test_imported_checkpoint_cannot_start_before_transfer_receipt(self):
        with patch.object(Path, "exists", return_value=False):
            self.assertFalse(suite.ready({"name": "saved", "import_required": True}))
            self.assertTrue(suite.ready({"name": "new"}))
        with patch.object(Path, "exists", return_value=True):
            self.assertTrue(suite.ready({"name": "saved", "import_required": True}))


if __name__ == "__main__":
    unittest.main()
