import copy
import unittest
from inference_lab.core import (Configuration, episode_metrics, feature_key, make_tasks,
                                score, select_configuration, train_policy)


class ResearchContracts(unittest.TestCase):
    def test_exact_scoring_rejects_near_misses_and_duplicate_keys(self):
        truth = [{"units": 1}]
        self.assertTrue(score('[{"units":1}]', truth))
        for wrong in ['[{"units":true}]', '[{"units":1.0}]', '[{"units":2,"units":1}]',
                      'Answer: [{"units":1}]', '[{"units":1,"extra":0}]']:
            self.assertFalse(score(wrong, truth))

    def test_deadline_failures_stay_in_denominator(self):
        rows = [{"correct": True, "finish_s": 2, "ttft_s": 1},
                {"correct": True, "finish_s": 7, "ttft_s": 3},
                {"correct": False, "finish_s": 1, "ttft_s": None}]
        result = episode_metrics(rows, 10, 5)
        self.assertEqual(result["requests"], 3)
        self.assertEqual(result["on_time_correct"], 1)
        self.assertEqual(result["goodput_per_min"], 6)

    def test_calibration_does_not_read_test_results(self):
        def row(name, goodput, accuracy=1, phase="calibration"):
            return {"phase": phase, "feature": "solo_short", "configuration": name,
                    "peak_active_gib": 6, "prompt_lengths": [100],
                    "metrics": {"correct": int(accuracy * 4), "requests": 4,
                                "goodput_per_min": goodput, "mean_completion_s": 60 / goodput}}
        calibration = [row("serial", 10), row("kv8", 20), row("kv4", 100, .5)]
        p = train_policy(calibration)
        p2 = train_policy(calibration + [row("serial", 10000, phase="test")])
        self.assertEqual(p, p2)
        self.assertEqual(p["best_fixed"], "kv8")
        self.assertNotIn("kv4", p["eligible"]["solo_short"])
        failed = copy.deepcopy(calibration)
        failed[1]["error"] = "allocator error"
        self.assertEqual(train_policy(failed)["best_fixed"], "serial")
        with self.assertRaises(ValueError):
            select_configuration(p, [100], budget_gib=1)
        with self.assertRaises(ValueError):
            select_configuration(p, [10000])

    def test_fixture_splits_are_disjoint(self):
        a = make_tasks("calibration", 0, "burst_mixed")
        b = make_tasks("test", 0, "burst_mixed")
        self.assertFalse({r["prompt"] for r in a} & {r["prompt"] for r in b})
        self.assertEqual(feature_key([300, 2800]), "burst_long")

    def test_unsupported_combinations_rejected_before_gpu_work(self):
        # The Engine.run check precedes tokenization and GPU operations.
        from inference_lab.mlx_backend import Engine
        instance = object.__new__(Engine)
        with self.assertRaisesRegex(ValueError, "Unsupported composition"):
            instance.run([], Configuration("invalid", "batch", 4, kv_bits=8))


if __name__ == "__main__":
    unittest.main()
