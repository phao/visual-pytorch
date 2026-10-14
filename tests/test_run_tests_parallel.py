"""Pure parts of tools/run_tests_parallel.py: summary parsing and CPU grouping."""
import importlib.util
import unittest
from pathlib import Path

path = Path(__file__).resolve().parents[1] / 'tools' / 'run_tests_parallel.py'
spec = importlib.util.spec_from_file_location('run_tests_parallel', path)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class RunnerTests(unittest.TestCase):
    def test_summary_is_read_from_unittest_output_and_exit_status(self):
        self.assertEqual(runner.parse_summary('..\nRan 17 tests in 2.1s\n\nOK (skipped=2)\n', 0), (17, 2, True))
        self.assertEqual(runner.parse_summary('Ran 1 test in 0.0s\n\nFAILED (failures=1)\n', 1), (1, 0, False))
        self.assertEqual(runner.parse_summary('Ran 0 tests in 0.0s\n\nNO TESTS RAN\n', 5), (0, 0, False))
        # An interpreter crash or timeout has no summary: never counted as passing.
        self.assertEqual(runner.parse_summary('Traceback ...', 1), (0, 0, False))
        self.assertEqual(runner.parse_summary('', 0), (0, 0, False))

    def test_cpu_groups_never_exceed_workers_or_overlap(self):
        groups = runner.cpu_groups([0, 2, 4, 5, 6, 7, 8, 9], per_worker=2, workers=4)
        self.assertEqual(groups, [[0, 2], [4, 5], [6, 7], [8, 9]])
        self.assertEqual(runner.cpu_groups([0, 2, 4, 5, 6], 2, 8), [[0, 2], [4, 5]])
        self.assertEqual(runner.cpu_groups([3], 2, 4), [[3]])  # fewer CPUs than one worker needs: one worker

    def test_detection_picks_distinct_allowed_cpus_within_the_limit(self):
        picked = runner.one_cpu_per_core(3)
        self.assertLessEqual(len(picked), 3)
        self.assertEqual(len(set(picked)), len(picked))


if __name__ == '__main__':
    unittest.main()
