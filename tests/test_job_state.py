"""No Qt dependency: execution plus queued admission define job ownership."""
import unittest
from types import SimpleNamespace
from app.job_state import job_busy


class JobStateTests(unittest.TestCase):
    def test_no_job(self): self.assertFalse(job_busy(None))
    def test_legacy_job_keeps_thread_semantics(self):
        for running in (True, False):
            self.assertEqual(job_busy(SimpleNamespace(isRunning=lambda: running)), running)
    def test_terminal_delivery_keeps_finished_thread_busy(self):
        job = SimpleNamespace(isRunning=lambda: False, awaiting_terminal=True)
        self.assertTrue(job_busy(job)); job.awaiting_terminal=False
        self.assertFalse(job_busy(job))
    def test_thread_remains_busy_after_admission_until_it_finishes(self):
        self.assertTrue(job_busy(SimpleNamespace(isRunning=lambda: True, awaiting_terminal=False)))
