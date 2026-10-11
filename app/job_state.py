"""Busy means the execution AND its queued terminal admission are unfinished."""


def job_busy(job):
    return bool(job and (job.isRunning() or getattr(job, 'awaiting_terminal', False)))
