"""Cooperative cross-process save locking and content revisions."""
import fcntl
import hashlib
import os
from contextlib import contextmanager
from pathlib import Path


def revision(path):
    path = Path(path)
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def stamp(path):
    try:
        stat = Path(path).stat()
        return stat.st_ino, stat.st_size, stat.st_mtime_ns
    except FileNotFoundError:
        return None


@contextmanager
def save_lock(path):
    # Sidecar survives atomic replacement of the SQLite inode; never delete it.
    fd = os.open(str(path) + '.lock', os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError('Workspace is being saved by another process. Retry shortly.') from error
        yield
    finally:
        os.close(fd)
