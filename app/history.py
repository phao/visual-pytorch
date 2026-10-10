"""Bounded, session-only SQLite edit history, stored on disk rather than in RAM."""
from pathlib import Path
from contextlib import closing
import sqlite3
import tempfile
import uuid


class History:
    def __init__(self, max_steps=50, max_bytes=128*1024**2):
        self.undo_stack=[];self.redo_stack=[]
        self.max_steps=max_steps;self.max_bytes=max_bytes
        self.directory=None;self.notice=None

    def summary(self):
        return {'undo':bool(self.undo_stack),'redo':bool(self.redo_stack),'notice':self.notice}

    def capture(self, db):
        size=db.execute('PRAGMA page_count').fetchone()[0]*db.execute('PRAGMA page_size').fetchone()[0]
        if size > self.max_bytes:return None
        if self.directory is None:self.directory=tempfile.TemporaryDirectory(prefix='tensor-loom-history-')
        path=Path(self.directory.name)/(uuid.uuid4().hex+'.sqlite')
        with closing(sqlite3.connect(path)) as target:db.backup(target)
        return path

    @staticmethod
    def discard(path):
        if path is not None:path.unlink(missing_ok=True)

    def record(self, before, db):
        for path in self.redo_stack:self.discard(path)
        self.redo_stack.clear()
        size=db.execute('PRAGMA page_count').fetchone()[0]*db.execute('PRAGMA page_size').fetchone()[0]
        if before is None or size > self.max_bytes:
            self.discard(before)
            for path in self.undo_stack:self.discard(path)
            self.undo_stack.clear()
            self.notice=f'Workspace exceeds the {self.max_bytes//1024**2} MiB session-history budget; this edit is not undoable.'
            return
        self.notice=None;self.undo_stack.append(before)
        self.trim()

    def trim(self):
        while len(self.undo_stack)+len(self.redo_stack)>self.max_steps or sum(p.stat().st_size for p in self.undo_stack+self.redo_stack)>self.max_bytes:
            stack=self.undo_stack if self.undo_stack else self.redo_stack
            self.discard(stack.pop(0))

    def restore(self, workspace, redo=False):
        source=self.redo_stack if redo else self.undo_stack
        destination=self.undo_stack if redo else self.redo_stack
        if not source:return False
        current=self.capture(workspace.db)
        if current is None:raise ValueError('Workspace exceeds the undo/redo history budget.')
        candidate=sqlite3.connect(':memory:')
        try:
            with closing(sqlite3.connect(source[-1])) as snapshot:snapshot.backup(candidate)
            candidate.execute('PRAGMA foreign_keys=ON')
            # New training evidence survives reverting earlier edits for models
            # that exist in the restored collection. Deleted-model evidence is
            # recoverable through its history snapshot/redo, not an orphan FK row.
            ids={r[0] for r in candidate.execute('SELECT id FROM architectures')}
            for row in workspace.db.execute('SELECT * FROM runs'):
                if row[1] not in ids:continue
                if candidate.execute('SELECT 1 FROM runs WHERE architecture_id=? AND created_at=? AND graph_snapshot=? AND result=?',row[1:]).fetchone():continue
                candidate.execute('INSERT INTO runs(architecture_id,created_at,graph_snapshot,result) VALUES (?,?,?,?)',row[1:])
            candidate.commit()
            if candidate.execute('PRAGMA foreign_key_check').fetchone():raise ValueError('History reference integrity failed.')
            candidate.backup(workspace.db)
        except Exception:
            self.discard(current);raise
        finally:candidate.close()
        destination.append(current);self.discard(source.pop())
        self.trim()
        workspace.dirty=True
        return True
