"""Byte-bound undo proof reuse; schema/heads/changed rows always fail closed."""
import sqlite3
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.core import Workspace,GraphError,default_model
from app.revisions import (SCHEMA,RegistryValidationCache,_registry_fingerprint,
                           frozen_document,validate_all)
from test_datasets import data


class UndoValidationCacheTests(unittest.TestCase):
    def setUp(self):
        self.w=Workspace();self.addCleanup(self.w.db.close)
        self.addCleanup(lambda:self.w.history.directory.cleanup() if self.w.history.directory else None)
        self.d=default_model();self.w.put(self.d);self.ref=self.w.put_spec('dataset',data())

    def edit(self):
        before=self.w.history.capture(self.w.db);self.d['name']='Changed'
        self.w.put(self.d);self.w.history.record(before,self.w.db)

    def warm(self):validate_all(self.w.db,cache=self.w.history.registry_validation)

    def unguarded(self,sql,args=()):
        self.w.db.execute('DROP TRIGGER spec_revisions_no_update')
        self.w.db.execute(sql,args);self.w.db.execute(SCHEMA['spec_revisions_no_update']);self.w.db.commit()

    def test_unchanged_registry_undo_redo_needs_no_document_revalidation(self):
        self.warm();self.edit()
        with patch('app.revisions.frozen_document',side_effect=AssertionError('unchanged bytes already proved')):
            self.assertTrue(self.w.history.restore(self.w));self.assertTrue(self.w.history.restore(self.w,redo=True))
        self.assertEqual(self.w.models()[self.d['id']]['name'],'Changed')
        validate_all(self.w.db)

    def test_loaded_file_seeds_first_undo_but_open_and_save_force_complete_proof(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'test.vtw'
            with patch('app.revisions.frozen_document',wraps=frozen_document) as validator:
                self.w.save(path);self.assertGreater(validator.call_count,0)
            before=path.read_bytes()
            with patch('app.revisions.frozen_document',wraps=frozen_document) as validator:
                loaded=Workspace.load(path);self.assertGreater(validator.call_count,0)
            self.addCleanup(loaded.db.close)
            self.addCleanup(lambda:loaded.history.directory.cleanup() if loaded.history.directory else None)
            self.assertEqual(path.read_bytes(),before)
            snap=loaded.history.capture(loaded.db);d=loaded.models()[self.d['id']];d['name']='Moved';loaded.put(d);loaded.history.record(snap,loaded.db)
            with patch('app.revisions.frozen_document',side_effect=AssertionError('loaded immutable bytes already proved')):loaded.history.restore(loaded)
            with patch('app.revisions.frozen_document',wraps=frozen_document) as validator:
                loaded.save(path);self.assertGreater(validator.call_count,0)

    def test_changed_document_name_timestamp_and_revision_never_reuse_stale_proof(self):
        for column,value in (('document','{}'),('name','Wrong row name'),('created_at','not UTC'),('revision','f'*64)):
            with self.subTest(column=column):
                cache=RegistryValidationCache();validate_all(self.w.db,cache=cache)
                candidate=sqlite3.connect(':memory:');self.w.db.backup(candidate)
                try:
                    candidate.execute('DROP TRIGGER spec_revisions_no_update')
                    candidate.execute('UPDATE spec_revisions SET '+column+'=?',(value,))
                    candidate.execute(SCHEMA['spec_revisions_no_update']);candidate.commit()
                    with self.assertRaises(GraphError):validate_all(candidate,cache=cache)
                finally:candidate.close()

    def test_schema_and_heads_are_checked_even_when_row_fingerprint_matches(self):
        self.warm();key=_registry_fingerprint(self.w.db)
        self.w.db.execute('DROP TRIGGER spec_revisions_no_update')
        with self.assertRaisesRegex(GraphError,'guard'):validate_all(self.w.db,cache=self.w.history.registry_validation)
        self.w.db.execute(SCHEMA['spec_revisions_no_update'])
        self.w.db.execute('PRAGMA foreign_keys=OFF')
        self.w.db.execute("UPDATE spec_heads SET revision=?",('0'*64,));self.w.db.commit()
        self.assertEqual(_registry_fingerprint(self.w.db),key)
        with self.assertRaisesRegex(GraphError,'missing revision'):validate_all(self.w.db,cache=self.w.history.registry_validation)

    def test_history_failure_preserves_database_dirty_flag_and_stacks(self):
        self.warm();self.edit();self.w.dirty=False
        self.unguarded("UPDATE spec_revisions SET document='{}'")
        before=self.w.db.serialize();undo=list(self.w.history.undo_stack);redo=list(self.w.history.redo_stack)
        with self.assertRaises(GraphError):self.w.history.restore(self.w)
        self.assertEqual(self.w.db.serialize(),before);self.assertFalse(self.w.dirty)
        self.assertEqual(self.w.history.undo_stack,undo);self.assertEqual(self.w.history.redo_stack,redo)

    def test_new_registry_rows_get_fully_proved_and_survive_undo(self):
        self.warm();self.edit();new=data();new['id']='new';self.w.put_spec('dataset',new)
        with patch('app.revisions.frozen_document',wraps=frozen_document) as validator:
            self.w.history.restore(self.w);self.assertGreater(validator.call_count,0)
        self.assertEqual(len(self.w.list_specs('dataset',all_revisions=True)),2)
        with patch('app.revisions.frozen_document',side_effect=AssertionError('new union already proved')):self.w.history.restore(self.w,redo=True)
        validate_all(self.w.db)

    def test_fingerprints_include_deleted_rows_and_cache_is_bounded(self):
        self.warm();key=_registry_fingerprint(self.w.db)
        candidate=sqlite3.connect(':memory:');self.w.db.backup(candidate)
        try:
            candidate.execute('DROP TRIGGER spec_revisions_no_delete');candidate.execute('DELETE FROM spec_revisions');candidate.commit()
            self.assertNotEqual(_registry_fingerprint(candidate),key)
            with self.assertRaises(GraphError):validate_all(candidate,cache=self.w.history.registry_validation)
        finally:candidate.close()
        cache=RegistryValidationCache(max_entries=2)
        for value in (b'a',b'b',b'c'):cache.remember(value)
        self.assertFalse(cache.contains(b'a'));self.assertTrue(cache.contains(b'b'));self.assertEqual(len(cache._proofs),2)

    def test_untrusted_history_snapshot_cannot_reuse_current_byte_proof(self):
        self.warm();self.edit();self.w.dirty=False
        with sqlite3.connect(self.w.history.undo_stack[-1]) as snapshot:
            snapshot.execute('DROP TRIGGER spec_revisions_no_update')
            snapshot.execute("UPDATE spec_revisions SET document='{}'")
            snapshot.execute(SCHEMA['spec_revisions_no_update'])
        before=self.w.db.serialize();paths=list(self.w.history.undo_stack)
        with self.assertRaises(GraphError):self.w.history.restore(self.w)
        self.assertEqual(before,self.w.db.serialize());self.assertFalse(self.w.dirty)
        self.assertEqual(paths,self.w.history.undo_stack)

    def test_generated_columns_cannot_splice_a_second_valid_row_into_a_proof(self):
        new=data();new['id']='second';self.w.put_spec('dataset',new);self.warm()
        rows=list(self.w.db.execute('SELECT * FROM spec_revisions ORDER BY kind,id,revision'))
        self.assertEqual(len(rows),2)
        def old_unframed_fingerprint(db):
            h=hashlib.sha256(b'visual-pytorch-registry-proof-v1\0')
            for row in db.execute('SELECT * FROM spec_revisions ORDER BY kind,id,revision'):
                for field in row:
                    raw=field.encode('utf-8');h.update(len(raw).to_bytes(8,'big'));h.update(raw)
            return h.digest()
        candidate=sqlite3.connect(':memory:')
        try:
            columns=', '.join('shadow'+str(i)+" TEXT GENERATED ALWAYS AS ('"+value.replace("'","''")+"') VIRTUAL" for i,value in enumerate(rows[1]))
            for name,sql in SCHEMA.items():
                if name=='spec_revisions':sql=sql.replace('PRIMARY KEY(kind,id,revision)',columns+', PRIMARY KEY(kind,id,revision)')
                candidate.execute(sql)
            candidate.execute('INSERT INTO spec_revisions(kind,id,revision,name,document,created_at) VALUES (?,?,?,?,?,?)',rows[0])
            candidate.execute('INSERT INTO spec_heads VALUES (?,?,?)',rows[0][:3]);candidate.commit()
            # Reproduce the reviewer's noncryptographic row-splicing collision.
            self.assertEqual(old_unframed_fingerprint(candidate),old_unframed_fingerprint(self.w.db))
            for force in (False,True):
                with self.subTest(force=force),self.assertRaisesRegex(GraphError,'hidden/generated'):
                    validate_all(candidate,cache=self.w.history.registry_validation,force=force)
            with self.assertRaisesRegex(GraphError,'six fields'):_registry_fingerprint(candidate)
        finally:candidate.close()

    def test_union_of_separately_valid_divergent_run_histories_is_rejected_atomically(self):
        from contextlib import closing
        from test_experiment_objects import experiment_workspace
        from app.experiment_objects import execute_candidate,object_reference,selection_at
        w,e=experiment_workspace();other=Workspace()
        try:
            w.db.backup(other.db);snap=w.history.capture(w.db)
            ref=object_reference('experiment',e);choice=selection_at(e,0)
            for target,missing in ((w,'/no-such-undo-python-A'),(other,'/no-such-undo-python-B')):
                done=execute_candidate(target,ref,choice,'same-run','Actual failed worker',python=missing,timeout=5)
                self.assertEqual(done['record']['attempts'][0]['status'],'failed')
                validate_all(target.db,cache=w.history.registry_validation)
            with closing(sqlite3.connect(snap)) as db:other.db.backup(db)
            w.history.record(snap,w.db);w.dirty=False;before=w.db.serialize();paths=list(w.history.undo_stack)
            with self.assertRaisesRegex(GraphError,'diverge'):w.history.restore(w)
            self.assertEqual(w.db.serialize(),before);self.assertFalse(w.dirty);self.assertEqual(w.history.undo_stack,paths)
        finally:
            w.db.close();other.db.close()
            if w.history.directory:w.history.directory.cleanup()


if __name__=='__main__':unittest.main()
