"""Preview/atomic admission use actual edit executor without source mutation."""
import copy
import hashlib
import random
import tempfile
import unittest
from pathlib import Path
from app.core import Workspace,GraphError
from app.edit_planning import preview,admit
from app.discovery import operator_schema
from vpw_cli import apply_edits


class EditPlanningTests(unittest.TestCase):
    def setUp(self):
        self.workspace=Workspace();self.addCleanup(self.workspace.db.close)
        self.addCleanup(lambda:self.workspace.history.directory.cleanup() if self.workspace.history.directory else None)
        example=operator_schema('Linear')['example'];self.mid=example['root']
        self.workspace.put(example['models'][self.mid])

    def call(self,edits,**kwargs):return preview(self.workspace,edits,executor=apply_edits,**kwargs)

    def test_preview_preserves_bytes_dirty_rng_selections_and_history(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'test.vtw';self.workspace.save(path)
            disk=path.read_bytes();db=self.workspace.db.serialize();rng=random.getstate()
            history=(list(self.workspace.history.undo_stack),list(self.workspace.history.redo_stack))
            result=self.call([{'op':'set-node','id':self.mid,'node':'op','values':{'features':4}}])
            self.assertEqual(result['status'],'ready');self.assertTrue(result['effects'][0]['port_contract_changed'])
            self.assertEqual(result['effects'][0]['before']['parameters'],9)
            self.assertEqual(result['effects'][0]['after']['parameters'],12)
            self.assertEqual(path.read_bytes(),disk);self.assertEqual(self.workspace.db.serialize(),db)
            self.assertFalse(self.workspace.dirty);self.assertEqual(random.getstate(),rng)
            self.assertEqual(history,(self.workspace.history.undo_stack,self.workspace.history.redo_stack))

    def test_late_failure_is_located_and_does_not_publish_partial_edits(self):
        edits=[{'op':'rename','id':self.mid,'name':'First edit'},
               {'op':'set-node','id':self.mid,'node':'missing','values':{'features':2}}]
        before=self.workspace.db.serialize();result=self.call(edits)
        self.assertFalse(result['admissible']);self.assertTrue(result['partial_effects'])
        self.assertEqual(result['findings'][0]['path'],['edits',1]);self.assertEqual(result['findings'][0]['node'],'missing')
        with self.assertRaises(GraphError):admit(self.workspace,edits,executor=apply_edits)
        self.assertEqual(self.workspace.db.serialize(),before)

    def test_actual_admission_rechecks_snapshot_and_commits_all_edits(self):
        edits=[{'op':'rename','id':self.mid,'name':'New name'},{'op':'move','id':self.mid,'node':'op','x':71,'y':99}]
        result=self.call(edits);self.workspace.put({**self.workspace.models()[self.mid],'name':'Concurrent change'})
        before=self.workspace.db.serialize()
        with self.assertRaisesRegex(GraphError,'stale'):admit(self.workspace,edits,executor=apply_edits,expected_snapshot=result['snapshot_sha256'])
        self.assertEqual(self.workspace.db.serialize(),before)
        fresh=self.call(edits);admit(self.workspace,edits,executor=apply_edits,expected_snapshot=fresh['snapshot_sha256'])
        self.assertEqual(self.workspace.models()[self.mid]['name'],'New name')
        self.assertEqual(next(n for n in self.workspace.models()[self.mid]['graph']['nodes'] if n['id']=='op')['x'],71)

    def test_existing_and_introduced_invalid_graphs_are_distinct(self):
        doc=self.workspace.models()[self.mid];doc['graph']['edges']=[];self.workspace.put(doc)
        result=self.call([]);self.assertFalse(result['admissible'])
        self.assertTrue(all(f['origin']=='existing' for f in result['findings']))
        self.assertEqual(self.call([],allow_invalid=True)['status'],'draft')
        restored=operator_schema('Linear')['example']['models'][self.mid];self.workspace.put(restored)
        result=self.call([{'op':'disconnect','id':self.mid,'edge':restored['graph']['edges'][0]}])
        self.assertTrue(all(f['origin']=='introduced' for f in result['findings']))

    def test_budget_and_closed_controls_refuse_before_source_copy(self):
        before=self.workspace.db.serialize()
        with self.assertRaisesRegex(GraphError,'budget'):self.call([],max_bytes=1)
        with self.assertRaises(GraphError):self.call([],allow_invalid='yes')
        with self.assertRaises(GraphError):self.call([{}]*257)
        self.assertEqual(self.workspace.db.serialize(),before)

    def test_empty_admission_is_noop_and_preference_changes_are_visible(self):
        self.workspace.dirty=False;before=self.workspace.db.serialize()
        admit(self.workspace,[],executor=apply_edits)
        self.assertEqual(self.workspace.db.serialize(),before);self.assertFalse(self.workspace.dirty)
        result=self.call([{'op':'mode','enabled':False}])
        self.assertEqual(result['metadata_changes'],[{'key':'constrained_mode','before':'true','after':'false'}])
        self.assertEqual(self.workspace.db.serialize(),before)


if __name__=='__main__':unittest.main()
