"""Real native Window dispatch shares recipe admission and undo with the CLI."""
import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from app.core import Workspace


@unittest.skipUnless(importlib.util.find_spec('PyQt6'),'Run in the native Qt environment.')
class DesktopConstructionTests(unittest.TestCase):
    def setUp(self):
        from app.desktop import Window
        self.dispatch=Window.dispatch
        self.workspace=Workspace();self.addCleanup(self.workspace.db.close)
        self.addCleanup(lambda:self.workspace.history.directory.cleanup() if self.workspace.history.directory else None)
        self.messages=[]
        self.window=SimpleNamespace(workspace=self.workspace,job=None,
            broadcast=lambda *args,**kw:self.messages.append((args,kw)))

    def call(self,**request):return self.dispatch(self.window,request,None)

    def test_preview_is_readonly_and_atomic_creation_has_one_undo_step(self):
        before=self.workspace.db.serialize()
        proposed=self.call(action='previewTemplate',name='recurrent-cell',id='sequence')
        self.assertEqual(self.workspace.db.serialize(),before)
        self.assertFalse(self.workspace.history.summary()['undo']);self.assertFalse(self.workspace.history.summary()['redo'])
        self.assertEqual(self.messages,[])
        result=self.call(action='instantiateTemplate',name='recurrent-cell',id='sequence',expected_snapshot=proposed['snapshot_sha256'])
        self.assertTrue(result['changed']);self.assertEqual(len(self.workspace.history.undo_stack),1)
        self.assertEqual(set(self.workspace.models()),{'sequence/cell','sequence'})
        self.call(action='undo');self.assertEqual(self.workspace.models(),{})
        self.call(action='redo');self.assertEqual(set(self.workspace.models()),{'sequence/cell','sequence'})

    def test_failure_keeps_bytes_dirty_and_history_and_stale_preview_is_refused(self):
        preview=self.call(action='previewTemplate',name='branching-mlp',id='mlp')
        self.call(action='instantiateTemplate',name='branching-mlp',id='other')
        before=self.workspace.db.serialize();history=list(self.workspace.history.undo_stack)
        with self.assertRaisesRegex(ValueError,'stale'):
            self.call(action='instantiateTemplate',name='branching-mlp',id='mlp',expected_snapshot=preview['snapshot_sha256'])
        self.assertEqual(self.workspace.db.serialize(),before);self.assertEqual(self.workspace.history.undo_stack,history)
        with self.assertRaises(ValueError):
            self.call(action='applyEdits',edits=[{'op':'rename','id':'other','name':'Partial'},
                {'op':'set-node','id':'other','node':'missing','values':{}}])
        self.assertEqual(self.workspace.db.serialize(),before);self.assertEqual(self.workspace.history.undo_stack,history)

    def test_template_declarations_are_available_without_a_workspace(self):
        self.window.workspace=None
        self.assertEqual(len(self.call(action='templateList')),5)
        self.assertEqual(self.call(action='templateShow',name='neural-msvar')['kind'],'exact-model')
        with self.assertRaises(ValueError):self.call(action='instantiateTemplate',name='neural-msvar',id='msvar')


if __name__=='__main__':unittest.main()
