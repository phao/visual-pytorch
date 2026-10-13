"""Actual CLI recipe/planning admission; file conflict and no-execution contracts."""
import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch
import vpw_cli
from app.core import Workspace


class ConstructionCLITests(unittest.TestCase):
    def call(self,*args):
        out=io.StringIO()
        with redirect_stdout(out):code=vpw_cli.main(list(map(str,args)))
        return code,json.loads(out.getvalue())

    def setUp(self):
        self.directory=tempfile.TemporaryDirectory();self.addCleanup(self.directory.cleanup)
        self.path=Path(self.directory.name)/'test.vtw'
        self.assertEqual(self.call(self.path,'init')[0],0)

    def test_catalogue_without_workspace_and_default_instantiation_without_training(self):
        for command in (('template-list',),('template-show','neural-msvar')):
            with patch('vpw_cli.Workspace.load',side_effect=AssertionError('workspace read')):
                self.assertEqual(self.call(*command)[0],0)
        before=self.path.read_bytes()
        code,result=self.call(self.path,'instantiate-template','neural-msvar','--id','msvar','--dry-run')
        self.assertEqual(code,0,result);self.assertEqual(self.path.read_bytes(),before)
        self.assertEqual(result['value']['effects'][0]['after']['parameters'],1315)
        self.assertEqual(self.call(self.path,'instantiate-template','neural-msvar','--id','msvar')[0],0)
        workspace=Workspace.load(self.path)
        try:
            self.assertIn('msvar',workspace.models())
            self.assertEqual(workspace.db.execute('SELECT count(*) FROM runs').fetchone()[0],0)
            self.assertEqual(workspace.db.execute("SELECT count(*) FROM spec_heads WHERE kind IN ('run','dataset')").fetchone()[0],0)
        finally:workspace.db.close()
        saved=self.path.read_bytes()
        code,result=self.call(self.path,'instantiate-template','neural-msvar','--id','msvar','--dry-run')
        self.assertEqual(code,1,result);self.assertEqual(result['value']['findings'][0]['code'],'EDIT_REFUSED')
        self.assertEqual(self.path.read_bytes(),saved)

    def test_apply_dryrun_plan_and_exact_snapshot_recheck(self):
        self.assertEqual(self.call(self.path,'instantiate-template','branching-mlp','--id','mlp')[0],0)
        edits=json.dumps([{'op':'rename','id':'mlp','name':'Renamed'}]);before=self.path.read_bytes()
        with patch('sys.stdin',io.StringIO(edits)):code,result=self.call(self.path,'plan-edits','-')
        self.assertEqual(code,0,result);self.assertEqual(self.path.read_bytes(),before)
        token=result['value']['snapshot_sha256']
        with patch('sys.stdin',io.StringIO(edits)):code,other=self.call(self.path,'apply','-','--dry-run')
        self.assertEqual(code,0,other);self.assertEqual(other['value']['snapshot_sha256'],token)
        with patch('sys.stdin',io.StringIO(edits)):code,actual=self.call(self.path,'apply','-','--expect-snapshot',token)
        self.assertEqual(code,0,actual)
        saved=self.path.read_bytes()
        with patch('sys.stdin',io.StringIO(edits)):code,refused=self.call(self.path,'apply','-','--expect-snapshot',token)
        self.assertEqual(code,1,refused);self.assertIn('stale',refused['error']);self.assertEqual(self.path.read_bytes(),saved)
        with patch('sys.stdin',io.StringIO(edits)):
            code,refused=self.call(self.path,'--expect-revision',hashlib.sha256(before).hexdigest(),'apply','-')
        self.assertEqual(code,1,refused);self.assertEqual(self.path.read_bytes(),saved)

    def test_dependency_order_and_late_batch_failure_are_atomic(self):
        self.assertEqual(self.call(self.path,'instantiate-template','recurrent-cell','--id','sequence')[0],0)
        before=self.path.read_bytes();edits=json.dumps([{'op':'rename','id':'sequence','name':'Changed'},
            {'op':'set-node','id':'sequence','node':'missing','values':{}}])
        with patch('sys.stdin',io.StringIO(edits)):code,result=self.call(self.path,'apply','-')
        self.assertEqual(code,1,result);self.assertEqual(self.path.read_bytes(),before)


if __name__=='__main__':unittest.main()
