import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from app.core import Workspace, default_model, check_document, GraphError, generate_code, validate_model
from app.themes import THEME_NAMES
import vpw_cli


class NodePresentationTests(unittest.TestCase):
    def test_folding_is_boolean_and_roundtrips_without_computation_changes(self):
        w=Workspace();d=default_model();w.put(d)
        try:
            d=w.models()[d['id']]
            d['dataset']={'x':[[1,2],[3,4]],'y':[[1],[2]]}
            w.put(d);d=w.models()[d['id']]
            d['result']={'mse':0};w.put(d)
            code=generate_code(w.models(),d['id']);contract=validate_model(w.models(),d['id'])
            before=w.history.capture(w.db)
            d['graph']['nodes'][1].update(collapsed=True,width=300,height=150)
            w.put(d);w.history.record(before,w.db)
            self.assertEqual(generate_code(w.models(),d['id']),code)
            self.assertEqual(validate_model(w.models(),d['id']),contract)
            self.assertEqual(w.models()[d['id']]['result'],{'mse':0})
            self.assertTrue(w.history.restore(w));self.assertNotIn('collapsed',w.models()[d['id']]['graph']['nodes'][1])
            self.assertTrue(w.history.restore(w,redo=True))
            with tempfile.TemporaryDirectory() as td:
                w.save(Path(td)/'folded.vtw');loaded=Workspace.load(w.path)
                try:
                    n=loaded.models()[d['id']]['graph']['nodes'][1]
                    self.assertIs(n['collapsed'],True);self.assertEqual((n['width'],n['height']),(300,150))
                    self.assertEqual(loaded.models()[d['id']]['dataset'],d['dataset'])
                finally:loaded.db.close()
            for value in (0,1,None,'true',[],{}):
                bad=copy.deepcopy(d);bad['graph']['nodes'][1]['collapsed']=value
                with self.assertRaises(GraphError):check_document(bad)
        finally:
            if w.history.directory is not None:w.history.directory.cleanup()
            w.db.close()

    def test_dimensions_are_optional_finite_positive_and_not_booleans(self):
        d=default_model();check_document(d)
        for value in (0,-1,True,False,None,'300',100001,float('inf'),float('nan')):
            for field in ('width','height'):
                bad=copy.deepcopy(d);bad['graph']['nodes'][1][field]=value
                with self.assertRaises(GraphError):check_document(bad)
        d['graph']['nodes'][1].update(width=1.25,height=100000);check_document(d)

    def test_resize_roundtrip_preserves_code_contract_data_and_results(self):
        w=Workspace();d=default_model();d['dataset']={'x':[[1,2],[3,4]],'y':[[1],[2]],'source':'fixture'}
        d['result']={'mse':0};w.put(d)
        code=generate_code(w.models(),d['id']);contract=validate_model(w.models(),d['id'])
        d['graph']['nodes'][1].update(width=333.5,height=180.25);w.put(d)
        with tempfile.TemporaryDirectory() as td:
            w.save(Path(td)/'geometry.vtw');loaded=Workspace.load(w.path)
            after=loaded.models()[d['id']]
            self.assertEqual(after['graph']['nodes'][1]['width'],333.5)
            self.assertEqual(after['dataset'],d['dataset']);self.assertEqual(after['result'],d['result'])
            self.assertEqual(generate_code(loaded.models(),d['id']),code)
            self.assertEqual(validate_model(loaded.models(),d['id']),contract)
            loaded.db.close()
        w.db.close()

    def test_resize_undo_redo_restores_presentation(self):
        w=Workspace();d=default_model();w.put(d)
        before=w.history.capture(w.db);d['graph']['nodes'][1].update(width=300,height=150)
        w.put(d);w.history.record(before,w.db)
        self.assertTrue(w.history.restore(w));self.assertNotIn('width',w.models()[d['id']]['graph']['nodes'][1])
        self.assertTrue(w.history.restore(w,redo=True));self.assertEqual(w.models()[d['id']]['graph']['nodes'][1]['width'],300)
        w.history.directory.cleanup();w.db.close()

    def test_cli_resize_is_atomic_and_browsing_is_sorted(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'agent.vtw';w=Workspace();d=default_model('zeta');w.put(d);w.put(default_model('Alpha'))
            w.save(path);w.db.close()
            def cli(*args):
                out=io.StringIO()
                with redirect_stdout(out):code=vpw_cli.main([str(path),*args])
                return code,json.loads(out.getvalue())
            self.assertEqual([v['name'] for v in cli('inspect')[1]['value']['architectures']],['Alpha','zeta'])
            edits=[{'op':'resize','id':d['id'],'node':'linear1','width':300.5,'height':120}]
            with patch('vpw_cli.read_json',return_value=edits):self.assertEqual(cli('apply','-')[0],0)
            before=path.read_bytes();edits[0]['width']=-1
            with patch('vpw_cli.read_json',return_value=edits):self.assertEqual(cli('apply','-')[0],1)
            self.assertEqual(path.read_bytes(),before)
            doc=cli('show',d['id'])[1]['value'];self.assertEqual(doc['graph']['nodes'][1]['width'],300.5)
            edits=[{'op':'set-node','id':d['id'],'node':'linear1','values':{},'remove_fields':['width','height']}]
            with patch('vpw_cli.read_json',return_value=edits):self.assertEqual(cli('apply','-')[0],0)
            restored=cli('show',d['id'])[1]['value']['graph']['nodes'][1]
            self.assertNotIn('width',restored);self.assertNotIn('height',restored)

    def test_theme_choices_are_alphabetical(self):
        self.assertEqual(list(THEME_NAMES),sorted(THEME_NAMES,key=str.casefold))
