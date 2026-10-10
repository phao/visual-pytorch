"""Stable column meaning, legacy migration, explicit CLI and evidence status."""
import copy
import io
import json
import sqlite3
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from app.core import Workspace, default_model, GraphError, train_model, validate_model
from app.evidence import dataset_report, checkpoint_report
import vpw_cli


def two_inputs():
    doc=default_model()
    doc['graph']={'nodes':[
        {'id':'a','type':'Input','features':1,'label':'first'},
        {'id':'b','type':'Input','features':1,'label':'second'},
        {'id':'join','type':'Concat','sources':['b','a'],'flatten':True},
        {'id':'linear','type':'Linear','features':1},
        {'id':'out','type':'Output'}],
        'edges':[{'from':a,'to':b} for a,b in [('a','join'),('b','join'),('join','linear'),('linear','out')]]}
    doc['training']['epochs']=2
    doc['dataset']={'x':[[i/10.,(i+3)/8.] for i in range(10)],'y':[[i/7.] for i in range(10)]}
    return doc


class BindingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Native Qt smoke imports only two_inputs, never this test lifecycle.
        import torch
        torch.set_num_threads(1)

    def workspace(self):
        w=Workspace();self.addCleanup(w.db.close)
        d=two_inputs();w.put(d);return w,d['id']

    def report(self,w,mid):
        models=w.models();return dataset_report(models[mid],validate_model(models,mid))

    def test_legacy_root_concat_order_is_pinned_with_real_column_offsets(self):
        w,mid=self.workspace();d=w.models()[mid]
        self.assertEqual(d['graph']['input_order'],['b','a'])
        self.assertEqual([(g['id'],g['start'],g['stop']) for g in d['dataset_binding']['inputs']],
                         [('b',0,1),('a',1,2)])
        self.assertTrue(self.report(w,mid)['allowed'])

    def test_node_storage_reorder_and_labels_never_reinterpret_columns_or_checkpoint(self):
        w,mid=self.workspace();d=w.models()[mid]
        result=train_model(w.models(),mid);d['result']=result;w.put(d)
        before=copy.deepcopy(w.models()[mid]['dataset_binding'])
        d=w.models()[mid];d['graph']['nodes'].reverse()
        for n in d['graph']['nodes']:n['label']='renamed '+n['id']
        w.put(d)
        self.assertEqual(w.models()[mid]['dataset_binding'],before)
        self.assertEqual(w.models()[mid]['result'],result)
        self.assertTrue(self.report(w,mid)['allowed'])
        self.assertEqual(checkpoint_report(w.models(),mid)['status'],'current')

    def test_explicit_order_change_same_width_blocks_training_until_rebind(self):
        w,mid=self.workspace();d=w.models()[mid];original=copy.deepcopy(d['dataset'])
        d['graph']['input_order']=['a','b'];w.put(d)
        report=self.report(w,mid)
        self.assertFalse(report['allowed']);self.assertTrue(report['can_rebind'])
        self.assertEqual(report['status'],'incompatible')
        self.assertEqual(w.models()[mid]['dataset'],original)
        with self.assertRaisesRegex(GraphError,'binding'):train_model(w.models(),mid)

    def test_replacing_port_id_same_width_blocks_training(self):
        w,mid=self.workspace();d=w.models()[mid]
        next(n for n in d['graph']['nodes'] if n['id']=='a')['id']='new'
        next(n for n in d['graph']['nodes'] if n['id']=='join')['sources']=['b','new']
        next(e for e in d['graph']['edges'] if e['from']=='a')['from']='new'
        w.put(d)
        self.assertEqual(self.report(w,mid)['status'],'incompatible')
        with self.assertRaisesRegex(GraphError,'binding'):train_model(w.models(),mid)

    def test_shape_change_with_same_flat_width_still_requires_binding_review(self):
        w=Workspace();self.addCleanup(w.db.close);d=default_model(inputs=2,outputs=1)
        d['dataset']={'x':[[1.,2.],[3.,4.]],'y':[[1.],[2.]]};w.put(d)
        d=w.models()[d['id']];d['graph']['nodes'][0]['shape']=[1,2];w.put(d)
        self.assertEqual(self.report(w,d['id'])['status'],'incompatible')

    def test_invalid_values_or_width_cannot_be_fixed_by_rebind(self):
        w,mid=self.workspace();models=w.models();d=models[mid];contract=validate_model(models,mid)
        for data in ({'x':[[1.],[2.]],'y':[[1.],[2.]]},
                     {'x':[[True,2.],[1.,2.]],'y':[[1.],[2.]]},
                     {'x':[[1.,2.]],'y':[[1.]]},None):
            with self.subTest(data=data):
                d['dataset']=data;report=dataset_report(d,contract)
                self.assertFalse(report['allowed']);self.assertFalse(report['can_rebind'])

    def test_missing_binding_is_explicitly_unbound_without_mutation(self):
        w,mid=self.workspace();models=w.models();d=models[mid];d['dataset_binding']=None
        original=copy.deepcopy(d)
        report=dataset_report(d,validate_model(models,mid))
        self.assertEqual(report['status'],'unbound');self.assertTrue(report['can_rebind'])
        self.assertEqual(d,original)

    def test_v1_and_v2_migration_pins_original_meaning_without_rewriting_disk(self):
        for version in (1,2):
            with self.subTest(version=version),tempfile.TemporaryDirectory() as directory:
                w,mid=self.workspace();path=Path(directory)/'legacy.vtw';w.save(path)
                with sqlite3.connect(path) as db:
                    raw=json.loads(db.execute('SELECT document FROM architectures WHERE id=?',(mid,)).fetchone()[0])
                    raw['graph'].pop('input_order');raw.pop('dataset_binding')
                    db.execute('UPDATE architectures SET document=? WHERE id=?',(json.dumps(raw),mid))
                    db.execute('PRAGMA user_version='+str(version))
                before=path.read_bytes();loaded=Workspace.load(path)
                try:
                    self.assertEqual(path.read_bytes(),before)
                    d=loaded.models()[mid];self.assertEqual(d['graph']['input_order'],['b','a'])
                    self.assertEqual(d['dataset_binding']['inputs'][0]['id'],'b')
                    self.assertTrue(self.report(loaded,mid)['allowed'])
                    loaded.save(path)
                finally:loaded.db.close()
                with sqlite3.connect(path) as db:self.assertEqual(db.execute('PRAGMA user_version').fetchone()[0],3)

    def test_cli_rebind_is_explicit_atomic_preserves_values_and_historical_runs(self):
        w,mid=self.workspace();d=w.models()[mid];result=train_model(w.models(),mid)
        d['result']=result;w.put(d);w.record_run(mid,result,w.models())
        d=w.models()[mid];d['graph']['input_order']=['a','b'];w.put(d)
        dataset=copy.deepcopy(w.models()[mid]['dataset'])
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'bindings.vtw';w.save(path);stream=io.StringIO()
            with patch('vpw_cli.read_json',return_value=[{'op':'rebind-dataset','id':mid}]),redirect_stdout(stream):
                code=vpw_cli.main([str(path),'apply','-'])
            self.assertEqual(code,0,stream.getvalue());loaded=Workspace.load(path)
            try:
                self.assertEqual(loaded.models()[mid]['dataset'],dataset)
                self.assertIsNone(loaded.models()[mid]['result'])
                self.assertTrue(self.report(loaded,mid)['allowed'])
                self.assertEqual(loaded.models()[mid]['dataset_binding']['inputs'][0]['id'],'a')
                self.assertEqual(json.loads(loaded.db.execute('SELECT result FROM runs').fetchone()[0]),result)
            finally:loaded.db.close()

    def test_cli_rebind_cannot_admit_wrong_width_and_leaves_disk_unchanged(self):
        w,mid=self.workspace();d=w.models()[mid];d['dataset']['x']=[[1.] for _ in range(10)];w.put(d)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'bad.vtw';w.save(path);before=path.read_bytes();stream=io.StringIO()
            with patch('vpw_cli.read_json',return_value=[{'op':'rebind-dataset','id':mid}]),redirect_stdout(stream):
                code=vpw_cli.main([str(path),'apply','-'])
            self.assertEqual(code,1);self.assertIn('rows must have',stream.getvalue())
            self.assertEqual(path.read_bytes(),before)

    def test_honest_checkpoint_states_are_read_only(self):
        w,mid=self.workspace();models=w.models();original=copy.deepcopy(models)
        self.assertEqual(checkpoint_report(models,mid)['status'],'none')
        models[mid]['result']={'mse':.5}
        self.assertEqual(checkpoint_report(models,mid)['status'],'unverified')
        result=train_model(original,mid);models[mid]['result']=result
        self.assertEqual(checkpoint_report(models,mid)['status'],'current')
        result['identity']['dataset']='0'*64
        self.assertEqual(checkpoint_report(models,mid)['status'],'stale')
        result=train_model(original,mid);result['weights']['layer_0.bias'][0]=float('nan');models[mid]['result']=result
        before=json.dumps(models,sort_keys=True)
        self.assertEqual(checkpoint_report(models,mid)['status'],'invalid')
        self.assertEqual(json.dumps(models,sort_keys=True),before)

    def test_binding_version_and_offsets_are_not_coerced_by_python_equality(self):
        w,mid=self.workspace();models=w.models();contract=validate_model(models,mid)
        for field,value in [('version',True),('version',1.),('start',False),('start',0.)]:
            d=copy.deepcopy(models[mid])
            if field=='version':d['dataset_binding'][field]=value
            else:d['dataset_binding']['inputs'][0][field]=value
            with self.subTest(field=field,value=value):
                self.assertEqual(dataset_report(d,contract)['status'],'incompatible')

    def test_invalid_preprocessing_is_invalid_not_stale_identity(self):
        w,mid=self.workspace();models=w.models();result=train_model(models,mid)
        result['preprocessing']['kind']='unsupported-transform';models[mid]['result']=result
        self.assertEqual(checkpoint_report(models,mid)['status'],'invalid')
