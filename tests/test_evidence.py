"""Independent reviewed counterexamples; private workspaces and bounded CPU fits."""
import copy
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import torch
from app.core import Workspace, default_model, compile_model, validate_model, train_model, restore_predictor, generate_predictor

torch.set_num_threads(1)


def independent(widths=(1, 1)):
    doc=default_model()
    doc['graph']={'nodes':[
        {'id':'a','type':'Input','features':1},
        {'id':'b','type':'Input','features':1},
        {'id':'la','type':'Linear','features':widths[0]},
        {'id':'lb','type':'Linear','features':widths[1]},
        {'id':'ya','type':'Output'}, {'id':'yb','type':'Output'}],
        'edges':[{'from':a,'to':b} for a,b in [('a','la'),('b','lb'),('la','ya'),('lb','yb')]]}
    doc['training']['epochs']=2
    doc['dataset']={'x':[[float(i),float(i+2)] for i in range(10)],
                    'y':[[float(i)]*sum(widths) for i in range(10)]}
    return doc


class EvidenceTests(unittest.TestCase):
    def test_bulk_presentation_keeps_real_checkpoint_predictor_and_export(self):
        d=default_model();d['training']['epochs']=2
        d['dataset']={'x':[[0.,1.],[1.,0.],[2.,1.]],'y':[[1.],[2.],[3.]]}
        w=Workspace()
        try:
            w.put(d);d=w.models()[d['id']];d['result']=train_model(w.models(),d['id']);w.put(d)
            result=w.models()[d['id']]['result']
            x=torch.tensor([[3.,2.]])
            before=restore_predictor(w.models(),d['id'],result).forward_tensor(x)
            exported=generate_predictor(w.models(),d['id'],result)
            for collapsed in (True,False):
                current=w.models()[d['id']]
                for i,n in enumerate(current['graph']['nodes']):
                    n.update(collapsed=collapsed,x=i*100,y=i*24)
                w.put(current)
                self.assertEqual(w.models()[d['id']]['result'],result)
                after=restore_predictor(w.models(),d['id'],result).forward_tensor(x)
                torch.testing.assert_close(after,before,rtol=0,atol=0)
                self.assertEqual(generate_predictor(w.models(),d['id'],result),exported)
        finally:w.db.close()

    def test_independent_affine_paths_do_not_fake_serial_composition(self):
        for widths in [(1,1),(2,3)]:
            with self.subTest(widths=widths):
                d=independent(widths)
                result=train_model({d['id']:d},d['id'])
                self.assertNotIn('A',result)
                self.assertEqual(result['affine_analysis']['status'],'unavailable')
                self.assertTrue(result['weights'])
                self.assertTrue(result['mse']>=0)

    def test_metrics_describe_returned_checkpoint(self):
        d=default_model()
        d['training']['epochs']=1
        d['dataset']={'x':[[float(i),float(i-2)] for i in range(10)],
                      'y':[[float(i+3)] for i in range(10)]}
        r=train_model({d['id']:d},d['id'])
        m=compile_model({d['id']:d},d['id'])
        m.load_state_dict({k:torch.tensor(v) for k,v in r['weights'].items()})
        x=torch.tensor(d['dataset']['x']);y=torch.tensor(d['dataset']['y'])
        order=torch.randperm(10,generator=torch.Generator().manual_seed(7))
        expected=torch.nn.functional.mse_loss(m(x[order[:8]]),y[order[:8]]).item()
        self.assertAlmostEqual(r['history'][-1]['train'],expected,places=5)
        self.assertEqual(r['history'][-1]['test'],r['mse'])

    def test_actual_updates_not_plot_sample_count(self):
        d=default_model();d['training']['epochs']=800
        d['dataset']={'x':[[0.,1.],[1.,0.],[2.,1.]],'y':[[1.],[2.],[3.]]}
        class Cancel:
            count=0
            def is_set(self):
                self.count+=1
                return self.count>13
        r=train_model({d['id']:d},d['id'],cancel=Cancel())
        self.assertEqual(r['epochs'],13)
        self.assertEqual(r['completed_updates'],13)
        self.assertEqual(r['history'][-1]['epoch'],13)

    def test_replacing_dataset_invalidates_current_not_historical_result(self):
        w=Workspace();d=independent()
        try:
            w.put(d);d=w.models()[d['id']]
            d['result']={'mse':.1}
            w.put(d);w.record_run(d['id'],d['result'],w.models())
            d['dataset']['y'][0][0]+=10
            w.put(d)
            self.assertIsNone(w.models()[d['id']]['result'])
            self.assertEqual(w.db.execute('SELECT COUNT(*) FROM runs').fetchone()[0],1)
        finally:w.db.close()

    def test_reordering_storage_preserves_input_column_identity(self):
        w=Workspace();d=independent()
        try:
            w.put(d);d=w.models()[d['id']]
            m=compile_model({d['id']:d},d['id'])
            with torch.no_grad():
                m.layer_0.weight.fill_(1);m.layer_0.bias.zero_()
                m.layer_1.weight.fill_(10);m.layer_1.bias.zero_()
            before=m.forward_tensor(torch.tensor([[2.,7.]]))
            d['graph']['nodes'].reverse()
            w.put(d);reordered=w.models()[d['id']]
            self.assertEqual([g['id'] for g in validate_model({d['id']:reordered},d['id'])['input_groups']],['a','b'])
            after=compile_model({d['id']:reordered},d['id'])
            after.load_state_dict(m.state_dict())
            torch.testing.assert_close(after.forward_tensor(torch.tensor([[2.,7.]])),before)
            self.assertEqual(before.tolist(),[[2.,70.]])
        finally:w.db.close()
