"""Actual filter diagrams through native/export/family/runtime and storage."""
import copy
import tempfile
import unittest
from pathlib import Path
import torch
from app.core import Workspace,GraphError,validate_model,compile_model,generate_code
from app.evidence import parameter_schema
from app.family_records import retain,compile_specialization
from app.family_export import family_source
from gaussian_graph_fixture import model


class GaussianGraphTests(unittest.TestCase):
    def test_component_authoring_preserves_existing_graph_and_refuses_bad_edit(self):
        from app.core import default_model
        from app.evidence import identity
        from app.gaussian_authoring import add,edit
        original=default_model();models={original['id']:original}
        created,nid=add(models,original['id'],identity(models,original['id'])['architecture'])
        self.assertEqual(original['graph']['nodes'],created['graph']['nodes'][:3])
        self.assertEqual(len(created['graph']['nodes']),17)
        c=validate_model({original['id']:created},original['id']);self.assertEqual(c['parameters'],13)
        filter_node=next(n for n in created['graph']['nodes'] if n['id']==nid)
        fields={k:copy.deepcopy(filter_node[k]) for k in ('inputs','initial_representation','transition_representation','transition_mode','mean_mode','variance_mode')}
        fields['mean_mode']='sequence'
        snapshot=copy.deepcopy(created)
        with self.assertRaises(GraphError):edit({original['id']:created},original['id'],nid,identity({original['id']:created},original['id'])['architecture'],fields)
        self.assertEqual(created,snapshot)

    def test_native_plain_export_all_roles_and_gradients(self):
        doc=model();models={'hmm':doc};c=validate_model(models,'hmm')
        self.assertEqual(c['parameters'],10)
        self.assertEqual(sum(torch.tensor(row['shape']).prod().item() for row in parameter_schema(models,'hmm').values()),10)
        native=compile_model(models,'hmm');ns={};exec(generate_code(models,'hmm'),ns)
        x={'observations':torch.tensor([[[.1],[.3],[-.2],[.5]],[[.2],[.1],[0.],[.4]]]),
           'lengths':torch.tensor([[4.],[3.]])}
        expected=native(x);actual=ns['model'](x)
        for key in expected:torch.testing.assert_close(actual[key],expected[key],rtol=0,atol=0)
        for runner in (native,ns['model']):
            loss=-runner(x)['sequence_evidence'].mean();loss.backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in runner.parameters()))
        torch.testing.assert_close(expected['filtered'].exp().sum(-1),torch.ones(2,4))
        self.assertTrue(torch.equal(expected['step_evidence'][1,3:],torch.zeros(1)))

    def test_family12_specialization11_and_temporal_runtime_keep_shared_sources(self):
        doc=model();family={'version':12,'id':'hmm-family','name':'HMM family','root':'hmm',
            'models':{'hmm':doc},'symbols':[],'assignments':[],'repeats':[],
            'runtime_policy':{'version':2,'kind':'runtime-temporal-v2','inputs':{'observations':[0]}}}
        workspace=Workspace();self.addCleanup(workspace.db.close)
        retained=retain(workspace,family=family,materialize=False)
        ref=retained['specialization'];self.assertEqual(workspace.get_spec(ref['kind'],ref['id'],ref['revision'])['version'],11)
        native=compile_specialization(workspace,ref)
        ns={};exec(family_source(family)['code'],ns);portable=ns['ArchitectureFamily']()
        for steps in (2,6):
            x={'observations':torch.zeros(3,steps,1),'lengths':torch.full((3,1),float(steps))}
            for key,value in native(x).items():torch.testing.assert_close(value,portable(x)[key],rtol=0,atol=0)
        self.assertEqual(sum(p.numel() for p in native.parameters()),10)

    def test_old_container_refuses_filter_drawings_and_archived_model8(self):
        import sqlite3
        workspace=Workspace();workspace.put(model());ref=workspace.freeze_model('hmm')
        self.assertEqual(workspace.get_spec(ref['kind'],ref['id'],ref['revision'])['version'],8)
        workspace.archive_spec('model','hmm')
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'filter.vtw';workspace.save(path)
            loaded=Workspace.load(path);loaded.db.close()
            with sqlite3.connect(path) as db:db.execute('PRAGMA user_version=27')
            with self.assertRaisesRegex(GraphError,'version-28'):Workspace.load(path)


if __name__=='__main__':unittest.main()
