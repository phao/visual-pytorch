"""All explicit controls through actual full NN, compiler and training paths."""
import copy
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
import torch
from app.core import Workspace,GraphError,compile_model,generate_code,validate_model
from app.neural_msvar import controlled_drawing,controlled_dataset,controlled_task
from app.revisions import model_document,validate_model_document
from app.datasets import make_split
from app.task_execution import task_bundle,training_protocol,restore_task_predictor,generate_task_predictor
from app.task_training import train_task
from app.task_result import validate_result
from app.task_runtime import forward_ports
from app.transport import run_json_worker
from app.family_records import retain,compile_specialization
from app.family_export import family_source
from examples.build_neural_msvar import build

torch.set_num_threads(1)


def bundle(updates=3, *, all_invalid=False):
    doc=controlled_drawing(steps=6);data=controlled_dataset(steps=6,trajectories=12)
    if all_invalid:
        for row in data['records']:row['values']['loss_valid']=False
    return task_bundle(model_document({doc['id']:doc},doc['id']),data,controlled_task(),
        make_split(data,{'method':'group','field':'trajectory','seed':31,'fractions':[.5,.25,.25]}),
        training_protocol(epochs=None,updates=updates,batch_size=3,lr=.01,evaluate_every=1))


class ControlledMSVARTests(unittest.TestCase):
    def test_controlled_example_builder_is_no_run_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'controlled.vtw'
            build(path,controlled=True)
            before=path.read_bytes();workspace=Workspace.load(path)
            try:
                doc=workspace.models()['neural-msvar']
                nodes={n['id']:n for n in doc['graph']['nodes']}
                self.assertEqual(len(nodes),27)
                self.assertEqual(nodes['variances']['transform'],'exp')
                self.assertEqual(nodes['variances']['positive_floor'],0.)
                self.assertEqual(nodes['lags']['op_version'],2)
                self.assertEqual(workspace.db.execute("SELECT count(*) FROM spec_heads WHERE kind='run'").fetchone()[0],0)
            finally:workspace.db.close()
            with self.assertRaisesRegex(GraphError,'never overwrites'):build(path,controlled=True,run=True)
            self.assertEqual(path.read_bytes(),before)

    def test_full_native_and_export_controls_carry_reset_score_and_gradients(self):
        doc=controlled_drawing(steps=6);native=compile_model({doc['id']:doc},doc['id']);namespace={};exec(generate_code({doc['id']:doc},doc['id']),namespace)
        portable=namespace['model'];portable.load_state_dict(native.state_dict(),strict=True)
        row=controlled_dataset(steps=6,trajectories=1)['records'][0]['values']
        x={g['id']:torch.tensor([row[g['id']]],dtype=torch.float32) for g in native.contract['input_groups']}
        outputs=forward_ports(native,x,native.contract);actual=forward_ports(portable,x,portable.contract)
        for key,value in outputs.items():torch.testing.assert_close(actual[key],value,rtol=0,atol=0)
        torch.testing.assert_close(outputs['out_filtered'][:,1],outputs['out_filtered'][:,0],rtol=0,atol=0)
        self.assertEqual(float(outputs['out_step_evidence'][0,1]),0.)
        self.assertEqual(float(outputs['out_step_evidence'][0,2]),0.)
        torch.testing.assert_close(outputs['out_sequence_evidence'][:,0],outputs['out_step_evidence'].sum(1))
        source=next(item for item in native.contract['plan'] if item['id']=='raw_variances')
        raw=getattr(native,source['layer']).value
        torch.testing.assert_close(outputs['out_variances'][0],raw.exp(),rtol=0,atol=0)
        (-outputs['out_sequence_evidence'].mean()).backward();(-actual['out_sequence_evidence'].mean()).backward()
        for (name,p),(other,q) in zip(native.named_parameters(),portable.named_parameters()):
            self.assertEqual(name,other);self.assertIsNotNone(p.grad);self.assertTrue(torch.isfinite(p.grad).all())
            torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0)

    def test_real_worker_checkpoint_and_standalone_predictor_match(self):
        frozen=bundle();result=train_task(frozen,constrained=False)
        self.assertEqual(result['status'],'completed',result['failure']);self.assertEqual(frozen['model']['version'],10)
        validate_result(frozen,{k:v for k,v in result.items() if k!='bundle'})
        worker=run_json_worker([sys.executable,'-u','-m','app.worker'],{'kind':'task-v5','bundle':frozen,'constrained':False},cwd=str(Path(__file__).parents[1]))
        self.assertEqual(worker['checkpoint'],result['checkpoint']);self.assertEqual(worker['history'],result['history'])
        native=restore_task_predictor(frozen,result['checkpoint'],constrained=False)
        namespace={};exec(generate_task_predictor(frozen,result['checkpoint'],constrained=False),namespace)
        rows=[r['values'] for r in frozen['dataset']['records'][-2:]]
        for key,value in native(rows).items():torch.testing.assert_close(namespace['predict_records'](rows)[key],value,rtol=0,atol=0)

    def test_all_invalid_loss_is_unavailable_not_a_fabricated_training_fit(self):
        frozen=bundle(all_invalid=True);result=train_task(frozen,constrained=False)
        self.assertEqual(result['status'],'unavailable',result['failure'])
        self.assertEqual(result['counts']['completed_updates'],0)
        self.assertIsNone(result['final_metrics']['train']['objective'])
        self.assertEqual(result['final_metrics']['train']['heads']['sequence-nll']['denominator'],0.)
        validate_result(frozen,{k:v for k,v in result.items() if k!='bundle'})

    def test_controlled_family14_variable_time_with_no_new_parameter_owners(self):
        doc=controlled_drawing(steps=6)
        declaration={key:[0] for key in ('observations','execution_mask','reset_mask','emission_mask','reset_history')}
        family={'version':14,'id':'controlled-family','name':'Controlled neural MS-VAR','root':doc['id'],
            'models':{doc['id']:doc},'symbols':[],'assignments':[],'repeats':[],
            'runtime_policy':{'version':2,'kind':'runtime-temporal-v2','inputs':declaration}}
        workspace=Workspace();self.addCleanup(workspace.db.close);refs=retain(workspace,family=family,materialize=False)
        native=compile_specialization(workspace,refs['specialization'],constrained=False)
        namespace={};exec(family_source(family)['code'],namespace);portable=namespace['ArchitectureFamily']()
        portable.network.load_state_dict(native.state_dict(),strict=True)
        for steps in (3,8):
            values=controlled_dataset(steps=steps,trajectories=1)['records'][0]['values']
            x={g['id']:torch.tensor([values[g['id']]],dtype=torch.float32) for g in native.contract['input_groups']}
            expected=forward_ports(native,x,native.contract);actual=forward_ports(portable,x,portable.network.contract)
            for key,value in expected.items():torch.testing.assert_close(actual[key],value,rtol=0,atol=0)
        self.assertEqual(sum(p.numel() for p in native.parameters()),1315)

    def test_prior_grammars_and_headers_cannot_gain_controls_or_logvariance(self):
        doc=controlled_drawing();frozen=model_document({doc['id']:doc},doc['id']);frozen['version']=9
        with self.assertRaisesRegex(GraphError,'ModelRevision10'):validate_model_document(frozen)
        workspace=Workspace();self.addCleanup(workspace.db.close);workspace.put(doc)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'controlled.vtw';workspace.save(path)
            with sqlite3.connect(path) as db:db.execute('PRAGMA user_version=31')
            with self.assertRaisesRegex(GraphError,'version-32'):Workspace.load(path)


if __name__=='__main__':unittest.main()
