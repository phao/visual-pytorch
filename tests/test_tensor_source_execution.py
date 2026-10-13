"""Real source task workers, portable predictors and captured notebook kernels."""
import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from app.core import Workspace, compile_model, generate_code
from app.datasets import make_split
from app.evidence import parameter_schema
from app.experiment_objects import execute_candidate, selection_at
from app.family_export import family_source
from app.notebooks import create_bundle, execute_bundle, verify_bundle
from app.revisions import model_document
from app.task_execution import task_bundle, training_protocol, restore_task_predictor, check_task_checkpoint
from app.task_training import train_task
from app.transport import run_json_worker
from test_tensor_source_graph import source_graph


def bundle(updates=6):
    doc=source_graph();mid=doc['id']
    rows=[{'id':'row-'+str(i),'values':{'x':[i/20,-i/30],'y':[i/20+1.5,-i/30-.5]}} for i in range(24)]
    dataset={'version':1,'id':'source-data','name':'Learn a shared tensor offset',
        'storage':{'kind':'inline-records','version':1},
        'fields':[{'id':k,'dtype':'float32','shape':[2]} for k in ('x','y')],'records':rows}
    task={'version':1,'id':'source-task','name':'Offset sanity check',
        'inputs':[{'port':'input','field':'x','transforms':[]}],
        'heads':[{'id':'target','port':'output','field':'y','loss':'mse','weight':1.,'transforms':[]}]}
    manifest=make_split(dataset,{'method':'row','seed':13,'fractions':[.75,.125,.125]})
    return task_bundle(model_document({mid:doc},mid),dataset,task,manifest,
        training_protocol(epochs=None,updates=updates,batch_size=6,lr=.1,evaluate_every=3))


class TensorSourceExecution(unittest.TestCase):
    def test_retained_source_values_bind_to_frozen_new_tensor_owner(self):
        from evidence_fixture import experiment
        from app.parameter_sources import inventory,read
        from app.parameter_binding import preview,admit,check_dependencies
        from app.policy_authoring import prepare,TOKEN
        from app.policy_graph import plan
        b=bundle(3);workspace=Workspace();workspace.set_constrained_mode(False)
        self.addCleanup(workspace.db.close)
        self.addCleanup(lambda:workspace.history.directory.cleanup() if workspace.history.directory else None)
        dataset={k:copy.deepcopy(v) for k,v in b['dataset'].items() if k!='revision'}
        task={k:copy.deepcopy(v) for k,v in b['task'].items() if k!='revision'}
        ref=experiment(workspace,source_graph(),dataset,task,'source-binding',updates=3)
        exp=workspace.get_spec(ref['kind'],ref['id'],ref['revision'])
        done=execute_candidate(workspace,ref,selection_at(exp,0),'source-fit','Source fit',python=sys.executable,timeout=30)
        listing=inventory(workspace,done['reference'],attempt=1)
        parameter=listing['parameters'][0]
        selection={'run':listing['run'],'attempt':1,'checkpoint_id':listing['checkpoint_id'],
                   'parameter':{k:parameter[k] for k in ('path','id')}}
        destination=source_graph(input=False);destination['id']='destination';destination['name']='Frozen tensor destination'
        destination['graph']['parameter_policy']={'version':1,'parameters':{'["tensor","value"]':{'trainable':False}}}
        workspace.put(destination)
        info=prepare(workspace.models(),'destination','tensor');token={k:info[k] for k in TOKEN}
        before=workspace.db.serialize();candidate=preview(workspace,token,'["tensor","value"]',selection)
        self.assertEqual(workspace.db.serialize(),before)
        retained=admit(workspace,token,'["tensor","value"]',selection)
        check_dependencies(candidate['binding'],workspace.get_spec)
        owner=plan(workspace.models(),'destination')['resolved']['owners']['["destination","tensor","value"]']
        self.assertFalse(owner['trainable'])
        self.assertEqual(owner['initializer']['values'],read(workspace,selection)['values'])
        native=compile_model(workspace.models(),'destination')
        self.assertFalse(native.layer_0.value.requires_grad)
        torch.testing.assert_close(native(batch_size=2),torch.tensor(owner['initializer']['values']).expand(2,2))

    def test_real_worker_checkpoint_restore_policy_and_prediction_parity(self):
        b=bundle();local=train_task(b,constrained=False)
        self.assertEqual(local['status'],'completed',local['failure'])
        check_task_checkpoint(b,local['checkpoint'],constrained=False)
        worker=run_json_worker([sys.executable,'-u','-m','app.worker'],
            {'kind':'task-v1','bundle':b,'constrained':False},cwd=str(Path(__file__).parents[1]))
        self.assertEqual(worker['checkpoint'],local['checkpoint'])
        self.assertEqual(worker['history'],local['history'])
        self.assertEqual(local['checkpoint']['version'],2)
        self.assertEqual(set(local['checkpoint']['weights']),{'layer_0.value'})
        predictions=restore_task_predictor(b,local['checkpoint'],constrained=False)([r['values'] for r in b['dataset']['records'][-3:]])
        self.assertEqual(tuple(predictions['output'].shape),(3,2))
        self.assertNotEqual(local['checkpoint']['weights']['layer_0.value'],[0.,0.])

    def test_fresh_app_free_source_and_nullary_family_with_real_parameters(self):
        f={'version':11,'id':'source-family','name':'Source family','root':'sources',
            'models':{'sources':source_graph(input=False)},'symbols':[],'assignments':[],
            'repeats':[],'runtime_policy':None}
        probe='''import runpy,sys,torch
s=runpy.run_path(sys.argv[1]);n=s['ArchitectureFamily']() if 'ArchitectureFamily' in s else s['model']
assert not any(k=='app' or k.startswith('app.') for k in sys.modules)
out=n(batch_size=3);assert list(out.shape)==[3,2]
out.sum().backward();assert torch.equal(next(n.parameters()).grad,torch.tensor([3.,3.]))
print('PASS genuine app-free source batch and gradients')
'''
        with tempfile.TemporaryDirectory() as temporary:
            for kind,code in (('concrete',generate_code(f['models'],'sources')),('family',family_source(f)['code'])):
                path=Path(temporary)/(kind+'.py');path.write_text(code)
                result=subprocess.run([sys.executable,'-I','-B','-c',probe,str(path)],cwd=temporary,capture_output=True,text=True,timeout=30)
                self.assertEqual(result.returncode,0,result.stderr)
                self.assertIn('PASS genuine',result.stdout)

    def test_actual_saved_and_rerun_notebooks_preserve_source_checkpoint(self):
        b=bundle(3);workspace=Workspace();workspace.set_constrained_mode(False)
        self.addCleanup(workspace.db.close)
        self.addCleanup(lambda:workspace.history.directory.cleanup() if workspace.history.directory else None)
        refs={kind:workspace.put_spec(kind,b[kind]) for kind in ('model','dataset','task')}
        experiment={'version':1,'id':'source-trial','name':'Tensor source trial','question':'Are source values reproducible?',
            'models':[refs['model']],'data_tasks':[{'id':'pair','dataset':refs['dataset'],'task':refs['task']}],
            'splits':[{'id':'split','protocol':{k:v for k,v in b['manifest']['protocol'].items() if k!='algorithm'}}],
            'initialization_seeds':[7],'protocols':[{'id':'adam','protocol':b['protocol']}],
            'selection':{'kind':'crossed','pairs':[],'exclusions':[]}}
        ref=workspace.put_spec('experiment',experiment)
        done=execute_candidate(workspace,ref,selection_at(experiment,0),'source-fit','Tensor source fit',python=sys.executable,timeout=30)
        result=done['record']['attempts'][0]['result']
        self.assertEqual(result['status'],'completed')
        records=[r['values'] for r in b['dataset']['records'][-3:]]
        expected=restore_task_predictor(b,result['checkpoint'],constrained=False)(records)
        views=[]
        with tempfile.TemporaryDirectory() as temporary:
            for mode in ('result-report','rerun'):
                recipe=workspace.put_spec('report',{'version':1,'id':mode,'name':'Source report','question':'Are source checkpoint and runtime captured?',
                    'mode':mode,'experiment':ref if mode=='rerun' else None,
                    'runs':[done['reference']] if mode=='result-report' else [],
                    'sections':['models','training','evaluation','checkpoints','failures'],
                    'analysis':{'kind':'descriptive','inferential_unit':'run'}})
                before=workspace.db.serialize();capture=create_bundle(workspace,recipe,Path(temporary)/mode)
                verify_bundle(capture['path'])
                self.assertTrue((Path(capture['path'])/'runtime/app/tensor_sources.py').is_file())
                actual=execute_bundle(capture['path'],timeout=60)
                self.assertEqual(actual['execution']['status'],'completed',actual['execution']['error'])
                self.assertGreater(actual['execution']['executed_code_cells'],0)
                directory=Path(actual['path']);view=json.loads((directory/'report-data.json').read_text());views.append(view)
                self.assertEqual(view['runs'][0]['checkpoint_id'],result['checkpoint_id'])
                self.assertEqual(view['notebook_execution']['training_requested'],mode=='rerun')
                self.assertEqual(workspace.db.serialize(),before)
                predictors=list(directory.glob('predictor-*.py'));self.assertEqual(len(predictors),1)
                namespace={};exec(predictors[0].read_text(),namespace)
                torch.testing.assert_close(namespace['predict_records'](records)['output'],expected['output'],rtol=0,atol=0)
            self.assertEqual(views[0]['runs'][0]['metrics'],views[1]['runs'][0]['metrics'])


if __name__=='__main__': unittest.main()
