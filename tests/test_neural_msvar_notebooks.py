"""Requested full model through actual worker and both real notebook kernels."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
import torch
from app.core import Workspace
from app.neural_msvar import drawing,dataset,likelihood_task,controlled_drawing,controlled_dataset,controlled_task
from app.family_records import retain
from app.task_execution import training_protocol,restore_task_predictor
from app.experiment_objects import execute_candidate,selection_at,candidate_bundle
from app.notebooks import create_bundle,execute_bundle,verify_bundle


class NeuralMSVARNotebookTests(unittest.TestCase):
    def test_full_k5_p3_d3_worker_and_actual_saved_rerun_kernels(self):
        self._kernels(False)

    def test_controlled_full_model_literal_logvariance_and_actual_kernels(self):
        self._kernels(True)

    def _kernels(self,controlled):
        workspace=Workspace();workspace.set_constrained_mode(False);self.addCleanup(workspace.db.close)
        self.addCleanup(lambda:workspace.history.directory.cleanup() if workspace.history.directory else None)
        model=(controlled_drawing if controlled else drawing)(steps=5)
        declaration={key:[0] for key in ('observations','execution_mask','reset_mask','emission_mask','reset_history')} if controlled else {'observations':[0]}
        retained=retain(workspace,family={'version':14 if controlled else 13,'id':'full-msvar-family','name':'Full neural MS-VAR',
            'root':model['id'],'models':{model['id']:model},'symbols':[],'assignments':[],'repeats':[],
            'runtime_policy':{'version':2,'kind':'runtime-temporal-v2','inputs':declaration}},materialize=True)
        d=workspace.put_spec('dataset',(controlled_dataset if controlled else dataset)(steps=5,trajectories=12))
        t=workspace.put_spec('task',(controlled_task if controlled else likelihood_task)())
        experiment={'version':7,'id':'full-msvar-trial','name':'Conditional sequence density',
            'question':'Can the full causal 5-state 3-lag trivariate model replay exact learned density?',
            'models':[{'id':'neural','family':retained['family'],'bindings':{},'axes':[],'execution':'runtime-temporal-v2'}],
            'data_tasks':[{'id':'observed','dataset':d,'task':t}],
            'splits':[{'id':'subjects','protocol':{'method':'group','field':'trajectory','seed':31,'fractions':[.5,.25,.25]}}],
            'initialization_seeds':[7],
            'protocols':[{'id':'sanity','protocol':training_protocol(epochs=None,updates=3,batch_size=3,lr=.01,evaluate_every=1)}],
            'selection':{'kind':'crossed','pairs':[],'exclusions':[]}}
        ref=workspace.put_spec('experiment',experiment)
        selected=selection_at(experiment,0)
        executed=execute_candidate(workspace,ref,selected,'fit','Bounded full MS-VAR check',python=sys.executable,timeout=30)
        attempt=executed['record']['attempts'][0];self.assertEqual(attempt['status'],'completed',attempt['error'])
        result=attempt['result'];frozen=candidate_bundle(workspace.get_spec('experiment',ref['id'],ref['revision']),selected,workspace.get_spec)
        self.assertEqual(frozen['model']['version'],10 if controlled else 9);self.assertEqual(frozen['version'],6)
        rows=[r['values'] for r in frozen['dataset']['records'][-2:]]
        expected=restore_task_predictor(frozen,result['checkpoint'],constrained=False)(rows)
        with tempfile.TemporaryDirectory() as temporary:
            for mode in ('result-report','rerun'):
                recipe=workspace.put_spec('report',{'version':1,'id':mode,'name':'Full neural MS-VAR report',
                    'question':'Exact likelihood/diagnostic replay, not supervised regimes','mode':mode,
                    'experiment':ref if mode=='rerun' else None,'runs':[executed['reference']] if mode=='result-report' else [],
                    'sections':['models','training','evaluation','checkpoints'],
                    'analysis':{'kind':'descriptive','inferential_unit':'run'}})
                before=workspace.db.serialize();capture=create_bundle(workspace,recipe,Path(temporary)/mode);verify_bundle(capture['path'])
                actual=execute_bundle(capture['path'],timeout=60)
                self.assertEqual(actual['execution']['status'],'completed',actual['execution']['error'])
                root=Path(actual['path']);view=json.loads((root/'report-data.json').read_text())
                self.assertEqual(view['runs'][0]['checkpoint_id'],result['checkpoint_id'])
                self.assertEqual(view['runs'][0]['metrics'],{'train':result['final_metrics']['train'],'validation':result['final_metrics']['validation'],'test':result['final_metrics']['test']})
                namespace={};exec(next(root.glob('predictor-*.py')).read_text(),namespace)
                for key,value in expected.items():torch.testing.assert_close(namespace['predict_records'](rows)[key],value,rtol=0,atol=0)
                self.assertEqual(workspace.db.serialize(),before)
                print(json.dumps({'test':'full-K5-p3-D3-msvar-kernel','controlled':controlled,'mode':mode,'checkpoint_id':result['checkpoint_id'],
                                  'cells':actual['execution']['executed_code_cells'],'parameters':1315}),flush=True)


if __name__=='__main__':unittest.main()
