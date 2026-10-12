"""Arithmetic in fresh app-free source and actual saved/rerun Jupyter kernels."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from app.core import Workspace, generate_code, compile_model
from app.family_export import family_source
from app.experiment_objects import execute_candidate, selection_at
from app.notebooks import create_bundle, execute_bundle, verify_bundle
from app.task_execution import restore_task_predictor
from test_arithmetic import family
from test_arithmetic_execution import arithmetic_bundle


class ArithmeticNotebookExecutionTests(unittest.TestCase):
    def test_fresh_generated_concrete_and_family_imports_are_application_free(self):
        f=family();values={'x':[[.1,.2]],'other':[[-.3,.4]]}
        native=compile_model(f['models'],f['root'],constrained=False)
        # Persist explicit weights, not accidental agreement between RNG states.
        weights={k:v.tolist() for k,v in native.state_dict().items()}
        expected=native.forward_tensor({k:torch.tensor(v) for k,v in values.items()}).detach().tolist()
        probe='''import json,runpy,sys,torch
torch.set_num_threads(1)
s=runpy.run_path(sys.argv[1]);p=json.loads(sys.argv[2])
n=s['ArchitectureFamily']().network if 'ArchitectureFamily' in s else s['model']
n.load_state_dict({k:torch.tensor(v) for k,v in p['weights'].items()},strict=True)
out=n.forward_tensor({k:torch.tensor(v) for k,v in p['values'].items()})
assert not any(k=='app' or k.startswith('app.') for k in sys.modules)
print(json.dumps(out.detach().tolist()))
'''
        with tempfile.TemporaryDirectory() as td:
            for kind,code in (('concrete',generate_code(f['models'],f['root'])),('family',family_source(f,constrained=False)['code'])):
                target=Path(td)/(kind+'.py');target.write_text(code)
                result=subprocess.run([sys.executable,'-I','-B','-c',probe,str(target),json.dumps({'weights':weights,'values':values})],cwd=td,capture_output=True,text=True,timeout=25)
                self.assertEqual(result.returncode,0,result.stderr);self.assertEqual(json.loads(result.stdout),expected)

    def test_actual_saved_and_rerun_kernels_preserve_arithmetic_checkpoint_and_predictions(self):
        b=arithmetic_bundle(updates=3);w=Workspace();w.set_constrained_mode(False)
        self.addCleanup(w.db.close)
        self.addCleanup(lambda:w.history.directory.cleanup() if w.history.directory else None)
        refs={kind:w.put_spec(kind,b[kind]) for kind in ('model','dataset','task')}
        e={'version':1,'id':'arithmetic-trial','name':'Arithmetic trial','question':'Does arithmetic survive actual captured notebook execution?',
           'models':[refs['model']],'data_tasks':[{'id':'pair','dataset':refs['dataset'],'task':refs['task']}],
           'splits':[{'id':'split','protocol':{k:v for k,v in b['manifest']['protocol'].items() if k!='algorithm'}}],'initialization_seeds':[7],
           'protocols':[{'id':'adam','protocol':b['protocol']}],'selection':{'kind':'crossed','pairs':[],'exclusions':[]}}
        ref=w.put_spec('experiment',e)
        done=execute_candidate(w,ref,selection_at(e,0),'fit','Arithmetic fit',python=sys.executable,timeout=30)
        result=done['record']['attempts'][0]['result'];self.assertEqual(done['record']['attempts'][0]['status'],'completed')
        records=[r['values'] for r in b['dataset']['records'][-3:]]
        expected=restore_task_predictor(b,result['checkpoint'],constrained=False)(records)
        views=[]
        with tempfile.TemporaryDirectory() as td:
            for mode in ('result-report','rerun'):
                recipe=w.put_spec('report',{'version':1,'id':mode,'name':'Arithmetic report','question':'Are the exact arithmetic run and source reproducible?',
                    'mode':mode,'experiment':ref if mode=='rerun' else None,'runs':[done['reference']] if mode=='result-report' else [],
                    'sections':['models','training','evaluation','checkpoints','failures'],'analysis':{'kind':'descriptive','inferential_unit':'run'}})
                before=w.db.serialize();capture=create_bundle(w,recipe,Path(td)/mode);verify_bundle(capture['path'])
                actual=execute_bundle(capture['path'],timeout=60)
                self.assertEqual(actual['execution']['status'],'completed',actual['execution']['error'])
                self.assertGreater(actual['execution']['executed_code_cells'],0)
                directory=Path(actual['path']);view=json.loads((directory/'report-data.json').read_text());views.append(view)
                self.assertEqual(view['runs'][0]['checkpoint_id'],result['checkpoint_id'])
                self.assertEqual(view['notebook_execution']['training_requested'],mode=='rerun')
                self.assertEqual(w.db.serialize(),before)
                predictors=list(directory.glob('predictor-*.py'));self.assertEqual(len(predictors),1)
                scope={};exec(predictors[0].read_text(),scope);output=scope['predict_records'](records)
                torch.testing.assert_close(output['y'],expected['y'],rtol=0,atol=0)
            self.assertEqual(views[0]['runs'][0]['metrics'],views[1]['runs'][0]['metrics'])


if __name__=='__main__':unittest.main()
