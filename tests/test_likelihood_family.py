"""Real signed likelihood family, experiment and portable kernel execution."""
import copy
import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
import torch
from app.core import Workspace, GraphError
from app.family_records import retain
from app.experiment_objects import (candidate, candidate_bundle, preview_experiment, selection_at, execute_candidate)
from app.task_execution import (task_bundle, training_protocol, run_identity, validate_bundle,
                                restore_task_predictor, generate_task_predictor)
from app.task_training import train_task
from app.task_result import validate_result
from app.transport import run_json_worker
from app.notebooks import create_bundle, execute_bundle, verify_bundle
from test_likelihood_task import fixture

torch.set_num_threads(1)


def experiment(*, runtime=True, steps=6, updates=3):
    drawing, dataset, task = fixture()
    # Narrow but nonsingular emissions: continuous NLL legitimately becomes negative.
    owners = drawing['graph']['parameter_policy']['parameters']
    owners['["means","value"]']['initializer']['values'] = [[-.04], [.04]]
    owners['["raw_variances","value"]'] = {
        'initializer': {'kind': 'values', 'dtype': 'float32', 'values': [[-4.], [-4.]]}}
    family = {'version':12, 'id':'hmm-family', 'name':'Unlabelled Gaussian family',
              'root':'hmm', 'models':{'hmm':drawing}, 'symbols':[], 'assignments':[], 'repeats':[],
              'runtime_policy':{'version':2,'kind':'runtime-temporal-v2','inputs':{'observations':[0]}} if runtime else None}
    if runtime:
        dataset['fields'][0]['shape'] = [steps, 1]
        for row in dataset['records']:
            row['values']['observations'] = [[row['values']['observations'][0][0] + .01*t] for t in range(steps)]
            row['values']['lengths'] = [float(steps)]
    workspace = Workspace(); workspace.set_constrained_mode(False)
    retained = retain(workspace, family=family, materialize=False)
    refs = {kind:workspace.put_spec(kind, doc) for kind,doc in (('dataset',dataset),('task',task))}
    doc = {'version':7,'id':'likelihood-trial','name':'Exact Gaussian likelihood trial',
           'question':'Can unlabelled sequence evidence be learned and replayed?',
           'models':[{'id':'hmm','family':retained['family'],'bindings':{},'axes':[],
                      'execution':'runtime-temporal-v2' if runtime else 'concrete-v1'}],
           'data_tasks':[{'id':'unlabelled','dataset':refs['dataset'],'task':refs['task']}],
           'splits':[{'id':'split','protocol':{'method':'row','seed':13,'fractions':[.5,.25,.25]}}],
           'initialization_seeds':[7],
           'protocols':[{'id':'adam','protocol':training_protocol(epochs=None,updates=updates,batch_size=2,lr=.02,evaluate_every=1)}],
           'selection':{'kind':'crossed','pairs':[],'exclusions':[]}}
    ref = workspace.put_spec('experiment',doc)
    return workspace, workspace.get_spec('experiment',ref['id'],ref['revision']), ref


class LikelihoodFamilyTests(unittest.TestCase):
    def make(self, **kwargs):
        workspace, doc, ref = experiment(**kwargs)
        self.addCleanup(workspace.db.close)
        self.addCleanup(lambda:workspace.history.directory.cleanup() if workspace.history.directory else None)
        return workspace, doc, ref

    def test_runtime6_native_worker_export_and_variable_time_keep_same_owners(self):
        workspace, doc, _ = self.make()
        before = workspace.db.serialize()
        preview = preview_experiment(doc,workspace.get_spec)['candidates'][0]
        self.assertEqual(preview['status'],'compatible',preview['reason'])
        frozen = candidate_bundle(doc,selection_at(doc,0),workspace.get_spec)
        self.assertEqual(frozen['version'],6)
        self.assertEqual(run_identity(frozen)['task_runtime'],'signed-additive-v2')
        self.assertEqual(validate_bundle(frozen)['parameters'],10)
        self.assertEqual(next(g['shape'] for g in validate_bundle(frozen)['input_groups'] if g['id']=='observations'),[6,1])
        result = train_task(frozen,constrained=False)
        self.assertEqual(result['status'],'completed',result['failure'])
        self.assertLess(result['final_metrics']['train']['objective'],0.)
        validate_result(frozen,{k:v for k,v in result.items() if k!='bundle'})
        worker = run_json_worker([sys.executable,'-u','-m','app.worker'],
                                {'kind':'task-v6','bundle':frozen,'constrained':False},cwd=str(Path(__file__).parents[1]))
        self.assertEqual(worker['checkpoint'],result['checkpoint'])
        self.assertEqual(worker['history'],result['history'])
        predictor = restore_task_predictor(frozen,result['checkpoint'],constrained=False)
        namespace = {};exec(generate_task_predictor(frozen,result['checkpoint'],constrained=False),namespace)
        for steps in (2,7):
            rows = [{'observations':[[.05]]*steps,'lengths':[float(steps)]}]
            expected = predictor(rows);actual = namespace['predict_records'](rows)
            self.assertEqual(tuple(actual['out_filtered'].shape),(1,steps,2))
            for key in expected:torch.testing.assert_close(actual[key],expected[key],rtol=0,atol=0)
        self.assertEqual(workspace.db.serialize(),before)

    def test_old_experiments_cannot_gain_target_free_meaning_on_either_preview_path(self):
        workspace, doc, _ = self.make()
        old = copy.deepcopy(doc);old.pop('revision');old['version']=6
        ref = workspace.put_spec('experiment',old);old=workspace.get_spec('experiment',ref['id'],ref['revision'])
        self.assertEqual(preview_experiment(old,workspace.get_spec)['candidates'][0]['status'],'incompatible')
        self.assertEqual(candidate(old,selection_at(old,0),workspace.get_spec)['status'],'incompatible')
        with self.assertRaises(GraphError):candidate_bundle(old,selection_at(old,0),workspace.get_spec)

    def test_archived_experiment7_cannot_fit_an_old_container_header(self):
        workspace, doc, ref = self.make()
        workspace.archive_spec('experiment',ref['id'])
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'likelihood.vtw';workspace.save(path)
            loaded=Workspace.load(path);loaded.db.close()
            with sqlite3.connect(path) as db:db.execute('PRAGMA user_version=29')
            with self.assertRaisesRegex(GraphError,'version-30'):Workspace.load(path)

    def test_real_saved_and_rerun_notebook_kernels_match_native_likelihood(self):
        workspace, doc, ref = self.make()
        executed=execute_candidate(workspace,ref,selection_at(doc,0),'fit','Unlabelled density fit',python=sys.executable,timeout=30)
        result=executed['record']['attempts'][0]['result']
        self.assertEqual(executed['record']['attempts'][0]['status'],'completed',executed['record']['attempts'][0]['error'])
        self.assertEqual(executed['record']['version'],7)
        frozen=candidate_bundle(doc,selection_at(doc,0),workspace.get_spec)
        rows=[r['values'] for r in frozen['dataset']['records'][-2:]]
        expected=restore_task_predictor(frozen,result['checkpoint'],constrained=False)(rows)
        views=[]
        with tempfile.TemporaryDirectory() as directory:
            for mode in ('result-report','rerun'):
                recipe=workspace.put_spec('report',{'version':1,'id':mode,'name':'Gaussian likelihood report',
                    'question':'Do target-free likelihood and posterior diagnostics replay?', 'mode':mode,
                    'experiment':ref if mode=='rerun' else None,'runs':[executed['reference']] if mode=='result-report' else [],
                    'sections':['models','training','evaluation','checkpoints','failures'],
                    'analysis':{'kind':'descriptive','inferential_unit':'run'}})
                before=workspace.db.serialize();captured=create_bundle(workspace,recipe,Path(directory)/mode)
                verify_bundle(captured['path'])
                run=execute_bundle(captured['path'],timeout=60)
                self.assertEqual(run['execution']['status'],'completed',run['execution']['error'])
                root=Path(run['path']);view=json.loads((root/'report-data.json').read_text());views.append(view)
                self.assertEqual(view['runs'][0]['checkpoint_id'],result['checkpoint_id'])
                namespace={};exec(next(root.glob('predictor-*.py')).read_text(),namespace)
                self.assertEqual(namespace['target_fields'],[])
                for key,value in expected.items():torch.testing.assert_close(namespace['predict_records'](rows)[key],value,rtol=0,atol=0)
                self.assertEqual(workspace.db.serialize(),before)
                print(json.dumps({'test':'actual-target-free-gaussian-kernel','mode':mode,
                    'checkpoint_id':result['checkpoint_id'],'code_cells':run['execution']['executed_code_cells']}),flush=True)
            self.assertEqual(views[0]['runs'][0]['metrics'],views[1]['runs'][0]['metrics'])


if __name__=='__main__':unittest.main()
