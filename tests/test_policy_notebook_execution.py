"""Policy-bearing task checkpoints through actual saved/rerun Jupyter kernels."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
import torch
from app.core import Workspace
from app.experiment_objects import execute_candidate, selection_at
from app.notebooks import create_bundle, execute_bundle, verify_bundle
from app.task_execution import restore_task_predictor
from policy_runtime_fixture import bundle

torch.set_num_threads(1)


class PolicyNotebookExecutionTests(unittest.TestCase):
    def test_actual_saved_and_rerun_kernels_capture_policy_and_reproduce_checkpoint_and_predictions(self):
        b = bundle(updates=3); workspace = Workspace(); workspace.set_constrained_mode(False)
        self.addCleanup(workspace.db.close)
        self.addCleanup(lambda: workspace.history.directory.cleanup() if workspace.history.directory else None)
        refs = {kind: workspace.put_spec(kind, b[kind]) for kind in ('model', 'dataset', 'task')}
        experiment = {'version': 1, 'id': 'policy-trial', 'name': 'LoRA policy trial',
            'question': 'Does frozen-base adaptation survive a portable captured notebook?',
            'models': [refs['model']], 'data_tasks': [{'id': 'pair', 'dataset': refs['dataset'], 'task': refs['task']}],
            'splits': [{'id': 'split', 'protocol': {k: v for k, v in b['manifest']['protocol'].items() if k != 'algorithm'}}],
            'initialization_seeds': [7], 'protocols': [{'id': 'adam', 'protocol': b['protocol']}],
            'selection': {'kind': 'crossed', 'pairs': [], 'exclusions': []}}
        ref = workspace.put_spec('experiment', experiment)
        executed = execute_candidate(workspace, ref, selection_at(experiment, 0), 'fit', 'Tiny adapter fit', python=sys.executable, timeout=30)
        self.assertEqual(executed['record']['attempts'][0]['status'], 'completed', executed['record']['attempts'][0]['error'])
        result = executed['record']['attempts'][0]['result']
        self.assertEqual(result['checkpoint']['version'], 2)
        records = [row['values'] for row in b['dataset']['records'][-3:]]
        expected = restore_task_predictor(b, result['checkpoint'], constrained=False)(records)
        views = []
        with tempfile.TemporaryDirectory() as directory:
            for mode in ('result-report', 'rerun'):
                recipe = workspace.put_spec('report', {'version': 1, 'id': mode, 'name': 'LoRA policy report',
                    'question': 'Are initialization, freezing and evidence portable?', 'mode': mode,
                    'experiment': ref if mode == 'rerun' else None, 'runs': [executed['reference']] if mode == 'result-report' else [],
                    'sections': ['models', 'training', 'evaluation', 'checkpoints', 'failures'],
                    'analysis': {'kind': 'descriptive', 'inferential_unit': 'run'}})
                before = workspace.db.serialize(); capture = create_bundle(workspace, recipe, Path(directory) / mode)
                manifest = verify_bundle(capture['path'])
                for name in ('parameter_policy', 'policy_graph', 'parameter_execution'):
                    self.assertTrue((Path(capture['path']) / 'runtime/app' / (name + '.py')).is_file())
                actual = execute_bundle(capture['path'], timeout=60)
                self.assertEqual(actual['execution']['status'], 'completed', actual['execution']['error'])
                self.assertGreater(actual['execution']['executed_code_cells'], 0)
                root = Path(actual['path']); view = json.loads((root / 'report-data.json').read_text()); views.append(view)
                self.assertEqual(view['runs'][0]['checkpoint_id'], result['checkpoint_id'])
                self.assertEqual(view['notebook_execution']['training_requested'], mode == 'rerun')
                self.assertEqual(workspace.db.serialize(), before)
                predictors = list(root.glob('predictor-*.py')); self.assertEqual(len(predictors), 1)
                namespace = {}; exec(predictors[0].read_text(), namespace)
                torch.testing.assert_close(namespace['predict_records'](records)['y'], expected['y'], rtol=0, atol=0)
                print(json.dumps({'test': 'actual-policy-notebook-kernel', 'mode': mode,
                    'checkpoint_id': result['checkpoint_id'], 'code_cells': actual['execution']['executed_code_cells'],
                    'policy_identity': result['checkpoint']['parameter_policy_receipt']['policy_identity'],
                    'untouched_workspace': True}), flush=True)
            self.assertEqual(views[0]['runs'][0]['metrics'], views[1]['runs'][0]['metrics'])


if __name__ == '__main__': unittest.main()
