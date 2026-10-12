"""Two genuine notebook kernels from a retained Family7 experiment.

Run with PYTHONPATH=.:tests in the numerical environment. A new output folder is
optional; otherwise all artifacts are temporary. No user workspace is opened.
"""
import json
import sys
import tempfile
from pathlib import Path

from app.core import Workspace
from app.experiment_objects import execute_candidate, object_reference, preview_experiment
from app.notebooks import create_bundle, execute_bundle, verify_bundle
from app.revisions import validate_all
from examples.lstm_family import definition
from test_lstm_task_execution import lstm_bundle


def check(folder):
    base = lstm_bundle(2); workspace = Workspace(); workspace.set_constrained_mode(False)
    try:
        refs = {kind: workspace.put_spec(kind, base[kind]) for kind in ('dataset', 'task')}
        family = workspace.put_spec('family', definition())
        experiment = {'version': 3, 'id': 'lstm-family-trial', 'name': 'Explicit LSTM family trial',
                      'question': 'Do retained constructors and typed bias axes reproduce?',
                      'models': [{'id': 'chosen', 'family': family,
                                  'bindings': {'time': 3, 'width': 1, 'cell-width': 3, 'layers': 1,
                                               'projection': 2, 'directions': True},
                                  'axes': [{'symbol': 'bias', 'values': [True]}]}],
                      'data_tasks': [{'id': 'pair', 'dataset': refs['dataset'], 'task': refs['task']}],
                      'splits': [{'id': 'split', 'protocol': {'method': 'row', 'seed': 31, 'fractions': [.65, .15, .2]}}],
                      'initialization_seeds': [7], 'protocols': [{'id': 'adam', 'protocol': base['protocol']}],
                      'selection': {'kind': 'crossed', 'pairs': [], 'exclusions': []}}
        ref = workspace.put_spec('experiment', experiment)
        frozen = workspace.get_spec('experiment', ref['id'], ref['revision'])
        selected = preview_experiment(frozen, workspace.get_spec)['candidates'][0]
        assert selected['status'] == 'compatible', selected
        run = execute_candidate(workspace, object_reference('experiment', frozen), selected['coordinates'],
                                'actual-family-run', 'Actual family trial', timeout=30)
        attempt = run['record']['attempts'][-1]
        assert attempt['status'] == 'completed', attempt['error']
        spec = run['record']['resolved']['references']['specialization']
        retained = workspace.get_spec(spec['kind'], spec['id'], spec['revision'])
        assert retained['version'] == 6 and retained['runtime'] is None
        assert retained['bindings']['bias'] is True
        assert workspace.get_spec('model', retained['model']['id'], retained['model']['revision'])['version'] == 4
        views = []
        for mode in ('result-report', 'rerun'):
            recipe = {'version': 1, 'id': mode, 'name': 'LSTM family evidence',
                      'question': 'Can the exact family candidate be reproduced?', 'mode': mode,
                      'runs': [run['reference']] if mode == 'result-report' else [],
                      'experiment': object_reference('experiment', frozen) if mode == 'rerun' else None,
                      'sections': ['models', 'training', 'evaluation', 'checkpoints', 'failures'],
                      'analysis': {'kind': 'descriptive', 'inferential_unit': 'run'}}
            report = workspace.put_spec('report', recipe)
            exported = create_bundle(workspace, report, folder/mode); verify_bundle(exported['path'])
            assert (Path(exported['path'])/'runtime/app/lstm_nodes.py').is_file()
            result = execute_bundle(exported['path'], timeout=60)
            assert result['execution']['status'] == 'completed', result['execution']['error']
            view = json.loads((Path(result['path'])/'report-data.json').read_text())
            assert view['runs'][0]['status'] == 'completed'
            assert view['runs'][0]['checkpoint_id'] == attempt['result']['checkpoint_id']
            assert view['notebook_execution']['training_requested'] == (mode == 'rerun')
            views.append(view)
        assert views[0]['runs'][0]['metrics'] == views[1]['runs'][0]['metrics']
        validate_all(workspace.db); workspace.save(folder/'family-experiment.vtw')
        loaded = Workspace.load(folder/'family-experiment.vtw')
        try: assert loaded.get_spec('specialization', spec['id'], spec['revision']) == retained
        finally: loaded.db.close()
        print(json.dumps({'ok': True, 'family7': True, 'specialization6': True, 'experiment3_typed_axis': True,
                          'model4': True, 'real_worker': True, 'actual_kernels': ['result-report', 'rerun'],
                          'metrics_exact': True, 'checkpoint_exact': True, 'vtw18_reload': True,
                          'runtime_T': False, 'output': str(folder)}))
    finally:
        workspace.db.close()
        if workspace.history.directory: workspace.history.directory.cleanup()


if __name__ == '__main__':
    if len(sys.argv) == 2:
        folder = Path(sys.argv[1]); folder.mkdir(); check(folder)
    elif len(sys.argv) == 1:
        with tempfile.TemporaryDirectory() as temporary: check(Path(temporary))
    else: raise SystemExit('Usage: probe_lstm_family_experiment.py [NEW_OUTPUT_FOLDER]')
