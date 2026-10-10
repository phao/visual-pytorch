"""Create a concrete experiment workspace without overwriting existing files.

From repository root: python -m examples.experiment_workspace /path/to/new.vtw
Default: save two models x two datasets x two seeds and preview, no training.
--run-first explicitly executes one bounded real worker and stores report evidence.
"""
import argparse
import json
import math
import sys
from pathlib import Path

from app.core import Workspace, default_model, default_mlp
from app.experiment_objects import execute_candidate, object_reference, preview_experiment
from app.task_execution import training_protocol


def populate(workspace):
    models = []
    for model in (default_model('Linear', inputs=1), default_mlp('Tanh MLP', inputs=1, hidden=(8,))):
        workspace.put(model)
        models.append(workspace.freeze_model(model['id']))
    pairs = []
    for kind, teacher in (('affine', lambda x: 1.5 * x - .25),
                          ('nonlinear', lambda x: math.sin(2. * x) + .2 * x)):
        dataset = {'version': 1, 'id': kind, 'name': kind.title() + ' observations',
            'storage': {'kind': 'inline-records', 'version': 1},
            'fields': [{'id': 'signal', 'dtype': 'float32', 'shape': [1]},
                       {'id': 'target', 'dtype': 'float32', 'shape': [1]}],
            'records': [{'id': 'row-' + str(i), 'values': {'signal': [(i - 15.5) / 8.],
                'target': [teacher((i - 15.5) / 8.)]}} for i in range(32)]}
        task = {'version': 1, 'id': kind + '-task', 'name': kind.title() + ' regression',
            'inputs': [{'port': 'input', 'field': 'signal', 'transforms': [{'kind': 'standardize'}]}],
            'heads': [{'id': 'prediction', 'port': 'output', 'field': 'target', 'loss': 'mse',
                       'weight': 1., 'transforms': []}]}
        pairs.append({'id': kind, 'dataset': workspace.put_spec('dataset', dataset),
                      'task': workspace.put_spec('task', task)})
    experiment = {'version': 1, 'id': 'linear-versus-mlp', 'name': 'Linear versus Tanh MLP',
        'question': 'How do Linear and Tanh MLP compare on affine and nonlinear data under the same update budget?',
        'models': models, 'data_tasks': pairs,
        'splits': [{'id': 'split-31', 'protocol': {'method': 'row', 'seed': 31, 'fractions': [.625, .125, .25]}}],
        'initialization_seeds': [7, 13],
        'protocols': [{'id': 'adam-100', 'protocol': training_protocol(
            epochs=None, updates=100, batch_size=5, lr=.03, evaluate_every=20)}],
        'selection': {'kind': 'crossed', 'pairs': [], 'exclusions': []}}
    ref = workspace.put_spec('experiment', experiment)
    workspace.put_spec('report', {'version': 1, 'id': 'rerun-comparison', 'name': 'Explicit rerun comparison',
        'question': experiment['question'], 'mode': 'rerun', 'runs': [], 'experiment': ref,
        'sections': ['question', 'protocol', 'models', 'datasets', 'training', 'evaluation', 'failures'],
        'analysis': {'kind': 'descriptive', 'inferential_unit': 'run'}})
    return workspace.get_spec('experiment', ref['id'], ref['revision'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('workspace')
    parser.add_argument('--run-first', action='store_true')
    parser.add_argument('--torch-python', default=sys.executable)
    args = parser.parse_args(argv)
    path = Path(args.workspace).expanduser().resolve()
    if path.suffix.lower() != '.vtw' or path.exists():
        parser.error('Choose a new .vtw filename; existing files are never overwritten.')
    workspace = Workspace()
    try:
        experiment = populate(workspace)
        workspace.save(path, require_new=True)
        preview = preview_experiment(experiment, workspace.get_spec)
        output = {'path': str(path), 'experiment': object_reference('experiment', experiment),
                  'preview': preview, 'run': None}
        if args.run_first:
            result = execute_candidate(workspace, output['experiment'], preview['candidates'][0]['coordinates'],
                'first-trial', 'First bounded trial', python=args.torch_python, timeout=60)
            output['run'] = result['reference']
            workspace.put_spec('report', {'version': 1, 'id': 'first-trial-report', 'name': 'First trial evidence',
                'question': experiment['question'], 'mode': 'result-report', 'runs': [result['reference']],
                'experiment': None, 'sections': ['question', 'protocol', 'evaluation', 'failures', 'checkpoints'],
                'analysis': {'kind': 'descriptive', 'inferential_unit': 'run'}})
            workspace.save(path)
        print(json.dumps(output, allow_nan=False))
    finally:
        workspace.db.close()


if __name__ == '__main__':
    main()
