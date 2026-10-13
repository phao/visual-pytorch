"""Target-free Gaussian evidence through real task execution, not fake targets."""
import copy
import json
import math
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from app.core import GraphError, Workspace, validate_model
from app.datasets import prepare, validate_task, validate_task_spec
from app.revisions import model_document
from app.task_execution import (task_bundle, training_protocol, validate_bundle,
                               restore_task_predictor, generate_task_predictor)
from app.task_runtime import head_terms, MetricAccumulator, tensor_batch, validate_forward_outputs
from app.task_training import train_task
from app.task_result import validate_result
from app.transport import run_json_worker
from gaussian_graph_fixture import model
from test_datasets import data, manifest

torch.set_num_threads(1)


def fixture():
    drawing = model()
    fields = [{'id': 'observations', 'dtype': 'float32', 'shape': [4, 1], 'axes': ['T', 'D']},
              {'id': 'lengths', 'dtype': 'float32', 'shape': [1]}]
    rows = [{'observations': [[.04 * (i - 3) + .01 * t] for t in range(4)], 'lengths': [4.]}
            for i in range(8)]
    dataset = data(fields, rows)
    task = {'version': 2, 'id': 'likelihood', 'name': 'Unlabelled Gaussian sequence density',
            'inputs': [{'port': p, 'field': p, 'transforms': []} for p in ('observations', 'lengths')],
            'heads': [{'id': 'density', 'port': 'out_sequence_evidence',
                       'loss': 'negative-log-likelihood', 'weight': 1.}],
            'diagnostic_outputs': ['out_filtered', 'out_final', 'out_step_evidence']}
    return drawing, dataset, task


def bundle(updates=4, *, narrow=False):
    drawing, dataset, task = fixture()
    if narrow:
        owners = drawing['graph']['parameter_policy']['parameters']
        owners['["means","value"]']['initializer']['values'] = [[-.04], [.04]]
        owners['["raw_variances","value"]'] = {
            'initializer': {'kind': 'values', 'dtype': 'float32', 'values': [[-4.], [-4.]]}}
    return task_bundle(model_document({'hmm': drawing}, 'hmm'), dataset, task, manifest(dataset),
                       training_protocol(epochs=None, updates=updates, batch_size=2, lr=.02, evaluate_every=1))


class LikelihoodTaskTests(unittest.TestCase):
    def test_target_free_preparation_does_not_make_target_columns(self):
        drawing, dataset, task = fixture()
        contract = validate_model({'hmm': drawing}, 'hmm')
        prepared = prepare(dataset, task, contract, manifest(dataset), constrained=False)
        for population in prepared['populations'].values():
            self.assertEqual(population['targets'], {})
        batch = tensor_batch(prepared, task, 'train', [0, 1])
        self.assertEqual(batch['targets'], {})
        self.assertEqual(prepared['fitted']['heads']['density'], {'kind': 'target-free-log-evidence'})

    def test_signed_density_reduction_and_gradient_have_independent_oracle(self):
        log_evidence = torch.tensor([[2.], [3.], [-1.]], dtype=torch.float64, requires_grad=True)
        terms = head_terms(log_evidence, None, torch.tensor([1., 1., 0.]), torch.tensor([1., 2., 5.]),
                           'negative-log-likelihood')
        self.assertEqual(float(terms['numerator'].detach()), -8.)
        self.assertEqual(float(terms['denominator']), 3.)
        terms['mean'].backward()
        torch.testing.assert_close(log_evidence.grad, torch.tensor([[-1/3], [-2/3], [0.]], dtype=torch.float64))
        accumulator = MetricAccumulator('negative-log-likelihood', 'fixed', 'train')
        accumulator.add(terms, ['a', 'b', 'c'], checkpoint='fixed', population='train')
        self.assertEqual(accumulator.report()['value'], -8/3)
        with self.assertRaises(GraphError):
            head_terms(log_evidence, torch.zeros_like(log_evidence), torch.ones(3), torch.ones(3), 'negative-log-likelihood')
        empty = head_terms(log_evidence, None, torch.zeros(3), torch.ones(3), 'negative-log-likelihood')
        self.assertIsNone(empty['mean']); self.assertEqual(float(empty['numerator'].detach()), 0.)

    def test_version_and_representation_fences_do_not_guess_from_names(self):
        drawing, dataset, task = fixture(); contract = validate_model({'hmm': drawing}, 'hmm')
        for change in ('v1', 'target', 'missing-diagnostic', 'overlap', 'posterior', 'unproven'):
            bad = copy.deepcopy(task); plan = copy.deepcopy(contract)
            if change == 'v1': bad['version'] = 1
            if change == 'target': bad['heads'][0]['field'] = 'observations'
            if change == 'missing-diagnostic': bad['diagnostic_outputs'].pop()
            if change == 'overlap': bad['diagnostic_outputs'].append('out_sequence_evidence')
            if change == 'posterior': bad['heads'][0]['port'] = 'out_filtered'
            if change == 'unproven':
                next(g for g in plan['output_groups'] if g['id'] == 'out_sequence_evidence').pop('representation')
            with self.subTest(change=change), self.assertRaises(GraphError): validate_task(bad, dataset, plan)
        frozen = bundle(); self.assertEqual(frozen['version'], 5)
        frozen['version'] = 1
        with self.assertRaises(GraphError): validate_bundle(frozen)
        for version in (True, 0, 3):
            bad = copy.deepcopy(task); bad['version'] = version
            with self.assertRaises(GraphError): validate_task_spec(bad)

    def test_real_training_checkpoint_export_and_signed_result_admission(self):
        frozen = bundle(30, narrow=True); result = train_task(frozen, constrained=False)
        self.assertEqual(result['status'], 'completed', result['failure'])
        self.assertEqual(result['counts']['completed_updates'], 30)
        self.assertLess(result['final_metrics']['train']['objective'], 0.)
        stored = {k: v for k, v in result.items() if k != 'bundle'}
        validate_result(frozen, stored)
        predictor = restore_task_predictor(frozen, result['checkpoint'], constrained=False)
        namespace = {}; exec(generate_task_predictor(frozen, result['checkpoint'], constrained=False), namespace)
        records = [r['values'] for r in frozen['dataset']['records']]
        expected = predictor(records); actual = namespace['predict_records'](records)
        self.assertEqual(namespace['target_fields'], [])
        for key in expected: torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
        for bad_value in (math.nan, math.inf, -math.inf, 999.):
            corrupt = copy.deepcopy(stored)
            corrupt['final_metrics']['train']['heads']['density']['value'] = bad_value
            with self.assertRaises(GraphError): validate_result(frozen, corrupt)
        corrupt = copy.deepcopy(stored); corrupt['version'] = 1
        with self.assertRaises(GraphError): validate_result(frozen, corrupt)

    def test_negative_nll_curve_coordinates_stay_inside_axes(self):
        import re
        from app.report_runtime import curve_svg
        result = train_task(bundle(4, narrow=True), constrained=False)
        svg = curve_svg({'reference': {'id': 'density-fit', 'revision': 'a' * 64}}, result, 'density',
                        {'model': 'Gaussian HMM', 'dataset': 'Unlabelled trajectories', 'unit': '', 'transforms': []})
        self.assertIn('negative-log-likelihood', svg)
        self.assertIn('>-', svg)
        import xml.etree.ElementTree as ET
        metadata = json.loads(next(e.text for e in ET.fromstring(svg) if e.tag.endswith('metadata')))
        self.assertLess(metadata['y_range'][0],0.)
        self.assertEqual(metadata['y_range'][1],0.)
        axes = re.search(r'<path d="M 72 ([\d.]+) V ([\d.]+) H 732"', svg)
        top, bottom = map(float, axes.groups())
        for coordinates in re.findall(r'<polyline[^>]*points="([^"]+)"', svg):
            for pair in coordinates.split():
                x, y = map(float, pair.split(','))
                self.assertTrue(72 <= x <= 732)
                self.assertTrue(top - 1e-5 <= y <= bottom + 1e-5)

    def test_structural_minus_infinity_is_only_valid_for_proven_diagnostics(self):
        contract = {'output_groups': [{'id': 'posterior', 'shape': [2], 'representation': 'log-probabilities'},
                                      {'id': 'evidence', 'shape': [1], 'representation': 'log-evidence'}]}
        task = {'version': 2, 'diagnostic_outputs': ['posterior']}
        values = {'posterior': torch.tensor([[0., -math.inf]]), 'evidence': torch.tensor([[2.]])}
        validate_forward_outputs(values, contract, task)
        for value in (math.nan, math.inf):
            bad = copy.deepcopy(values); bad['posterior'][0, 1] = value
            with self.assertRaises(GraphError): validate_forward_outputs(bad, contract, task)
        with self.assertRaises(GraphError): validate_forward_outputs(values, contract, {'version': 1})
        bad = copy.deepcopy(values); bad['evidence'][0, 0] = -math.inf
        with self.assertRaises(GraphError): validate_forward_outputs(bad, contract, task)

    def test_actual_worker_matches_parent_and_exact_request_version(self):
        frozen = bundle()
        parent = train_task(frozen, constrained=False)
        worker = run_json_worker([sys.executable, '-u', '-m', 'app.worker'],
                                {'kind': 'task-v5', 'bundle': frozen, 'constrained': False},
                                cwd=str(Path(__file__).resolve().parents[1]))
        self.assertEqual(worker['checkpoint_id'], parent['checkpoint_id'])
        self.assertEqual(worker['history'], parent['history'])
        self.assertEqual(worker['final_metrics'], parent['final_metrics'])
        validate_result(frozen, {k: v for k, v in worker.items() if k != 'bundle'})

    def test_vtw_archived_task2_requires_its_container_version(self):
        _, _, task = fixture(); workspace = Workspace(); self.addCleanup(workspace.db.close)
        workspace.put_spec('task', task); workspace.archive_spec('task', task['id'])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'likelihood.vtw'; workspace.save(path)
            loaded = Workspace.load(path); loaded.db.close()
            with sqlite3.connect(path) as database: database.execute('PRAGMA user_version=28')
            with self.assertRaisesRegex(GraphError, 'version-29'): Workspace.load(path)


if __name__ == '__main__': unittest.main()
