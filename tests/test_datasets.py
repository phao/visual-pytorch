"""Dataset/task tests use named observations and independently computed values."""
import copy
import json
import io
import math
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from contextlib import redirect_stdout
from unittest.mock import patch

import torch
from app.core import GraphError, Workspace, compile_model, default_model, default_cnn, validate_model
from app.datasets import (capabilities, check_manifest, freeze_dataset, make_split,
                          prepare, preview, validate_dataset, validate_task)

torch.set_num_threads(1)


def data(fields=None, values=None):
    fields = fields or [{'id': 'x', 'dtype': 'float32', 'shape': [1]},
                        {'id': 'y', 'dtype': 'float32', 'shape': [1]},
                        {'id': 'subject', 'dtype': 'string', 'shape': []},
                        {'id': 'time', 'dtype': 'int64', 'shape': []}]
    values = values or [{'x': [float(i)], 'y': [float(2 * i + 1)], 'subject': 's' + str(i // 2), 'time': i} for i in range(8)]
    return {'version': 1, 'id': 'observations', 'name': 'Paired measurements',
            'storage': {'kind': 'inline-records', 'version': 1}, 'fields': fields,
            'records': [{'id': 'r' + str(i), 'values': row} for i, row in enumerate(values)],
            'provenance': {'procedure': 'deterministic test fixture'}}


def task(transforms=None, target_transforms=None):
    return {'version': 1, 'id': 'task', 'name': 'Predict measurements',
            'inputs': [{'port': 'input', 'field': 'x', 'transforms': transforms or []}],
            'heads': [{'id': 'prediction', 'port': 'output', 'field': 'y', 'loss': 'mse', 'weight': 1., 'transforms': target_transforms or []}]}


def manifest(dataset, train=None, validation=None, test=None, excluded=None):
    return make_split(dataset, {'method': 'external', 'roles': {
        'train': train or ['r0', 'r1'], 'validation': ['r2'] if validation is None else validation,
        'test': [r['id'] for r in dataset['records'][3:]] if test is None else test}, 'excluded': excluded or []})


def contract(model=None):
    model = model or default_model(inputs=1, outputs=1)
    return model, validate_model({model['id']: model}, model['id'])


class DatasetTests(unittest.TestCase):
    def test_all_dtype_shapes_and_context_are_preserved(self):
        d = data()
        d['fields'] += [{'id': 'flag', 'dtype': 'bool', 'shape': []},
                        {'id': 'label', 'dtype': 'category', 'shape': [], 'vocabulary': {'version': 1, 'values': ['1', 1]}},
                        {'id': 'image', 'dtype': 'float32', 'shape': [1, 2, 3], 'axes': ['C', 'H', 'W'], 'unit': 'kelvin'}]
        for row in d['records']:
            row['values'].update(flag=True, label='1', image=[[[1., 2., 3.], [4., 5., 6.]]])
        f = freeze_dataset(d)
        self.assertNotIn('revision', d)
        self.assertEqual(validate_dataset(f)['image']['unit'], 'kelvin')
        self.assertEqual(capabilities(d), {'adapter': 'inline-records-v1', 'random_access': True, 'known_cardinality': 8, 'repeatable': True, 'resumable_position': False})
        f['records'][0]['values']['x'][0] = 9.
        with self.assertRaisesRegex(GraphError, 'revision'):
            validate_dataset(f)

    def test_revisions_change_for_bytes_schema_vocabulary_and_context(self):
        d = data(); revision = freeze_dataset(d)['revision']
        for change in ('bytes', 'unit', 'order', 'provenance', 'name'):
            candidate = copy.deepcopy(d)
            if change == 'bytes': candidate['records'][0]['values']['x'][0] += .1
            if change == 'unit': candidate['fields'][0]['unit'] = 'meters'
            if change == 'order': candidate['fields'].reverse()
            if change == 'provenance': candidate['provenance']['procedure'] = 'another procedure'
            if change == 'name': candidate['name'] = 'Renamed dataset'
            self.assertNotEqual(revision, freeze_dataset(candidate)['revision'])

    def test_malformed_datasets_reject_without_scalar_coercion(self):
        changes = [lambda d: d.update(version=True), lambda d: d.update(version=2),
            lambda d: d['storage'].update(version=True), lambda d: d['storage'].update(kind='imaginary-stream'),
            lambda d: d['fields'][0].update(dtype=['float32']), lambda d: d['fields'][0].update(shape=[True]),
            lambda d: d['fields'][0].update(shape=[1, -1]), lambda d: d['fields'][0].update(axes=['F', 'F']),
            lambda d: d['fields'][0].update(nullable=1), lambda d: d['records'][0]['values'].update(x=[True]),
            lambda d: d['records'][0]['values'].update(x=[float('inf')]),
            lambda d: d['records'][0]['values'].update(x=[2**2000]),
            lambda d: d['records'][0]['values'].update(x=[None]), lambda d: d['records'][0]['values'].update(x=[]),
            lambda d: d['records'][0]['values'].update(extra=1), lambda d: d['records'][1].update(id='r0'),
            lambda d: d['fields'].append(copy.deepcopy(d['fields'][0])), lambda d: d.update(unrecognized=True)]
        for change in changes:
            d = data(); change(d)
            with self.subTest(document=repr(d)[:100]), self.assertRaises(GraphError): validate_dataset(d)

    def test_category_ontology_is_fixed_not_inferred_from_observed_labels(self):
        fields = [{'id': 'label', 'dtype': 'category', 'shape': [], 'vocabulary': {'version': 1, 'values': ['z', 'a', 'never-observed']}}]
        d = data(fields, [{'label': 'z'}, {'label': 'a'}])
        validate_dataset(d)
        for labels in (['z', 'z'], ['z', True], ['z'], ['z', []]):
            bad = copy.deepcopy(d); bad['fields'][0]['vocabulary']['values'] = labels
            with self.assertRaises(GraphError): validate_dataset(bad)
        d['records'][0]['values']['label'] = 'unknown'
        with self.assertRaises(GraphError): validate_dataset(d)

    def test_nullable_whole_fields_and_tensor_leaves_are_explicit(self):
        d = data(); d['fields'][0]['nullable'] = True
        d['records'][0]['values']['x'] = None; d['records'][1]['values']['x'] = [None]
        validate_dataset(d)
        _, c = contract()
        with self.assertRaisesRegex(GraphError, 'missing'): prepare(d, task(), c, manifest(d), constrained=False)

    def test_preview_is_bounded_deep_owned_and_rng_neutral(self):
        d = data(); before = copy.deepcopy(d); rng = random.getstate()
        a = preview(d, 3); d['records'].reverse(); b = preview(d, 3)
        self.assertEqual(a, b); self.assertEqual(random.getstate(), rng)
        self.assertEqual(a['fields']['x']['range'], [0., 2.]); self.assertEqual(a['selected_records'], 3)
        self.assertEqual(a['scope'], 'bounded-preview-not-a-dataset-audit')
        a['records'][0]['values']['x'][0] = 200.
        self.assertEqual(d['records'][::-1], before['records'])
        for limit in (True, 0, 101, 1.5):
            with self.assertRaises(GraphError): preview(d, limit)

    def test_module_remains_importable_without_qt_torch_numpy(self):
        probe = subprocess.run([sys.executable, '-c', "import sys; import app.datasets, app.task_runtime; assert not any(k in sys.modules for k in ('torch','numpy','PyQt6'))"], capture_output=True, text=True, timeout=10)
        self.assertEqual(probe.returncode, 0, probe.stderr)

    def test_real_cli_preparation_manifest_replay_is_pure_and_read_only(self):
        with tempfile.TemporaryDirectory() as td:
            directory = Path(td); path = directory / 'lab.vtw'; w = Workspace(); model, _ = contract()
            w.put(model); w.save(path); w.db.close(); before = path.read_bytes()
            dataset = data(); binding = task([{'kind': 'standardize'}]); split = {'method': 'row', 'seed': 3, 'fractions': [.5, .25, .25]}
            for name, value in (('data', dataset), ('task', binding), ('split', split)):
                (directory / (name + '.json')).write_text(json.dumps(value))
            base = [str(path), 'prepare-task', model['id'], '--dataset', str(directory / 'data.json'), '--task', str(directory / 'task.json'), '--memory-limit-percent', '10']
            pure_probe = "import sys, vpw_cli; code=vpw_cli.main(sys.argv[1:]); assert not any(k in sys.modules for k in ('torch','numpy','PyQt6')); raise SystemExit(code)"
            result = subprocess.run([sys.executable, '-c', pure_probe, *base, '--split', str(directory / 'split.json')], capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            value = json.loads(result.stdout)['value']; self.assertEqual(path.read_bytes(), before)
            saved = directory / 'manifest.json'; saved.write_text(json.dumps(value['manifest']))
            replay = subprocess.run([sys.executable, '-c', pure_probe, *base, '--manifest', str(saved)], capture_output=True, text=True, timeout=10)
            self.assertEqual(replay.returncode, 0, replay.stdout + replay.stderr)
            self.assertEqual(json.loads(replay.stdout)['value'], value); self.assertEqual(path.read_bytes(), before)
            dataset['records'][0]['values']['x'][0] += .1; (directory / 'data.json').write_text(json.dumps(dataset))
            stale = subprocess.run([sys.executable, '-c', pure_probe, *base, '--manifest', str(saved)], capture_output=True, text=True, timeout=10)
            self.assertEqual(stale.returncode, 1); self.assertIn('revision', json.loads(stale.stdout)['error']); self.assertEqual(path.read_bytes(), before)

    def test_real_cli_preview_is_bounded_and_does_not_fit_or_edit(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'empty.vtw'; w = Workspace(); w.save(path); w.db.close(); before = path.read_bytes()
            observations = Path(td) / 'data.json'; observations.write_text(json.dumps(data()))
            result = subprocess.run([sys.executable, 'vpw_cli.py', str(path), 'preview-dataset', str(observations), '--limit', '2'], capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            value = json.loads(result.stdout)['value']; self.assertEqual(value['selected_records'], 2)
            self.assertNotIn('fitted', value); self.assertEqual(path.read_bytes(), before)

    def test_cli_uses_readonly_user_ram_preference_and_honors_override(self):
        import vpw_cli
        from app.config import Config
        with tempfile.TemporaryDirectory() as td:
            directory = Path(td); path = directory / 'lab.vtw'; w = Workspace(); model, _ = contract()
            w.put(model); w.save(path); w.db.close(); before = path.read_bytes()
            for name, value in (('data', data()), ('task', task()), ('split', {'method': 'row', 'seed': 3, 'fractions': [.5, .25, .25]})):
                (directory / (name + '.json')).write_text(json.dumps(value))
            config_path = directory / 'config.json'; config_path.write_text(json.dumps({'memory_limit_percent': 5.}))
            args = [str(path), 'prepare-task', model['id'], '--dataset', str(directory / 'data.json'), '--task', str(directory / 'task.json'), '--split', str(directory / 'split.json')]
            def run(extra=()):
                stream = io.StringIO()
                with patch('app.config.Config', return_value=Config(config_path)), patch('app.memory.available_memory', return_value=128 * 1024**2), redirect_stdout(stream):
                    code = vpw_cli.main(args + list(extra))
                return code, json.loads(stream.getvalue())
            self.assertEqual(run()[0], 1)  # 5% < the preparation runtime allowance
            self.assertEqual(run(('--memory-limit-percent', '10'))[0], 0)
            self.assertEqual(json.loads(config_path.read_text()), {'memory_limit_percent': 5.})
            config_path.write_text('{broken'); invalid_before = config_path.read_bytes()
            self.assertEqual(run()[0], 1)
            self.assertEqual(run(('--memory-limit-percent', '10'))[0], 0)
            self.assertEqual(config_path.read_bytes(), invalid_before); self.assertEqual(path.read_bytes(), before)


class SplitTests(unittest.TestCase):
    def test_row_split_is_reproducible_and_records_actual_ids(self):
        d = data(); proto = {'method': 'row', 'seed': 17, 'fractions': [.5, .25, .25]}
        rng = random.getstate(); a = make_split(d, proto); d['records'].reverse(); b = make_split(d, proto)
        self.assertEqual(a['roles'], b['roles']); self.assertNotEqual(a['dataset_revision'], b['dataset_revision'])
        self.assertEqual(random.getstate(), rng); self.assertNotIn('algorithm', proto)
        self.assertEqual([len(a['roles'][r]) for r in ('train', 'validation', 'test')], [4, 2, 2])
        self.assertEqual(a['roles'], {'train': ['r3', 'r1', 'r0', 'r4'], 'validation': ['r5', 'r2'], 'test': ['r7', 'r6']})

    def test_group_split_is_subject_disjoint_and_counts_units_not_rows(self):
        d = data()
        for row, subject in zip(d['records'], ('s0', 's0', 's0', 's0', 's1', 's2', 's3', 's3')):
            row['values']['subject'] = subject
        m = make_split(d, {'method': 'group', 'field': 'subject', 'seed': 2, 'fractions': [.5, .25, .25]})
        self.assertEqual([len(m['roles'][r]) for r in ('train', 'validation', 'test')], [2, 2, 4])
        rows = {r['id']: r['values'] for r in d['records']}
        owners = {}
        for role, ids in m['roles'].items():
            for rid in ids:
                subject = rows[rid]['subject']; self.assertEqual(owners.setdefault(subject, role), role)
        broken = copy.deepcopy(m); a = broken['roles']['train'][0]; b = broken['roles']['test'][0]
        broken['roles']['train'][0] = b; broken['roles']['test'][0] = a
        with self.assertRaisesRegex(GraphError, 'group'): check_manifest(d, broken)

    def test_temporal_split_keeps_ties_together_and_rejects_future_mixing(self):
        d = data()
        for row in d['records']: row['values']['time'] //= 2
        m = make_split(d, {'method': 'temporal', 'field': 'time', 'fractions': [.5, .25, .25]})
        self.assertEqual(m['roles'], {'train': ['r0', 'r1', 'r2', 'r3'], 'validation': ['r4', 'r5'], 'test': ['r6', 'r7']})
        m['roles']['train'][0], m['roles']['test'][0] = m['roles']['test'][0], m['roles']['train'][0]
        with self.assertRaisesRegex(GraphError, 'chronological'): check_manifest(d, m)

    def test_external_split_requires_complete_disjoint_selection(self):
        d = data(); m = manifest(d, test=['r3', 'r4', 'r5', 'r6'], excluded=['r7'])
        self.assertEqual(m['excluded'], ['r7'])
        for mutation in ('overlap', 'missing', 'phantom', 'duplicate', 'empty-train', 'revision'):
            bad = copy.deepcopy(m)
            if mutation == 'overlap': bad['roles']['test'].append('r0')
            if mutation == 'missing': bad['excluded'] = []
            if mutation == 'phantom': bad['roles']['test'][0] = 'phantom'
            if mutation == 'duplicate': bad['roles']['train'].append('r0')
            if mutation == 'empty-train': bad['roles']['test'] += bad['roles']['train']; bad['roles']['train'] = []
            if mutation == 'revision': bad['dataset_revision'] = '0' * 64
            with self.subTest(mutation=mutation), self.assertRaises(GraphError): check_manifest(d, bad)

    def test_malformed_split_protocols_are_not_guessed(self):
        d = data()
        for proto in ({'method': 'row'}, {'method': 'random'},
                      {'method': 'row', 'seed': True, 'fractions': [.5, .25, .25]},
                      {'method': 'row', 'seed': 1, 'fractions': [.5, .25, .3]},
                      {'method': 'group', 'seed': 1, 'field': 'x', 'fractions': [.5, .25, .25]},
                      {'method': 'temporal', 'field': 'subject', 'fractions': [.5, .25, .25]},
                      {'method': 'row', 'seed': 1, 'fractions': [.99, .005, .005]}):
            with self.subTest(protocol=proto), self.assertRaises(GraphError): make_split(d, proto)

    def test_zero_validation_fraction_is_explicit(self):
        d = data(); m = make_split(d, {'method': 'row', 'seed': 3, 'fractions': [.75, 0., .25]})
        self.assertEqual(m['roles']['validation'], [])


class PreparationTests(unittest.TestCase):
    def test_train_only_standardization_known_scalar_forward_and_gradient(self):
        d = data(); d['records'][1]['values']['x'] = [2.]; d['records'][1]['values']['y'] = [5.]
        d['records'][-1]['values']['x'] = [10000.]; d['records'][-1]['values']['y'] = [20001.]
        model, c = contract(); t = task([{'kind': 'standardize'}]); m = manifest(d)
        before = json.dumps((d, t, c, m), sort_keys=True)
        p = prepare(d, t, c, m, constrained=False)
        state = p['fitted']['inputs']['input']['steps'][0]
        self.assertEqual(state['mean'], [1.]); self.assertEqual(state['scale'], [1.])
        self.assertEqual(p['populations']['train']['inputs']['input'], [[-1.], [1.]])
        self.assertEqual(p['populations']['test']['inputs']['input'][-1], [9999.])
        self.assertEqual(p['fitted']['fit_observation_ids'], ['r0', 'r1'])
        module = compile_model({model['id']: model}, model['id'])
        with torch.no_grad(): module.layer_0.weight.fill_(2.); module.layer_0.bias.fill_(3.)
        x = torch.tensor(p['populations']['test']['inputs']['input'], dtype=torch.float32, requires_grad=True)
        y = module({c['input_groups'][0]['name']: x})
        self.assertEqual(y.tolist(), p['populations']['test']['targets']['prediction'])
        y.sum().backward(); self.assertEqual(x.grad.tolist(), [[2.]] * len(x))
        self.assertEqual(json.dumps((d, t, c, m), sort_keys=True), before)

    def test_fitted_transform_chain_impute_units_then_scale_never_uses_test(self):
        d = data(); d['fields'][0]['nullable'] = True
        d['records'][0]['values']['x'] = [10.]; d['records'][1]['values']['x'] = [None]
        d['records'][2]['values']['x'] = [500.]; d['records'][3]['values']['x'] = None
        _, c = contract(); transforms = [{'kind': 'impute-mean'}, {'kind': 'affine', 'scale': 2., 'offset': -10.}, {'kind': 'standardize'}]
        p = prepare(d, task(transforms), c, manifest(d), constrained=False)
        state = p['fitted']['inputs']['input']['steps']
        self.assertEqual(state[0]['mean'], [10.]); self.assertEqual(state[2]['mean'], [10.]); self.assertEqual(state[2]['scale'], [1.])
        self.assertEqual(p['populations']['validation']['inputs']['input'], [[980.]])
        self.assertEqual(p['populations']['test']['inputs']['input'][0], [0.])
        d['records'][0]['values']['x'] = None
        with self.assertRaisesRegex(GraphError, 'every training'): prepare(d, task(transforms), c, manifest(d), constrained=False)

    def test_target_transforms_are_fitted_independently_on_training(self):
        d = data(); _, c = contract()
        p = prepare(d, task(target_transforms=[{'kind': 'standardize'}]), c, manifest(d), constrained=False)
        self.assertEqual(p['fitted']['heads']['prediction']['steps'][0]['mean'], [2.])
        self.assertEqual(p['populations']['train']['targets']['prediction'], [[-1.], [1.]])

    def test_bindings_use_ids_and_fail_explicitly_on_shape_axes_change(self):
        d = data(); model, c = contract(); t = task()
        model['graph']['nodes'][0]['label'] = 'New human name'
        c2 = validate_model({model['id']: model}, model['id']); validate_task(t, d, c2)
        for key, value in (('port', 'unknown'), ('field', 'unknown')):
            bad = copy.deepcopy(t); bad['inputs'][0][key] = value
            with self.assertRaises(GraphError): validate_task(bad, d, c)
        changed = copy.deepcopy(c); changed['input_groups'][0]['shape'] = [2]
        with self.assertRaises(GraphError): validate_task(t, d, changed)
        changed = copy.deepcopy(c); changed['input_groups'][0]['axes'] = ['F']
        with self.assertRaisesRegex(GraphError, 'axes'): validate_task(t, d, changed)
        d['fields'][0]['axes'] = ['F']; validate_task(t, d, changed)

    def test_classification_targets_use_full_fixed_vocab_with_real_torch_loss(self):
        d = data(); d['fields'][1] = {'id': 'y', 'dtype': 'category', 'shape': [], 'vocabulary': {'version': 1, 'values': ['right', 'left', 'unseen']}}
        for i, row in enumerate(d['records']): row['values']['y'] = 'left' if i % 2 else 'right'
        model, c = contract(default_model(inputs=1, outputs=3)); t = task(); t['heads'][0]['loss'] = 'cross-entropy'
        p = prepare(d, t, c, manifest(d), constrained=False)
        self.assertEqual(p['populations']['train']['targets']['prediction'], [0, 1])
        logits = torch.tensor([[0., 1., 2.], [1., 2., 3.]], requires_grad=True)
        labels = torch.tensor(p['populations']['train']['targets']['prediction'], dtype=torch.int64)
        loss = torch.nn.functional.cross_entropy(logits, labels); loss.backward()
        expected = ((math.log(1 + math.e + math.exp(2))) + (math.log(math.e + math.exp(2) + math.exp(3)) - 2)) / 2
        self.assertAlmostEqual(float(loss.detach()), expected, places=6)
        probs = [math.exp(v) / (1 + math.e + math.exp(2)) for v in (0, 1, 2)]
        self.assertAlmostEqual(float(logits.grad[0, 0]), (probs[0] - 1) / 2, places=6)
        wrong = default_model(inputs=1, outputs=2)
        with self.assertRaisesRegex(GraphError, 'vocabulary'): validate_task(t, d, contract(wrong)[1])

    def test_onehot_is_ordered_and_numeric_scalar_input_becomes_vector(self):
        d = data(); d['fields'][0] = {'id': 'x', 'dtype': 'category', 'shape': [], 'vocabulary': {'version': 1, 'values': ['b', 'a']}}
        for i, row in enumerate(d['records']): row['values']['x'] = 'a' if i % 2 else 'b'
        _, c = contract(default_model(inputs=2, outputs=1))
        p = prepare(d, task([{'kind': 'onehot'}]), c, manifest(d), constrained=False)
        self.assertEqual(p['populations']['train']['inputs']['input'], [[1., 0.], [0., 1.]])
        d = data(); d['fields'][0]['shape'] = []
        for r in d['records']: r['values']['x'] = r['values']['x'][0]
        p = prepare(d, task(), contract()[1], manifest(d), constrained=False)
        self.assertEqual(p['populations']['train']['inputs']['input'], [[0.], [1.]])

    def test_same_image_field_input_and_reconstruction_target_no_source_duplication(self):
        model = default_cnn(); model['graph']['nodes'] = [{'id': 'input', 'type': 'Input', 'shape': [1, 2, 3], 'axes': ['C', 'H', 'W']}, {'id': 'output', 'type': 'Output'}]
        model['graph']['edges'] = [{'from': 'input', 'to': 'output'}]
        d = data([{'id': 'image', 'dtype': 'float32', 'shape': [1, 2, 3], 'axes': ['C', 'H', 'W']}],
                 [{'image': [[[i, i + 1, i + 2], [i + 3, i + 4, i + 5]]]} for i in range(8)])
        t = task(); t['inputs'][0]['field'] = 'image'; t['heads'][0]['field'] = 'image'
        _, c = contract(model); p = prepare(d, t, c, manifest(d), constrained=False)
        self.assertEqual(p['populations']['test']['inputs']['input'], p['populations']['test']['targets']['prediction'])
        module = compile_model({model['id']: model}, model['id']); x = torch.tensor(p['populations']['test']['inputs']['input'])
        self.assertTrue(torch.equal(module(x), x)); self.assertEqual(len(d['fields']), 1)

    def test_ragged_padding_derived_length_mask_and_multimodal_real_forward(self):
        fields = [{'id': 'seq', 'dtype': 'float32', 'shape': [-1, 2]}, {'id': 'other', 'dtype': 'float32', 'shape': [1]}, {'id': 'y', 'dtype': 'float32', 'shape': [1]}]
        d = data(fields, [{'seq': [[i, i + 1]] * (i % 3), 'other': [i], 'y': [i + 1]} for i in range(8)])
        model = default_model(inputs=1, outputs=1)
        model['graph'] = {'nodes': [{'id': 'sequence', 'type': 'Input', 'shape': [3, 2]}, {'id': 'length', 'type': 'Input', 'shape': [1]}, {'id': 'mask', 'type': 'Input', 'shape': [3]}, {'id': 'other', 'type': 'Input', 'shape': [1]}, {'id': 'join', 'type': 'Concat', 'sources': ['sequence', 'length', 'mask', 'other']}, {'id': 'linear', 'type': 'Linear', 'features': 1, 'bias': False}, {'id': 'output', 'type': 'Output'}],
            'edges': [{'from': n, 'to': 'join'} for n in ('sequence', 'length', 'mask', 'other')] + [{'from': 'join', 'to': 'linear'}, {'from': 'linear', 'to': 'output'}]}
        _, c = contract(model); t = task()
        t['inputs'] = [{'port': 'sequence', 'field': 'seq', 'transforms': [{'kind': 'pad', 'length': 3, 'value': -5.}]},
            {'port': 'length', 'field': 'seq', 'transforms': [], 'derived': 'length', 'length': 3},
            {'port': 'mask', 'field': 'seq', 'transforms': [], 'derived': 'mask', 'length': 3},
            {'port': 'other', 'field': 'other', 'transforms': []}]
        p = prepare(d, t, c, manifest(d), constrained=False); pop = p['populations']['train']
        self.assertEqual(pop['inputs']['sequence'][0], [[-5., -5.]] * 3)
        self.assertEqual(pop['inputs']['sequence'][1], [[1., 2.], [-5., -5.], [-5., -5.]])
        self.assertEqual(pop['inputs']['length'], [[0.], [1.]])
        self.assertEqual(pop['inputs']['mask'], [[0., 0., 0.], [1., 0., 0.]])
        module = compile_model({model['id']: model}, model['id'])
        with torch.no_grad(): module.layer_0.weight.fill_(1.)
        tensor_inputs = {g['name']: torch.tensor(pop['inputs'][g['id']]) for g in c['input_groups']}
        self.assertEqual(module(tensor_inputs).tolist(), [[-30.], [-14.]])
        mismatch = copy.deepcopy(t); mismatch['inputs'][1]['length'] = 2
        with self.assertRaises(GraphError): validate_task(mismatch, d, c)
        d['records'][0]['values']['seq'] = [[1., 2.]] * 4
        with self.assertRaisesRegex(GraphError, 'truncate'): prepare(d, t, c, manifest(d), constrained=False)

    def test_masks_and_weights_preserve_target_population_and_reject_invalid_values(self):
        d = data(); d['fields'] += [{'id': 'valid', 'dtype': 'bool', 'shape': []}, {'id': 'weight', 'dtype': 'float32', 'shape': []}]
        for i, r in enumerate(d['records']): r['values'].update(valid=bool(i % 2), weight=i / 2.)
        t = task(); t['heads'][0].update(mask='valid', sample_weight='weight')
        p = prepare(d, t, contract()[1], manifest(d), constrained=False)
        self.assertEqual(p['populations']['train']['masks']['prediction'], [False, True])
        self.assertEqual(p['populations']['train']['sample_weights']['prediction'], [0., .5])
        d['records'][0]['values']['weight'] = -.1
        with self.assertRaisesRegex(GraphError, 'nonnegative'): prepare(d, t, contract()[1], manifest(d), constrained=False)

    def test_excluded_missing_records_are_not_materialized_or_fitted(self):
        d = data(); d['fields'][0]['nullable'] = True; d['records'][-1]['values']['x'] = None
        p = prepare(d, task(), contract()[1], manifest(d, test=['r3', 'r4', 'r5', 'r6'], excluded=['r7']), constrained=False)
        self.assertNotIn('r7', [rid for pop in p['populations'].values() for rid in pop['observation_ids']])

    def test_constraints_preflight_expanded_data_before_padding_and_do_not_cap_authoring(self):
        d = data(); t = task(); _, c = contract(); m = manifest(d)
        with self.assertRaisesRegex(GraphError, 'budget'): prepare(d, t, c, m, available_bytes=1)
        prepare(d, t, c, m, available_bytes=1024**3)
        for value in (True, 0., 101.):
            with self.assertRaises(GraphError): prepare(d, t, c, m, memory_limit_percent=value)
        fields = [{'id': 'x', 'dtype': 'float32', 'shape': [-1]}, {'id': 'y', 'dtype': 'float32', 'shape': [1]}]
        d = data(fields, [{'x': [1.], 'y': [1.]}, {'x': [2.], 'y': [2.]}])
        t = task([{'kind': 'pad', 'length': 10**12, 'value': 0.}]); _, c = contract(default_model(inputs=10**12, outputs=1))
        validate_task(t, d, c)
        m = manifest(d, train=['r0'], validation=[], test=['r1'])
        with self.assertRaisesRegex(GraphError, 'budget'): prepare(d, t, c, m, available_bytes=1024**3)

    def test_unsupported_transform_fields_versions_and_bad_heads_do_not_guess(self):
        d = data(); _, c = contract()
        for mutation in ('version', 'unknown', 'duplicate', 'head-loss', 'head-weight', 'mask-id', 'bad-transform', 'zero-affine', 'extra-transform', 'metadata'):
            t = task()
            if mutation == 'version': t['version'] = True
            if mutation == 'unknown': t['unknown'] = 1
            if mutation == 'duplicate': t['inputs'] *= 2
            if mutation == 'head-loss': t['heads'][0]['loss'] = 'implicit-objective'
            if mutation == 'head-weight': t['heads'][0]['weight'] = False
            if mutation == 'mask-id': t['heads'][0]['mask'] = []
            if mutation == 'bad-transform': t['inputs'][0]['transforms'] = [{'kind': 'fit-everything'}]
            if mutation == 'zero-affine': t['inputs'][0]['transforms'] = [{'kind': 'affine', 'scale': 0., 'offset': 1.}]
            if mutation == 'extra-transform': t['inputs'][0]['transforms'] = [{'kind': 'standardize', 'silent-targets': True}]
            if mutation == 'metadata': t['description'] = float('nan')
            with self.subTest(mutation=mutation), self.assertRaises(GraphError): validate_task(t, d, c)


if __name__ == '__main__':
    unittest.main()
