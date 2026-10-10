"""CNN values/gradients use scalar loop oracles, not our executor or F.conv2d."""
import copy
import io
import json
import math
import random
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import torch
from app.core import (GraphError, Workspace, VERSION, check_document, compile_model,
                      default_model, default_cnn, generate_dataset, generate_code, generate_predictor,
                      restore_predictor, train_model, validate_model)
from app.evidence import binding_for, dataset_report, parameter_schema
from app.operators import normalized
from app.memory import estimate_memory
import vpw_cli

torch.set_num_threads(1)


def chain(shape, *operations, axes=None):
    doc = default_model('Tensor probe')
    nodes = [{'id': 'input', 'type': 'Input', 'shape': list(shape)}]
    if axes is not None:
        nodes[0]['axes'] = list(axes)
    nodes += [dict(op, id='op' + str(i)) for i, op in enumerate(operations)]
    nodes += [{'id': 'output', 'type': 'Output'}]
    doc['graph'] = {'nodes': nodes, 'edges': [
        {'from': a['id'], 'to': b['id']} for a, b in zip(nodes, nodes[1:])]}
    return doc


def scalar_conv(x, weight, bias, stride, padding, dilation, groups, upstream=None):
    """Independent Python cross-correlation and analytic VJP, all float64 scalars."""
    bsz, cin, hin, win = len(x), len(x[0]), len(x[0][0]), len(x[0][0][0])
    cout, cg, kh, kw = len(weight), len(weight[0]), len(weight[0][0]), len(weight[0][0][0])
    sh, sw = stride; ph, pw = padding; dh, dw = dilation
    oh = (hin + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    ow = (win + 2 * pw - dw * (kw - 1) - 1) // sw + 1
    out = [[[[0. for _ in range(ow)] for _ in range(oh)] for _ in range(cout)] for _ in range(bsz)]
    gx = [[[[0. for _ in range(win)] for _ in range(hin)] for _ in range(cin)] for _ in range(bsz)]
    gw = [[[[0. for _ in range(kw)] for _ in range(kh)] for _ in range(cg)] for _ in range(cout)]
    gb = [0.] * cout
    for b in range(bsz):
        for oc in range(cout):
            first = (oc // (cout // groups)) * cg
            for h in range(oh):
                for w in range(ow):
                    value = bias[oc] if bias is not None else 0.
                    u = upstream[b][oc][h][w] if upstream is not None else 0.
                    gb[oc] += u
                    for c in range(cg):
                        for i in range(kh):
                            for j in range(kw):
                                ih, iw = h * sh - ph + i * dh, w * sw - pw + j * dw
                                if 0 <= ih < hin and 0 <= iw < win:
                                    value += x[b][first + c][ih][iw] * weight[oc][c][i][j]
                                    gx[b][first + c][ih][iw] += u * weight[oc][c][i][j]
                                    gw[oc][c][i][j] += u * x[b][first + c][ih][iw]
                    out[b][oc][h][w] = value
    return out, gx, gw, gb


class CNNTests(unittest.TestCase):
    def contract(self, doc):
        check_document(doc)
        return validate_model({doc['id']: doc}, doc['id'])

    def compile(self, doc):
        return compile_model({doc['id']: doc}, doc['id'])

    def test_non_square_shape_unequal_parameters_and_exact_counts(self):
        d = chain([4, 7, 9], {'type': 'Conv2d', 'out_channels': 6, 'groups': 2,
                  'kernel_size': [2, 3], 'stride': [2, 1], 'padding': [1, 0], 'dilation': [2, 1]})
        c = self.contract(d)
        self.assertEqual(c['output_shape'], [6, 4, 7])
        self.assertEqual(c['parameters'], 6 * 2 * 2 * 3 + 6)
        self.assertEqual(c['output_axes'], ['C', 'H', 'W'])
        self.assertTrue(c['affine']); self.assertFalse(c['matrix_affine'])
        m = self.compile(d)
        self.assertEqual(list(m(torch.zeros(2, 4, 7, 9)).shape), [2, 6, 4, 7])
        self.assertEqual(sum(p.numel() for p in m.parameters()), c['parameters'])

    def test_memory_counts_convolution_workspace_without_dense_global_affine_map(self):
        d = chain([64, 128, 128], {'type': 'Conv2d', 'out_channels': 128, 'kernel_size': [7, 5]})
        c = self.contract(d); expected = 64 * 7 * 5 * 122 * 124
        self.assertEqual(c['convolution_workspace_elements'], expected)
        memory = estimate_memory(c, samples=10, training=True)
        self.assertEqual(memory['version'], 2)
        self.assertEqual(memory['components']['convolution_workspace'], 32 * 8 * expected)
        self.assertEqual(memory['components']['affine_report'], 0)
        self.assertEqual(estimate_memory(c)['components']['convolution_workspace'], 8 * expected)
        self.assertTrue(c['affine']); self.assertFalse(c['matrix_affine'])

    def test_32_bounded_unequal_spatial_configs_match_runtime_shapes_and_parameter_counts(self):
        rng = random.Random(31)
        for case in range(32):
            groups = rng.choice([1, 2]); ci = groups * rng.choice([1, 2]); co = groups * rng.choice([1, 3])
            op = {'type': 'Conv2d', 'out_channels': co, 'groups': groups, 'bias': bool(case % 2),
                  'kernel_size': [rng.randint(1, 3), rng.randint(1, 4)],
                  'stride': [rng.randint(1, 3), rng.randint(1, 2)],
                  'padding': [rng.randint(0, 2), rng.randint(0, 2)], 'dilation': [rng.randint(1, 2), 1]}
            d = chain([ci, 8, 9], op); c = self.contract(d); m = self.compile(d)
            with self.subTest(case=case, op=op):
                self.assertEqual(list(m(torch.ones(1, ci, 8, 9)).shape)[1:], c['output_shape'])
                self.assertEqual(sum(p.numel() for p in m.parameters()), c['parameters'])

    def test_grouped_depthwise_bias_and_batches_independent_values_and_gradients(self):
        for groups, out_channels in ((1, 3), (2, 4), (2, 6)):
            for bias in (False, True):
                for batch in (1, 2):
                    with self.subTest(groups=groups, bias=bias, batch=batch):
                        d = chain([2, 4, 5], {'type': 'Conv2d', 'out_channels': out_channels,
                                  'kernel_size': [2, 2], 'stride': [1, 2], 'padding': [1, 0],
                                  'dilation': [2, 1], 'groups': groups, 'bias': bias})
                        m = self.compile(d).double()
                        x = (torch.arange(batch * 2 * 4 * 5, dtype=torch.float64).reshape(batch, 2, 4, 5) / 37 - .6).requires_grad_()
                        with torch.no_grad():
                            m.layer_0.weight.copy_(torch.arange(m.layer_0.weight.numel(), dtype=torch.float64).reshape_as(m.layer_0.weight) / 19 - .3)
                            if bias: m.layer_0.bias.copy_(torch.arange(out_channels, dtype=torch.float64) / 11)
                        y = m(x)
                        u = torch.linspace(-.8, .9, y.numel(), dtype=torch.float64).reshape_as(y)
                        expected, gx, gw, gb = scalar_conv(x.tolist(), m.layer_0.weight.tolist(),
                            m.layer_0.bias.tolist() if bias else None, [1, 2], [1, 0], [2, 1], groups, u.tolist())
                        torch.testing.assert_close(y, torch.tensor(expected, dtype=torch.float64), atol=1e-12, rtol=1e-12)
                        (y * u).sum().backward()
                        torch.testing.assert_close(x.grad, torch.tensor(gx, dtype=torch.float64), atol=1e-12, rtol=1e-12)
                        torch.testing.assert_close(m.layer_0.weight.grad, torch.tensor(gw, dtype=torch.float64), atol=1e-12, rtol=1e-12)
                        if bias:
                            torch.testing.assert_close(m.layer_0.bias.grad, torch.tensor(gb, dtype=torch.float64), atol=1e-12, rtol=1e-12)

    def test_invalid_groups_or_spatial_dimensions_fail_before_torch_allocation(self):
        cases = [
            ([3, 4, 5], {'groups': 2, 'out_channels': 4}),
            ([4, 4, 5], {'groups': 2, 'out_channels': 3}),
            ([1, 2, 2], {'kernel_size': 5}),
            ([4, 5], {}),
        ]
        for shape, extra in cases:
            d = chain(shape, {'type': 'Conv2d', 'out_channels': 2, 'kernel_size': 3, **extra})
            with self.subTest(shape=shape, extra=extra), patch('torch.nn.Conv2d', side_effect=AssertionError('allocated')):
                with self.assertRaises(GraphError): self.compile(d)

    def test_schema_rejects_unsupported_options_versions_and_numeric_coercion(self):
        base = {'id': 'c', 'type': 'Conv2d', 'out_channels': 2, 'kernel_size': 3}
        for values in ({'op_version': 2}, {'op_version': True}, {'kernel_size': True},
                       {'kernel_size': [1, 2.0]}, {'stride': 0}, {'padding': -1},
                       {'dilation': 0}, {'groups': False}, {'bias': 1},
                       {'padding_mode': 'reflect'}, {'padding': 'same'}, {'in_channels': 2},
                       {'kernel_size': [1]}, {'out_channels': 0}):
            with self.subTest(values=values), self.assertRaises(GraphError):
                normalized(dict(base, **values))
        for kind, values in (('MaxPool2d', {'return_indices': True}),
                             ('AvgPool2d', {'ceil_mode': True}),
                             ('AvgPool2d', {'count_include_pad': 1}),
                             ('AvgPool2d', {'divisor_override': 2}),
                             ('MaxPool2d', {'padding': 2})):
            with self.subTest(kind=kind, values=values), self.assertRaises(GraphError):
                normalized({'id': 'p', 'type': kind, 'kernel_size': 2, **values})

    def test_scalar_defaults_persist_as_explicit_pairs_without_mutating_caller(self):
        d = chain([1, 5, 7], {'type': 'Conv2d', 'out_channels': 2, 'kernel_size': 3})
        original = copy.deepcopy(d)
        w = Workspace(); self.addCleanup(w.db.close); w.put(d)
        n = w.models()[d['id']]['graph']['nodes'][1]
        self.assertEqual(n['kernel_size'], [3, 3]); self.assertEqual(n['stride'], [1, 1])
        self.assertEqual(n['padding'], [0, 0]); self.assertEqual(n['dilation'], [1, 1])
        self.assertEqual(n['op_version'], 1); self.assertTrue(n['bias'])
        self.assertEqual(d, original)

    def test_real_cli_creates_cnn_and_invalid_group_edit_preserves_disk_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'cli-cnn.vtw'
            base = [sys.executable, 'vpw_cli.py', str(path)]
            init = subprocess.run(base + ['init'], text=True, capture_output=True, timeout=10)
            self.assertEqual(init.returncode, 0, init.stderr)
            d = default_cnn(); d['graph']['nodes'][1]['kernel_size'] = 2
            create = subprocess.run(base + ['apply', '-'], input=json.dumps([{'op': 'create', 'document': d}]),
                                    text=True, capture_output=True, timeout=10)
            self.assertEqual(create.returncode, 0, create.stdout + create.stderr)
            loaded = Workspace.load(path)
            try: self.assertEqual(loaded.models()[d['id']]['graph']['nodes'][1]['kernel_size'], [2, 2])
            finally: loaded.db.close()
            before = path.read_bytes()
            bad = subprocess.run(base + ['apply', '-'],
                input=json.dumps([{'op': 'set-node', 'id': d['id'], 'node': 'conv', 'values': {'groups': 2}}]),
                text=True, capture_output=True, timeout=10)
            self.assertEqual(bad.returncode, 1)
            self.assertIn('divisible', json.loads(bad.stdout)['error'])
            self.assertEqual(before, path.read_bytes())

    def test_runnable_example_creates_trained_workspace_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'example.vtw'
            cmd = [sys.executable, '-m', 'examples.cnn_regression', str(path), '--train']
            run = subprocess.run(cmd, text=True, capture_output=True, timeout=20)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            value = json.loads(run.stdout); loaded = Workspace.load(path)
            try:
                d = loaded.models()[value['id']]
                self.assertEqual(d['result']['completed_updates'], 450)
                self.assertEqual(loaded.db.execute('SELECT count(*) FROM runs').fetchone()[0], 1)
                self.assertIn('nn.Conv2d', generate_predictor(loaded.models(), value['id'], d['result']))
            finally: loaded.db.close()
            before = path.read_bytes()
            again = subprocess.run(cmd, text=True, capture_output=True, timeout=10)
            self.assertNotEqual(again.returncode, 0)
            self.assertEqual(before, path.read_bytes())

    def test_pool_values_and_input_gradients_include_and_exclude_padding(self):
        for kind in ('AvgPool2d', 'MaxPool2d'):
            for include in (True, False):
                op = {'type': kind, 'kernel_size': [2, 3], 'stride': [1, 2], 'padding': [1, 1]}
                if kind == 'AvgPool2d': op['count_include_pad'] = include
                d = chain([1, 3, 5], op)
                m = self.compile(d).double()
                x = torch.arange(15, dtype=torch.float64).reshape(1, 1, 3, 5).sub(20).requires_grad_()
                y = m(x); expected = []; grad = [[0.] * 5 for _ in range(3)]
                for h in range(4):
                    row = []
                    for w in range(3):
                        cells = [(h - 1 + i, w * 2 - 1 + j) for i in range(2) for j in range(3)
                                 if 0 <= h - 1 + i < 3 and 0 <= w * 2 - 1 + j < 5]
                        values = [float(x[0, 0, i, j].detach()) for i, j in cells]
                        if kind == 'MaxPool2d':
                            value = max(values); i, j = cells[values.index(value)]; grad[i][j] += 1.
                        else:
                            divisor = 6 if include else len(cells)
                            value = sum(values) / divisor
                            for i, j in cells: grad[i][j] += 1. / divisor
                        row.append(value)
                    expected.append(row)
                torch.testing.assert_close(y, torch.tensor([[expected]], dtype=torch.float64), atol=1e-12, rtol=1e-12)
                y.sum().backward()
                torch.testing.assert_close(x.grad, torch.tensor([[grad]], dtype=torch.float64), atol=1e-12, rtol=1e-12)
                self.assertEqual(self.contract(d)['affine'], kind != 'MaxPool2d')

    def test_adaptive_pool_unequal_bins_independent_values_and_gradients(self):
        d = chain([1, 5, 7], {'type': 'AdaptiveAvgPool2d', 'output_size': [2, 3]})
        x = torch.arange(35, dtype=torch.float64).reshape(1, 1, 5, 7).requires_grad_()
        y = self.compile(d).double()(x); expected = []; grad = [[0.] * 7 for _ in range(5)]
        for h in range(2):
            row = []
            for w in range(3):
                cells = [(i, j) for i in range(h * 5 // 2, math.ceil((h + 1) * 5 / 2))
                         for j in range(w * 7 // 3, math.ceil((w + 1) * 7 / 3))]
                row.append(sum(i * 7 + j for i, j in cells) / len(cells))
                for i, j in cells: grad[i][j] += 1 / len(cells)
            expected.append(row)
        torch.testing.assert_close(y, torch.tensor([[expected]], dtype=torch.float64))
        y.sum().backward()
        torch.testing.assert_close(x.grad, torch.tensor([[grad]], dtype=torch.float64))

    def test_permute_identifiable_values_is_not_reshape_and_preserves_batch(self):
        d = chain([2, 3, 4], {'type': 'Permute', 'dims': [2, 0, 1]}, axes=['H', 'W', 'C'])
        x = torch.arange(48.).reshape(2, 2, 3, 4)
        y = self.compile(d)(x)
        expected = [[[[float(x[b, h, w, c]) for w in range(3)] for h in range(2)] for c in range(4)] for b in range(2)]
        torch.testing.assert_close(y, torch.tensor(expected), rtol=0, atol=0)
        self.assertFalse(torch.equal(y, x.reshape(2, 4, 2, 3)))
        self.assertEqual(self.contract(d)['output_axes'], ['C', 'H', 'W'])

    def test_hwc_requires_permute_and_named_input_axes_reach_conv(self):
        conv = {'type': 'Conv2d', 'out_channels': 2, 'kernel_size': 1}
        d = chain([3, 4, 2], conv, axes=['H', 'W', 'C'])
        with self.assertRaisesRegex(GraphError, 'Use Permute'): self.contract(d)
        d = chain([3, 4, 2], {'type': 'Permute', 'dims': [2, 0, 1]}, conv, axes=['H', 'W', 'C'])
        m = self.compile(d)
        self.assertEqual(list(m({'input': torch.ones(1, 3, 4, 2)}).shape), [1, 2, 3, 4])
        self.assertEqual(binding_for(self.contract(d))['inputs'][0]['axes'], ['H', 'W', 'C'])

    def test_invalid_axes_permutations_and_flatten_ranges_are_rejected(self):
        for op in ({'type': 'Permute', 'dims': [0, 0, 1]}, {'type': 'Permute', 'dims': [0, True]},
                   {'type': 'Permute', 'dims': [0, 1]}, {'type': 'Flatten', 'start_dim': 3},
                   {'type': 'Flatten', 'start_dim': 2, 'end_dim': 1}, {'type': 'Flatten', 'start_dim': True}):
            with self.subTest(op=op), self.assertRaises(GraphError): self.contract(chain([1, 3, 4], op))
        for axes in (['C', 'H'], ['C', 'C', 'W'], ['C', 1, 'W']):
            with self.subTest(axes=axes), self.assertRaises(GraphError): self.contract(chain([1, 3, 4], axes=axes))

    def test_flatten_partial_negative_axes_preserve_batch_and_value_order(self):
        x = torch.arange(48.).reshape(2, 2, 3, 4)
        for start, end, shape in ((0, -1, [24]), (1, -1, [2, 12]), (-3, -2, [6, 4]), (1, 1, [2, 3, 4])):
            with self.subTest(start=start, end=end):
                d = chain([2, 3, 4], {'type': 'Flatten', 'start_dim': start, 'end_dim': end})
                torch.testing.assert_close(self.compile(d)(x), x.reshape(2, *shape), rtol=0, atol=0)
                self.assertEqual(self.contract(d)['output_shape'], shape)

    def test_native_export_pool_flatten_head_and_default_dtype(self):
        for kind in ('AvgPool2d', 'MaxPool2d', 'AdaptiveAvgPool2d'):
            for bias in (False, True):
                pool = {'type': kind, **({'output_size': [2, 2]} if kind == 'AdaptiveAvgPool2d' else {'kernel_size': 2})}
                d = chain([2, 6, 7], {'type': 'Conv2d', 'out_channels': 4, 'groups': 2,
                          'kernel_size': [2, 3], 'bias': bias}, {'type': 'Activation', 'activation': 'Tanh'},
                          pool, {'type': 'Flatten'}, {'type': 'Linear', 'features': 2, 'bias': bias})
                models = {d['id']: d}; native = self.compile(d)
                default = torch.get_default_dtype()
                try:
                    torch.set_default_dtype(torch.float64)
                    scope = {}; exec(generate_code(models, d['id']), scope)
                    self.assertTrue(all(p.dtype == torch.float32 for p in scope['model'].parameters()))
                finally: torch.set_default_dtype(default)
                scope['model'].load_state_dict(native.state_dict())
                for batch in (1, 3):
                    x = torch.linspace(-1., 1., batch * 2 * 6 * 7).reshape(batch, 2, 6, 7)
                    torch.testing.assert_close(scope['model'](x), native(x), rtol=0, atol=0)
                self.assertEqual(set(parameter_schema(models, d['id'])), set(native.state_dict()))

    def test_tensor_axes_changes_invalidate_dataset_even_when_width_unchanged(self):
        d = chain([2, 2, 2], {'type': 'Flatten'}, {'type': 'Linear', 'features': 1}, axes=['C', 'H', 'W'])
        d['dataset'] = {'x': [[0.] * 8, [1.] * 8], 'y': [[0.], [1.]]}
        w = Workspace(); self.addCleanup(w.db.close); w.put(d); d = w.models()[d['id']]
        self.assertTrue(dataset_report(d, self.contract(d))['allowed'])
        d['graph']['nodes'][0]['axes'] = ['H', 'W', 'C']; w.put(d); d = w.models()[d['id']]
        self.assertEqual(dataset_report(d, self.contract(d))['status'], 'incompatible')

    def test_spatial_data_adapter_reproducible_and_teacher_matches_independent_kernel_loops(self):
        d = default_cnn(); models = {d['id']: d}
        original = copy.deepcopy(models)
        data = generate_dataset(models, d['id'], samples=8, kind='spatial')
        self.assertEqual(data, generate_dataset(models, d['id'], samples=8, kind='auto'))
        self.assertEqual(models, original)
        t = data['truth']
        images = torch.tensor(data['x'], dtype=torch.float64).reshape(8, 1, 6, 5).tolist()
        values = scalar_conv(images, t['weight'], t['bias'], [1, 1], [0, 0], [1, 1], 1)[0]
        expected = []
        for image in values:
            pooled = [sum(math.tanh(v) for row in channel for v in row) / 16 for channel in image]
            expected.append([sum(a * v for a, v in zip(weights, pooled)) + bias
                             for weights, bias in zip(t['readout'], t['offset'])])
        torch.testing.assert_close(torch.tensor(data['y'], dtype=torch.float64),
                                   torch.tensor(expected, dtype=torch.float64), atol=1e-12, rtol=1e-12)
        self.assertEqual(data['adapter'], {'version': 1, 'kind': 'declared-row-major',
                         'sample_shape': [1, 6, 5], 'axes': ['C', 'H', 'W']})

    def test_invalid_recorded_adapter_cannot_be_repaired_by_column_rebind(self):
        d = default_cnn(); models = {d['id']: d}
        d['dataset'] = generate_dataset(models, d['id'], samples=8, kind='spatial')
        w = Workspace(); self.addCleanup(w.db.close); w.put(d); d = w.models()[d['id']]
        for key, value in (('version', True), ('sample_shape', [5, 6, 1]), ('axes', ['H', 'W', 'C']),
                           ('kind', 'nhwc-to-nchw')):
            bad = copy.deepcopy(d); bad['dataset']['adapter'][key] = value
            c = self.contract(bad); report = dataset_report(bad, c)
            self.assertFalse(report['allowed']); self.assertFalse(report['can_rebind'])
            with self.assertRaisesRegex(GraphError, 'adapter'):
                train_model({bad['id']: bad}, bad['id'])
        bad = copy.deepcopy(d); bad['dataset']['adapter']['axes'] = ['H', 'W', 'C']; w.put(bad)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'adapter.vtw'; w.save(path); before = path.read_bytes(); stream = io.StringIO()
            with patch('vpw_cli.read_json', return_value=[{'op': 'rebind-dataset', 'id': d['id']}]), redirect_stdout(stream):
                self.assertEqual(vpw_cli.main([str(path), 'apply', '-']), 1)
            self.assertEqual(before, path.read_bytes())

    def test_nested_cnn_references_have_independent_state_schema_and_matching_export(self):
        child = chain([2, 4, 5], {'type': 'Conv2d', 'out_channels': 2, 'kernel_size': 1, 'groups': 2})
        parent = chain([2, 4, 5], {'type': 'Module', 'ref': child['id']},
                       {'type': 'Module', 'ref': child['id']}, {'type': 'Flatten'},
                       {'type': 'Linear', 'features': 1})
        models = {child['id']: child, parent['id']: parent}; mid = parent['id']
        native = compile_model(models, mid); scope = {}; exec(generate_code(models, mid), scope)
        scope['model'].load_state_dict(native.state_dict()); x = torch.randn(3, 2, 4, 5)
        torch.testing.assert_close(scope['model'](x), native(x), rtol=0, atol=0)
        self.assertNotEqual(native.layer_0.layer_0.weight.data_ptr(), native.layer_1.layer_0.weight.data_ptr())
        schema = parameter_schema(models, mid)
        self.assertEqual(schema['layer_0.layer_0.weight']['shape'], [2, 1, 1, 1])
        self.assertNotEqual(schema['layer_0.layer_0.weight']['id'], schema['layer_1.layer_0.weight']['id'])

    def test_v3_checkpoint_meaning_survives_read_only_load_and_explicit_save_upgrade(self):
        d = default_model(); d['training']['epochs'] = 2
        d['dataset'] = {'x': [[float(i), float(i - 1)] for i in range(8)], 'y': [[float(i)] for i in range(8)]}
        w = Workspace(); self.addCleanup(w.db.close); w.put(d); d = w.models()[d['id']]
        result = train_model(w.models(), d['id']); d['result'] = result; w.put(d)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'v3.vtw'; w.save(path)
            with sqlite3.connect(path) as db: db.execute('PRAGMA user_version=3')
            before = path.read_bytes(); loaded = Workspace.load(path)
            try:
                self.assertEqual(path.read_bytes(), before)
                self.assertEqual(loaded.models(), w.models())
                self.assertEqual(loaded.db.execute('PRAGMA user_version').fetchone()[0], VERSION)
                restored = restore_predictor(loaded.models(), d['id'], result)
                x = torch.ones(2, 2)
                torch.testing.assert_close(restored(x), restore_predictor(w.models(), d['id'], result)(x), rtol=0, atol=0)
                loaded.save(path)
            finally: loaded.db.close()
            with sqlite3.connect(path) as db: self.assertEqual(db.execute('PRAGMA user_version').fetchone()[0], VERSION)

    def test_cnn_training_reload_cli_trained_export_and_corrupt_tensor_rejection(self):
        d = chain([1, 4, 5], {'type': 'Conv2d', 'out_channels': 2, 'kernel_size': [2, 2]},
                  {'type': 'Activation', 'activation': 'Tanh'},
                  {'type': 'AdaptiveAvgPool2d', 'output_size': 1}, {'type': 'Flatten'},
                  {'type': 'Linear', 'features': 1, 'bias': True}, axes=['C', 'H', 'W'])
        rng = random.Random(47)
        x = [[rng.uniform(-1, 1) for _ in range(20)] for _ in range(96)]
        # A spatial teacher evaluated independently with scalar kernel loops.
        weights = [[[[.3, -.2], [.5, .1]]], [[[.1, .6], [-.4, .2]]]]
        structured = torch.tensor(x).reshape(96, 1, 4, 5).tolist()
        values = scalar_conv(structured, weights, [.2, -.1], [1, 1], [0, 0], [1, 1], 1)[0]
        targets = [[sum(math.tanh(v) for row in channels[0] for v in row) / 12
                    - .7 * sum(math.tanh(v) for row in channels[1] for v in row) / 12 + .15]
                   for channels in values]
        d['dataset'] = {'x': x, 'y': targets, 'adapter': {'version': 1, 'kind': 'declared-row-major',
                        'sample_shape': [1, 4, 5], 'axes': ['C', 'H', 'W']}}
        d['training'] = {'epochs': 350, 'lr': .03, 'seed': 7}
        models = {d['id']: d}; result = train_model(models, d['id'])
        train = result['split_indices']['train']; held = result['split_indices']['held_out']
        mean = sum(targets[i][0] for i in train) / len(train)
        baseline = sum((targets[i][0] - mean) ** 2 for i in held) / len(held)
        self.assertLess(result['mse'], baseline * .05)
        self.assertEqual(result['completed_updates'], 350)
        self.assertEqual(result['history'][-1]['test'], result['mse'])
        self.assertFalse(result['matrix_affine']); self.assertNotIn('A', result)
        restored = restore_predictor(models, d['id'], result)
        inputs = torch.tensor(x).reshape(96, 1, 4, 5)
        predicted = restored(inputs).detach()
        measured = ((predicted[held] - torch.tensor(targets)[held]) ** 2).mean().item()
        self.assertEqual(result['mse'], measured)
        w = Workspace(); self.addCleanup(w.db.close); d['result'] = result; w.put(d)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'cnn.vtw'; w.save(path); loaded = Workspace.load(path)
            try:
                current = loaded.models()[d['id']]
                torch.testing.assert_close(restore_predictor(loaded.models(), d['id'], current['result'])(inputs), predicted, rtol=0, atol=0)
                stream = io.StringIO()
                with redirect_stdout(stream): self.assertEqual(vpw_cli.main([str(path), 'export-trained', d['id']]), 0)
                scope = {}; exec(json.loads(stream.getvalue())['value']['code'], scope)
                torch.testing.assert_close(scope['model'](inputs), predicted, rtol=0, atol=0)
            finally: loaded.db.close()
        bad = copy.deepcopy(result); bad['weights']['layer_0.weight'][0][0][0][0] = True
        for use in (restore_predictor, generate_predictor):
            with self.assertRaisesRegex(GraphError, 'float32'): use(models, d['id'], bad)
        print(f'CNN independent spatial task: held-out MSE={result["mse"]:.8g}; train-mean baseline={baseline:.8g}')


if __name__ == '__main__':
    unittest.main()
