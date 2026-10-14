"""CNNs from simpler pieces: Patches and Reduce, and the exact compositions they promise."""
import itertools
import sqlite3
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from app.core import GraphError, Workspace, compile_model, default_model, generate_code, validate_model
from app.revisions import model_document


def drawing(nodes, shape=(3, 9, 8), name='pieces'):
    doc = default_model(name, inputs=2, outputs=1)
    nodes = [{'id': 'input', 'type': 'Input', 'shape': list(shape), 'axes': ['C', 'H', 'W']}, *nodes, {'id': 'output', 'type': 'Output'}]
    doc['graph'] = {'nodes': nodes, 'edges': [{'from': a['id'], 'to': b['id']} for a, b in zip(nodes, nodes[1:])]}
    return {doc['id']: doc}, doc['id']


def run(models, mid, x):
    return compile_model(models, mid).forward_tensor(x)


class PatchesTests(unittest.TestCase):
    def test_patches_equal_unfold_with_patch_values_last(self):
        for k, s, p, d in itertools.product([(1, 1), (3, 3), (3, 2), (2, 4)], [1, 2], [0, 1], [1, 2]):
            n = (5, 8, 7)
            try:
                models, mid = drawing([{'id': 'p', 'type': 'Patches', 'kernel_size': list(k), 'stride': [s, s],
                                        'padding': [p, p], 'dilation': [d, d]}], n)
                shape = validate_model(models, mid)['shapes']['p']
            except GraphError:
                continue  # window does not fit: validation refuses, as Conv2d does
            x = torch.randn(4, *n)
            cols = F.unfold(x, k, dilation=d, padding=p, stride=s)
            expected = cols.transpose(1, 2).reshape(4, *shape['output_shape'])
            self.assertTrue(torch.equal(run(models, mid, x), expected), (k, s, p, d))
            self.assertEqual(shape['output_shape'][-1], 5 * k[0] * k[1])
            self.assertEqual(shape['output_axes'], ['H', 'W', 'C'])
            self.assertEqual(validate_model(models, mid)['parameters'], 0)

    def test_window_must_fit_and_needs_chw(self):
        with self.assertRaisesRegex(GraphError, 'window does not fit'):
            validate_model(*drawing([{'id': 'p', 'type': 'Patches', 'kernel_size': [9, 9]}], (1, 4, 4)))
        models, mid = drawing([{'id': 'p', 'type': 'Patches', 'kernel_size': [2, 2]}], (1, 4, 4))
        models[mid]['graph']['nodes'][0]['axes'] = ['H', 'W', 'C']
        with self.assertRaisesRegex(GraphError, 'needs axes \\[C,H,W\\]'):
            validate_model(models, mid)


class CompositionTests(unittest.TestCase):
    def conv_pair(self, c, o, k, s, p, d, bias, shape):
        conv, _ = drawing([{'id': 'conv', 'type': 'Conv2d', 'out_channels': o, 'kernel_size': list(k), 'stride': [s, s],
                             'padding': [p, p], 'dilation': [d, d], 'bias': bias}], (c, *shape))
        pieces, _ = drawing([{'id': 'p', 'type': 'Patches', 'kernel_size': list(k), 'stride': [s, s], 'padding': [p, p], 'dilation': [d, d]},
                             {'id': 'l', 'type': 'Linear', 'features': o, 'bias': bias},
                             {'id': 'perm', 'type': 'Permute', 'dims': [2, 0, 1]}], (c, *shape))
        return conv, pieces

    def test_conv2d_equals_patches_linear_permute_in_values_and_gradients(self):
        torch.manual_seed(0)
        for c, o, k, s, p, d, bias in [(1, 2, (3, 3), 1, 0, 1, True), (3, 4, (5, 5), 2, 2, 1, True), (2, 3, (3, 2), 1, 1, 2, False),
                                       (4, 1, (1, 1), 1, 0, 1, True), (2, 5, (2, 3), 3, 1, 1, True)]:
            conv, pieces = self.conv_pair(c, o, k, s, p, d, bias, (9, 8))
            cm, pm = compile_model(conv, next(iter(conv))), compile_model(pieces, next(iter(pieces)))
            with torch.no_grad():
                pm.layer_1.weight.copy_(cm.layer_0.weight.reshape(o, -1))
                if bias:
                    pm.layer_1.bias.copy_(cm.layer_0.bias)
            x1 = torch.randn(3, c, 9, 8, requires_grad=True); x2 = x1.detach().clone().requires_grad_(True)
            a, b = cm.forward_tensor(x1), pm.forward_tensor(x2)
            self.assertEqual(a.shape, b.shape)
            self.assertTrue(torch.allclose(a, b, atol=1e-5), (c, o, k, s, p, d))
            a.square().sum().backward(); b.square().sum().backward()
            self.assertTrue(torch.allclose(x1.grad, x2.grad, atol=1e-5))
            self.assertTrue(torch.allclose(cm.layer_0.weight.grad.reshape(o, -1), pm.layer_1.weight.grad, atol=1e-4))
            self.assertEqual(validate_model(conv, next(iter(conv)))['parameters'], validate_model(pieces, next(iter(pieces)))['parameters'])

    def test_pooling_compositions_where_exact(self):
        torch.manual_seed(1)
        x = torch.randn(2, 3, 8, 6)
        def pooled(kind, k, s, p, mode, **extra):
            models, mid = drawing([{'id': 'pool', 'type': kind, 'kernel_size': [k, k], 'stride': [s, s], 'padding': [p, p], **extra}], (3, 8, 6))
            h, w = validate_model(models, mid)['shapes']['pool']['output_shape'][1:]
            pieces, pid = drawing([{'id': 'p', 'type': 'Patches', 'kernel_size': [k, k], 'stride': [s, s], 'padding': [p, p]},
                                   {'id': 'rs', 'type': 'Reshape', 'shape': [h, w, 3, k * k]},
                                   {'id': 'r', 'type': 'Reduce', 'mode': mode},
                                   {'id': 'perm', 'type': 'Permute', 'dims': [2, 0, 1]}], (3, 8, 6))
            return run(models, mid, x), run(pieces, pid, x)
        for k, s in [(2, 2), (3, 1), (3, 2)]:
            a, b = pooled('MaxPool2d', k, s, 0, 'max')
            self.assertTrue(torch.allclose(a, b, atol=1e-6), ('max', k, s))
            a, b = pooled('AvgPool2d', k, s, 1 if k == 3 else 0, 'mean', count_include_pad=True)
            self.assertTrue(torch.allclose(a, b, atol=1e-6), ('avg', k, s))
        # Not exact, and therefore not offered by the expansion: padded max pooling pads with -inf, Patches with zeros.
        a, b = pooled('MaxPool2d', 3, 1, 1, 'max')
        self.assertFalse(torch.allclose(a, b))
        adaptive, aid = drawing([{'id': 'a', 'type': 'AdaptiveAvgPool2d', 'output_size': [1, 1]}], (3, 8, 6))
        pieces, pid = drawing([{'id': 'rs', 'type': 'Reshape', 'shape': [3, 48]}, {'id': 'r', 'type': 'Reduce', 'mode': 'mean'}], (3, 8, 6))
        self.assertTrue(torch.allclose(run(adaptive, aid, x).reshape(2, 3), run(pieces, pid, x), atol=1e-6))

    def test_export_is_standalone_and_equal(self):
        torch.manual_seed(2)
        models, mid = drawing([{'id': 'p', 'type': 'Patches', 'kernel_size': [3, 3], 'padding': [1, 1]},
                               {'id': 'l', 'type': 'Linear', 'features': 4, 'bias': True},
                               {'id': 'perm', 'type': 'Permute', 'dims': [2, 0, 1]},
                               {'id': 'p2', 'type': 'Patches', 'kernel_size': [2, 2], 'stride': [2, 2]},
                               {'id': 'rs', 'type': 'Reshape', 'shape': [4, 4, 4, 4]},
                               {'id': 'r', 'type': 'Reduce', 'mode': 'max'}], (3, 8, 8))
        native = compile_model(models, mid)
        namespace = {}
        exec(generate_code(models, mid), namespace)
        exported = namespace['model']
        exported.load_state_dict(native.state_dict())
        x = torch.randn(3, 3, 8, 8)
        self.assertTrue(torch.equal(native.forward_tensor(x), exported.forward_tensor(x)))
        self.assertEqual(tuple(native.forward_tensor(torch.zeros(0, 3, 8, 8)).shape), (0, 4, 4, 4))

    def test_patches_output_is_counted_for_memory_admission(self):
        models, mid = drawing([{'id': 'p', 'type': 'Patches', 'kernel_size': [5, 5], 'padding': [2, 2]}], (3, 32, 32))
        contract = validate_model(models, mid)
        self.assertGreaterEqual(contract['activation_elements'], 32 * 32 * 3 * 25)


class ReduceTests(unittest.TestCase):
    def test_reduce_validation(self):
        for node, message in [({'mode': 'sum'}, 'max" or "mean'), ({'axis': 9}, 'integer sample axis'), ({'axis': 3}, 'outside the incoming')]:
            models, mid = drawing([{'id': 'r', 'type': 'Reduce', **node}], (3, 4, 4))
            with self.assertRaisesRegex(GraphError, message):
                validate_model(models, mid)
        models, mid = drawing([{'id': 'rs', 'type': 'Reshape', 'shape': [48]}, {'id': 'r', 'type': 'Reduce'}], (3, 4, 4))
        with self.assertRaisesRegex(GraphError, 'at least two sample axes'):
            validate_model(models, mid)
        models, mid = drawing([{'id': 'r', 'type': 'Reduce', 'mode': 'max', 'axis': 0}], (3, 4, 4))
        shape = validate_model(models, mid)['shapes']['r']
        self.assertEqual((shape['output_shape'], shape['output_axes']), ([4, 4], ['H', 'W']))


class FenceTests(unittest.TestCase):
    def test_model_revision_11_and_container_33(self):
        models, mid = drawing([{'id': 'p', 'type': 'Patches', 'kernel_size': [2, 2]},
                               {'id': 'l', 'type': 'Linear', 'features': 2, 'bias': True}], (1, 4, 4))
        self.assertEqual(model_document(models, mid)["version"], 11)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'pieces.vtw'
            workspace = Workspace(); workspace.put(models[mid]); workspace.save(path); workspace.db.close()
            Workspace.load(path).db.close()
            db = sqlite3.connect(path); db.execute('PRAGMA user_version=32'); db.commit(); db.close()
            with self.assertRaisesRegex(GraphError, 'require a version-33 container'):
                Workspace.load(path)


if __name__ == '__main__':
    unittest.main()
