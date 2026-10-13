"""Runtime source proofs and admission, plus native graph integration checks.

The graph integration cases require core/graph source admission and the explicit
_graph_values(..., batch_size=B) interface implemented by the integrating worker.
This module does not change process-wide Torch thread settings on import.
"""
import copy
import math
import unittest
from unittest.mock import patch

import torch

from app.core import GraphError
from app import runtime_shapes as spatial
from app import temporal_shapes as temporal
from app import temporal_shapes_v2 as temporal_v2
from app.tensor_sources import build


def fixture(mode='spatial', kind='Constant'):
    shape = [1, 3, 4] if mode == 'spatial' else [3, 2]
    axes = ['C', 'H', 'W'] if mode == 'spatial' else ['T', 'F']
    nodes = [
        {'id': 'input', 'type': 'Input', 'shape': shape, 'axes': axes},
        {'id': 'source', 'type': kind, 'shape': [2, 3], 'axes': ['T', 'F'],
         **({'fill': 2.5} if kind == 'Constant' else {})},
        {'id': 'dynamic', 'type': 'Output'},
        {'id': 'fixed', 'type': 'Output'},
    ]
    model = {'id': 'root', 'name': 'Fixed source beside runtime axes', 'graph': {
        'nodes': nodes, 'input_order': ['input'], 'edges': [
            {'from': 'input', 'to': 'dynamic'}, {'from': 'source', 'to': 'fixed'}]}}
    return {'root': model}, 'root', {'input': [1, 2] if mode == 'spatial' else [0]}


def admitted_contract(models, mid):
    """Small admitted boundary fixture, independent of pending graph admission."""
    nodes = models[mid]['graph']['nodes']
    inp = nodes[0]
    group = dict(id='input', name='input', shape=inp['shape'], axes=inp['axes'],
                 features=math.prod(inp['shape']), start=0, stop=math.prod(inp['shape']))
    shapes = {n['id']: {'output_shape': n.get('shape', inp['shape'] if n['id'] == 'dynamic' else [2, 3])}
              for n in nodes}
    plan = [{'id': n['id'], 'type': n['type'], 'sources':
             [] if n['type'] in {'Input', 'Constant', 'TensorParameter'} else
             [{'node': 'input' if n['id'] == 'dynamic' else 'source', 'port': None}]}
            for n in nodes]
    return {'input_groups': [group], 'shapes': shapes, 'plan': plan,
            'tensor_source': None, 'output_groups': [{'id': 'dynamic', 'name': 'dynamic'},
                                                   {'id': 'fixed', 'name': 'fixed'}],
            'parameters': 0, 'buffer_elements': 6, 'activation_elements': 20,
            'inputs': group['features'], 'outputs': group['features'] + 6}


MODES = [('spatial', spatial), ('temporal-v1', temporal), ('temporal-v2', temporal_v2)]


class SourceRuntimeBoundaryTests(unittest.TestCase):
    def test_nullary_module_sources_do_not_depend_on_parent_runtime_axes(self):
        for mode, runtime in MODES:
            models, mid, declared = fixture(mode)
            source = models[mid]['graph']['nodes'][1]
            child = {'id': 'source-child', 'name': 'Nullary source child', 'graph': {
                'nodes': [copy.deepcopy(source), {'id': 'out', 'type': 'Output'}],
                'edges': [{'from': 'source', 'to': 'out'}], 'input_order': []}}
            models['source-child'] = child
            models[mid]['graph']['nodes'][1] = {'id': 'source', 'type': 'Module',
                'ref': 'source-child', 'batch_only': True}
            c = admitted_contract(models, mid)
            c['plan'][1]['sources'] = []
            child_contract = {
                'input_groups': [], 'tensor_source': 'out', 'output_groups': [{'id': 'out'}],
                'shapes': {'source': {'output_shape': [2, 3]}, 'out': {'output_shape': [2, 3]}},
                'plan': [{'id': 'source', 'type': 'Constant', 'sources': []},
                         {'id': 'out', 'type': 'Output', 'sources': [{'node': 'source', 'port': None}]}]}

            def contract(collection, key, **kwargs):
                return c if key == mid else child_contract

            with self.subTest(mode=mode), patch('app.core.check_document'), \
                 patch.object(spatial, 'validate_model', side_effect=contract), \
                 patch.object(spatial, 'contract', side_effect=contract), \
                 patch.object(spatial, 'topology', side_effect=lambda model: model['graph']['nodes']):
                spec = spatial._dependency_specification(models, mid, declared,
                    temporal=mode != 'spatial', temporal_version=2 if mode == 'temporal-v2' else 1)
            expected = ['input.H', 'input.W'] if mode == 'spatial' else ['input.T']
            self.assertEqual(spec['output_dependencies'], [expected])

    def test_runtime_wrapper_passes_validated_batch_to_genuine_source_kernel(self):
        for mode, runtime in MODES:
            for kind in ('Constant', 'TensorParameter'):
                models, mid, declared = fixture(mode, kind)
                c = admitted_contract(models, mid)
                spec = {'scope': 'explicit-eager-runtime-' +
                        ('spatial-contract' if mode == 'spatial' else
                         'temporal-contract' if mode == 'temporal-v1' else 'temporal-v2-contract'),
                        'baseline_input_groups': c['input_groups'], 'runtime_inputs': declared}
                order = []

                class Baseline(torch.nn.Module):
                    def __init__(self, layers):
                        super().__init__()
                        for name, layer in layers.items():
                            self.add_module(name, layer)
                        self.contract = c

                def construct(*args, **kwargs):
                    order.append('construct')
                    return Baseline({'layer_0': build(models[mid]['graph']['nodes'][1], admitted_elements=6)})

                def preflight(*args, **kwargs):
                    order.append('admit')

                def plan(collection, key, specification, shapes):
                    result = copy.deepcopy(c)
                    result['shapes']['dynamic']['output_shape'] = shapes['input']
                    return result

                def execute(module, x, actual, *, batch_size):
                    return {'input': x, 'source': module.layer_0(batch_size),
                            'dynamic': x, 'fixed': module.layer_0(batch_size)}

                with self.subTest(mode=mode, kind=kind), \
                     patch('app.core.preflight', side_effect=preflight), \
                     patch('app.core.compile_model', side_effect=construct), \
                     patch('app.core._graph_values', side_effect=execute) as values, \
                     patch('app.core._graph_tensor', side_effect=lambda v, actual: torch.cat(
                         [v['dynamic'].flatten(1), v['fixed'].flatten(1)], 1)), \
                     patch.object(spatial, 'shape_contract', side_effect=plan):
                    model = spatial._compile_runtime_model(models, mid, spec, constrained=False)
                    if mode != 'temporal-v2':
                        self.assertEqual(order, ['admit', 'construct'])
                    owner = model.layer_0.value
                    for batch, extent in ((1, 7), (3, 2)):
                        shape = [1, extent, 2] if mode == 'spatial' else [extent, 2]
                        x = torch.ones(batch, *shape)
                        result = model(x)
                        torch.testing.assert_close(result['fixed'], owner.unsqueeze(0).expand(batch, 2, 3))
                        self.assertEqual(values.call_args.kwargs, {'batch_size': batch})
                        self.assertEqual(model.intermediate_values(x)['source'].shape, (batch, 2, 3))
                        self.assertEqual(model.forward_tensor(x).shape, (batch, math.prod(shape) + 6))
                        if kind == 'TensorParameter':
                            owner.grad = None
                            result['fixed'].sum().backward()
                            torch.testing.assert_close(owner.grad, torch.full_like(owner, float(batch)))

    def test_admitted_zero_incoming_sources_have_fixed_axis_dependencies(self):
        for mode, runtime in MODES:
            for kind in ('Constant', 'TensorParameter'):
                models, mid, declared = fixture(mode, kind)
                c = admitted_contract(models, mid)
                # A fixed source can feed a learned last-axis constructor even
                # when its named T axis resembles a root runtime axis.
                linear = {'id': 'head', 'type': 'Linear', 'features': 4}
                models[mid]['graph']['nodes'].insert(-1, linear)
                c['plan'].insert(-1, {'id': 'head', 'type': 'Linear', 'sources': [{'node': 'source', 'port': None}]})
                c['plan'][-1]['sources'][0]['node'] = 'head'
                before = copy.deepcopy(models)
                with self.subTest(mode=mode, kind=kind), \
                     patch('app.core.check_document'), \
                     patch.object(spatial, 'validate_model', return_value=c), \
                     patch.object(spatial, 'contract', return_value=c), \
                     patch.object(spatial, 'topology', return_value=models[mid]['graph']['nodes']):
                    spec = spatial._dependency_specification(models, mid, declared,
                        temporal=mode != 'spatial', temporal_version=2 if mode == 'temporal-v2' else 1)
                expected = ['input.H', 'input.W'] if mode == 'spatial' else ['input.T']
                self.assertEqual(spec['output_dependencies'], [expected])
                self.assertEqual(models, before)

    def test_buffer_storage_is_charged_independently_of_batch_and_gradients(self):
        models, mid, _ = fixture()
        c = admitted_contract(models, mid)
        for _, runtime in MODES:
            with self.subTest(runtime=runtime.__name__):
                def report(batch, gradients):
                    args = (c, batch) if runtime is spatial else (models, mid, c, batch)
                    return runtime.admission(*args, constrained=False, gradients=gradients)['memory']
                small = report(1, False)
                large = report(7, True)
                self.assertEqual(small['components']['buffers'], 8 * c['buffer_elements'])
                self.assertEqual(large['components']['buffers'], small['components']['buffers'])
                self.assertGreater(large['components']['activations'], small['components']['activations'])
                self.assertEqual(large['estimated_bytes'], sum(large['components'].values()))

    def test_buffer_storage_can_block_every_runtime_at_exact_ram_boundary(self):
        models, mid, _ = fixture()
        c = admitted_contract(models, mid)
        c['buffer_elements'] = 10**8
        for _, runtime in MODES:
            args = (c, 1) if runtime is spatial else (models, mid, c, 1)
            with self.subTest(runtime=runtime.__name__):
                amount = runtime.admission(*args, constrained=False)['memory']['estimated_bytes']
                with patch.object(spatial, 'available_memory', return_value=10 * amount):
                    self.assertTrue(runtime.admission(*args)['allowed'])
                with patch.object(spatial, 'available_memory', return_value=10 * amount - 1):
                    with self.assertRaisesRegex(GraphError, 'Constrained'):
                        runtime.admission(*args)

    def test_old_contracts_without_buffers_keep_zero_buffer_charge(self):
        models, mid, _ = fixture()
        c = admitted_contract(models, mid)
        c.pop('buffer_elements')
        self.assertEqual(spatial.admission(c, 1, constrained=False)['memory']['components']['buffers'], 0)


class NativeSourceRuntimeTests(unittest.TestCase):
    def test_explicit_runtime_batch_and_no_grad_admission_match_preview(self):
        for mode, runtime in MODES:
            models,mid,declared=fixture(mode)
            factory=runtime.compile_runtime_model if runtime is spatial else runtime.compile_temporal_model
            model=factory(models,mid,declared,constrained=True)
            shape=models[mid]['graph']['nodes'][0]['shape'];x=torch.ones(2,*shape)
            self.assertTrue(torch.equal(model(x,batch_size=2)['fixed'],model(x)['fixed']))
            self.assertTrue(torch.equal(model.forward_tensor(x,batch_size=2),model.forward_tensor(x)))
            self.assertTrue(torch.equal(model.intermediate_values(x,batch_size=2)['source'],model.intermediate_values(x)['source']))
            for batch in (True,1,3):
                with self.assertRaises(GraphError):model(x,batch_size=batch)
            with torch.no_grad():
                model.eval()
                estimated=model.preview_shapes({'input':shape},batch_size=2,gradients=False)['safety']['memory']['estimated_bytes']
                # The runtime and additional source gate must agree at the
                # exact preview admission boundary, including inference mode.
                with patch.object(spatial,'available_memory',return_value=10*estimated), \
                     patch('app.core.available_memory',return_value=10*estimated):
                    self.assertTrue(torch.isfinite(model(x,batch_size=2)['fixed']).all())

    def test_all_runtime_factories_expand_fixed_sources_and_reuse_parameter_owners(self):
        for mode, runtime in MODES:
            for kind in ('Constant', 'TensorParameter'):
                with self.subTest(mode=mode, kind=kind):
                    models, mid, declared = fixture(mode, kind)
                    factory = runtime.compile_runtime_model if runtime is spatial else runtime.compile_temporal_model
                    model = factory(models, mid, declared, constrained=False)
                    owner = next(m for m in model.modules() if getattr(m, 'kind', None) == kind)
                    pointer = owner.value.data_ptr()
                    optimizer = torch.optim.SGD(model.parameters(), lr=.1) if kind == 'TensorParameter' else None
                    for batch, extent in ((1, 2), (3, 7), (2, 1)):
                        shape = [1, extent, extent + 1] if mode == 'spatial' else [extent, 2]
                        x = torch.ones(batch, *shape)
                        out = model(x)
                        torch.testing.assert_close(out['dynamic'], x)
                        torch.testing.assert_close(out['fixed'], owner.value.unsqueeze(0).expand(batch, 2, 3))
                        self.assertEqual(list(out['fixed'].shape), [batch, 2, 3])
                        self.assertEqual(model.preview_shapes({'input': shape})['contract']['shapes']['source']['output_shape'], [2, 3])
                        torch.testing.assert_close(model.intermediate_values(x)['source'], out['fixed'])
                        packed = model.forward_tensor(x)
                        torch.testing.assert_close(packed, torch.cat([x.flatten(1), out['fixed'].flatten(1)], 1))
                        if optimizer is not None:
                            optimizer.zero_grad()
                            out['fixed'].sum().backward()
                            torch.testing.assert_close(owner.value.grad, torch.full((2, 3), float(batch)))
                            optimizer.step()
                        self.assertEqual(owner.value.data_ptr(), pointer)
                    with self.assertRaises(GraphError):
                        model(torch.ones(2, *shape, dtype=torch.float64))

    def test_whole_model_ram_gate_precedes_any_source_construction(self):
        for mode, runtime in MODES:
            models, mid, declared = fixture(mode)
            factory = runtime.compile_runtime_model if runtime is spatial else runtime.compile_temporal_model
            for initial in (None, {'input': [1, 1, 2] if mode == 'spatial' else [1, 2]}):
                with self.subTest(mode=mode, initial=initial), \
                     patch('app.core.available_memory', return_value=1), \
                     patch.object(spatial, 'available_memory', return_value=1), \
                     patch('app.tensor_sources.build', side_effect=AssertionError('Source allocated before admission')):
                    with self.assertRaisesRegex(GraphError, 'Constrained'):
                        factory(models, mid, declared, constrained=True, initial_shapes=initial)

    def test_native_factory_uses_exact_admitted_shape_product(self):
        for kind in ('Constant', 'TensorParameter'):
            models, mid, declared = fixture(kind=kind)
            with self.subTest(kind=kind):
                with patch('app.tensor_sources.build', wraps=build) as construct:
                    model = spatial.compile_runtime_model(models, mid, declared, constrained=False)
                self.assertTrue(construct.called)
                self.assertEqual(construct.call_args.kwargs['admitted_elements'], 6)
                self.assertEqual(model.contract['buffer_elements'], 6 if kind == 'Constant' else 0)

    def test_sources_cannot_be_declared_as_runtime_inputs(self):
        for mode, runtime in MODES:
            models, mid, _ = fixture(mode)
            with self.subTest(mode=mode), self.assertRaises(GraphError):
                runtime.specification(models, mid, {'source': [0]})

    def test_nullary_module_expands_source_with_parent_batch_at_variable_axes(self):
        for mode, runtime in MODES:
            models, mid, declared = fixture(mode)
            source = models[mid]['graph']['nodes'][1]
            models['source-child'] = {'id': 'source-child', 'name': 'Nullary child', 'graph': {
                'nodes': [copy.deepcopy(source), {'id': 'out', 'type': 'Output'}],
                'edges': [{'from': 'source', 'to': 'out'}], 'input_order': []}}
            models[mid]['graph']['nodes'][1] = {'id': 'source', 'type': 'Module',
                'ref': 'source-child', 'batch_only': True}
            factory = runtime.compile_runtime_model if runtime is spatial else runtime.compile_temporal_model
            with self.subTest(mode=mode):
                model = factory(models, mid, declared, constrained=False)
                for batch, extent in ((1, 7), (4, 2)):
                    shape = [1, extent, 2] if mode == 'spatial' else [extent, 2]
                    result = model(torch.ones(batch, *shape))
                    torch.testing.assert_close(result['fixed'], torch.full((batch, 2, 3), 2.5))


if __name__ == '__main__':
    unittest.main()
