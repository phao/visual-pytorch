"""Checked exported classes, semantic warm starts and independent numerical oracles."""
import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F
from app.core import compile_model
from app.family_export import family_source, write_new_family
from app.families import specialize
from test_constructor_choices import choices
from test_families import mlp, cnn
from test_scoped_constructors import scoped, same_context

torch.set_num_threads(1)


def exported(source):
    scope = {}; exec(family_source(source, constrained=False)['code'], scope)
    return scope


class FamilyExportExecutionTests(unittest.TestCase):
    def test_four_grammars_outputs_input_parameter_gradients_and_native_state_agree(self):
        tied = mlp(); tied['version'] = 2; tied['repeats'][0]['mode'] = 'tied'
        cases = [(mlp(), [{'hidden-width': 2, 'hidden-depth': 1}, {'hidden-width': 4, 'hidden-depth': 3}]),
                 (tied, [{'hidden-width': 3, 'hidden-depth': 2}]),
                 (choices(), [{'activation-kind': 'GELU', 'learn-bias': False}, {'activation-kind': 'Softmax', 'learn-bias': True}]),
                 (scoped(), [{'left-width': 2, 'right-width': 4}, {}])]
        for source, selections in cases:
            scope = exported(source)
            for bindings in selections:
                r = specialize(source, bindings); native = compile_model(r['models'], r['root'], constrained=False).double()
                target = scope['ArchitectureFamily'](bindings=bindings).double(); target.network.load_state_dict(native.state_dict(), strict=True)
                self.assertEqual(target.contract, r['contract']); self.assertEqual(target.provenance()['architecture_hash'], r['architecture_hash'])
                self.assertEqual(sum(p.numel() for p in target.parameters()), r['contract']['parameters'])
                for batch in (1, 3):
                    x = torch.linspace(-.7, .9, batch*r['contract']['inputs'], dtype=torch.float64).reshape(batch, *r['contract']['input_shape']).requires_grad_()
                    xx = x.detach().clone().requires_grad_(); actual = target(x); expected = native(xx)
                    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
                    for a, b in zip(torch.autograd.grad(actual.square().sum(), [x, *target.parameters()]),
                                    torch.autograd.grad(expected.square().sum(), [xx, *native.parameters()])):
                        torch.testing.assert_close(a, b, rtol=1e-11, atol=1e-11)

    def test_exported_scoped_branches_match_independent_functional_oracle(self):
        scope = exported(scoped()); model = scope['ArchitectureFamily']().double()
        params = [p.detach().clone().requires_grad_() for p in model.parameters()]
        w1, w2, b2, wh, bh = params
        x = torch.linspace(-.6, .8, 8, dtype=torch.float64).reshape(4, 2).requires_grad_(); xx = x.detach().clone().requires_grad_()
        expected = F.linear(torch.cat([torch.tanh(F.linear(xx, w1)), F.relu(F.linear(xx, w2, b2))], -1), wh, bh)
        actual = model(x); torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
        for a, b in zip(torch.autograd.grad(actual.square().sum(), [x, *model.parameters()]),
                        torch.autograd.grad(expected.square().sum(), [xx, *params])):
            torch.testing.assert_close(a, b, rtol=1e-11, atol=1e-11)

    def test_constructor_uses_cpu_float32_even_under_double_meta_process_defaults(self):
        old_dtype, old_device = torch.get_default_dtype(), torch.get_default_device()
        try:
            torch.set_default_dtype(torch.float64); torch.set_default_device('meta')
            for source in (choices(), cnn()):
                scope = exported(source); model = scope['ArchitectureFamily']()
                self.assertTrue(all(p.dtype == torch.float32 and p.device.type == 'cpu' for p in model.parameters()))
                x = torch.zeros((1, *model.contract['input_shape']), dtype=torch.float32, device='cpu')
                self.assertEqual(model(x).dtype, torch.float32)
                self.assertEqual(model(x).device.type, 'cpu')
        finally:
            torch.set_default_device(old_device); torch.set_default_dtype(old_dtype)

    def test_grouped_convolution_valid_and_invalid_constructor_before_allocations(self):
        scope = exported(cnn()); model = scope['ArchitectureFamily']().double()
        x = torch.linspace(-.5, .6, 2*4*5*6, dtype=torch.float64).reshape(2, 4, 5, 6).requires_grad_()
        params = [p.detach().clone().requires_grad_() for p in model.parameters()]; xx = x.detach().clone().requires_grad_()
        convw, convb, headw, headb = params
        y = torch.tanh(F.conv2d(xx, convw, convb, padding=(1,1), groups=2))
        expected = F.linear(F.adaptive_avg_pool2d(y, (1,1)).flatten(1), headw, headb)
        actual = model(x); torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
        for a, b in zip(torch.autograd.grad(actual.square().sum(), [x, *model.parameters()]),
                        torch.autograd.grad(expected.square().sum(), [xx, *params])):
            torch.testing.assert_close(a, b, rtol=1e-11, atol=1e-11)
        with patch('torch.nn.Conv2d', side_effect=AssertionError('No allocation')):
            with self.assertRaisesRegex(ValueError, 'groups|divis'): scope['ArchitectureFamily'](bindings={'g': 3})

    def test_multiple_named_inputs_split_outputs_and_packed_forward_match_functional_oracle(self):
        nodes = [{'id': 'a', 'type': 'Input', 'features': 2, 'label': 'first'},
                 {'id': 'b', 'type': 'Input', 'features': 1, 'label': 'second'},
                 {'id': 'la', 'type': 'Linear', 'features': 2, 'bias': False},
                 {'id': 'ta', 'type': 'Activation', 'activation': 'Tanh'},
                 {'id': 'lb', 'type': 'Linear', 'features': 2, 'bias': True},
                 {'id': 'rb', 'type': 'Activation', 'activation': 'ReLU'},
                 {'id': 'join', 'type': 'Concat', 'sources': ['ta','rb'], 'flatten': False},
                 {'id': 'split', 'type': 'Splitter', 'parts': [{'output': 'one', 'features': 1}, {'output': 'rest', 'features': 3}]},
                 {'id': 'one', 'type': 'Output', 'label': 'one'}, {'id': 'rest', 'type': 'Output', 'label': 'rest'}]
        edges = [{'from': a,'to':b} for a,b in [('a','la'),('la','ta'),('b','lb'),('lb','rb'),('ta','join'),('rb','join'),('join','split')]]
        edges += [{'from':'split','to':p,'from_port':p} for p in ('one','rest')]
        source = {'version':1,'id':'ports-family','name':'Named split ports','root':'ports',
                  'models':{'ports':{'id':'ports','name':'Named split ports','graph':{'nodes':nodes,'edges':edges,'input_order':['a','b']}}},
                  'symbols':[],'assignments':[],'repeats':[]}
        model = exported(source)['ArchitectureFamily']().double()
        a = torch.tensor([[.2,-.3],[-.4,.5]],dtype=torch.float64,requires_grad=True)
        b = torch.tensor([[.7],[-.6]],dtype=torch.float64,requires_grad=True)
        aa=a.detach().clone().requires_grad_();bb=b.detach().clone().requires_grad_()
        weights=[p.detach().clone().requires_grad_() for p in model.parameters()];wa,wb,bias=weights
        joined=torch.cat([torch.tanh(F.linear(aa,wa)),F.relu(F.linear(bb,wb,bias))],dim=-1)
        actual=model({'first':a,'second':b})
        torch.testing.assert_close(actual['one'],joined[:,:1],rtol=1e-12,atol=1e-12)
        torch.testing.assert_close(actual['rest'],joined[:,1:],rtol=1e-12,atol=1e-12)
        for observed,oracle in zip(torch.autograd.grad(sum(v.square().sum() for v in actual.values()),[a,b,*model.parameters()]),
                                  torch.autograd.grad(joined.square().sum(),[aa,bb,*weights])):
            torch.testing.assert_close(observed,oracle,rtol=1e-11,atol=1e-11)
        torch.testing.assert_close(model.forward_tensor({'first':a,'second':b}),joined,rtol=1e-12,atol=1e-12)
        packed=torch.cat([a,b],dim=1);out=model(packed)
        torch.testing.assert_close(out['rest'],joined[:,1:],rtol=1e-12,atol=1e-12)

    def test_invalid_unresolved_types_bounds_ram_and_runtime_tampering_fail_before_layers(self):
        source = scoped(); scope = exported(source)
        selections = ({'left-width': 0}, {'left-width': True}, {'unknown': 3}, {'left-width': 17}, {'child-bias': 0})
        with patch('torch.nn.Linear', side_effect=AssertionError('No allocation')):
            for bindings in selections:
                with self.subTest(bindings=bindings), self.assertRaises(ValueError): scope['ArchitectureFamily'](bindings=bindings)
            with patch.object(scope['_runtime']().core, 'available_memory', return_value=1):
                with self.assertRaisesRegex(ValueError, 'Constrained'): scope['ArchitectureFamily'](constrained=True)
            for kwargs in ({'constrained': 0}, {'memory_limit_percent': 0}):
                with self.assertRaises(ValueError): scope['ArchitectureFamily'](**kwargs)
            source['symbols'][0].pop('default'); missing = exported(source)
            self.assertEqual(missing['preview_family']()['status'], 'conditional')
            with self.assertRaisesRegex(ValueError, 'parent'): missing['ArchitectureFamily']()
            broken = exported(scoped()); broken['_SOURCES']['core'] += '\n'
            with self.assertRaisesRegex(ValueError, 'runtime bytes'): broken['ArchitectureFamily']()
            broken = exported(scoped()); broken['_FAMILY']['name'] = 'Changed'
            with self.assertRaisesRegex(ValueError, 'definition changed'): broken['ArchitectureFamily']()

    def test_instance_independence_sharing_hooks_optimizer_and_input_contract(self):
        source = same_context(scoped()); source['models']['parent']['graph']['nodes'][2]['share_with'] = 'left'
        scope = exported(source); a = scope['ArchitectureFamily'](); b = scope['ArchitectureFamily']()
        self.assertEqual(len([m for m in a.modules() if isinstance(m, torch.nn.Linear)]), 2)
        self.assertTrue(all(p is not q for p, q in zip(a.parameters(), b.parameters())))
        hooks = []; handle = a.network.register_forward_hook(lambda *_: hooks.append(True))
        try:
            opt = torch.optim.Adam(a.parameters(), lr=.01)
            loss = a({'input': torch.tensor([[.2, -.3], [.4, .5]])}).square().mean(); loss.backward(); opt.step()
            self.assertEqual(hooks, [True]); self.assertEqual(len(opt.state), 3)
            with self.assertRaisesRegex(ValueError, 'shape'): a(torch.ones(2, 3))
            with self.assertRaisesRegex(ValueError, 'named'): a({'wrong': torch.ones(2,2)})
        finally: handle.remove()

    def test_semantic_warm_start_matches_relocated_head_not_registration_index(self):
        scope = exported(mlp()); a = scope['ArchitectureFamily'](bindings={'hidden-width': 3, 'hidden-depth': 1})
        b = scope['ArchitectureFamily'](bindings={'hidden-width': 3, 'hidden-depth': 3})
        with torch.no_grad():
            for i, p in enumerate(a.parameters()): p.fill_(.11*(i+1))
        snapshot = a.semantic_state(); snapshot['schema'] = dict(reversed(list(snapshot['schema'].items())))
        snapshot['weights'] = dict(reversed(list(snapshot['weights'].items())))
        result = b.warm_start(snapshot)
        moved = [row for row in result['loaded'] if row['source'] != row['target']]
        self.assertEqual(len(moved), 2)
        for row in result['loaded']: torch.testing.assert_close(b.state_dict()[row['target']], snapshot['weights'][row['source']])
        self.assertTrue(result['missing']); self.assertFalse(result['optimizer_resume'])
        before = {k:v.clone() for k,v in b.state_dict().items()}
        with self.assertRaisesRegex(ValueError, 'Strict'): b.warm_start(snapshot, strict=True)
        self.assertTrue(all(torch.equal(v,b.state_dict()[k]) for k,v in before.items()))

    def test_warm_start_shape_and_dtype_skips_forgery_nonfinite_and_validation_atomicity(self):
        scope = exported(mlp()); a = scope['ArchitectureFamily'](bindings={'hidden-width': 2}); b = scope['ArchitectureFamily'](bindings={'hidden-width': 4})
        result = b.warm_start(a.semantic_state()); self.assertTrue(result['skipped']); self.assertEqual(len(result['loaded']), 1)
        checkpoint = a.semantic_state(); before = {k:v.clone() for k,v in b.state_dict().items()}
        for mutation in ('family', 'version', 'runtime', 'id', 'shape', 'binding', 'nan', 'paths'):
            bad = copy.deepcopy(checkpoint); first = next(iter(bad['schema']))
            if mutation == 'family': bad['family']['revision'] = '0'*64
            if mutation == 'version': bad['version'] = True
            if mutation == 'runtime': bad['runtime_sha256'] = '0'*64
            if mutation == 'id': bad['schema'][first]['id'] = 'forged'
            if mutation == 'shape': bad['schema'][first]['shape'] = [True]
            if mutation == 'binding': bad['bindings']['hidden-width'] = 4
            if mutation == 'nan': bad['weights'][first].flatten()[0] = float('nan')
            if mutation == 'paths': bad['weights'].pop(first)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError): b.warm_start(bad)
            self.assertTrue(all(torch.equal(v,b.state_dict()[k]) for k,v in before.items()))
        double = scope['ArchitectureFamily'](bindings={'hidden-width': 4}).double()
        self.assertEqual(len(double.warm_start(b.semantic_state())['loaded']), 0)
        protected = scope['ArchitectureFamily'](bindings={'hidden-width': 4}, constrained=True)
        before = {k:v.clone() for k,v in protected.state_dict().items()}
        with (patch.object(scope['_runtime']().core, 'available_memory', return_value=1),
              patch('torch.isfinite', side_effect=AssertionError('No finite tensor allocation before admission'))):
            with self.assertRaisesRegex(ValueError, 'Constrained'): protected.warm_start(b.semantic_state())
        self.assertTrue(all(torch.equal(v,protected.state_dict()[k]) for k,v in before.items()))

    def test_semantic_components_distinguish_slashes_from_nested_call_paths_and_copies(self):
        source = scoped(); source['models']['parent']['graph']['nodes'][1]['id'] = 'left/x'
        for edge in source['models']['parent']['graph']['edges']:
            for key in ('from','to'):
                if edge[key] == 'left': edge[key] = 'left/x'
        source['models']['parent']['graph']['nodes'][3]['sources'][0] = 'left/x'
        for call in source['constructors']:
            if call['node'] == 'left': call['node'] = 'left/x'
        scope = exported(source); model = scope['ArchitectureFamily'](); schema = model.semantic_parameter_schema()
        self.assertEqual(len({d['id'] for d in schema.values()}), len(schema))
        components = [json.loads(d['id']) for d in schema.values()]; self.assertTrue(any('left/x' in c for c in components))
        schema.clear(); self.assertTrue(model.semantic_parameter_schema())
        metadata = model.provenance(); metadata['bindings'].clear(); self.assertTrue(model.provenance()['bindings'])

    def test_fresh_import_is_app_free_has_no_initial_model_or_rng_mutation_then_constructs_and_trains(self):
        source = family_source(scoped(), constrained=False)
        with tempfile.TemporaryDirectory() as td:
            write_new_family(source, Path(td) / 'portable_family.py')
            code = '''import sys,torch
torch.set_num_threads(1)
before=torch.random.get_rng_state().clone()
import portable_family as f
assert torch.equal(before,torch.random.get_rng_state())
assert not any(k=='app' or k.startswith('app.') for k in sys.modules)
assert not hasattr(f,'model')
assert f.preview_family()['status']=='verified'
assert torch.equal(before,torch.random.get_rng_state())
m=f.ArchitectureFamily(bindings={'left-width':2,'right-width':3})
opt=torch.optim.Adam(m.parameters(),lr=.01)
x=torch.tensor([[.2,-.3],[.4,.7]])
for _ in range(3):
 opt.zero_grad();loss=m(x).square().mean();loss.backward();opt.step()
n=f.ArchitectureFamily(bindings={'left-width':2,'right-width':3})
n.warm_start(m.semantic_state(),strict=True)
assert torch.equal(n(x),m(x))
assert not any(k=='app' or k.startswith('app.') for k in sys.modules)
print('app-free checked family: 3 updates; strict semantic warm-start matches')'''
            ran = subprocess.run([sys.executable, '-c', code], cwd=td, capture_output=True, text=True, timeout=20)
            self.assertEqual(ran.returncode, 0, ran.stderr)


if __name__ == '__main__': unittest.main()
