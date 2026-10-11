"""Actual scoped modules: independent functional/gradient oracles, fits and kernel."""
import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F
from app.core import compile_model, generate_code
from app.families import specialize
from app.experiment_objects import candidate_bundle, execute_candidate, object_reference, preview_experiment
from app.revisions import validate_all
from test_scoped_constructors import scoped, scoped_experiment, same_context

torch.set_num_threads(1)


class ScopedExecutionTests(unittest.TestCase):
    def test_branches_constructor_choices_outputs_gradients_and_generated_code(self):
        for source, tied in [(scoped(), False), (same_context(scoped()), False), (same_context(scoped()), True)]:
            if tied: source['models']['parent']['graph']['nodes'][2]['share_with'] = 'left'
            r = specialize(source); net = compile_model(r['models'], r['root'], constrained=False).double()
            self.assertEqual(sum(p.numel() for p in net.parameters()), r['contract']['parameters'])
            if source['constructors'][3]['value'] == source['constructors'][0]['value']:
                self.assertEqual(net.layer_0 is getattr(net, 'layer_1', None), False)
            leaves = [p.detach().clone().requires_grad_() for p in net.parameters()]
            if tied: w1, wh, bh = leaves; w2, b2 = w1, None
            elif len(leaves) == 4: w1, w2, wh, bh = leaves; b2 = None
            else: w1, w2, b2, wh, bh = leaves
            for batch in (1, 4):
                x = torch.linspace(-.7, .9, batch*2, dtype=torch.float64).reshape(batch, 2).requires_grad_()
                xx = x.detach().clone().requires_grad_(); a = torch.tanh(F.linear(xx, w1))
                b = torch.tanh(F.linear(xx, w2)) if b2 is None else F.relu(F.linear(xx, w2, b2))
                expected = F.linear(torch.cat([a, b], dim=-1), wh, bh); actual = net(x)
                torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
                for observed, oracle in zip(torch.autograd.grad(actual.square().sum(), [x, *net.parameters()]),
                                            torch.autograd.grad(expected.square().sum(), [xx, *leaves])):
                    torch.testing.assert_close(observed, oracle, rtol=1e-11, atol=1e-11)
                scope = {}; exec(generate_code(r['models'], r['root']), scope)
                exported = scope['model'].double(); exported.load_state_dict(net.state_dict(), strict=True)
                torch.testing.assert_close(exported(x), expected, rtol=1e-12, atol=1e-12)

    def test_equal_specs_are_independent_module_and_optimizer_objects_unless_explicitly_tied(self):
        for tied in (False, True):
            source = same_context(scoped())
            if tied: source['models']['parent']['graph']['nodes'][2]['share_with'] = 'left'
            r = specialize(source); net = compile_model(r['models'], r['root'], constrained=False)
            linears = [m for m in net.modules() if isinstance(m, torch.nn.Linear)]
            self.assertEqual(len(linears), 2 if tied else 3)
            if not tied: self.assertIsNot(net.layer_0, net.layer_1); self.assertIsNot(linears[0].weight, linears[1].weight)
            opt = torch.optim.Adam(net.parameters(), lr=.01)
            net(torch.tensor([[.2, -.3], [.4, .1]])).square().mean().backward(); opt.step()
            self.assertEqual(len(opt.state), 3 if tied else 4)
            self.assertEqual(len(list(net.named_parameters(remove_duplicate=False))), len(list(net.parameters())))

    def test_worker_checkpoint_retry_and_app_free_predictor_have_nested_context_provenance(self):
        from app.task_execution import check_task_checkpoint, restore_task_predictor
        from app.task_result import validate_result
        from app.run_export import predictor_source, write_new_predictor
        w, e = scoped_experiment(); self.addCleanup(w.db.close)
        self.addCleanup(lambda: w.history.directory.cleanup() if w.history.directory else None)
        ref = object_reference('experiment', e); rows = preview_experiment(e, w.get_spec)['candidates']
        for index in (0, 12):
            result = execute_candidate(w, ref, rows[index]['coordinates'], 'scoped-'+str(index), 'Scoped branches', python=sys.executable, timeout=20)
            record = result['record']; attempt = record['attempts'][0]
            self.assertEqual(attempt['status'], 'completed', attempt['error']); self.assertEqual(attempt['result']['counts']['completed_updates'], 3)
            bundle = candidate_bundle(e, rows[index]['coordinates'], w.get_spec); validate_result(bundle, attempt['result'])
            checkpoint = attempt['result']['checkpoint']; check_task_checkpoint(bundle, checkpoint, constrained=False)
            sr = record['resolved']['references']['specialization']; spec = w.get_spec('specialization', sr['id'], sr['revision'])
            self.assertEqual(spec['version'], 3); self.assertEqual(len(spec['instances']), 3)
            self.assertEqual(len([i for i in spec['instances'] if i['template'] == 'child']), 2)
            retry = execute_candidate(w, ref, rows[index]['coordinates'], record['id'], 'Retry', retry=True, python=sys.executable, timeout=20)
            self.assertEqual(retry['record']['attempts'][1]['result']['checkpoint'], checkpoint)
            records = [{'x': [.2]}, {'x': [-.6]}]
            expected = {k: v.tolist() for k, v in restore_task_predictor(bundle, checkpoint, constrained=False)(records).items()}
            with tempfile.TemporaryDirectory() as td:
                write_new_predictor(predictor_source(w, result['reference']), Path(td) / 'predictor.py')
                code = 'import predictor,json,sys;assert "app" not in sys.modules;print(json.dumps({k:v.tolist() for k,v in predictor.predict_records('+repr(records)+').items()}))'
                ran = subprocess.run([sys.executable, '-c', code], cwd=td, capture_output=True, text=True, timeout=20)
                self.assertEqual(ran.returncode, 0, ran.stderr); self.assertEqual(json.loads(ran.stdout), expected)
        validate_all(w.db)

    def test_actual_jupyter_scoped_rerun_exports_each_resolved_concrete_closure(self):
        from app.notebooks import create_bundle, execute_bundle, verify_bundle
        w, e = scoped_experiment(); self.addCleanup(w.db.close)
        self.addCleanup(lambda: w.history.directory.cleanup() if w.history.directory else None)
        e.pop('revision'); e['models'][0]['axes'] = [{'symbol': 'left-width', 'values': [2, 3]}]
        e['models'][0]['bindings'] = {'right-width': 5}; e['data_tasks'] = e['data_tasks'][:1]; e['initialization_seeds'] = [7]
        ref = w.put_spec('experiment', e)
        recipe = w.put_spec('report', {'version': 1, 'id': 'scoped-rerun', 'name': 'Scoped calls', 'question': 'Do local constructors remain reproducible?',
            'mode': 'rerun', 'experiment': ref, 'runs': [], 'sections': ['models', 'training', 'evaluation', 'checkpoints', 'failures'],
            'analysis': {'kind': 'descriptive', 'inferential_unit': 'run'}})
        with tempfile.TemporaryDirectory() as td:
            before = w.db.serialize(); export = create_bundle(w, recipe, Path(td) / 'bundle'); self.assertEqual(w.db.serialize(), before)
            result = execute_bundle(Path(export['path']), timeout=60)
            self.assertEqual(result['execution']['status'], 'completed', result['execution']['error'])
            output = Path(result['path']); data = json.loads((output / 'report-data.json').read_text())
            self.assertEqual(len(data['runs']), 2); self.assertTrue(all(r['status'] == 'completed' for r in data['runs']))
            self.assertEqual(len(list(output.glob('architecture-*.py'))), 2)
            self.assertEqual(len(list(output.glob('predictor-*.py'))), 2); self.assertEqual(len(list(output.glob('curve-*.svg'))), 2)
            verify_bundle(Path(export['path'])); self.assertEqual(w.db.serialize(), before)


if __name__ == '__main__': unittest.main()
