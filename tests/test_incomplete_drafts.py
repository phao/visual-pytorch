"""Located structural problems and display-only partial shapes for invalid drafts."""
import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from app.core import GraphError, Workspace, default_model, validate_model
from app.graph import partial_shapes


def reported_draft():
    """Structure of an owner-reported draft: skip connection into Concat, Output left unwired."""
    doc = default_model('Skip', inputs=2, outputs=1)
    doc['graph'] = {'nodes': [
        {'id': 'input', 'type': 'Input', 'features': 2}, {'id': 'output', 'type': 'Output'},
        {'id': 'h', 'type': 'Linear', 'features': 16, 'bias': True}, {'id': 't', 'type': 'Activation', 'activation': 'Tanh'},
        {'id': 'h2', 'type': 'Linear', 'features': 16, 'bias': True}, {'id': 't2', 'type': 'Activation', 'activation': 'Tanh'},
        {'id': 'r', 'type': 'Linear', 'features': 1, 'bias': True},
        {'id': 'c', 'type': 'Concat', 'sources': ['input', 't'], 'flatten': True}],
        'edges': [{'from': 'input', 'to': 'c'}, {'from': 'input', 'to': 'h'}, {'from': 'h', 'to': 't'},
                  {'from': 't', 'to': 'c'}, {'from': 'c', 'to': 'h2'}, {'from': 'h2', 'to': 't2'}, {'from': 't2', 'to': 'r'}]}
    column = {'input': 0, 'h': 1, 't': 2, 'c': 3, 'h2': 4, 't2': 5, 'r': 6, 'output': 7}
    for n in doc['graph']['nodes']:  # a readable left-to-right drawing, as in the reported file
        n.update(x=90 + 230 * column[n['id']], y=320 if n['id'] in ('h', 't') else 170)
    return {doc['id']: doc}, doc['id']


def located(models, mid):
    try:
        validate_model(models, mid)
    except GraphError as error:
        return error
    raise AssertionError('draft unexpectedly valid')


class StructuralProblemTests(unittest.TestCase):
    def test_reported_draft_names_both_blocks(self):
        models, mid = reported_draft()
        error = located(models, mid)
        self.assertEqual({(p['code'], p['node']) for p in error.problems},
                         {('no_input', 'output'), ('no_output_path', 'r')})
        self.assertIn("Output 'output' has no incoming wire.", [p['message'] for p in error.problems])
        self.assertEqual((error.code, error.node), (error.problems[0]['code'], error.problems[0]['node']))

    def test_labels_name_blocks_and_extra_inputs_are_located(self):
        models, mid = reported_draft()
        graph = models[mid]['graph']
        graph['edges'].append({'from': 'r', 'to': 'output'})
        graph['edges'].append({'from': 'h', 'to': 'h2'})
        next(n for n in graph['nodes'] if n['id'] == 'h2')['label'] = 'Mixer'
        error = located(models, mid)
        self.assertEqual([p['code'] for p in error.problems], ['extra_inputs'])
        self.assertEqual(str(error), "Linear 'Mixer' has 2 incoming wires; use Concat or Add to join branches.")

    def test_join_and_output_problems(self):
        models, mid = reported_draft()
        graph = models[mid]['graph']
        graph['edges'].remove({'from': 'input', 'to': 'c'})
        graph['edges'] += [{'from': 'r', 'to': 'output'}, {'from': 'output', 'to': 'h'}]
        codes = {(p['code'], p['node']) for p in located(models, mid).problems}
        self.assertIn(('join_sources', 'c'), codes)
        self.assertIn(('output_sends', 'output'), codes)

    def test_shape_errors_are_located_on_their_block(self):
        models, mid = reported_draft()
        graph = models[mid]['graph']
        graph['edges'].append({'from': 'r', 'to': 'output'})
        graph['nodes'].append({'id': 'bad', 'type': 'Reshape', 'shape': [7]})
        graph['edges'].remove({'from': 'h2', 'to': 't2'})
        graph['edges'] += [{'from': 'h2', 'to': 'bad'}, {'from': 'bad', 'to': 't2'}]
        self.assertEqual(located(models, mid).node, 'bad')


class PartialShapeTests(unittest.TestCase):
    def test_reported_draft_shapes_everything_but_the_unwired_output(self):
        models, mid = reported_draft()
        shapes = partial_shapes(models, mid)
        self.assertEqual({k: (v['input_shape'], v['output_shape']) for k, v in shapes.items() if k != 'input'},
                         {'h': ([2], [16]), 't': ([16], [16]), 'c': ([2], [18]), 'h2': ([18], [16]),
                          't2': ([16], [16]), 'r': ([16], [1])})
        self.assertNotIn('output', shapes)
        self.assertNotIn('__probe_output_0', shapes)

    def test_downstream_of_a_problem_is_undetermined_upstream_is_kept(self):
        models, mid = reported_draft()
        graph = models[mid]['graph']
        graph['edges'] += [{'from': 'r', 'to': 'output'}, {'from': 'h', 'to': 'h2'}]  # h2 now has two inputs
        shapes = partial_shapes(models, mid)
        self.assertEqual(set(shapes), {'input', 'h', 't', 'c'})

    def test_shape_error_drops_only_that_block_and_its_descendants(self):
        models, mid = reported_draft()
        graph = models[mid]['graph']
        graph['nodes'].append({'id': 'bad', 'type': 'Reshape', 'shape': [7]})
        graph['edges'].remove({'from': 'h2', 'to': 't2'})
        graph['edges'] += [{'from': 'h2', 'to': 'bad'}, {'from': 'bad', 'to': 't2'}]
        self.assertEqual(set(partial_shapes(models, mid)), {'input', 'h', 't', 'c', 'h2'})

    def test_valid_graph_matches_the_contract_and_input_is_untouched(self):
        models, mid = reported_draft()
        models[mid]['graph']['edges'].append({'from': 'r', 'to': 'output'})
        before = copy.deepcopy(models)
        self.assertEqual(partial_shapes(models, mid), validate_model(models, mid)['shapes'])
        self.assertEqual(models, before)

    def test_whole_graph_problems_give_no_shapes(self):
        models, mid = reported_draft()
        models[mid]['graph']['nodes'] = [n for n in models[mid]['graph']['nodes'] if n['type'] != 'Input']
        models[mid]['graph']['edges'] = [e for e in models[mid]['graph']['edges'] if e['from'] != 'input']
        self.assertEqual(partial_shapes(models, mid), {})


class CommandLineProblemTests(unittest.TestCase):
    def test_validate_reports_located_problems_with_block_ids(self):
        models, mid = reported_draft()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'draft.vtw'
            workspace = Workspace(); workspace.put(models[mid]); workspace.save(path)
            out = subprocess.run([sys.executable, 'vpw_cli.py', str(path), 'validate', mid], capture_output=True,
                                 text=True, cwd=Path(__file__).resolve().parents[1], timeout=60)
        self.assertEqual(out.returncode, 1)
        reply = json.loads(out.stdout)
        self.assertEqual(reply['error'], "Output 'output' has no incoming wire.")
        self.assertEqual([(p['code'], p['node']) for p in reply['problems']], [('no_input', 'output'), ('no_output_path', 'r')])


if __name__ == '__main__':
    unittest.main()
