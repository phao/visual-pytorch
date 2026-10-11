"""Independent pure branch-authoring, preservation and stale-dialog checks."""
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from contextlib import redirect_stdout
from unittest.mock import patch
import vpw_cli
from app.core import GraphError, Workspace, default_model, validate_model
from app.recurrent_authoring import prepare, build
from app.evidence import identity
from test_recurrent_contract import fixture


def authored(cnn=False):
    models, ref, _ = fixture(cnn=cnn)
    parent = default_model('Existing architecture');parent['id'] = 'parent'
    models[parent['id']] = parent
    info = prepare(models, parent['id'], ref)
    token = {key: info[key] for key in ('id', 'ref', 'root_graph', 'transition')}
    options = {'version': 1, 'steps': 4, 'label': 'Explicit scan',
               'inputs': [{'input': 'frame' if cnn else 'x', 'role': 'sequence', 'output': None},
                          {'input': 'h', 'role': 'state', 'output': 'next'}],
               'sequence_outputs': ['next'], 'controls': ['lengths', 'mask', 'reset']}
    return models, token, options


class RecurrentAuthoringTests(unittest.TestCase):
    def test_prepare_and_build_do_not_mutate_sources_or_choices(self):
        models, token, options = authored();before = copy.deepcopy((models, token, options))
        doc, nid = build(models, token, options)
        self.assertEqual((models, token, options), before)
        self.assertEqual(doc['graph']['nodes'][:3], models['parent']['graph']['nodes'])
        self.assertEqual(doc['graph']['edges'][:2], models['parent']['graph']['edges'])
        c = validate_model({**models, 'parent': doc}, 'parent')
        self.assertEqual(c['parameters'], 13)
        self.assertEqual(c['shapes'][nid]['ports'], {'sequence/next': [4, 2], 'state/h': [2]})

    def test_complete_control_and_axis_declarations_in_original_input_order(self):
        models, token, options = authored(cnn=True);doc, nid = build(models, token, options)
        n = next(n for n in doc['graph']['nodes'] if n['id'] == nid)
        inputs = {n['id']: n for n in doc['graph']['nodes'] if n['type'] == 'Input'}
        sequence = inputs[n['sequence_inputs'][0]['source']]
        self.assertEqual(sequence['shape'], [4, 1, 4, 5]);self.assertEqual(sequence['axes'], ['T', 'C', 'H', 'W'])
        self.assertEqual(doc['graph']['input_order'][0], 'input')
        self.assertEqual(inputs[n['controls']['lengths']['source']]['shape'], [1])
        self.assertEqual(n['controls']['reset']['adapter'], 'exact-bool-v1')

    def test_final_only_and_no_controls_are_valid_explicit_choices(self):
        models, token, options = authored();options.update(sequence_outputs=[], controls=[])
        doc, nid = build(models, token, options);c = validate_model({**models, 'parent': doc}, 'parent')
        self.assertEqual(c['shapes'][nid]['ports'], {'state/h': [2]})

    def test_repeated_branches_have_unique_ids_and_boundary_names(self):
        models, token, options = authored();models['parent'], _ = build(models, token, options)
        info = prepare(models, 'parent', token['ref']);token = {k: info[k] for k in token}
        doc, _ = build(models, token, options);validate_model({**models, 'parent': doc}, 'parent')
        self.assertEqual(len(doc['graph']['nodes']), len({n['id'] for n in doc['graph']['nodes']}))
        for kind in ('Input', 'Output'):
            names = [n.get('label', n['id']) for n in doc['graph']['nodes'] if n['type'] == kind]
            self.assertEqual(len(names), len(set(names)))

    def test_stale_parent_geometry_and_transition_semantics_or_labels_refuse(self):
        for where in ('parent', 'body', 'name'):
            models, token, options = authored()
            if where == 'parent':models['parent']['graph']['nodes'][0]['x'] += 24
            elif where == 'body':models[token['ref']]['graph']['nodes'][3]['bias'] = False
            else:models[token['ref']]['name'] = 'Renamed'
            before = copy.deepcopy(models)
            with self.subTest(where=where), self.assertRaisesRegex(GraphError, 'changed'):build(models, token, options)
            self.assertEqual(models, before)

    def test_closed_tokens_options_and_exact_positive_T(self):
        for field, value in [('version', True), ('steps', True), ('steps', 0), ('steps', 1.5), ('steps', 2**53), ('label', ''), ('label', 'x'*61), ('controls', ['mask', 'mask']), ('sequence_outputs', ['missing'])]:
            models, token, options = authored();options[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(GraphError):build(models, token, options)
        models, token, options = authored()
        for bad in ({**token, 'extra': 1}, {k: v for k, v in token.items() if k != 'transition'}):
            with self.assertRaises(GraphError):build(models, bad, options)
        options['extra'] = 'no'
        with self.assertRaises(GraphError):build(models, token, options)

    def test_roles_are_complete_and_explicit_with_matching_state_outputs(self):
        for changes in ([{'input': 'x', 'role': '', 'output': None}, {'input': 'h', 'role': 'state', 'output': 'next'}],
                        [{'input': 'x', 'role': 'sequence', 'output': 'next'}, {'input': 'h', 'role': 'state', 'output': 'next'}],
                        [{'input': 'x', 'role': 'state', 'output': 'next'}, {'input': 'h', 'role': 'state', 'output': 'next'}],
                        [{'input': 'x', 'role': 'sequence', 'output': None}, {'input': 'x', 'role': 'state', 'output': 'next'}],
                        [{'input': 'x', 'role': 'sequence', 'output': None}, {'input': 'h', 'role': 'state', 'output': 'missing'}]):
            models, token, options = authored();options['inputs'] = changes
            with self.subTest(changes=changes), self.assertRaises(GraphError):build(models, token, options)

    def test_incompatible_state_shape_is_refused(self):
        models, token, options = authored();models[token['ref']]['graph']['nodes'][3]['features'] = 3
        info = prepare(models, 'parent', token['ref']);token = {k: info[k] for k in token}
        with self.assertRaisesRegex(GraphError, 'shape'):build(models, token, options)

    def test_reference_cycle_is_refused(self):
        models, token, options = authored()
        models[token['ref']]['graph'] = {'nodes': [{'id': 'x', 'type': 'Input', 'features': 2},
          {'id': 'm', 'type': 'Module', 'ref': 'parent'}, {'id': 'y', 'type': 'Output'}],
          'edges': [{'from': 'x', 'to': 'm'}, {'from': 'm', 'to': 'y'}]}
        with self.assertRaisesRegex(GraphError, 'cycle'):prepare(models, 'parent', token['ref'])

    def test_exact_control_and_node_capacity_guards(self):
        models, token, options = authored();options['steps'] = 2**24+1
        with self.assertRaisesRegex(GraphError, 'lengths'):build(models, token, options)
        options['steps'] = 4
        models['parent']['graph']['nodes'].extend({'id': 'spare'+str(i), 'type': 'Linear', 'features': 1} for i in range(60))
        info = prepare(models, 'parent', token['ref']);token = {k: info[k] for k in token}
        with self.assertRaisesRegex(GraphError, '64 blocks'):build(models, token, options)

    def test_unrelated_invalid_draft_is_preserved_while_a_valid_branch_is_authored(self):
        models, token, options = authored();models['parent']['graph']['edges'].pop()
        info = prepare(models, 'parent', token['ref']);token = {k: info[k] for k in token}
        doc, _ = build(models, token, options)
        self.assertEqual(doc['graph']['edges'][0], models['parent']['graph']['edges'][0])
        self.assertNotIn({'from': 'linear1', 'to': 'output'}, doc['graph']['edges'])
        with self.assertRaises(GraphError):validate_model({**models, 'parent': doc}, 'parent')

    def test_workspace_save_reload_identity_and_old_dataset_clear(self):
        models, token, options = authored();models['parent']['dataset'] = {'x': [[1, 2]], 'y': [[1]]}
        doc, _ = build(models, token, options);self.assertIsNone(doc['dataset']);self.assertIsNone(doc['result'])
        models['parent'] = doc
        with tempfile.TemporaryDirectory() as tmp:
            w = Workspace()
            try:
                for model in models.values():w.put(model)
                path = Path(tmp)/'author.vtw';w.save(path);loaded = Workspace.load(path)
                try:self.assertEqual(identity(loaded.models(), 'parent'), identity(models, 'parent'))
                finally:loaded.db.close()
            finally:w.db.close()

    def test_cli_preparation_is_readonly_and_apply_uses_the_same_guarded_builder(self):
        models, token, options = authored()
        with tempfile.TemporaryDirectory() as tmp:
            w=Workspace();path=Path(tmp)/'cli.vtw'
            try:
                for model in models.values():w.put(model)
                w.save(path)
            finally:w.db.close()
            before=path.read_bytes();stream=io.StringIO()
            with redirect_stdout(stream):self.assertEqual(vpw_cli.main([str(path),'prepare-recurrent','parent','transition']),0)
            info=json.loads(stream.getvalue())['value'];self.assertEqual(path.read_bytes(),before)
            token={k:info[k] for k in token};edits=[{'op':'add-recurrent','token':token,'options':options}]
            with patch('sys.stdin',io.StringIO(json.dumps(edits))),redirect_stdout(io.StringIO()):
                self.assertEqual(vpw_cli.main([str(path),'apply','-']),0)
            after=path.read_bytes();self.assertNotEqual(after,before)
            with patch('sys.stdin',io.StringIO(json.dumps(edits))),redirect_stdout(io.StringIO()):
                self.assertEqual(vpw_cli.main([str(path),'apply','-']),1)
            self.assertEqual(path.read_bytes(),after)
            loaded=Workspace.load(path)
            try:self.assertEqual(validate_model(loaded.models(),'parent')['parameters'],13)
            finally:loaded.db.close()

    def test_cli_batch_failure_leaves_disk_unchanged_after_a_provisional_branch(self):
        models, token, options = authored()
        with tempfile.TemporaryDirectory() as tmp:
            w=Workspace();path=Path(tmp)/'atomic.vtw'
            try:
                for model in models.values():w.put(model)
                w.save(path)
                info=prepare(w.models(),'parent','transition');token={k:info[k] for k in token}
            finally:w.db.close()
            before=path.read_bytes();edits=[{'op':'add-recurrent','token':token,'options':options},{'op':'unsupported','id':'parent'}]
            with patch('sys.stdin',io.StringIO(json.dumps(edits))),redirect_stdout(io.StringIO()):
                self.assertEqual(vpw_cli.main([str(path),'apply','-']),1)
            self.assertEqual(path.read_bytes(),before)
