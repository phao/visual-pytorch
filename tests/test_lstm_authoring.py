"""Guarded LSTM creation/editing, independent shapes and real serialized CLI."""
import copy
import hashlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import vpw_cli
from app.core import GraphError, Workspace, default_model, validate_model
from app.evidence import parameter_schema
from app.lstm_authoring import prepare, build, prepare_edit, build_edit, CREATE_TOKEN, EDIT_TOKEN
from app.presentation import block_parameter_counts


ROOT = Path(__file__).resolve().parents[1]


def creation(**options):
    parent = default_model('Parent'); parent['id'] = 'parent'
    models = {'parent': parent}; info = prepare(models, 'parent')
    choices = info['options']; choices.update(options)
    return models, {k: info[k] for k in CREATE_TOKEN}, choices


def authored(**options):
    models, token, choices = creation(**options)
    models['parent'], nid = build(models, token, choices)
    info = prepare_edit(models, 'parent', nid)
    return models, {k: info[k] for k in EDIT_TOKEN}, info['options']


def by_id(doc, nid):
    return next(n for n in doc['graph']['nodes'] if n['id'] == nid)


class LSTMAuthoringTests(unittest.TestCase):
    def test_prepare_creation_defaults_are_explicit_and_do_not_mutate(self):
        models, token, choices = creation(); before = copy.deepcopy(models)
        self.assertEqual(set(token), set(CREATE_TOKEN))
        self.assertEqual((choices['steps'], choices['input_size'], choices['hidden_size']), (4, 3, 8))
        self.assertEqual((choices['bias'], choices['bidirectional'], choices['lengths']), (True, False, False))
        self.assertEqual(models, before)

    def test_creation_shapes_axes_and_counts_are_independently_computed(self):
        for h, p, layers, directions, bias in ((3, 0, 1, 1, False), (7, 4, 2, 2, True),
                                               (4, 1, 3, 1, True), (2, 0, 2, 2, False)):
            with self.subTest(h=h, p=p, layers=layers, directions=directions, bias=bias):
                models, token, options = creation(steps=6, input_size=5, hidden_size=h,
                    proj_size=p, num_layers=layers, bidirectional=directions == 2, bias=bias)
                before = copy.deepcopy((models, token, options)); doc, nid = build(models, token, options)
                self.assertEqual((models, token, options), before)
                node = by_id(doc, nid); q = p or h; ld = layers * directions
                for role, shape, axes in (('sequence', [6, 5], ['T', 'F']),
                                          ('hidden', [ld, q], ['LD', 'Hout']),
                                          ('cell', [ld, h], ['LD', 'Hcell'])):
                    source = by_id(doc, node['inputs'][role])
                    self.assertEqual((source['shape'], source['axes']), (shape, axes))
                c = validate_model({**models, 'parent': doc}, 'parent')
                self.assertEqual(c['shapes'][nid]['ports'],
                    {'sequence': [6, directions*q], 'hidden': [ld, q], 'cell': [ld, h]})
                count = sum(directions * (4*h*((5 if i == 0 else directions*q)+q) +
                                          (8*h if bias else 0) + (p*h if p else 0))
                            for i in range(layers))
                self.assertEqual(c['parameters'], 3 + count)
                self.assertEqual(block_parameter_counts(doc, c, {})[nid], str(count))
                ids = {entry['id'] for entry in parameter_schema({**models, 'parent': doc}, 'parent').values()}
                self.assertIn('parent/'+nid+'/weight_ih_l0', ids)

    def test_creation_appends_roots_without_changing_old_packing_or_nodes(self):
        models, token, options = creation(lengths=True); old = copy.deepcopy(models['parent'])
        old['graph']['input_order'] = ['input']; models['parent'] = old
        info = prepare(models, 'parent'); token = {k: info[k] for k in CREATE_TOKEN}
        doc, nid = build(models, token, options)
        self.assertEqual(doc['graph']['nodes'][:3], old['graph']['nodes'])
        self.assertEqual(doc['graph']['edges'][:2], old['graph']['edges'])
        self.assertEqual(doc['graph']['input_order'][0], 'input')
        self.assertEqual(len(doc['graph']['input_order']), 5)
        node = by_id(doc, nid)
        self.assertEqual(node['lengths']['adapter'], 'exact-int64-v1')
        self.assertEqual(by_id(doc, node['lengths']['source'])['shape'], [1])
        self.assertEqual([e['from_port'] for e in doc['graph']['edges'] if e['from'] == nid],
                         ['sequence', 'hidden', 'cell'])

    def test_repeated_branches_have_distinct_ids_and_boundary_names(self):
        models, token, options = creation(label='a'*60)
        models['parent'], first = build(models, token, options)
        info = prepare(models, 'parent'); token = {k: info[k] for k in CREATE_TOKEN}
        doc, second = build(models, token, options)
        self.assertNotEqual(first, second)
        self.assertEqual(len({n['id'] for n in doc['graph']['nodes']}), len(doc['graph']['nodes']))
        for kind in ('Input', 'Output'):
            names = [n.get('label') or n['id'] for n in doc['graph']['nodes'] if n['type'] == kind]
            self.assertEqual(len(set(names)), len(names)); self.assertTrue(all(len(n) <= 60 for n in names))
        validate_model({**models, 'parent': doc}, 'parent')

    def test_stale_graph_name_and_reference_closure_refuse(self):
        for kind in ('geometry', 'name', 'reference', 'missing-reference-now-present'):
            models, _, options = creation()
            if 'reference' in kind:
                models['parent']['graph']['nodes'].append({'id': 'ref', 'type': 'Module', 'ref': 'child'})
                if kind == 'reference':
                    child = default_model(); child['id'] = 'child'; models['child'] = child
            info = prepare(models, 'parent'); token = {k: info[k] for k in CREATE_TOKEN}
            if kind == 'geometry': models['parent']['graph']['nodes'][0]['x'] += 1
            elif kind == 'name': models['parent']['name'] = 'Changed'
            elif kind == 'reference': models['child']['graph']['nodes'][1]['features'] = 3
            else:
                child = default_model(); child['id'] = 'child'; models['child'] = child
            before = copy.deepcopy(models)
            with self.subTest(kind=kind), self.assertRaisesRegex(GraphError, 'changed'):
                build(models, token, options)
            self.assertEqual(models, before)

    def test_data_and_result_changes_do_not_stale_the_architecture_token(self):
        models, token, options = creation()
        models['parent'].update(dataset={'keep': 'new data'}, result={'keep': 'new result'})
        doc, _ = build(models, token, options)
        self.assertIsNone(doc['dataset']); self.assertIsNone(doc['result'])
        self.assertEqual(models['parent']['dataset'], {'keep': 'new data'})

    def test_closed_tokens_and_constructor_options_reject(self):
        models, token, options = creation()
        for candidate in ({}, {**token, 'extra': True}, {**token, 'id': []}):
            with self.subTest(token=candidate), self.assertRaises(GraphError):build(models, candidate, options)
        for field, value in (('steps', True), ('steps', 0), ('input_size', 1.5), ('version', True),
                             ('hidden_size', 0), ('num_layers', 65), ('bias', 1), ('bidirectional', 'no'),
                             ('lengths', 1), ('proj_size', 8), ('proj_size', -1), ('label', ''), ('label', 'x'*61)):
            with self.subTest(field=field, value=value), self.assertRaises(GraphError):
                build(models, token, {**options, field: value})
        for candidate in ({**options, 'extra': 1}, {k: v for k, v in options.items() if k != 'bias'}):
            with self.assertRaises(GraphError):build(models, token, candidate)

    def test_supplied_lengths_exactness_limit_and_node_capacity(self):
        models, token, options = creation(lengths=True, steps=2**24+1)
        with self.assertRaisesRegex(GraphError, r'2\*\*24'):build(models, token, options)
        models, _, options = creation()
        models['parent']['graph']['nodes'] += [{'id': 'draft'+str(i), 'type': 'Linear', 'features': 1} for i in range(55)]
        info = prepare(models, 'parent'); token = {k: info[k] for k in CREATE_TOKEN}
        with self.assertRaisesRegex(GraphError, '64 blocks'):build(models, token, options)

    def test_invalid_unrelated_draft_is_preserved_not_repaired(self):
        models, _, options = creation()
        models['parent']['graph']['nodes'].append({'id': 'draft', 'type': 'Linear', 'features': 0})
        info = prepare(models, 'parent'); token = {k: info[k] for k in CREATE_TOKEN}
        doc, nid = build(models, token, options)
        self.assertEqual(by_id(doc, 'draft'), by_id(models['parent'], 'draft'))
        models['parent'] = doc; info = prepare_edit(models, 'parent', nid)
        self.assertTrue(all(s['shape'] is None for s in info['sources']))
        token = {k: info[k] for k in EDIT_TOKEN}; choices = {**info['options'], 'label': 'Edited'}
        edited, _ = build_edit(models, token, choices)
        self.assertEqual(by_id(edited, 'draft'), by_id(doc, 'draft'))
        with self.assertRaises(GraphError):validate_model({**models, 'parent': edited}, 'parent')

    def test_noop_preserves_graph_data_result_and_does_not_mutate_options(self):
        models, token, options = authored()
        models['parent'].update(dataset={'keep': True}, result={'keep': True})
        before = copy.deepcopy((models, token, options)); doc, nid = build_edit(models, token, options)
        self.assertEqual(doc, models['parent']); self.assertEqual(nid, token['node'])
        self.assertEqual((models, token, options), before)

    def test_noop_preserves_implicit_operator_defaults_and_missing_label(self):
        models, token, _ = authored()
        node = by_id(models['parent'], token['node'])
        for key in ('label', 'num_layers', 'bidirectional', 'proj_size', 'bias', 'op_version', 'lengths'):
            node.pop(key)
        models['parent'].update(dataset={'keep': True}, result={'keep': True})
        info = prepare_edit(models, 'parent', token['node']); token = {k: info[k] for k in EDIT_TOKEN}
        doc, _ = build_edit(models, token, info['options'])
        self.assertEqual(doc, models['parent'])

    def test_label_edit_keeps_non_target_nodes_geometry_and_outgoing_wires(self):
        models, token, options = authored(lengths=True); old = copy.deepcopy(models['parent'])
        node = by_id(models['parent'], token['node']); node.update(width=190, height=120, collapsed=True)
        info = prepare_edit(models, 'parent', token['node']); token = {k: info[k] for k in EDIT_TOKEN}
        old = copy.deepcopy(models['parent']); doc, _ = build_edit(models, token, {**options, 'label': 'Changed'})
        self.assertEqual(doc['graph']['edges'], old['graph']['edges'])
        self.assertEqual(doc['graph']['input_order'], old['graph']['input_order'])
        self.assertEqual([n for n in doc['graph']['nodes'] if n['id'] != token['node']],
                         [n for n in old['graph']['nodes'] if n['id'] != token['node']])
        for field in ('x', 'y', 'width', 'height', 'collapsed'):
            self.assertEqual(by_id(doc, token['node'])[field], node[field])

    def test_constructor_change_never_resizes_existing_state_sources_implicitly(self):
        models, token, options = authored(); before = copy.deepcopy(models)
        with self.assertRaisesRegex(GraphError, 'initial hidden'):
            build_edit(models, token, {**options, 'hidden_size': 5})
        self.assertEqual(models, before)

    def test_explicit_matching_input_shapes_allow_projection_direction_and_layers_edit(self):
        models, token, options = authored(); node = by_id(models['parent'], token['node'])
        by_id(models['parent'], node['inputs']['hidden'])['shape'] = [4, 3]
        by_id(models['parent'], node['inputs']['cell'])['shape'] = [4, 7]
        info = prepare_edit(models, 'parent', token['node']); token = {k: info[k] for k in EDIT_TOKEN}
        doc, _ = build_edit(models, token, {**options, 'hidden_size': 7, 'proj_size': 3,
                                           'num_layers': 2, 'bidirectional': True, 'bias': False})
        c = validate_model({**models, 'parent': doc}, 'parent')
        self.assertEqual(c['shapes'][token['node']]['ports'],
                         {'sequence': [4, 6], 'hidden': [4, 3], 'cell': [4, 7]})

    def test_catalog_is_alphabetical_and_never_exposes_output_or_self_as_source(self):
        models, token, _ = authored(); info = prepare_edit(models, 'parent', token['node'])
        self.assertEqual(info['sources'], sorted(info['sources'], key=lambda s: (s['name'].casefold(), s['key'])))
        self.assertTrue(all(s['key'] != token['node'] and by_id(models['parent'], s['key'])['type'] != 'Output'
                            for s in info['sources']))

    def test_named_source_rebinding_preserves_physical_port_identity_and_other_nodes(self):
        models, token, options = authored(); source = options['inputs']['hidden']
        models['parent']['graph']['nodes'].append({'id': 'router', 'type': 'Replicator', 'outputs': ['left', 'right']})
        models['parent']['graph']['edges'].append({'from': source, 'to': 'router'})
        info = prepare_edit(models, 'parent', token['node']); token = {k: info[k] for k in EDIT_TOKEN}
        choices = copy.deepcopy(options); choices['inputs']['hidden'] = 'router@left'
        doc, _ = build_edit(models, token, choices)
        self.assertIn({'from': 'router', 'from_port': 'left', 'to': token['node']}, doc['graph']['edges'])
        self.assertNotIn({'from': source, 'to': token['node']}, doc['graph']['edges'])
        self.assertEqual(by_id(doc, source), by_id(models['parent'], source))
        validate_model({**models, 'parent': doc}, 'parent')

    def test_same_hidden_cell_source_has_one_wire_when_axes_are_not_confused(self):
        models, token, options = authored(); node = by_id(models['parent'], token['node'])
        by_id(models['parent'], node['inputs']['hidden']).pop('axes')
        info = prepare_edit(models, 'parent', token['node']); token = {k: info[k] for k in EDIT_TOKEN}
        choices = copy.deepcopy(options); choices['inputs']['cell'] = choices['inputs']['hidden']
        doc, _ = build_edit(models, token, choices)
        self.assertEqual(len([e for e in doc['graph']['edges'] if e['to'] == token['node']]), 2)
        self.assertIn(node['inputs']['cell'], {n['id'] for n in doc['graph']['nodes']})
        # Unused former cell Input is preserved and still blocks whole-model execution.
        with self.assertRaises(GraphError):validate_model({**models, 'parent': doc}, 'parent')

    def test_invalid_self_missing_role_adapter_and_owner_refuse_without_mutation(self):
        for alteration in ('self', 'missing', 'role', 'adapter', 'owner', 'version', 'extra'):
            models, token, options = authored(lengths=True); before = copy.deepcopy(models)
            if alteration == 'self':options['inputs']['sequence'] = token['node']+'@sequence'
            elif alteration == 'missing':options['inputs']['hidden'] = 'missing'
            elif alteration == 'role':options['inputs'].pop('cell')
            elif alteration == 'adapter':options['lengths']['adapter'] = 'guess'
            elif alteration == 'owner':options['share_with'] = 'linear1'
            elif alteration == 'version':options['version'] = True
            else:options['extra'] = 'unsupported'
            with self.subTest(alteration=alteration), self.assertRaises(GraphError):build_edit(models, token, options)
            self.assertEqual(models, before)

    def test_downstream_declared_shape_incompatibility_is_refused(self):
        models, token, options = authored(); graph = models['parent']['graph']
        dest = next(e['to'] for e in graph['edges'] if e['from'] == token['node'] and e['from_port'] == 'sequence')
        by_id(models['parent'], dest)['shape'] = [4, 7]
        info = prepare_edit(models, 'parent', token['node']); token = {k: info[k] for k in EDIT_TOKEN}
        before = copy.deepcopy(models)
        with self.assertRaises(GraphError):build_edit(models, token, {**options, 'label': 'Changed'})
        self.assertEqual(models, before)

    def test_sharing_counts_canonical_owner_once_and_incompatible_owner_edit_refuses(self):
        models, token, options = authored(); first = token['node']
        info = prepare(models, 'parent'); choices = info['options']
        models['parent'], second = build(models, {k: info[k] for k in CREATE_TOKEN}, choices)
        info = prepare_edit(models, 'parent', second); token = {k: info[k] for k in EDIT_TOKEN}
        models['parent'], _ = build_edit(models, token, {**info['options'], 'share_with': first})
        c = validate_model(models, 'parent'); self.assertEqual(c['parameters'], 419)
        counts = block_parameter_counts(models['parent'], c, {})
        self.assertEqual((counts[first], counts[second]), ('416', '0'))
        info = prepare_edit(models, 'parent', first); token = {k: info[k] for k in EDIT_TOKEN}
        with self.assertRaisesRegex(GraphError, 'same constructor'):
            build_edit(models, token, {**info['options'], 'bias': False})

    def test_shared_alias_cycle_is_refused(self):
        models, token, options = authored()
        with self.assertRaisesRegex(GraphError, 'cycle'):
            build_edit(models, token, {**options, 'share_with': token['node']})

    def test_workspace_roundtrip_and_independent_frozen_revision_survive_edit(self):
        models, token, options = authored(lengths=True)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'lstm.vtw'; w = Workspace()
            try:
                w.put(models['parent']); ref = w.freeze_model('parent'); frozen = w.get_spec('model', ref['id'], ref['revision'])
                actual = w.models(); info = prepare_edit(actual, 'parent', token['node'])
                doc, _ = build_edit(actual, {k: info[k] for k in EDIT_TOKEN}, {**info['options'], 'bias': False})
                w.put(doc); w.save(path)
                self.assertEqual(w.get_spec('model', ref['id'], ref['revision']), frozen)
                loaded = Workspace.load(path)
                try:
                    self.assertEqual(loaded.models(), w.models()); self.assertEqual(loaded.get_spec('model', ref['id'], ref['revision']), frozen)
                finally:loaded.db.close()
            finally:w.db.close()


class LSTMSerializedCLITests(unittest.TestCase):
    def cli(self, path, *args, text=None):
        result = subprocess.run([sys.executable, str(ROOT/'vpw_cli.py'), str(path), *args],
                                input=text, text=True, capture_output=True, cwd=ROOT, timeout=20)
        return result.returncode, json.loads(result.stdout)

    def workspace(self, path):
        w = Workspace(); parent = default_model(); parent['id'] = 'parent'
        try:w.put(parent); w.save(path)
        finally:w.db.close()

    def test_real_stdin_create_prepare_edit_replay_and_file_conflict(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'lab.vtw'; self.workspace(path); old = path.read_bytes()
            code, response = self.cli(path, 'prepare-lstm', 'parent'); self.assertEqual(code, 0, response)
            self.assertEqual(path.read_bytes(), old); info = response['value']; token = {k: info[k] for k in CREATE_TOKEN}
            choices = {**info['options'], 'hidden_size': 5, 'bias': False}
            edit = {'op': 'add-lstm', 'token': token, 'options': choices}
            code, response = self.cli(path, '--expect-revision', hashlib.sha256(old).hexdigest(), 'apply', '-', text=json.dumps([edit]))
            self.assertEqual(code, 0, response); nid = response['value']['edits'][0]['node']; saved = path.read_bytes()
            code, response = self.cli(path, 'prepare-lstm-edit', 'parent', nid); self.assertEqual(code, 0, response)
            self.assertEqual(path.read_bytes(), saved); info = response['value']
            edit = {'op': 'edit-lstm', 'token': {k: info[k] for k in EDIT_TOKEN},
                    'options': {**info['options'], 'label': 'Agent edit'}}
            code, response = self.cli(path, 'apply', '-', text=json.dumps([edit])); self.assertEqual(code, 0, response)
            after = path.read_bytes()
            self.assertEqual(self.cli(path, 'apply', '-', text=json.dumps([edit]))[0], 1)
            self.assertEqual(path.read_bytes(), after)
            self.assertEqual(self.cli(path, '--expect-revision', hashlib.sha256(saved).hexdigest(), 'inspect')[0], 1)
            self.assertEqual(path.read_bytes(), after)

    def test_real_file_batch_late_failure_and_stale_token_are_atomic(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'lab.vtw'; self.workspace(path); before = path.read_bytes()
            info = self.cli(path, 'prepare-lstm', 'parent')[1]['value']
            edits = [{'op': 'add-lstm', 'token': {k: info[k] for k in CREATE_TOKEN}, 'options': info['options']},
                     {'op': 'not-a-valid-operation'}]
            batch = Path(folder)/'batch.json'; batch.write_text(json.dumps(edits))
            self.assertEqual(self.cli(path, 'apply', str(batch))[0], 1); self.assertEqual(path.read_bytes(), before)
            edits[1] = {'op': 'rename', 'id': 'parent', 'name': 'Changed'}
            edits.reverse(); batch.write_text(json.dumps(edits))
            self.assertEqual(self.cli(path, 'apply', str(batch))[0], 1); self.assertEqual(path.read_bytes(), before)

    def test_real_duplicate_key_and_all_nonfinite_json_refusals_leave_file_unchanged(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'lab.vtw'; self.workspace(path); before = path.read_bytes()
            for text in ('[{"op":"rename","op":"move","id":"parent"}]',
                         '[{"op":"rename","id":"parent","name":NaN}]',
                         '[{"op":"rename","id":"parent","name":Infinity}]',
                         '[{"op":"rename","id":"parent","name":1e309}]'):
                with self.subTest(text=text):
                    code, response = self.cli(path, 'apply', '-', text=text)
                    self.assertEqual(code, 1, response); self.assertEqual(path.read_bytes(), before)

    def test_read_json_file_and_stdin_share_strict_seam(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'options.json'
            for text in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":1e309}'):
                path.write_text(text)
                with self.assertRaises(ValueError):vpw_cli.read_json(str(path))
                with patch('vpw_cli.sys.stdin', io.StringIO(text)), self.assertRaises(ValueError):vpw_cli.read_json('-')
            text = '{"x":[1,2],"name":"exact"}'; path.write_text(text)
            with patch('vpw_cli.sys.stdin', io.StringIO(text)):
                self.assertEqual(vpw_cli.read_json(str(path)), vpw_cli.read_json('-'))


if __name__ == '__main__':unittest.main()
