"""Independent preservation, closure, source, port and atomic CLI edit checks."""
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from contextlib import redirect_stdout
from unittest.mock import patch

import vpw_cli
from app.core import GraphError, Workspace, validate_model
from app.evidence import identity
from app.recurrent_authoring import prepare, build
from app.recurrent_reconfiguration import prepare_edit, build_edit, TOKEN
from test_recurrent_authoring import authored


def editable(cnn=False):
    models, token, choices = authored(cnn)
    models['parent'], nid = build(models, token, choices)
    info = prepare_edit(models, 'parent', nid)
    return models, {key: info[key] for key in TOKEN}, info['options']


class RecurrentReconfigurationTests(unittest.TestCase):
    def test_changed_transition_can_be_rebound_to_multiple_sequences_and_states(self):
        models, token, options = editable();graph = models['transition']['graph'];cloned = copy.deepcopy(graph['nodes'])
        for node in cloned:
            node['id'] += '2'
            if 'label' in node:node['label'] += ' second'
            if node['type'] == 'Concat':node['sources'] = [s+'2' for s in node['sources']]
        graph['nodes'].extend(cloned)
        graph['edges'].extend({'from': e['from']+'2', 'to': e['to']+'2'} for e in list(graph['edges']))
        graph['input_order'].extend(['x2', 'h2'])
        info = prepare_edit(models, 'parent', token['node']);token = {k: info[k] for k in TOKEN}
        options['sequence_inputs'].append({'input': 'x2', 'source': options['sequence_inputs'][0]['source']})
        options['states'].append({'input': 'h2', 'output': 'next2', 'port': 'state/h2', 'source': options['states'][0]['source']})
        options['sequence_outputs'].append({'output': 'next2', 'port': 'sequence/next2'})
        doc, _ = build_edit(models, token, options)
        self.assertEqual(doc['graph']['edges'], models['parent']['graph']['edges'])
        c = validate_model({**models, 'parent': doc}, 'parent');self.assertEqual(c['parameters'], 23)
        self.assertEqual(c['shapes'][token['node']]['ports'], {'sequence/next': [4, 2], 'sequence/next2': [4, 2], 'state/h': [2], 'state/h2': [2]})

    def test_input_roles_can_change_explicitly_without_replacing_physical_sources(self):
        models, token, options = editable()
        options['sequence_inputs'][0]['input'] = 'h';options['states'][0]['input'] = 'x'
        doc, _ = build_edit(models, token, options)
        self.assertEqual(doc['graph']['edges'], models['parent']['graph']['edges'])
        self.assertEqual(validate_model({**models, 'parent': doc}, 'parent')['shapes'][token['node']]['ports'],
                         {'sequence/next': [4, 2], 'state/h': [2]})

    def test_named_router_port_is_an_explicit_source_and_keeps_the_port_identity(self):
        models, token, options = editable();source = options['states'][0]['source']
        graph = models['parent']['graph'];graph['nodes'].append({'id': 'router', 'type': 'Replicator', 'outputs': ['initial', 'other']})
        graph['edges'].append({'from': source, 'to': 'router'})
        info = prepare_edit(models, 'parent', token['node']);token = {k: info[k] for k in TOKEN}
        options['states'][0]['source'] = 'router@initial'
        doc, _ = build_edit(models, token, options)
        self.assertIn({'from': 'router', 'from_port': 'initial', 'to': token['node']}, doc['graph']['edges'])
        self.assertEqual(validate_model({**models, 'parent': doc}, 'parent')['parameters'], 13)

    def test_sharing_edits_validate_the_owner_and_all_affected_aliases(self):
        models, token, options = editable();first = token['node']
        _, _, creation = authored()
        prep = prepare(models, 'parent', 'transition');creation_token = {k: prep[k] for k in ('id', 'ref', 'root_graph', 'transition')}
        models['parent'], second = build(models, creation_token, creation)
        info = prepare_edit(models, 'parent', second);token = {k: info[k] for k in TOKEN};options = info['options'];options['share_with'] = first
        models['parent'], _ = build_edit(models, token, options)
        c = validate_model(models, 'parent');self.assertEqual(c['parameters'], 13)
        self.assertEqual(c['shapes'][second]['parameter_owner'], first)
        info = prepare_edit(models, 'parent', first);token = {k: info[k] for k in TOKEN};options = info['options']
        options['sequence_outputs'].append({'output': 'next', 'port': 'incompatible-owner-port'})
        before = copy.deepcopy(models)
        with self.assertRaisesRegex(GraphError, 'same constructor'):build_edit(models, token, options)
        self.assertEqual(models, before)

    def test_a_changed_downstream_shape_is_refused_even_with_unrelated_drafts(self):
        models, token, options = editable()
        output = next(e['to'] for e in models['parent']['graph']['edges'] if e['from'] == token['node'] and e['from_port'] == 'sequence/next')
        next(n for n in models['parent']['graph']['nodes'] if n['id'] == output)['shape'] = [9, 2]
        info = prepare_edit(models, 'parent', token['node']);token = {k: info[k] for k in TOKEN}
        before = copy.deepcopy(models)
        with self.assertRaises(GraphError):build_edit(models, token, options)
        self.assertEqual(models, before)

    def test_prepare_and_noop_preserve_all_sources_options_and_data(self):
        models, token, options = editable()
        models['parent']['dataset'] = {'keep': True};models['parent']['result'] = {'keep': True}
        before = copy.deepcopy((models, token, options))
        doc, nid = build_edit(models, token, options)
        self.assertEqual(doc, models['parent']);self.assertEqual(nid, token['node'])
        self.assertEqual((models, token, options), before)

    def test_label_edit_preserves_geometry_input_order_nodes_wires_and_clears_legacy(self):
        models, token, options = editable();node = next(n for n in models['parent']['graph']['nodes'] if n['id'] == token['node'])
        node.update(width=300, height=230, collapsed=True)
        info = prepare_edit(models, 'parent', token['node']);token = {k: info[k] for k in TOKEN}
        models['parent'].update(dataset={'old': True}, result={'old': True})
        options['label'] = 'Renamed scan'
        before = copy.deepcopy(models)
        doc, _ = build_edit(models, token, options)
        after = next(n for n in doc['graph']['nodes'] if n['id'] == token['node'])
        self.assertEqual(after, {**node, 'label': 'Renamed scan'})
        self.assertEqual(doc['graph']['edges'], before['parent']['graph']['edges'])
        self.assertEqual(doc['graph']['input_order'], before['parent']['graph']['input_order'])
        self.assertEqual([n for n in doc['graph']['nodes'] if n['id'] != token['node']], [n for n in before['parent']['graph']['nodes'] if n['id'] != token['node']])
        self.assertIsNone(doc['dataset']);self.assertIsNone(doc['result']);self.assertEqual(models, before)

    def test_explicit_existing_source_rebind_does_not_make_new_boundaries(self):
        models, token, options = editable();old = options['states'][0]['source'];options['states'][0]['source'] = 'input'
        doc, _ = build_edit(models, token, options)
        self.assertEqual(len(doc['graph']['nodes']), len(models['parent']['graph']['nodes']))
        self.assertNotIn({'from': old, 'to': token['node']}, doc['graph']['edges'])
        self.assertIn({'from': 'input', 'to': token['node']}, doc['graph']['edges'])
        # The old initial-state Input is intentionally not deleted. It is now
        # an unused draft; the whole compiler still requires explicit cleanup.
        with self.assertRaisesRegex(GraphError, 'unused branches'):
            validate_model({**models, 'parent': doc}, 'parent')

    def test_shared_physical_control_source_has_one_wire_and_explicit_roles(self):
        models, token, options = editable();options['controls']['reset']['source'] = options['controls']['mask']['source']
        doc, _ = build_edit(models, token, options)
        self.assertEqual(len([e for e in doc['graph']['edges'] if e['to'] == token['node']]), 4)
        self.assertEqual(len(doc['graph']['nodes']), len(models['parent']['graph']['nodes']))

    def test_control_removal_keeps_existing_input_and_all_downstream_wires(self):
        models, token, options = editable();source = options['controls'].pop('mask')['source']
        doc, _ = build_edit(models, token, options)
        self.assertTrue(any(n['id'] == source for n in doc['graph']['nodes']))
        self.assertEqual([e for e in doc['graph']['edges'] if e['from'] == token['node']], [e for e in models['parent']['graph']['edges'] if e['from'] == token['node']])

    def test_connected_return_port_removal_and_rename_refuse_without_mutation(self):
        for field in ('remove', 'rename'):
            models, token, options = editable();before = copy.deepcopy(models)
            if field == 'remove':options['sequence_outputs'] = []
            else:options['sequence_outputs'][0]['port'] = 'different'
            with self.subTest(field=field), self.assertRaisesRegex(GraphError, 'connected return port'):build_edit(models, token, options)
            self.assertEqual(models, before)

    def test_unconnected_return_port_can_be_added_and_explicitly_removed(self):
        models, token, options = editable();options['sequence_outputs'].append({'output': 'next', 'port': 'extra'})
        doc, _ = build_edit(models, token, options);models['parent'] = doc
        info = prepare_edit(models, 'parent', token['node']);token = {k: info[k] for k in TOKEN}
        options['sequence_outputs'].pop();doc, _ = build_edit(models, token, options)
        self.assertNotIn('extra', validate_model({**models, 'parent': doc}, 'parent')['shapes'][token['node']]['ports'])

    def test_exact_same_shape_transition_replacement_preserves_old_root_ids(self):
        models, token, options = editable();replacement = copy.deepcopy(models['transition']);replacement.update(id='replacement', name='Other transition')
        models['replacement'] = replacement
        info = prepare_edit(models, 'parent', token['node'], 'replacement');token = {k: info[k] for k in TOKEN}
        doc, _ = build_edit(models, token, options)
        changed = next(n for n in doc['graph']['nodes'] if n['id'] == token['node'])
        self.assertEqual(changed['ref'], 'replacement');self.assertEqual(doc['graph']['edges'], models['parent']['graph']['edges'])
        self.assertEqual([n['id'] for n in doc['graph']['nodes']], [n['id'] for n in models['parent']['graph']['nodes']])

    def test_root_geometry_current_transition_and_candidate_closures_are_guarded(self):
        for where in ('geometry', 'current', 'candidate'):
            models, token, options = editable();models['replacement'] = {**copy.deepcopy(models['transition']), 'id': 'replacement'}
            info = prepare_edit(models, 'parent', token['node'], 'replacement');token = {k: info[k] for k in TOKEN}
            if where == 'geometry':models['parent']['graph']['nodes'][0]['x'] += 24
            else:models['transition' if where == 'current' else 'replacement']['name'] += ' changed'
            before = copy.deepcopy(models)
            with self.subTest(where=where), self.assertRaisesRegex(GraphError, 'changed'):build_edit(models, token, options)
            self.assertEqual(models, before)

    def test_wrong_node_missing_reference_and_reference_cycle_are_refused(self):
        models, token, options = editable()
        for nid, ref in [('missing', None), ('input', None), (token['node'], 'missing'), (token['node'], 'parent')]:
            with self.subTest(nid=nid, ref=ref), self.assertRaises(GraphError):prepare_edit(models, 'parent', nid, ref)

    def test_closed_options_tokens_and_exact_version(self):
        models, token, options = editable()
        for bad in ({**token, 'extra': 1}, {k: v for k, v in token.items() if k != 'root_model'}):
            with self.assertRaises(GraphError):build_edit(models, bad, options)
        for changes in ({'extra': 1}, {'version': True}, {'version': 2}, {'label': ''}, {'label': 'a'*61}, {'share_with': True}, {'share_with': ''}):
            with self.subTest(changes=changes), self.assertRaises(GraphError):build_edit(models, token, {**options, **changes})

    def test_explicit_invalid_role_source_adapter_shape_or_sharing_is_refused(self):
        for change in ('role', 'source', 'adapter', 'shape', 'owner', 'self-source', 'port-duplicate'):
            models, token, options = editable();before = copy.deepcopy(models)
            if change == 'role':options['sequence_inputs'][0]['input'] = 'missing'
            elif change == 'source':options['states'][0]['source'] = 'missing'
            elif change == 'adapter':options['controls']['mask']['adapter'] = 'cast'
            elif change == 'shape':options['sequence_inputs'][0]['source'] = 'input'
            elif change == 'owner':options['share_with'] = 'linear1'
            elif change == 'self-source':options['states'][0]['source'] = token['node']+'@state/h'
            else:options['states'][0]['port'] = 'sequence/next'
            with self.subTest(change=change), self.assertRaises(GraphError):build_edit(models, token, options)
            self.assertEqual(models, before)

    def test_cnn_interface_edit_keeps_t_axes_and_transition_weights(self):
        models, token, options = editable(cnn=True);options['label'] = 'CNN scan'
        doc, _ = build_edit(models, token, options)
        c = validate_model({**models, 'parent': doc}, 'parent')
        self.assertEqual(c['parameters'], 17)
        self.assertEqual(c['shapes'][token['node']]['ports'], {'sequence/next': [4, 2], 'state/h': [2]})

    def test_invalid_unrelated_draft_is_retained_while_valid_paths_are_edited(self):
        models, token, options = editable();models['parent']['graph']['edges'].pop(0)
        info = prepare_edit(models, 'parent', token['node']);token = {k: info[k] for k in TOKEN}
        options['label'] = 'Change';before = copy.deepcopy(models)
        doc, _ = build_edit(models, token, options)
        self.assertEqual(doc['graph']['edges'], before['parent']['graph']['edges'])
        self.assertEqual(doc['graph']['nodes'][:3], before['parent']['graph']['nodes'][:3])
        with self.assertRaises(GraphError):validate_model({**models, 'parent': doc}, 'parent')
        self.assertEqual(models, before)

    def test_disk_cli_prepare_apply_replay_and_failure_batch_are_atomic(self):
        models, token, options = editable()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'edit.vtw';w = Workspace()
            try:
                for doc in models.values():w.put(doc)
                w.save(path)
            finally:w.db.close()
            before = path.read_bytes();stream = io.StringIO()
            with redirect_stdout(stream):self.assertEqual(vpw_cli.main([str(path), 'prepare-recurrent-edit', 'parent', token['node']]), 0)
            info = json.loads(stream.getvalue())['value'];self.assertEqual(path.read_bytes(), before)
            token = {k: info[k] for k in TOKEN};options = info['options'];options['label'] = 'CLI changed'
            edit = {'op': 'edit-recurrent', 'token': token, 'options': options}
            for edits, expected in [([edit, {'op': 'unsupported', 'id': 'parent'}], 1), ([edit], 0)]:
                with patch('sys.stdin', io.StringIO(json.dumps(edits))), redirect_stdout(io.StringIO()):self.assertEqual(vpw_cli.main([str(path), 'apply', '-']), expected)
                if expected:self.assertEqual(path.read_bytes(), before)
            after = path.read_bytes();self.assertNotEqual(after, before)
            with patch('sys.stdin', io.StringIO(json.dumps([edit]))), redirect_stdout(io.StringIO()):self.assertEqual(vpw_cli.main([str(path), 'apply', '-']), 1)
            self.assertEqual(path.read_bytes(), after)
            loaded = Workspace.load(path)
            try:
                self.assertEqual(validate_model(loaded.models(), 'parent')['parameters'], 13)
                self.assertEqual(identity(loaded.models(), 'parent')['architecture'], identity(models, 'parent')['architecture'])
            finally:loaded.db.close()
