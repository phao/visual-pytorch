"""Closed Add/Scale graph, version/storage/CLI and family arithmetic contracts."""
import copy
import io
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from contextlib import redirect_stdout

from app.core import Workspace, VERSION, SUPPORTED_VERSIONS, GraphError, check_document, validate_model
from app.operators import normalized
from app.revisions import model_document, frozen_document
from app.presentation import block_parameter_counts
from app.families import validate_family, specialize
from app.family_records import retain
import vpw_cli


def model(shape=(2,), coefficient=2., learned=False, axes=None):
    inputs = [{'id': key, 'type': 'Input', 'shape': list(shape)} for key in ('x', 'other')]
    if axes is not None:
        for item in inputs: item['axes'] = list(axes)
    nodes = inputs + [{'id': 'scale', 'type': 'Scale', 'coefficient': coefficient, 'op_version': 1},
        {'id': 'sum', 'type': 'Add', 'sources': ['base' if learned else 'x', 'scale'], 'op_version': 1},
        {'id': 'y', 'type': 'Output'}]
    if learned: nodes.append({'id': 'base', 'type': 'Linear', 'features': shape[-1], 'bias': True})
    edges = [{'from': 'other', 'to': 'scale'}, {'from': 'scale', 'to': 'sum'},
             {'from': 'base' if learned else 'x', 'to': 'sum'}, {'from': 'sum', 'to': 'y'}]
    if learned: edges.append({'from': 'x', 'to': 'base'})
    return {'id': 'arithmetic', 'name': 'Fixed arithmetic',
            'graph': {'nodes': nodes, 'edges': edges, 'input_order': ['x', 'other']}}


def family(learned=True):
    m = model(learned=learned)
    return {'version': 9, 'id': 'arithmetic-family', 'name': 'Arithmetic family', 'root': m['id'],
        'models': {m['id']: m}, 'symbols': [], 'assignments': [], 'repeats': [],
        'constructors': [], 'runtime_policy': None}


class ArithmeticContractTests(unittest.TestCase):
    def test_valid_scalar_and_multidimensional_shapes_preserve_axes_and_zero_counts(self):
        for shape, axes in (([2], None), ([3, 2], ['T', 'F']), ([1, 4, 5], ['C', 'H', 'W'])):
            m = model(shape, axes=axes); check_document(m); c = validate_model({m['id']: m}, m['id'])
            self.assertEqual(c['parameters'], 0); self.assertTrue(c['affine']); self.assertFalse(c['matrix_affine'])
            self.assertEqual(c['output_shape'], shape); self.assertEqual(c['output_axes'], axes)
            self.assertTrue(all(v == '0' for v in block_parameter_counts(m, c, {}).values()))
            self.assertEqual(next(i for i in c['plan'] if i['type'] == 'Scale')['coefficient'], 2.)

    def test_sum_requires_exact_shapes_axes_and_at_least_two_named_sources(self):
        for change in ('shape', 'axes', 'missing', 'duplicate', 'unknown'):
            m = model([3, 2], axes=['T', 'F']); add = next(n for n in m['graph']['nodes'] if n['type'] == 'Add')
            if change == 'shape': m['graph']['nodes'][1]['shape'] = [1, 2]  # PyTorch would broadcast; we refuse.
            if change == 'axes': m['graph']['nodes'][1]['axes'] = ['F', 'T']
            if change == 'missing': add['sources'] = ['x']
            if change == 'duplicate': add['sources'] = ['x', 'x']
            if change == 'unknown': add['sources'] = ['x', 'missing']
            with self.subTest(change=change), self.assertRaises(GraphError):
                check_document(m); validate_model({m['id']: m}, m['id'])

    def test_arithmetic_versions_and_fields_are_closed_not_ignored_metadata(self):
        for kind, n in (('Add', {'sources': ['a', 'b']}), ('Scale', {'coefficient': 2})):
            source = {'id': 'op', 'type': kind, **n}
            self.assertEqual(normalized(source)['op_version'], 1)
            for patch in ({'op_version': True}, {'op_version': 2}, {'bias': False}, {'share_with': 'a'}, {'trainable': False}):
                with self.subTest(kind=kind, patch=patch), self.assertRaises(GraphError): normalized({**source, **patch})
        for coefficient in (True, None, '2', [], float('nan'), float('inf'), float('-inf'), 1e100, 2**53, float(2**53), -float(2**53), 1e20, 10**400, -(10**400)):
            with self.subTest(coefficient=coefficient), self.assertRaises(GraphError):
                normalized({'id': 'op', 'type': 'Scale', 'coefficient': coefficient})
        for coefficient in (0, -2, 1.25, 2**53-1, -(2**53-1)):
            self.assertEqual(normalized({'id': 'op', 'type': 'Scale', 'coefficient': coefficient})['coefficient'], coefficient)

    def test_large_integer_coefficient_is_controlled_graph_refusal(self):
        for coefficient in (10**400,-(10**400)):
            with self.subTest(coefficient_sign=coefficient>0),self.assertRaises(GraphError):check_document(model(coefficient=coefficient))

    def test_owned_arithmetic_drawing_seed_upgrade_and_family9_rename(self):
        from app import authoring
        from app.authoring_sessions import Sessions
        w=Workspace();sessions=Sessions();owner=object()
        try:
            w.put(model(learned=True));draft=authoring.begin(w,'family',model='arithmetic',sid='seed')
            self.assertEqual(draft['document']['version'],9);self.assertIsNone(draft['document']['runtime_policy'])
            ref=authoring.save(w,draft);info=sessions.open(w,owner,'family',base=ref)
            changed=sessions.family_edit(w,info['session'],owner,info['sequence'],[{'op':'property','target':{'kind':'document'},'key':'name','text':'"Renamed Family9"'}])
            self.assertEqual(changed['sequence'],1);saved=sessions.commit(w,info['session'],owner,changed['sequence'])
            self.assertEqual(w.get_spec('family',saved['id'],saved['revision'])['name'],'Renamed Family9')
            info=sessions.open(w,owner,'family',sid='upgrade')
            info=sessions.patch(w,info['session'],owner,0,[{'op':'put','path':[],'key':'version','value':8}])
            changed=sessions.patch(w,info['session'],owner,info['sequence'],[{'op':'put','path':[],'key':'version','value':9}])
            self.assertEqual(changed['sequence'],2)
            with self.assertRaises(GraphError):sessions.patch(w,info['session'],owner,2,[{'op':'put','path':[],'key':'version','value':8}])
        finally:w.db.close()

    def test_structural_draft_can_wait_for_second_wire_but_cannot_run(self):
        m = model(); add = next(n for n in m['graph']['nodes'] if n['type'] == 'Add')
        add['sources'] = ['x']; m['graph']['edges'] = [e for e in m['graph']['edges'] if e != {'from': 'scale', 'to': 'sum'}]
        check_document(m)
        with self.assertRaises(GraphError): validate_model({m['id']: m}, m['id'])

    def test_model5_family9_specialization8_are_explicit_not_silent_old_contracts(self):
        m = model(learned=True); doc = model_document({m['id']: m}, m['id']); self.assertEqual(doc['version'], 5)
        for version in (1, 2, 3, 4):
            with self.subTest(version=version), self.assertRaises(GraphError): frozen_document('model', {**doc, 'version': version})
        f = family(); validate_family(f); self.assertGreater(specialize(f)['contract']['parameters'], 0)
        old = copy.deepcopy(f); old['version'] = 8
        with self.assertRaisesRegex(GraphError, 'Family9'): validate_family(old)
        w = Workspace()
        try:
            saved = retain(w, family=f, bindings={})
            self.assertEqual(w.get_spec('specialization', saved['specialization']['id'], saved['specialization']['revision'])['version'], 8)
            with tempfile.TemporaryDirectory() as td:
                path = Path(td) / 'family.vtw'; w.save(path); reopened = Workspace.load(path)
                try: self.assertEqual(reopened.get_spec('family', f['id'])['version'], 9)
                finally: reopened.db.close()
        finally: w.db.close()

    def test_vtw23_refuses_old_arithmetic_headers_and_reserved_E_without_rewriting(self):
        self.assertEqual(VERSION, 23); self.assertNotIn(21, SUPPORTED_VERSIONS); self.assertNotIn(22, SUPPORTED_VERSIONS)
        with tempfile.TemporaryDirectory() as td:
            for version in (20, 21, 22, 24):
                path = Path(td) / (str(version) + '.vtw'); w = Workspace(); w.put(model()); w.save(path); w.db.close()
                with sqlite3.connect(path) as db: db.execute('PRAGMA user_version=' + str(version))
                before = path.read_bytes()
                with self.assertRaises(GraphError): Workspace.load(path)
                self.assertEqual(path.read_bytes(), before)

    def test_vtw20_without_arithmetic_upgrades_only_working_copy_until_save(self):
        from app.core import default_model
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'old.vtw'; w = Workspace(); w.put(default_model()); w.save(path); w.db.close()
            with sqlite3.connect(path) as db: db.execute('PRAGMA user_version=20')
            before = path.read_bytes(); loaded = Workspace.load(path)
            try:
                self.assertEqual(loaded.db.execute('PRAGMA user_version').fetchone()[0], 23)
                self.assertEqual(path.read_bytes(), before); self.assertFalse(loaded.dirty)
            finally: loaded.db.close()

    def test_cli_apply_scale_change_is_atomic_and_roundtrips_new_container(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'cli.vtw'; w = Workspace(); w.put(model(learned=True)); w.save(path); w.db.close()
            patch_path = Path(td) / 'patch.json'
            for coefficient, expected in ((-1.5, 0), (True, 1)):
                edits = [{'op': 'set-node', 'id': 'arithmetic', 'node': 'scale', 'values': {'coefficient': coefficient}}]
                with patch_path.open('w') as stream: json.dump(edits, stream)
                before = path.read_bytes(); output = io.StringIO()
                with redirect_stdout(output): result = vpw_cli.main([str(path), 'apply', str(patch_path)])
                self.assertEqual(result, expected, output.getvalue())
                if expected: self.assertEqual(path.read_bytes(), before)
            reopened = Workspace.load(path)
            try: self.assertEqual(next(n for n in reopened.models()['arithmetic']['graph']['nodes'] if n['type'] == 'Scale')['coefficient'], -1.5)
            finally: reopened.db.close()


if __name__ == '__main__': unittest.main()
