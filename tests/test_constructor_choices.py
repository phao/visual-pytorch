"""Pure closed scalar grammar, version gates, retained provenance and UI sessions."""
import copy
import json
import random
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from app.core import GraphError, Workspace, VERSION, validate_model, generate_code
from app.families import preview, specialize, validate_family
from app.family_axes import at, count, selected, validate_choice
from app.family_records import retain
from app.experiment_objects import object_reference, preview_experiment, selection_at
from app.authoring_sessions import Sessions
from app.revisions import validate_all
from test_family_axes import family_experiment

ROOT = Path(__file__).resolve().parents[1]


def choices():
    return json.loads((ROOT / 'examples/families/activation-bias.json').read_text())


def choice_experiment():
    w, e = family_experiment(); source = choices(); ref = w.put_spec('family', source)
    e.pop('revision'); e.update(version=3, id='constructor-experiment', models=[{
        'id': 'constructor', 'family': ref, 'bindings': {}, 'axes': [
            {'symbol': 'activation-kind', 'values': ['Tanh', 'ReLU', 'Softmax']},
            {'symbol': 'learn-bias', 'values': [False, True]}]}])
    ref = w.put_spec('experiment', e)
    return w, w.get_spec('experiment', ref['id'], ref['revision'])


class ConstructorChoiceTests(unittest.TestCase):
    def workspace(self):
        w, e = choice_experiment(); self.addCleanup(w.db.close)
        self.addCleanup(lambda: w.history.directory.cleanup() if w.history.directory else None)
        return w, e

    def test_each_activation_and_bias_is_explicit_valid_concrete_code(self):
        source = choices(); original = copy.deepcopy(source); ids = set()
        for activation in ('GELU', 'ReLU', 'Sigmoid', 'Softmax', 'Tanh'):
            for bias in (False, True):
                resolved = specialize(source, {'activation-kind': activation, 'learn-bias': bias})
                nodes = resolved['models'][resolved['root']]['graph']['nodes']
                self.assertEqual(nodes[2]['activation'], activation)
                self.assertIs(nodes[1]['bias'], bias); self.assertIs(nodes[3]['bias'], bias)
                self.assertEqual(resolved['contract']['parameters'], 10 if bias else 6)
                self.assertIn('nn.' + activation + '(', generate_code(resolved['models'], resolved['root']))
                self.assertIn('bias=' + str(bias), generate_code(resolved['models'], resolved['root']))
                ids.add(resolved['specialization'])
        self.assertEqual(len(ids), 10); self.assertEqual(source, original)

    def test_enum_domain_bounds_types_and_unknown_constructor_are_not_coerced(self):
        for value in (0, True, 1., None, [], {}, '', ' Tanh'):
            with self.subTest(value=value), self.assertRaises(GraphError): preview(choices(), {'activation-kind': value})
        for value in (0, 1, 'false', None, []):
            with self.subTest(value=value), self.assertRaises(GraphError): specialize(choices(), {'learn-bias': value})
        self.assertEqual(preview(choices(), {'activation-kind': 'Unknown'})['status'], 'incompatible')
        source = choices(); source['symbols'][0]['values'].append('arbitrary-call()')
        result = preview(source, {'activation-kind': 'arbitrary-call()'})
        self.assertEqual(result['status'], 'incompatible'); self.assertIn('activation', result['reason'])

    def test_closed_symbol_grammar_defaults_domains_and_old_versions(self):
        for mutation in ('duplicate', 'empty', 'nonstring', 'too-many', 'default', 'bounds', 'boolean-domain', 'type', 'unknown-key'):
            source = choices(); symbol = source['symbols'][0]
            if mutation == 'duplicate': symbol['values'] += ['Tanh']
            if mutation == 'empty': symbol['values'] = []
            if mutation == 'nonstring': symbol['values'] = [False, 'Tanh']
            if mutation == 'too-many': symbol['values'] = ['v'+str(i) for i in range(65)]
            if mutation == 'default': symbol['default'] = 'Other'
            if mutation == 'bounds': symbol['minimum'] = 1
            if mutation == 'boolean-domain': source['symbols'][1]['values'] = [False, True]
            if mutation == 'type': symbol['type'] = ['enum']
            if mutation == 'unknown-key': symbol['python'] = 'eval'
            with self.subTest(mutation=mutation), self.assertRaises(GraphError): validate_family(source)
        for version in (1, 2, True, 5, 3.):
            source = choices(); source['version'] = version
            with self.subTest(version=version), self.assertRaises(GraphError): validate_family(source)

    def test_noninteger_symbols_cannot_enter_arithmetic_or_repeats(self):
        for value in ({'symbol': 'learn-bias'}, {'symbol': 'activation-kind'},
                      {'op': 'add', 'args': [1, {'symbol': 'learn-bias'}]}):
            source = choices(); source['assignments'][0].update(node='hidden', field='features', value=value)
            with self.subTest(value=value), self.assertRaisesRegex(GraphError, 'integer'): validate_family(source)
        source = choices(); source['repeats'] = [{'id': 'repeat', 'model': 'mlp', 'nodes': ['activation'],
            'count': {'symbol': 'learn-bias'}, 'mode': 'independent'}]
        with self.assertRaisesRegex(GraphError, 'integer'): validate_family(source)

    def test_assignment_type_scope_target_and_no_silent_missing_field_addition(self):
        for mutation in ('wrong-type', 'arithmetic', 'unknown', 'scope', 'missing-field', 'duplicate-target'):
            source = choices()
            if mutation == 'wrong-type': source['assignments'][0]['value'] = {'symbol': 'learn-bias'}
            if mutation == 'arithmetic': source['assignments'][0]['value'] = {'op': 'add', 'args': [1, 1]}
            if mutation == 'unknown': source['assignments'][0]['value'] = {'symbol': 'absent'}
            if mutation == 'scope': source['symbols'][0]['scope'] = 'absent'
            if mutation == 'missing-field': del source['models']['mlp']['graph']['nodes'][1]['bias']
            if mutation == 'duplicate-target': source['assignments'].append(copy.deepcopy(source['assignments'][0]))
            with self.subTest(mutation=mutation), self.assertRaises(GraphError): validate_family(source)

    def test_literal_choices_do_not_implicitly_bind_unused_symbols(self):
        source = choices(); source['symbols'] = []
        source['assignments'][0]['value'] = 'GELU'
        for assignment in source['assignments'][1:]: assignment['value'] = False
        resolved = specialize(source); self.assertEqual(resolved['bindings'], {})
        self.assertEqual(resolved['contract']['parameters'], 6)
        self.assertEqual(preview(source)['status'], 'verified')

    def test_conditional_default_and_visible_incompatibility_are_readonly(self):
        source = choices(); del source['symbols'][0]['default']; original = copy.deepcopy(source)
        self.assertEqual(preview(source)['status'], 'conditional')
        self.assertEqual(preview(source, {'activation-kind': 'Other'})['status'], 'incompatible')
        self.assertEqual(preview(source, {'activation-kind': 'ReLU'})['status'], 'verified')
        self.assertEqual(source, original)

    def test_tied_version_three_preserves_actual_shared_owner(self):
        source = choices(); source['repeats'] = [{'id': 'activations', 'model': 'mlp', 'nodes': ['activation'],
            'count': 3, 'mode': 'tied'}]
        resolved = specialize(source); self.assertEqual(resolved['contract']['parameters'], 10)
        self.assertEqual(len(resolved['occurrences']), 3)
        self.assertEqual({x['mode'] for x in resolved['occurrences']}, {'tied'})

    def test_retained_typed_bindings_canonical_model_and_forgery_rejection(self):
        w, e = self.workspace(); source = choices()
        result = retain(w, family=source, bindings={'activation-kind': 'ReLU', 'learn-bias': False}, materialize=False)
        record = w.get_spec('specialization', result['specialization']['id'], result['specialization']['revision'])
        self.assertEqual(record['version'], 2); self.assertEqual(record['bindings'], {'activation-kind': 'ReLU', 'learn-bias': False})
        self.assertEqual(w.get_spec('model', result['model']['id'])['version'], 1)
        for mutation in ('binding', 'version', 'omit', 'int-bool'):
            forged = copy.deepcopy(record); forged.pop('revision'); forged['id'] += '-forged'
            if mutation == 'binding': forged['bindings']['activation-kind'] = 'Tanh'
            if mutation == 'version': forged['version'] = 1
            if mutation == 'omit': forged['bindings'].pop('learn-bias')
            if mutation == 'int-bool': forged['bindings']['learn-bias'] = 0
            before = w.db.serialize()
            with self.subTest(mutation=mutation), self.assertRaises(GraphError): w.put_spec('specialization', forged)
            self.assertEqual(w.db.serialize(), before)
        validate_all(w.db)

    def test_axis_order_exact_scalar_membership_bool_is_not_integer_zero(self):
        w, e = self.workspace(); before = w.db.serialize(); rng = random.getstate()
        rows = preview_experiment(e, w.get_spec)['candidates']
        self.assertEqual(len(rows), 24); self.assertEqual({r['status'] for r in rows}, {'compatible'})
        self.assertEqual([rows[i]['coordinates']['model']['bindings'] for i in (0, 4, 8, 12)], [
            {'activation-kind': 'Tanh', 'learn-bias': False}, {'activation-kind': 'Tanh', 'learn-bias': True},
            {'activation-kind': 'ReLU', 'learn-bias': False}, {'activation-kind': 'ReLU', 'learn-bias': True}])
        self.assertEqual([rows[i]['parameters_exact'] for i in (0, 4)], ['6', '10'])
        bad = selection_at(e, 0)['model']; bad['bindings']['learn-bias'] = 0
        with self.assertRaisesRegex(GraphError, 'outside'): selected(e['models'], bad, version=3)
        choice = copy.deepcopy(e['models'][0]); choice['axes'][1]['values'] = [False, 0]
        validate_choice(choice, version=3); self.assertEqual(count(choice), 6)
        self.assertIs(at([choice], 0)['bindings']['learn-bias'], False)
        self.assertIs(type(at([choice], 1)['bindings']['learn-bias']), int)
        self.assertEqual(w.db.serialize(), before); self.assertEqual(random.getstate(), rng)

    def test_legacy_experiment_cannot_claim_typed_bindings_or_new_family(self):
        w, e = self.workspace()
        for version in (1, 2):
            doc = copy.deepcopy(e); doc.pop('revision'); doc['version'] = version
            with self.subTest(version=version), self.assertRaises(GraphError): w.put_spec('experiment', doc)
        e.pop('revision'); e['version'] = 2; e['models'][0].update(axes=[], bindings={})
        with self.assertRaisesRegex(GraphError, 'experiment version 3'): w.put_spec('experiment', e)

    def test_axis_bad_type_and_outside_enum_stay_visible_not_filtered(self):
        w, e = self.workspace(); e.pop('revision'); e['models'][0]['axes'][0]['values'] = ['ReLU', 'Unknown', 0]
        ref = w.put_spec('experiment', e); doc = w.get_spec('experiment', ref['id'], ref['revision'])
        rows = preview_experiment(doc, w.get_spec)['candidates']
        self.assertEqual([rows[i]['status'] for i in (0, 8, 16)], ['compatible', 'incompatible', 'incompatible'])
        self.assertIn('choices', rows[8]['reason']); self.assertIn('enum', rows[16]['reason'])

    def test_owned_atomic_version_upgrade_domains_defaults_and_no_downgrade(self):
        w = Workspace(); self.addCleanup(w.db.close); s = Sessions(); owner = object(); self.addCleanup(s.close_owner, owner)
        source = choices(); source['version'] = 1; source['symbols'] = []; source['assignments'] = []
        ref = w.put_spec('family', source); h = s.open(w, owner, 'family', base=ref); before = w.db.serialize()
        operations = [{'op': 'property', 'target': {'kind': 'document'}, 'key': 'version', 'text': '3'}]
        for index, symbol in enumerate(choices()['symbols']): operations.append({'op': 'insert', 'collection': 'symbols', 'index': index, 'value': symbol})
        for index, assignment in enumerate(choices()['assignments']): operations.append({'op': 'insert', 'collection': 'assignments', 'index': index, 'value': assignment})
        h = s.family_edit(w, h['session'], owner, 0, operations)
        self.assertEqual(s.family_preview(w, h['session'], owner, 1)['parameters'], 10)
        self.assertEqual(w.db.serialize(), before); s.check(w, h['session'], owner, 1)
        with self.assertRaises(GraphError): s.family_edit(w, h['session'], owner, 1, [operations[0] | {'text': '2'}])
        saved = s.commit(w, h['session'], owner, 1); self.assertEqual(w.get_spec('family', saved['id'])['version'], 3)

    def test_vtw_ten_roundtrip_and_old_header_reject_without_source_write(self):
        w, e = self.workspace()
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'choices.vtw'; w.save(path); original = path.read_bytes()
            reopened = Workspace.load(path)
            try: self.assertEqual(preview_experiment(e, reopened.get_spec), preview_experiment(e, w.get_spec))
            finally: reopened.db.close()
            self.assertEqual(path.read_bytes(), original)
            with closing(sqlite3.connect(path)) as db: db.execute('PRAGMA user_version=9'); db.commit()
            downgraded = path.read_bytes()
            with self.assertRaisesRegex(GraphError, 'version-10'): Workspace.load(path)
            self.assertEqual(path.read_bytes(), downgraded)
        independent = Workspace(); self.addCleanup(independent.db.close)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'older.vtw'; independent.save(path)
            with closing(sqlite3.connect(path)) as db: db.execute('PRAGMA user_version=9'); db.commit()
            original = path.read_bytes(); reopened = Workspace.load(path)
            try: self.assertEqual(reopened.db.execute('PRAGMA user_version').fetchone()[0], VERSION)
            finally: reopened.db.close()
            self.assertEqual(path.read_bytes(), original)

    def test_actual_agent_apply_specializes_and_retains_complete_typed_values(self):
        import vpw_cli
        w = Workspace(); self.addCleanup(w.db.close)
        result = vpw_cli.apply_edits(w, [{'op': 'specialize-family', 'family': choices(),
            'bindings': {'activation-kind': 'GELU', 'learn-bias': False}}])[0]
        self.assertEqual(result['resolved']['contract']['parameters'], 6)
        self.assertEqual(w.get_spec('specialization', result['specialization']['id'])['bindings']['activation-kind'], 'GELU')
        validate_all(w.db)

    def test_fresh_specialization_and_source_generation_import_no_tensor_runtime(self):
        code = 'import json,sys;from app.families import specialize;from app.core import generate_code;f=json.load(open("examples/families/activation-bias.json"));r=specialize(f);generate_code(r["models"],r["root"]);assert not any(m in sys.modules for m in ("torch","numpy","PyQt6"))'
        ran = subprocess.run([sys.executable, '-c', code], cwd=ROOT, text=True, capture_output=True, timeout=10)
        self.assertEqual(ran.returncode, 0, ran.stderr)


if __name__ == '__main__': unittest.main()
