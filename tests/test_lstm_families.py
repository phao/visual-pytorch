"""Pure family7: typed constructor shapes, historical provenance and .vtw18.

These checks never construct a tensor. Counts are derived independently from
gate/projection dimensions, not from the application's LSTM count helper.
"""
import copy
import itertools
import json
import random
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from app import authoring
from app.core import GraphError, VERSION, Workspace, default_model
from app.families import evaluate, expression, preview, specialize, validate_family
from app.family_export import family_source
from app.family_records import definition as record_definition, retain
from app.revisions import validate_all
from examples.lstm_family import definition

ROOT = Path(__file__).resolve().parents[1]


def parameter_count(width, hidden, projection, layers, directions, bias):
    recurrent = projection or hidden
    total = 0
    for layer in range(layers):
        incoming = width if layer == 0 else directions * recurrent
        total += directions * (4 * hidden * incoming + 4 * hidden * recurrent
                               + (8 * hidden if bias else 0)
                               + (hidden * projection if projection else 0))
    return total


class LSTMFamilyTests(unittest.TestCase):
    def setUp(self):
        self.w = Workspace()
        self.addCleanup(self.cleanup)

    def cleanup(self):
        self.w.db.close()
        if self.w.history.directory is not None:
            self.w.history.directory.cleanup()

    def record(self, result):
        ref = result['specialization']
        return self.w.get_spec(ref['kind'], ref['id'], ref['revision'])

    def test_constructor_cross_product_shapes_and_independent_counts(self):
        doc = definition(); original = copy.deepcopy(doc); rng = random.getstate()
        for layers, bidirectional, projection, bias in itertools.product((1, 2), (False, True), (0, 2), (False, True)):
            bindings = {'time': 4, 'width': 2, 'cell-width': 3, 'layers': layers,
                        'projection': projection, 'directions': bidirectional, 'bias': bias}
            with self.subTest(bindings=bindings):
                resolved = specialize(doc, bindings)
                directions = 2 if bidirectional else 1
                q = projection or 3
                self.assertEqual(resolved['version'], 5)
                self.assertIsNone(resolved['runtime']); self.assertIsNone(resolved['runtime_specification'])
                self.assertEqual(resolved['contract']['parameters'], parameter_count(2, 3, projection, layers, directions, bias))
                self.assertEqual(resolved['contract']['shapes']['lstm']['ports'],
                                 {'sequence': [4, directions * q], 'hidden': [layers * directions, q],
                                  'cell': [layers * directions, 3]})
                self.assertEqual(resolved['bindings'], bindings)
        self.assertEqual(doc, original); self.assertEqual(random.getstate(), rng)

    def test_one_hundred_bounded_constructor_points(self):
        rng = random.Random(804); doc = definition()
        for index in range(100):
            h = rng.randint(2, 12); p = rng.randrange(h); d = rng.choice((1, 2))
            t, width, layers = rng.randint(1, 12), rng.randint(1, 9), rng.randint(1, 4)
            bias = rng.choice((False, True))
            bindings = {'time': t, 'width': width, 'cell-width': h, 'layers': layers,
                        'projection': p, 'directions': d == 2, 'bias': bias}
            with self.subTest(index=index, bindings=bindings):
                result = specialize(doc, bindings)
                self.assertEqual(result['contract']['parameters'], parameter_count(width, h, p, layers, d, bias))
                self.assertEqual(result['contract']['shapes']['lstm']['ports']['hidden'], [layers * d, p or h])

    def test_old_family_versions_refuse_lstm_instead_of_reinterpreting_them(self):
        for version in range(1, 7):
            doc = definition(); doc['version'] = version
            if version < 6: doc.pop('runtime_policy')
            if version < 4: doc.pop('constructors')
            with self.subTest(version=version), self.assertRaisesRegex(GraphError, 'do not support LSTM'):
                validate_family(doc)

    def test_unsupported_runtime_policy_refused_at_retention_before_any_sql_changes(self):
        doc = definition()
        doc['runtime_policy'] = {'version': 1, 'kind': 'runtime-temporal-v1', 'inputs': {'sequence': [0]}}
        before = self.w.db.serialize()
        with self.assertRaisesRegex(GraphError, 'null fixed-T policy'):
            self.w.put_spec('family', doc)
        self.assertEqual(self.w.db.serialize(), before)
        self.assertEqual(self.w.list_specs(all_revisions=True), [])

    def test_required_fixed_policy_and_unknown_grammar_refuse(self):
        for mutation in ('missing', 'unknown', 'boolean', 'spatial'):
            doc = definition()
            if mutation == 'missing': doc.pop('runtime_policy')
            elif mutation == 'unknown': doc['version'] = 10
            elif mutation == 'boolean': doc['version'] = True
            else: doc['runtime_inputs'] = {'sequence': [1]}
            with self.subTest(mutation=mutation), self.assertRaises(GraphError): validate_family(doc)

    def test_projection_and_scalar_type_errors_are_explicit_incompatible_points(self):
        for bindings in ({'projection': 7}, {'projection': 9}, {'layers': 0}, {'time': 0}):
            with self.subTest(bindings=bindings):
                result = preview(definition(), bindings)
                self.assertEqual(result['status'], 'incompatible')
                with self.assertRaises(GraphError): specialize(definition(), bindings)
        # Malformed binding types refuse the preview request itself; validly
        # typed but inadmissible constructor points return an incompatibility.
        for bindings in ({'cell-width': True}, {'directions': 1}, {'bias': 'false'}, {'projection': 2.0}):
            with self.subTest(bindings=bindings), self.assertRaises(GraphError): preview(definition(), bindings)

    def test_conditional_binding_is_not_inferred_even_from_an_inactive_branch(self):
        doc = definition(); next(s for s in doc['symbols'] if s['id'] == 'cell-width').pop('default')
        result = preview(doc, {'projection': 2})
        self.assertEqual(result['status'], 'conditional')
        self.assertIn('cell-width', [s['id'] for s in result['missing']])
        with self.assertRaisesRegex(GraphError, 'Bind required architecture symbols'): specialize(doc, {'projection': 2})

    def test_repeat_lstm_is_not_a_unary_depth_macro(self):
        doc = definition()
        doc['repeats'] = [{'id': 'depth', 'model': doc['root'], 'nodes': ['lstm'], 'count': 2, 'mode': 'tied'}]
        with self.assertRaisesRegex(GraphError, 'unary operation'): validate_family(doc)

    def test_upstream_repeats_rewrite_all_explicit_input_roles_and_lengths(self):
        for role in ('sequence', 'hidden', 'cell', 'lengths'):
            doc = definition(); graph = doc['models'][doc['root']]['graph']
            lstm = next(n for n in graph['nodes'] if n['id'] == 'lstm')
            graph['nodes'].append({'id': 'activate', 'type': 'Activation', 'activation': 'Tanh'})
            next(e for e in graph['edges'] if e['from'] == role and e['to'] == 'lstm')['to'] = 'activate'
            graph['edges'].append({'from': 'activate', 'to': 'lstm'})
            if role == 'lengths': lstm['lengths']['source'] = 'activate'
            else: lstm['inputs'][role] = 'activate'
            doc['repeats'] = [{'id': 'repeat', 'model': doc['root'], 'nodes': ['activate'], 'count': 3, 'mode': 'independent'}]
            with self.subTest(role=role):
                result = specialize(doc)
                actual = next(n for n in result['models'][result['root']]['graph']['nodes'] if n['type'] == 'LSTM')
                self.assertEqual(actual['lengths']['source'] if role == 'lengths' else actual['inputs'][role], 'repeat/2/activate')

    def test_shared_owner_count_and_constructor_mismatch(self):
        doc = definition(); graph = doc['models'][doc['root']]['graph']
        alias = copy.deepcopy(next(n for n in graph['nodes'] if n['id'] == 'lstm'))
        alias.update(id='alias', share_with='lstm')
        graph['nodes'].extend([alias, {'id': 'alias-output', 'type': 'Output'}])
        graph['edges'].extend(dict(e, to='alias') for e in list(graph['edges']) if e['to'] == 'lstm')
        graph['edges'].append({'from': 'alias', 'from_port': 'sequence', 'to': 'alias-output'})
        self.assertEqual(specialize(doc)['contract']['parameters'], 1400)
        self.assertEqual(preview(doc, {'bias': False})['status'], 'incompatible')

    def test_retention_records_exact_versions_and_historical_defaults(self):
        doc = definition(); first = retain(self.w, family=doc, materialize=False)
        record = self.record(first)
        self.assertEqual(record['version'], 6); self.assertIsNone(record['runtime'])
        self.assertEqual(self.w.get_spec('model', record['model']['id'])['version'], 4)
        doc['name'] = 'Changed browsing head'
        next(s for s in doc['symbols'] if s['id'] == 'time')['default'] = 9
        self.w.put_spec('family', doc)
        self.assertEqual(self.record(first), record)
        validate_all(self.w.db)

    def test_wrong_record_version_binding_or_runtime_cannot_forge_same_model(self):
        result = retain(self.w, family=definition(), materialize=False)
        original = record_definition(self.record(result)); before = self.w.db.serialize()
        for mutation in ('version', 'bindings', 'runtime'):
            bad = copy.deepcopy(original)
            if mutation == 'version': bad['version'] = 5
            elif mutation == 'bindings': bad['bindings']['directions'] = False
            else: bad['runtime'] = {'version': 2, 'kind': 'runtime-temporal-v1',
                                    'inputs': {'sequence': [0]}, 'specification_sha256': '0' * 64}
            with self.subTest(mutation=mutation), self.assertRaises(GraphError): self.w.put_spec('specialization', bad)
            self.assertEqual(self.w.db.serialize(), before)

    def test_late_materialization_failure_restores_all_rows_and_dirty_flag(self):
        before = self.w.db.serialize(); dirty = self.w.dirty; real = self.w.put
        def fail(model):
            real(model)
            raise RuntimeError('late drawing insertion')
        with patch.object(self.w, 'put', side_effect=fail), self.assertRaisesRegex(RuntimeError, 'late drawing'):
            retain(self.w, family=definition(), materialize=True)
        self.assertEqual(self.w.db.serialize(), before); self.assertEqual(self.w.dirty, dirty)

    def test_vtw18_fence_applies_to_unmaterialized_family_and_old17_read_is_nonmutating(self):
        self.w.put_spec('family', definition())
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'family.vtw'; self.w.save(path)
            with closing(sqlite3.connect(path)) as db: db.execute('PRAGMA user_version=17'); db.commit()
            before = path.read_bytes()
            with self.assertRaisesRegex(GraphError, 'version-18'): Workspace.load(path)
            self.assertEqual(path.read_bytes(), before)
            legacy = Workspace(); self.addCleanup(legacy.db.close); legacy.put(default_model())
            old = Path(td) / 'old.vtw'; legacy.save(old)
            with closing(sqlite3.connect(old)) as db: db.execute('PRAGMA user_version=17'); db.commit()
            before = old.read_bytes(); opened = Workspace.load(old)
            try: self.assertEqual(opened.db.execute('PRAGMA user_version').fetchone()[0], VERSION)
            finally: opened.db.close()
            self.assertEqual(old.read_bytes(), before)

    def test_seed_from_concrete_lstm_is_family7_with_no_inferred_runtime_domain(self):
        doc = definition()
        for model in doc['models'].values(): self.w.put(model)
        before = self.w.db.serialize()
        draft = authoring.begin(self.w, 'family', model=doc['root'])
        self.assertEqual(draft['document']['version'], 7)
        self.assertIsNone(draft['document']['runtime_policy'])
        authoring.check(self.w, draft)
        self.assertEqual(self.w.db.serialize(), before)

    def test_fresh_stdlib_export_resolution_does_not_import_torch_or_qt(self):
        code = ('import sys;from examples.lstm_family import definition;'
                'from app.families import specialize;from app.family_export import family_source;'
                'doc=definition();assert specialize(doc)["contract"]["parameters"]==1400;'
                'source=family_source(doc);compile(source["code"],"<lstm-family>","exec");'
                'assert source["provenance"]["family_grammar"]==7;'
                'assert not any(m in sys.modules for m in ("torch","numpy","PyQt6"))')
        ran = subprocess.run(['/usr/bin/python3', '-c', code], cwd=ROOT, capture_output=True, text=True, timeout=25)
        self.assertEqual(ran.returncode, 0, ran.stderr)

    def test_real_example_cli_preview_export_no_overwrite_and_unchanged_source(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'example.vtw'
            command = ['/usr/bin/python3', 'examples/lstm_family.py', str(path)]
            made = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=25)
            self.assertEqual(made.returncode, 0, made.stderr)
            result = json.loads(made.stdout); self.assertEqual(result['parameters'], 1400)
            before = path.read_bytes(); family = result['family']; revision = family['revision']
            for args in (['preview-stored-family', family['id'], revision],
                         ['export-family', family['id'], revision, '--output', str(Path(td) / 'family.py')],
                         ['validate', result['model']['id']]):
                checked = subprocess.run(['/usr/bin/python3', 'vpw_cli.py', str(path), *args],
                                         cwd=ROOT, text=True, capture_output=True, timeout=25)
                self.assertEqual(checked.returncode, 0, checked.stdout + checked.stderr)
                self.assertTrue(json.loads(checked.stdout)['ok']); self.assertEqual(path.read_bytes(), before)
            refused = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=25)
            self.assertNotEqual(refused.returncode, 0); self.assertEqual(path.read_bytes(), before)


class ConstructorSelectionTests(unittest.TestCase):
    def setUp(self):
        doc = definition(); self.scope = doc['root']
        self.symbols = validate_family(doc)

    def check(self, value, bindings=None, *, choices=True):
        expression(value, self.symbols, self.scope, 'Selection test', choices=choices)
        return evaluate(value, bindings or {}, 'Selection test')

    def test_comparisons_and_boolean_symbol_choose_only_checked_scalar_branches(self):
        for cmp, expected in (('eq', 9), ('ne', 4), ('gt', 4), ('ge', 4), ('lt', 9), ('le', 9)):
            with self.subTest(cmp=cmp):
                self.assertEqual(self.check({'select': {'cmp': cmp, 'args': [3, 2]}, 'then': 4, 'else': 9}), expected)
        for value in (False, True):
            self.assertEqual(self.check({'select': {'symbol': 'directions'}, 'then': 2, 'else': 1},
                                        {'directions': value}), 2 if value else 1)

    def test_unchosen_arithmetic_is_not_evaluated_but_its_grammar_is_checked(self):
        value = {'select': True, 'then': 3, 'else': {'op': 'div', 'args': [1, 0]}}
        self.assertEqual(self.check(value), 3)
        value['select'] = False
        with self.assertRaisesRegex(GraphError, 'positive denominator'): self.check(value)
        value['select'] = True; value['else'] = 'arbitrary Python'
        with self.assertRaises(GraphError): self.check(value)

    def test_condition_and_branches_have_closed_typed_grammar(self):
        for predicate in (0, 1, 1.0, 'true', None, [], {'symbol': 'width'},
                          {'symbol': 'directions', 'extra': 1}, {'cmp': ['eq'], 'args': [1, 2]},
                          {'cmp': 'eq', 'args': [True, 2]}, {'cmp': 'eq', 'args': [1]},
                          {'cmp': 'call', 'args': [1, 2]}):
            with self.subTest(predicate=predicate), self.assertRaises(GraphError):
                self.check({'select': predicate, 'then': 2, 'else': 1})
        for value in ({'select': True, 'then': 2}, {'select': True, 'then': 2, 'else': 1, 'extra': 0},
                      {'select': True, 'then': True, 'else': 1}):
            with self.subTest(value=value), self.assertRaises(GraphError): self.check(value)

    def test_scopes_and_no_enum_or_integer_truthiness(self):
        for kind in ('integer', 'enum'):
            symbols = copy.deepcopy(self.symbols); symbols['directions']['type'] = kind
            if kind == 'enum': symbols['directions']['values'] = ['yes', 'no']
            with self.subTest(kind=kind), self.assertRaisesRegex(GraphError, 'type mismatch'):
                expression({'select': {'symbol': 'directions'}, 'then': 2, 'else': 1},
                           symbols, self.scope, 'typed', choices=True)
        symbols = copy.deepcopy(self.symbols); symbols['directions']['scope'] = 'other'
        with self.assertRaisesRegex(GraphError, 'out-of-scope'):
            expression({'select': {'symbol': 'directions'}, 'then': 2, 'else': 1}, symbols, self.scope, 'scoped', choices=True)

    def test_old_grammar_and_shared_depth_node_budgets_are_preserved(self):
        value = {'select': True, 'then': 2, 'else': 1}
        with self.assertRaisesRegex(GraphError, 'version 7'): self.check(value, choices=False)
        deep = 1
        for _ in range(25): deep = {'select': True, 'then': deep, 'else': 1}
        with self.assertRaisesRegex(GraphError, 'depth/node limit'): self.check(deep)
        broad = {'op': 'add', 'args': [{'op': 'add', 'args': [1] * 16}] * 16}
        with self.assertRaisesRegex(GraphError, 'depth/node limit'): self.check(broad)


if __name__ == '__main__': unittest.main()
