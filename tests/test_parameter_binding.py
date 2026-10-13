"""Durable copied initialization lineage and atomic edits from one retained fit."""
import copy
import json
import sqlite3
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.core import GraphError, Workspace, default_model, VERSION
from app.evidence import identity
from app.experiment_objects import execute_candidate, preview_experiment
from app.parameter_binding import preview, admit, validate, check_dependencies, current_bindings
from app.parameter_sources import inventory, inventory_page, read
from app.policy_authoring import TOKEN, prepare
from app.policy_graph import owner_key, plan
from app.revisions import frozen_document, validate_all
from evidence_fixture import experiment
from test_task_authoring import multi_fixture


class ParameterBindingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = Workspace(); cls.source.set_constrained_mode(False)
        model, data, task = multi_fixture()
        ref = experiment(cls.source, model, data, task, 'binding-source', updates=3)
        exp = cls.source.get_spec('experiment', ref['id'], ref['revision'])
        coords = preview_experiment(exp, cls.source.get_spec)['candidates'][0]['coordinates']
        cls.saved_run = execute_candidate(cls.source, ref, coords, 'source-run', 'Source', python=sys.executable, timeout=20)
        assert cls.saved_run['record']['attempts'][0]['status'] == 'completed', cls.saved_run
        listing = inventory(cls.source, cls.saved_run['reference'], attempt=1)
        weight = next(p for p in listing['parameters'] if p['path'].endswith('.weight'))
        cls.selection = {'run': listing['run'], 'attempt': 1, 'checkpoint_id': listing['checkpoint_id'],
                         'parameter': {k: weight[k] for k in ('path', 'id')}}

    @classmethod
    def tearDownClass(cls):
        if cls.source.history.directory: cls.source.history.directory.cleanup()
        cls.source.db.close()

    def setUp(self):
        self.w = Workspace(); self.source.db.backup(self.w.db)
        shape = read(self.w, self.selection)['source']['shape']
        model = default_model('Frozen destination', inputs=shape[1], outputs=shape[0]); model['id'] = 'destination'
        model['graph']['parameter_policy'] = {'version': 1, 'parameters': {owner_key(['linear1', 'weight']): {'trainable': False}}}
        self.w.put(model); self.w.dirty = False
        info = prepare(self.w.models(), 'destination', 'linear1')
        self.token = {k: info[k] for k in TOKEN}; self.selector = owner_key(['linear1', 'weight'])

    def tearDown(self):
        if self.w.history.directory: self.w.history.directory.cleanup()
        self.w.db.close()

    def test_preview_is_readonly_and_admit_preserves_trainability_and_exact_lineage(self):
        raw = self.w.db.serialize(); candidate = preview(self.w, self.token, self.selector, self.selection)
        self.assertEqual(self.w.db.serialize(), raw); self.assertFalse(self.w.dirty)
        self.assertFalse(plan({'destination': candidate['document']}, 'destination')['resolved']['owners'][owner_key(['destination', 'linear1', 'weight'])]['trainable'])
        retained = admit(self.w, self.token, self.selector, self.selection)
        self.assertTrue(retained['changed']); record = self.w.get_spec(**{'kind': retained['reference']['kind'], 'sid': retained['reference']['id'], 'revision': retained['reference']['revision']})
        self.assertEqual(record, candidate['binding']); check_dependencies(record, self.w.get_spec); validate_all(self.w.db)
        values = self.w.models()['destination']['graph']['parameter_policy']['parameters'][self.selector]['initializer']['values']
        self.assertEqual(values, read(self.w, self.selection)['values'])
        self.assertEqual(record['source']['run'], self.selection['run'])
        self.assertEqual(record['destination']['model'], retained['model']); self.assertFalse(record['optimizer_resume'])
        self.assertNotEqual(record['source']['model']['id'], record['destination']['model']['id'])
        self.assertEqual(current_bindings(self.w, 'destination')[self.selector]['reference'], retained['reference'])
        visual = self.w.models()['destination']; visual['graph']['nodes'][0]['x'] = 33; self.w.put(visual)
        self.assertEqual(current_bindings(self.w, 'destination')[self.selector]['reference'], retained['reference'])

    def test_fresh_token_repeat_is_true_noop_and_stale_token_refuses(self):
        first = admit(self.w, self.token, self.selector, self.selection)
        raw = self.w.db.serialize(); dirty = self.w.dirty; changes = self.w.db.total_changes
        info = prepare(self.w.models(), 'destination', 'linear1'); token = {k: info[k] for k in TOKEN}
        second = admit(self.w, token, self.selector, self.selection)
        self.assertFalse(second['changed']); self.assertEqual(first['reference'], second['reference'])
        self.assertEqual(raw, self.w.db.serialize()); self.assertEqual(changes, self.w.db.total_changes); self.assertEqual(self.w.dirty, dirty)
        with self.assertRaisesRegex(GraphError, 'changed'): admit(self.w, self.token, self.selector, self.selection)
        self.assertEqual(raw, self.w.db.serialize())

    def test_bad_source_shape_selector_and_name_refuse_without_writes(self):
        raw = self.w.db.serialize()
        for selection in (dict(self.selection, checkpoint_id='0' * 64), dict(self.selection, attempt=True),
                          dict(self.selection, parameter=dict(self.selection['parameter'], id='wrong'))):
            with self.assertRaises(GraphError): admit(self.w, self.token, self.selector, selection)
        with self.assertRaises(GraphError): admit(self.w, self.token, owner_key(['linear1', 'bias']), self.selection)
        with self.assertRaises(GraphError): admit(self.w, self.token, self.selector, self.selection, name='')
        with self.assertRaises(GraphError): admit(self.w, {**self.token, 'extra': 1}, self.selector, self.selection)
        self.assertEqual(raw, self.w.db.serialize()); self.assertFalse(self.w.dirty)

    def test_atomic_rollback_restores_draft_revisions_heads_and_dirty_flag(self):
        raw = self.w.db.serialize(); models = self.w.models(); rows = self.w.list_specs(all_revisions=True)
        original = self.w.put_spec
        def fail(kind, document, select=True):
            if kind == 'parameter-binding': raise RuntimeError('Injected receipt failure')
            return original(kind, document, select)
        with patch.object(self.w, 'put_spec', side_effect=fail), self.assertRaisesRegex(RuntimeError, 'receipt failure'):
            admit(self.w, self.token, self.selector, self.selection)
        self.assertEqual(self.w.models(), models); self.assertEqual(self.w.list_specs(all_revisions=True), rows)
        self.assertEqual(self.w.db.serialize(), raw); self.assertFalse(self.w.dirty)

    def test_pinned_receipt_survives_changed_drawing_model_heads_and_save_reload(self):
        first = admit(self.w, self.token, self.selector, self.selection); ref = first['reference']
        saved = self.w.get_spec('parameter-binding', ref['id'], ref['revision'])
        model = self.w.models()['destination']; model['name'] = 'New mutable head'
        next(n for n in model['graph']['nodes'] if n['type'] == 'Linear')['features'] += 1
        # Remove policy explicitly before changing shape; retained model is untouched.
        model['graph'].pop('parameter_policy'); self.w.put(model); self.w.freeze_model('destination')
        self.assertEqual(current_bindings(self.w, 'destination'), {})
        self.assertEqual(self.w.get_spec('parameter-binding', ref['id'], ref['revision']), saved)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'binding.vtw'; self.w.save(path); reopened = Workspace.load(path)
            try:
                self.assertEqual(reopened.get_spec('parameter-binding', ref['id'], ref['revision']), saved)
                self.assertEqual(reopened.db.execute('PRAGMA user_version').fetchone()[0], VERSION)
            finally: reopened.db.close()
            db = sqlite3.connect(path); db.execute('PRAGMA user_version=25'); db.close(); raw = path.read_bytes()
            reopened = Workspace.load(path)
            try:self.assertEqual(reopened.get_spec('parameter-binding', ref['id'], ref['revision']), saved)
            finally:reopened.db.close()
            self.assertEqual(path.read_bytes(), raw)
            # Deliberately downgrade only private test bytes; actual feature fence must refuse.
            db = sqlite3.connect(path); db.execute('PRAGMA user_version=24'); db.close()
            with self.assertRaisesRegex(GraphError, 'version-25'): Workspace.load(path)

    def test_forged_lineage_and_destination_values_are_refused(self):
        candidate = preview(self.w, self.token, self.selector, self.selection)
        self.w.put(candidate['document']); self.w.put_spec('model', candidate['model'])
        for field in ('checkpoint_id', 'values_sha256'):
            bad = copy.deepcopy(candidate['binding']); bad.pop('revision'); bad['source'][field] = '0' * 64
            if field == 'values_sha256': bad['destination'][field] = '0' * 64
            with self.subTest(field=field), self.assertRaises(GraphError): self.w.put_spec('parameter-binding', bad)
        bad = copy.deepcopy(candidate['binding']); bad.pop('revision'); bad['destination']['selector'] = owner_key(['linear1', 'bias'])
        bad['destination']['id'] = owner_key(['destination', 'linear1', 'bias'])
        with self.assertRaises(GraphError): self.w.put_spec('parameter-binding', bad)
        for change in ({'optimizer_resume': True}, {'version': 2}, {'extra': 1}):
            bad = {**candidate['binding'], **change}; bad.pop('revision')
            with self.assertRaises(GraphError): frozen_document('parameter-binding', bad)

    def test_fresh_admission_imports_no_numerical_or_Qt_packages(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'binding.vtw'; self.w.save(path)
            script = ('import sys,json;from app.core import Workspace;from app.parameter_binding import admit;'
                      'w=Workspace.load(sys.argv[1]);admit(w,*json.loads(sys.argv[2]));'
                      'print(json.dumps([n for n in sys.modules if n.split(".")[0] in ("torch","numpy","PyQt6")]))')
            proc = subprocess.run([sys.executable, '-c', script, str(path), json.dumps([self.token, self.selector, self.selection])],
                                  capture_output=True, text=True, timeout=20)
            self.assertEqual(proc.returncode, 0, proc.stderr); self.assertEqual(json.loads(proc.stdout), [])

    def test_cli_inventory_prepare_preview_atomic_apply_and_stale_disk_revision(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'binding.vtw'; self.w.save(path); before = path.read_bytes()
            def cli(*args, text=None):
                proc = subprocess.run([sys.executable, 'vpw_cli.py', str(path), *args], input=text,
                                      capture_output=True, text=True, timeout=20)
                return proc.returncode, json.loads(proc.stdout)
            code, prepared = cli('prepare-parameter-binding', 'destination', 'linear1'); self.assertEqual(code, 0, prepared)
            token = {k: prepared['value'][k] for k in TOKEN}; self.assertEqual(token, self.token)
            ref = self.selection['run']; code, rows = cli('retained-parameters', ref['id'], ref['revision'], '--attempt', '1')
            self.assertEqual(code, 0, rows); self.assertEqual(rows['value']['checkpoint_id'], self.selection['checkpoint_id'])
            self.assertEqual(path.read_bytes(), before)
            edit = {'op': 'bind-parameter', 'token': token, 'selector': self.selector, 'source': self.selection}
            # Later failure must prevent the entire ordered batch reaching disk.
            code, failed = cli('apply', '-', text=json.dumps([edit, {'op': 'future-edit'}]))
            self.assertEqual(code, 1, failed); self.assertEqual(path.read_bytes(), before)
            code, applied = cli('apply', '-', text=json.dumps([edit])); self.assertEqual(code, 0, applied)
            saved = path.read_bytes(); reopened = Workspace.load(path)
            try: self.assertEqual(len(reopened.list_specs('parameter-binding')), 1)
            finally: reopened.db.close()
            code, stale = cli('apply', '-', text=json.dumps([edit])); self.assertEqual(code, 1, stale)
            self.assertEqual(path.read_bytes(), saved)

    def test_source_pages_are_bounded_and_explicit_without_tensor_values(self):
        ref = self.selection['run']; page = inventory_page(self.w, ref, attempt=1, limit=1)
        self.assertEqual(len(page['parameters']), 1); self.assertEqual(page['next_offset'], 1)
        self.assertNotIn('values', page['parameters'][0]); self.assertLessEqual(len(json.dumps(page, ensure_ascii=True).encode()), 65536)
        next_page = inventory_page(self.w, ref, attempt=1, offset=1, limit=1)
        self.assertNotEqual(page['parameters'][0]['path'], next_page['parameters'][0]['path'])
        for bad in (0, True, 26):
            with self.assertRaises(GraphError): inventory_page(self.w, ref, attempt=1, limit=bad)

    def test_sequential_owner_copies_keep_each_tensor_proof_without_model_equality_claim(self):
        first = admit(self.w, self.token, self.selector, self.selection)
        listing = inventory(self.w, self.selection['run'], attempt=1)
        bias = next(p for p in listing['parameters'] if p['path'].endswith('.bias'))
        source = {**self.selection, 'parameter': {key: bias[key] for key in ('path', 'id')}}
        token = {key: prepare(self.w.models(), 'destination', 'linear1')[key] for key in TOKEN}
        second = admit(self.w, token, owner_key(['linear1', 'bias']), source)
        self.assertNotEqual(first['model']['revision'], second['model']['revision'])
        matches = current_bindings(self.w, 'destination')
        self.assertEqual(set(matches), {self.selector, owner_key(['linear1', 'bias'])})
        self.assertEqual(matches[self.selector]['receipt_model'], first['model'])
        self.assertIn('not-model-revision-equality', matches[self.selector]['scope'])
        model = self.w.models()['destination']; model['graph']['parameter_policy']['parameters'][self.selector]['initializer']['values'][0][0] += 1
        self.w.put(model); self.assertNotIn(self.selector, current_bindings(self.w, 'destination'))

    def test_real_signed_zero_source_replaces_positive_zero_and_fresh_repeat_is_noop(self):
        from test_authoring import measurements
        source = default_model(inputs=1, outputs=1)
        source['graph']['parameter_policy'] = {'version': 1, 'parameters': {self.selector: {
            'trainable': False, 'initializer': {'kind': 'values', 'dtype': 'float32', 'values': [[-0.]]}}}}
        task = {'version': 1, 'id': 'zero-task', 'name': 'Zero source task',
            'inputs': [{'port': 'input', 'field': 'x', 'transforms': []}],
            'heads': [{'id': 'y', 'port': 'output', 'field': 'y', 'loss': 'mse', 'weight': 1., 'transforms': []}]}
        expref = experiment(self.w, source, measurements(), task, 'negative-zero', updates=3)
        exp = self.w.get_spec('experiment', expref['id'], expref['revision'])
        coords = preview_experiment(exp, self.w.get_spec)['candidates'][0]['coordinates']
        run = execute_candidate(self.w, expref, coords, 'negative-zero-run', 'Negative-zero source', python=sys.executable, timeout=20)
        self.assertEqual(run['record']['attempts'][0]['status'], 'completed')
        listing = inventory(self.w, run['reference'], attempt=1); weight = next(p for p in listing['parameters'] if p['path'].endswith('.weight'))
        selected = {'run': run['reference'], 'attempt': 1, 'checkpoint_id': listing['checkpoint_id'], 'parameter': {k: weight[k] for k in ('path', 'id')}}
        destination = default_model(inputs=1, outputs=1); destination['id'] = 'zero-destination'
        destination['graph']['parameter_policy'] = {'version': 1, 'parameters': {self.selector: {
            'initializer': {'kind': 'values', 'dtype': 'float32', 'values': [[0.]]}}}}
        self.w.put(destination); token = {k: prepare(self.w.models(), destination['id'], 'linear1')[k] for k in TOKEN}
        bound = admit(self.w, token, self.selector, selected)
        value = self.w.models()[destination['id']]['graph']['parameter_policy']['parameters'][self.selector]['initializer']['values'][0][0]
        self.assertEqual(struct.pack('<f', value), b'\0\0\0\x80')
        self.assertIn(self.selector, current_bindings(self.w, destination['id']))
        token = {k: prepare(self.w.models(), destination['id'], 'linear1')[k] for k in TOKEN}; raw = self.w.db.serialize()
        self.assertFalse(admit(self.w, token, self.selector, selected)['changed']); self.assertEqual(raw, self.w.db.serialize())


if __name__ == '__main__': unittest.main()
