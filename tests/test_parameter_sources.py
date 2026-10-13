"""Exact source-half of base binding, using one real bounded retained worker fit."""
import copy
import hashlib
import json
import random
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.core import GraphError, Workspace
from app.experiment_objects import execute_candidate, preview_experiment
from app.parameter_sources import float32_values, inventory, read, VALUE_FORMAT
from evidence_fixture import experiment
from test_task_authoring import multi_fixture


class ParameterSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.w = Workspace(); cls.w.set_constrained_mode(False)
        model, data, task = multi_fixture()
        cls.exp = experiment(cls.w, model, data, task, 'base-source', updates=3)
        doc = cls.w.get_spec('experiment', cls.exp['id'], cls.exp['revision'])
        coordinates = preview_experiment(doc, cls.w.get_spec)['candidates'][0]['coordinates']
        cls.saved_run = execute_candidate(cls.w, cls.exp, coordinates, 'base-run', 'Base source', python=sys.executable, timeout=20)
        assert cls.saved_run['record']['attempts'][0]['status'] == 'completed', cls.saved_run
        cls.ref = cls.saved_run['reference']; listing = inventory(cls.w, cls.ref, attempt=1)
        cls.selection = {'run': cls.ref, 'attempt': 1, 'checkpoint_id': listing['checkpoint_id'],
                         'parameter': {k: listing['parameters'][0][k] for k in ('path', 'id')}}
        cls.retry = execute_candidate(cls.w, cls.exp, coordinates, 'base-run', 'Base source',
                                      python='/missing/parameter-source-python', retry=True)

    @classmethod
    def tearDownClass(cls):
        if cls.w.history.directory: cls.w.history.directory.cleanup()
        cls.w.db.close()

    def test_exact_values_and_shape_bound_float32_digest(self):
        answer = read(self.w, self.selection); source = answer['source']; path = self.selection['parameter']['path']
        original = self.saved_run['record']['attempts'][0]['result']['checkpoint']['weights'][path]
        self.assertEqual(answer['values'], original)
        h = hashlib.sha256(b'Visual PyTorch float32 tensor v1\0' + json.dumps(source['shape'], separators=(',', ':')).encode('ascii') + b'\0')
        def emit(value):
            if isinstance(value, list):
                for part in value: emit(part)
            else: h.update(struct.pack('<f', value))
        emit(original)
        self.assertEqual(source['values_sha256'], h.hexdigest())
        self.assertEqual(source['value_format'], VALUE_FORMAT)
        self.assertFalse(source['optimizer_resume']); self.assertEqual(source['completed_updates'], 3)
        self.assertEqual(source['run'], self.ref)
        answer['values'].clear(); self.assertEqual(read(self.w, self.selection)['values'], original)

    def test_sql_rng_readonly_and_no_execution_or_new_split(self):
        before = self.w.db.serialize(); rng = random.getstate()
        with (patch('app.transport.run_json_worker', side_effect=AssertionError('No worker')),
              patch('app.experiment_objects.make_split', side_effect=AssertionError('No split')),
              patch('app.task_execution.restore_task_predictor', side_effect=AssertionError('No inference'))):
            inventory(self.w, self.ref, attempt=1); read(self.w, self.selection)
        self.assertEqual(before, self.w.db.serialize()); self.assertEqual(random.getstate(), rng)

    def test_head_changes_do_not_change_exact_source_and_failed_attempt_never_substitutes(self):
        before = read(self.w, self.selection)
        modelref = before['source']['model']; original = self.w.models()[modelref['id']]
        changed = copy.deepcopy(original); changed['name'] = 'Mutable drawing is not the source'
        self.w.put(changed)
        try: self.assertEqual(read(self.w, self.selection), before)
        finally: self.w.put(original)
        self.assertNotEqual(self.retry['reference'], self.ref)
        self.assertEqual(inventory(self.w, self.retry['reference'], attempt=1)['checkpoint_id'], self.selection['checkpoint_id'])
        with self.assertRaisesRegex(GraphError, 'no parameter checkpoint'):
            inventory(self.w, self.retry['reference'], attempt=2)
        with self.assertRaises(TypeError): inventory(self.w, self.ref)

    def test_closed_or_stale_selection_refuses_without_mutation(self):
        before = self.w.db.serialize()
        choices = [dict(self.selection, attempt=value) for value in (0, True, 2, None)]
        choices += [dict(self.selection, checkpoint_id='0' * 64), dict(self.selection, extra=1),
                    dict(self.selection, run=dict(self.ref, revision=None)),
                    dict(self.selection, run=dict(self.ref, revision='0' * 64)),
                    dict(self.selection, parameter=dict(self.selection['parameter'], id='wrong')),
                    dict(self.selection, parameter=dict(self.selection['parameter'], path='missing')),
                    dict(self.selection, parameter=dict(self.selection['parameter'], path=[])),
                    dict(self.selection, parameter=dict(self.selection['parameter'], dtype='float64'))]
        for selection in choices:
            with self.subTest(selection=selection), self.assertRaises(GraphError): read(self.w, selection)
        self.assertEqual(before, self.w.db.serialize())

    def test_float32_roundtrip_preserves_signed_zero_and_distinguishes_shapes(self):
        values, a = float32_values([-.0, .1], [2]); _, b = float32_values([[-.0, .1]], [1, 2])
        self.assertEqual(struct.pack('<f', values[0]), b'\x00\x00\x00\x80')
        self.assertEqual(values[1], struct.unpack('<f', struct.pack('<f', .1))[0])
        self.assertNotEqual(a, b)
        for bad in ([True, 1.], [float('nan'), 1.], [1.], [float('inf'), 1.]):
            with self.assertRaises(GraphError): float32_values(bad, [2])
        for shape in ([True], [2.0], [-1], [0], [], [1] * 9, (2,), None):
            with self.assertRaises(GraphError): float32_values([1., 2.], shape)

    def test_fresh_source_reader_imports_no_numerical_or_Qt_packages(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'source.vtw'; self.w.save(path)
            script = ('import sys,json;from app.core import Workspace;from app.parameter_sources import read;'
                      'w=Workspace.load(sys.argv[1]);read(w,json.loads(sys.argv[2]));'
                      'print(json.dumps([n for n in sys.modules if n.split(".")[0] in ("torch","numpy","PyQt6")]))')
            result = subprocess.run([sys.executable, '-c', script, str(path), json.dumps(self.selection)],
                                    text=True, capture_output=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stderr); self.assertEqual(json.loads(result.stdout), [])


if __name__ == '__main__': unittest.main()
