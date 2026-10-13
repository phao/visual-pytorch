import copy
import json
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from app.core import GraphError, Workspace
from app.lora import populate
from app.experiment_objects import execute_candidate, preview_experiment
from app.parameter_sources import inventory, read
from app.parameter_derivation import admit, preview, merged_values, check_dependencies, MAX_PRODUCTS
from app.revisions import validate_all


class ParameterDerivationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = Workspace(); refs = populate(cls.source, updates=3)
        ref = refs['lora-linear-comparison']; exp = cls.source.get_spec('experiment', ref['id'], ref['revision'])
        coords = next(row['coordinates'] for row in preview_experiment(exp, cls.source.get_spec)['candidates'] if row['coordinates']['model']['id'] == 'lora-r2')
        cls.saved = execute_candidate(cls.source, ref, coords, 'merge-source', 'Merge source rank2', python=sys.executable, timeout=30)
        assert cls.saved['record']['attempts'][0]['status'] == 'completed', cls.saved
        listing = inventory(cls.source, cls.saved['reference'], attempt=1)
        cls.selections = {}
        for key in ('base', 'A', 'B'):
            row = next(row for row in listing['parameters'] if json.loads(row['id'])[1] == key)
            cls.selections[key] = {'run': cls.saved['reference'], 'attempt': 1, 'checkpoint_id': listing['checkpoint_id'],
                                   'parameter': {field: row[field] for field in ('path', 'id')}}

    @classmethod
    def tearDownClass(cls):
        if cls.source.history.directory: cls.source.history.directory.cleanup()
        cls.source.db.close()

    def setUp(self):
        self.w = Workspace(); self.source.db.backup(self.w.db); self.w.dirty = False
    def tearDown(self):
        if self.w.history.directory: self.w.history.directory.cleanup()
        self.w.db.close()

    def test_preview_readonly_and_explicit_admission_preserves_original_evidence(self):
        raw = self.w.db.serialize(); candidate = preview(self.w, self.selections)
        self.assertEqual(raw, self.w.db.serialize()); self.assertFalse(self.w.dirty)
        source = self.w.get_spec('run', self.saved['reference']['id'], self.saved['reference']['revision'])
        result = admit(self.w, self.selections)
        record = self.w.get_spec('parameter-derivation', result['reference']['id'], result['reference']['revision'])
        self.assertEqual(record, candidate['derivation']); check_dependencies(record, self.w.get_spec); validate_all(self.w.db)
        self.assertEqual(self.w.get_spec('run', source['id'], source['revision']), source)
        self.assertIn('lora-merged', self.w.models()); self.assertFalse(record['optimizer_resume'])
        self.assertEqual(record['operation']['rank'], 2); self.assertEqual(record['operation']['scale'], 2.)

    def test_repeated_id_wrong_alpha_owner_checkpoint_fields_refuse_without_writes(self):
        raw = self.w.db.serialize()
        for alpha in (True, 0, -1, 8, float('nan'), 10**1000):
            with self.subTest(alpha=str(alpha)[:20]), self.assertRaises(GraphError): admit(self.w, self.selections, alpha=alpha)
        for key in ('base', 'A', 'B'):
            bad = copy.deepcopy(self.selections); bad[key]['checkpoint_id'] = '0' * 64
            with self.assertRaises(GraphError): admit(self.w, bad)
        bad = copy.deepcopy(self.selections); bad['A'] = bad['B']
        with self.assertRaises(GraphError): admit(self.w, bad)
        with self.assertRaises(GraphError): admit(self.w, {**self.selections, 'extra': self.selections['A']})
        self.assertEqual(self.w.db.serialize(), raw); self.assertFalse(self.w.dirty)
        admit(self.w, self.selections); raw = self.w.db.serialize()
        with self.assertRaisesRegex(GraphError, 'never overwritten'): admit(self.w, self.selections)
        self.assertEqual(self.w.db.serialize(), raw)

    def test_savepoint_failure_rolls_back_drawing_revisions_heads_dirty(self):
        raw = self.w.db.serialize(); original = self.w.put_spec
        def fail(kind, document, select=True):
            if kind == 'parameter-derivation': raise RuntimeError('Injected derivation receipt failure')
            return original(kind, document, select)
        with patch.object(self.w, 'put_spec', side_effect=fail), self.assertRaises(RuntimeError): admit(self.w, self.selections)
        self.assertEqual(self.w.db.serialize(), raw); self.assertFalse(self.w.dirty)

    def test_forged_output_hash_refuses_and_older_container_with_archived_record_refuses(self):
        result = admit(self.w, self.selections); ref = result['reference']
        record = self.w.get_spec('parameter-derivation', ref['id'], ref['revision'])
        bad = copy.deepcopy(record); bad.pop('revision'); bad['destination']['values_sha256'] = '0' * 64
        with self.assertRaises(GraphError): self.w.put_spec('parameter-derivation', bad)
        self.w.archive_spec('parameter-derivation', ref['id'])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'merge.vtw'; self.w.save(path); loaded = Workspace.load(path)
            try: self.assertEqual(loaded.get_spec('parameter-derivation', ref['id'], ref['revision']), record)
            finally: loaded.db.close()
            db = sqlite3.connect(path); db.execute('PRAGMA user_version=25'); db.close(); raw = path.read_bytes()
            with self.assertRaisesRegex(GraphError, 'version-26'): Workspace.load(path)
            self.assertEqual(path.read_bytes(), raw)

    def test_float32_kernel_small_reference_overflow_shape_and_work_guards(self):
        self.assertEqual(merged_values([[1., 0.], [0., 1.]], [[2., 3.]], [[4.], [5.]], alpha=2.), [[17., 24.], [20., 31.]])
        with self.assertRaises(GraphError): merged_values([[0.]], [[3e38]], [[3e38]], alpha=1.)
        with self.assertRaises(GraphError): merged_values([[0., 0.]], [[1.]], [[2.]], alpha=1.)
        row = [0.] * 1001
        with self.assertRaisesRegex(GraphError, 'bounded local'): merged_values([row] * 1000, [row, row], [[0., 0.]], alpha=1.)

    def test_fresh_reader_preview_admission_import_no_numeric_or_Qt_packages(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'source.vtw'; self.w.save(path)
            script = ('import sys,json;from app.core import Workspace;from app.parameter_derivation import admit;'
                      'w=Workspace.load(sys.argv[1]);admit(w,json.loads(sys.argv[2]));'
                      'print(json.dumps([n for n in sys.modules if n.split(".")[0] in ("torch","numpy","PyQt6")]))')
            proc = subprocess.run([sys.executable, '-c', script, str(path), json.dumps(self.selections)], capture_output=True, text=True, timeout=30)
            self.assertEqual(proc.returncode, 0, proc.stderr); self.assertEqual(json.loads(proc.stdout), [])


if __name__ == '__main__': unittest.main()
