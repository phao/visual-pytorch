"""Exact allocation-free family source, exclusive files and Qt/agent fences."""
import copy
import hashlib
import importlib.util
import json
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app.authoring import workspace_id
from app.core import GraphError, Workspace
from app.family_export import family_source, runtime_sources, stored_source, write_new_family
from app.evidence import digest
from test_scoped_constructors import scoped

ROOT = Path(__file__).resolve().parents[1]


class FamilyExportTests(unittest.TestCase):
    def setUp(self):
        self.w = Workspace(); self.addCleanup(self.w.db.close)
        self.ref = self.w.put_spec('family', scoped())
        self.addCleanup(lambda: self.w.history.directory.cleanup() if self.w.history.directory else None)

    def test_generation_is_exact_deterministic_and_does_not_resolve_allocate_or_train(self):
        before = self.w.db.serialize(); rng = random.getstate()
        with (patch('app.families.specialize', side_effect=AssertionError('Do not instantiate')),
              patch('app.core.compile_model', side_effect=AssertionError('Do not allocate')),
              patch('app.transport.run_json_worker', side_effect=AssertionError('Do not train'))):
            a = stored_source(self.w, self.ref); b = stored_source(self.w, self.ref)
        self.assertEqual(a, b); self.assertEqual(before, self.w.db.serialize()); self.assertEqual(rng, random.getstate())
        self.assertEqual(a['provenance']['family'], self.ref); self.assertEqual(a['provenance']['family_grammar'], 4)
        self.assertEqual(a['provenance']['runtime_sha256'], digest(runtime_sources()))
        self.assertEqual(a['code_sha256'], hashlib.sha256(a['code'].encode()).hexdigest())
        self.assertEqual(a['code_bytes'], len(a['code'].encode())); compile(a['code'], '<checked-family>', 'exec')
        self.assertNotIn('class Workspace', a['code']); self.assertNotIn('CREATE TABLE', a['code'])

    def test_required_and_unresolved_families_can_export_but_malformed_ones_cannot(self):
        source = scoped(); source['symbols'][0].pop('default')
        result = family_source(source); self.assertEqual(result['provenance']['family']['revision'], digest(source))
        source['constructors'][0]['value'] = 'Python(width)'
        with self.assertRaises(GraphError): family_source(source)

    def test_all_four_grammars_export_their_own_versions_and_operator_runtime(self):
        for filename, version in [('width-depth.json', 1), ('width-depth-tied.json', 2), ('activation-bias.json', 3), ('scoped-branches.json', 4)]:
            source = json.loads((ROOT / 'examples/families' / filename).read_text())
            result = family_source(source)
            self.assertEqual(result['provenance']['family_grammar'], version)
            self.assertEqual(result['provenance']['operator_version'], 1)
            self.assertEqual(result['provenance']['export_format'], 1)

    def test_historical_exact_revision_not_current_head_or_mutable_drawing(self):
        a = stored_source(self.w, self.ref); newer = scoped(); newer['name'] = 'Different head'
        self.w.put_spec('family', newer)
        self.assertEqual(stored_source(self.w, self.ref), a)
        self.w.archive_spec('family', self.ref['id'])
        self.assertEqual(stored_source(self.w, self.ref), a)
        for bad in (dict(self.ref, revision='0'*64), dict(self.ref, kind='model'), dict(self.ref, revision=None)):
            with self.assertRaises(GraphError): stored_source(self.w, bad)

    def test_policy_is_explicit_checked_and_captured_not_a_current_model_run(self):
        self.w.set_constrained_mode(False); a = stored_source(self.w, self.ref, memory_limit_percent=7.5)
        self.assertEqual(a['provenance']['policy'], {'constrained': False, 'memory_limit_percent': 7.5})
        self.w.set_constrained_mode(True); self.assertFalse(a['provenance']['policy']['constrained'])
        for constrained in (0, 'true', None):
            with self.assertRaises(GraphError): family_source(scoped(), constrained=constrained)
        for percent in (True, 0, -1, 101, float('nan'), float('inf')):
            with self.assertRaises(ValueError): family_source(scoped(), memory_limit_percent=percent)

    def test_exclusive_export_no_overwrite_symlink_follow_or_hash_mismatch(self):
        source = stored_source(self.w, self.ref); before = self.w.db.serialize()
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'family.py'; receipt = write_new_family(source, path)
            self.assertEqual(path.read_text(), source['code']); self.assertEqual(receipt['provenance'], source['provenance'])
            self.assertNotIn('code', receipt)
            with self.assertRaises(FileExistsError): write_new_family(source, path)
            missing = Path(td) / 'missing.py'; link = Path(td) / 'link.py'; link.symlink_to(missing)
            with self.assertRaises(FileExistsError): write_new_family(source, link)
            self.assertFalse(missing.exists())
            tampered = Path(td) / 'tampered.py'
            with self.assertRaisesRegex(GraphError, 'byte hash'): write_new_family(dict(source, code=source['code']+'\n'), tampered)
            self.assertFalse(tampered.exists())
        self.assertEqual(before, self.w.db.serialize())

    def test_actual_cli_export_exact_receipt_file_and_unchanged_workspace(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'source.vtw'; self.w.save(path); before = path.read_bytes(); output = Path(td) / 'family.py'
            args = ['/usr/bin/python3', 'vpw_cli.py', str(path), 'export-family', self.ref['id'], self.ref['revision'], '--output', str(output)]
            ran = subprocess.run(args, cwd=ROOT, text=True, capture_output=True, timeout=20)
            self.assertEqual(ran.returncode, 0, ran.stderr); result = json.loads(ran.stdout)
            self.assertTrue(result['ok']); self.assertEqual(path.read_bytes(), before)
            self.assertEqual(result['value']['code_sha256'], hashlib.sha256(output.read_bytes()).hexdigest())
            self.assertEqual(result['value']['provenance']['family'], self.ref)
            again = subprocess.run(args, cwd=ROOT, text=True, capture_output=True, timeout=20)
            self.assertEqual(again.returncode, 1); self.assertEqual(path.read_bytes(), before)

    def test_source_generation_in_fresh_stdlib_process_imports_no_numerical_runtime(self):
        code = 'import json,sys;from app.family_export import family_source;s=family_source(json.load(open("examples/families/scoped-branches.json")));compile(s["code"],"<export>","exec");assert not any(m in sys.modules for m in ("torch","numpy","PyQt6"))'
        ran = subprocess.run(['/usr/bin/python3', '-c', code], cwd=ROOT, capture_output=True, text=True, timeout=20)
        self.assertEqual(ran.returncode, 0, ran.stderr)


@unittest.skipUnless(importlib.util.find_spec('PyQt6') is not None, 'Native export dispatch requires PyQt6; run explicitly in the Qt environment.')
class NativeFamilyExportTests(unittest.TestCase):
    def setUp(self):
        from app.desktop_experiments import dispatch
        self.dispatch = dispatch; self.w = Workspace(); self.addCleanup(self.w.db.close)
        self.ref = self.w.put_spec('family', scoped()); self.window = SimpleNamespace(workspace=self.w, job=None,
                                                                                  state=lambda: {'memory_limit_percent': 7.5})
        self.req = {'action': 'scienceExportFamily', 'workspace_id': workspace_id(self.w), 'family': self.ref}

    def test_direct_native_export_is_bounded_no_sql_writes_and_captures_policy(self):
        before = self.w.db.serialize()
        with tempfile.TemporaryDirectory() as td:
            output = Path(td) / 'family.py'; result = self.dispatch(self.window, dict(self.req, output=str(output)))
            self.assertNotIn('code', result); self.assertLess(len(json.dumps(result)), 2000)
            self.assertEqual(result['provenance']['policy']['memory_limit_percent'], 7.5); self.assertTrue(output.is_file())
        self.assertEqual(self.w.db.serialize(), before)

    def test_dialog_cancel_writes_nothing_and_modal_workspace_change_is_rejected(self):
        before = self.w.db.serialize()
        with patch('app.desktop_experiments.QFileDialog.getSaveFileName', return_value=('', '')):
            self.assertIsNone(self.dispatch(self.window, self.req))
        with tempfile.TemporaryDirectory() as td:
            output = Path(td) / 'family.py'; other = Workspace(); self.addCleanup(other.db.close)
            def choose(*_): self.window.workspace = other; return str(output), ''
            with patch('app.desktop_experiments.QFileDialog.getSaveFileName', side_effect=choose):
                with self.assertRaisesRegex(ValueError, 'changed'): self.dispatch(self.window, self.req)
            self.assertFalse(output.exists())
            self.window.workspace = self.w
            def pending(*_):
                self.window.job = SimpleNamespace(isRunning=lambda: False, awaiting_terminal=True)
                return str(output), ''
            with patch('app.desktop_experiments.QFileDialog.getSaveFileName', side_effect=pending):
                with self.assertRaisesRegex(ValueError, 'changed'): self.dispatch(self.window, self.req)
            self.assertFalse(output.exists())
        self.assertEqual(self.w.db.serialize(), before)

    def test_native_closed_fields_and_wrong_workspace_fail_before_export(self):
        for request in (dict(self.req, workspace_id='wrong'), dict(self.req, train=True), dict(self.req, output='')):
            with self.assertRaises(ValueError): self.dispatch(self.window, request)


if __name__ == '__main__': unittest.main()
