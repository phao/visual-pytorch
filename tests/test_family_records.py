"""Exact family/specialization meaning, SQLite atomicity and portable v7 records.

Pure standard-library tests: no Torch/Qt import, fit, tensor or hidden sidecar.
"""
import copy
import io
import json
import random
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from contextlib import closing, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from app.core import GraphError, VERSION, Workspace, default_model
from app.evidence import digest
from app.family_records import (definition, preview_stored, retain,
                                validate_specialization_dependencies)
from app.revisions import SCHEMA, frozen_document, validate_all
from app.science_browser import encoded, index_page, inspect_page, summary, MAX_BYTES
import vpw_cli

ROOT = Path(__file__).resolve().parents[1]


def family():
    return json.loads((ROOT / 'examples/families/width-depth.json').read_text())


class FamilyRecordTests(unittest.TestCase):
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

    def state(self):
        return list(self.w.db.iterdump()), self.w.dirty

    def cli(self, *args):
        stream = io.StringIO()
        with redirect_stdout(stream):
            code = vpw_cli.main([str(v) for v in args])
        return code, json.loads(stream.getvalue())

    def test_family_uses_same_external_digest_and_private_immutable_copy(self):
        source = family(); original = copy.deepcopy(source)
        ref = self.w.put_spec('family', source)
        self.assertEqual(ref['revision'], digest(source))
        self.assertEqual(source, original)
        stored = self.w.get_spec('family', ref['id'], ref['revision'])
        stored['symbols'][0]['name'] = 'mutated private copy'
        self.assertEqual(self.w.get_spec('family', ref['id'])['symbols'][0]['name'], 'width')
        validate_all(self.w.db)

    def test_family_revisions_may_be_unbound_not_silently_specialized(self):
        source = family(); del source['symbols'][0]['default']
        ref = self.w.put_spec('family', source)
        before, rng = self.state(), random.getstate()
        self.assertEqual(preview_stored(self.w, ref)['status'], 'conditional')
        with self.assertRaisesRegex(GraphError, 'Unresolved symbol'):
            retain(self.w, family_ref=ref)
        self.assertEqual(self.state(), before); self.assertEqual(random.getstate(), rng)

    def test_specialization_retains_full_binding_and_exact_frozen_model(self):
        result = retain(self.w, family=family(), bindings={'hidden-width': 4}, sid='chosen', name='Explicit W=4')
        record = self.record(result)
        self.assertEqual(record['bindings'], {'hidden-width': 4, 'hidden-depth': 2})
        self.assertEqual(record['family'], result['family']); self.assertEqual(record['model'], result['model'])
        self.assertEqual(set(self.w.models()), {result['resolved']['root']})
        frozen = self.w.get_spec('model', result['model']['id'], result['model']['revision'])
        self.assertEqual(frozen['architecture_hash'], result['resolved']['architecture_hash'])
        self.assertEqual(set(frozen['models'][frozen['root']]), {'id', 'name', 'graph'})
        self.assertTrue(all('x' not in n for n in frozen['models'][frozen['root']]['graph']['nodes']))
        validate_all(self.w.db)

    def test_registry_only_resolution_does_not_create_or_replace_a_drawing(self):
        self.w.put(default_model()); before = self.w.models()
        result = retain(self.w, family=family(), materialize=False)
        self.assertEqual(self.w.models(), before)
        self.assertFalse(result['materialized']); self.record(result); validate_all(self.w.db)

    def test_historical_family_is_not_replaced_by_current_browsing_head(self):
        source = family(); old = self.w.put_spec('family', source)
        source['symbols'][0]['default'] = 5
        current = self.w.put_spec('family', source)
        result = retain(self.w, family_ref=old, materialize=False)
        self.assertEqual(result['resolved']['contract']['parameters'], 37)
        self.assertEqual(self.record(result)['family'], old)
        self.assertEqual(self.w.get_spec('family', old['id'])['revision'], current['revision'])

    def test_default_bindings_are_pinned_when_the_family_later_changes(self):
        result = retain(self.w, family=family(), materialize=False)
        old = self.record(result)
        source = family(); source['symbols'][0]['default'] = 7
        self.w.put_spec('family', source)
        self.assertEqual(self.record(result), old)
        self.assertEqual(self.record(result)['bindings']['hidden-width'], 3)

    def test_canvas_edit_does_not_rewrite_specialization_or_frozen_model(self):
        result = retain(self.w, family=family())
        stored = self.record(result)
        frozen = self.w.get_spec('model', result['model']['id'], result['model']['revision'])
        drawing = self.w.models()[result['resolved']['root']]
        drawing['graph']['nodes'][1]['features'] = 9
        self.w.put(drawing)
        self.assertEqual(self.record(result), stored)
        self.assertEqual(self.w.get_spec('model', result['model']['id'], result['model']['revision']), frozen)
        validate_all(self.w.db)

    def test_missing_default_or_changed_binding_is_not_same_specialization(self):
        result = retain(self.w, family=family(), materialize=False)
        record = definition(self.record(result)); before = self.state()
        for change in ('missing-default', 'changed-value', 'boolean', 'unknown-key'):
            bad = copy.deepcopy(record)
            if change == 'missing-default': bad['bindings'].pop('hidden-depth')
            if change == 'changed-value': bad['bindings']['hidden-width'] = 4
            if change == 'boolean': bad['bindings']['hidden-width'] = True
            if change == 'unknown-key': bad['bindings']['invented'] = 3
            with self.subTest(change=change), self.assertRaises(GraphError):
                self.w.put_spec('specialization', bad)
            self.assertEqual(self.state(), before)

    def test_shape_equal_other_operation_is_not_an_admissible_frozen_model(self):
        first = retain(self.w, family=family(), materialize=False)
        second_family = family(); second_family['models']['mlp']['graph']['nodes'][3]['activation'] = 'ReLU'
        second = retain(self.w, family=second_family, materialize=False)
        a, b = self.record(first), self.record(second)
        self.assertEqual(first['resolved']['contract']['parameters'], second['resolved']['contract']['parameters'])
        wrong = definition(a); wrong['model'] = b['model']
        with self.assertRaisesRegex(GraphError, 'does not match'):
            self.w.put_spec('specialization', wrong)

    def test_missing_dependency_and_wrong_kind_are_rejected(self):
        result = retain(self.w, family=family(), materialize=False)
        record = definition(self.record(result))
        for field, mutation in [('family', 'revision'), ('model', 'id'), ('family', 'kind')]:
            wrong = copy.deepcopy(record)
            wrong[field][mutation] = '0'*64 if mutation == 'revision' else 'not-existing'
            with self.subTest(field=field, mutation=mutation), self.assertRaises(GraphError):
                self.w.put_spec('specialization', wrong)

    def test_failed_later_registry_write_rolls_back_every_row_and_dirty_flag(self):
        before, original = self.state(), self.w.db
        real = self.w.put_spec
        def fail(kind, doc, select=True):
            if kind == 'specialization': raise RuntimeError('Injected late record failure')
            return real(kind, doc, select)
        with patch.object(self.w, 'put_spec', side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, 'late record failure'):
                retain(self.w, family=family())
        self.assertEqual(self.state(), before); self.assertIs(self.w.db, original)
        self.assertFalse(self.w.db.in_transaction)
        validate_all(self.w.db)

    def test_failed_drawing_write_rolls_back_retained_science_too(self):
        before, original = self.state(), self.w.db
        real = self.w.put
        def fail(doc):
            real(doc)
            raise RuntimeError('Injected after drawing insertion')
        with patch.object(self.w, 'put', side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, 'after drawing'):
                retain(self.w, family=family())
        self.assertEqual(self.state(), before); self.assertIs(self.w.db, original)
        self.assertFalse(self.w.db.in_transaction)

    def test_outer_transaction_is_not_committed_by_inner_registry_writers(self):
        self.w.db.execute('BEGIN')
        self.w.db.execute("INSERT INTO metadata VALUES ('outer-fixture','retained-until-outer-rollback')")
        retain(self.w, family=family(), materialize=False)
        self.assertTrue(self.w.db.in_transaction)
        self.w.db.rollback()
        self.assertEqual(self.w.list_specs(all_revisions=True), [])
        self.assertIsNone(self.w.db.execute("SELECT 1 FROM metadata WHERE key='outer-fixture'").fetchone())

    def test_unresolved_or_collision_failure_precedes_all_admission(self):
        result = retain(self.w, family=family()); before = self.state()
        with self.assertRaisesRegex(GraphError, 'no implicit replacement'):
            retain(self.w, family=family())
        self.assertEqual(self.state(), before)
        source = family(); del source['symbols'][0]['default']
        with self.assertRaisesRegex(GraphError, 'Unresolved'):
            retain(self.w, family=source)
        self.assertEqual(self.state(), before)
        self.assertEqual(len(self.w.models()), 1); self.record(result)

    def test_family_and_specialization_sql_guards_prevent_overwrite_and_deletion(self):
        retain(self.w, family=family(), materialize=False)
        rows = list(self.w.db.execute('SELECT * FROM spec_revisions'))
        for kind in ('family', 'specialization'):
            row = next(r for r in rows if r[0] == kind)
            for sql, args in [('UPDATE spec_revisions SET name=? WHERE kind=?', ('corrupt',kind)),
                              ('DELETE FROM spec_revisions WHERE kind=?', (kind,)),
                              ('INSERT OR REPLACE INTO spec_revisions VALUES (?,?,?,?,?,?)', row)]:
                with self.subTest(kind=kind,sql=sql), self.assertRaises(sqlite3.IntegrityError):
                    self.w.db.execute(sql, args)
                self.w.db.rollback()
        validate_all(self.w.db)

    def test_archive_heads_does_not_remove_specialization_dependencies(self):
        result = retain(self.w, family=family(), materialize=False)
        self.w.archive_spec('family', result['family']['id'])
        self.w.archive_spec('model', result['model']['id'])
        record = self.record(result)
        self.assertEqual(record['family'], result['family']); validate_all(self.w.db)

    def test_undo_redo_retains_new_family_provenance_even_when_drawing_is_removed(self):
        before = self.w.history.capture(self.w.db)
        result = retain(self.w, family=family()); record = self.record(result)
        self.w.history.record(before, self.w.db)
        self.assertTrue(self.w.history.restore(self.w))
        self.assertEqual(self.w.models(), {})
        self.assertEqual(self.record(result), record)
        self.assertTrue(self.w.history.restore(self.w, redo=True))
        self.assertIn(result['resolved']['root'], self.w.models()); self.assertEqual(self.record(result), record)

    def test_v7_roundtrip_revalidates_exact_semantic_relationships(self):
        result = retain(self.w, family=family())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'family.vtw'; self.w.save(path)
            with closing(sqlite3.connect(path)) as db:
                self.assertEqual(db.execute('PRAGMA user_version').fetchone()[0], 7)
            opened = Workspace.load(path)
            try:
                self.assertEqual(opened.get_spec(**dict(kind='specialization', sid=result['specialization']['id'])), self.record(result))
                self.assertEqual(opened.models(), self.w.models()); self.assertFalse(opened.dirty)
            finally: opened.db.close()

    def test_old_header_cannot_hide_new_typed_records(self):
        retain(self.w, family=family(), materialize=False)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'wrong-header.vtw'; self.w.save(path)
            for version in (5, 6):
                with closing(sqlite3.connect(path)) as db: db.execute('PRAGMA user_version=' + str(version))
                before = path.read_bytes()
                with self.subTest(version=version), self.assertRaisesRegex(GraphError, 'version-7'):
                    Workspace.load(path)
                self.assertEqual(path.read_bytes(), before)

    def test_v6_migration_preserves_disk_and_all_existing_revision_hashes(self):
        ref = self.w.freeze_model(self._insert_concrete())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'old.vtw'; self.w.save(path)
            with closing(sqlite3.connect(path)) as db: db.execute('PRAGMA user_version=6')
            before = path.read_bytes(); hashes = self.w.list_specs(all_revisions=True)
            opened = Workspace.load(path)
            try:
                self.assertEqual(path.read_bytes(), before)
                self.assertEqual(opened.db.execute('PRAGMA user_version').fetchone()[0], VERSION)
                self.assertEqual(opened.list_specs(all_revisions=True), hashes)
                self.assertEqual(opened.get_spec('model', ref['id'], ref['revision']), self.w.get_spec('model', ref['id'], ref['revision']))
                opened.save(path)
            finally: opened.db.close()
            with closing(sqlite3.connect(path)) as db: self.assertEqual(db.execute('PRAGMA user_version').fetchone()[0], 7)

    def _insert_concrete(self):
        doc = default_model(); self.w.put(doc); return doc['id']

    def test_semantic_forgery_with_valid_content_hash_is_rejected_at_load(self):
        result = retain(self.w, family=family(), materialize=False)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'forged.vtw'; self.w.save(path)
            with closing(sqlite3.connect(path)) as db, db:
                db.execute('PRAGMA foreign_keys=ON'); db.execute('DROP TRIGGER spec_revisions_no_update')
                db.execute("DELETE FROM spec_heads WHERE kind='specialization'")
                row = db.execute("SELECT document FROM spec_revisions WHERE kind='specialization'").fetchone()[0]
                wrong = json.loads(row); wrong['bindings']['hidden-width'] = 5; wrong.pop('revision')
                wrong['revision'] = digest(wrong)
                db.execute("UPDATE spec_revisions SET document=?,revision=? WHERE kind='specialization'",
                           (json.dumps(wrong), wrong['revision']))
                db.execute(SCHEMA['spec_revisions_no_update'])
            before = path.read_bytes()
            with self.assertRaisesRegex(GraphError, 'does not match'):
                Workspace.load(path)
            self.assertEqual(path.read_bytes(), before)

    def test_summaries_and_ordinal_views_are_bounded_exact_and_readonly(self):
        result = retain(self.w, family=family(), materialize=False)
        before, rng = self.state(), random.getstate()
        for kind in ('family', 'specialization'):
            ref = result[kind]
            page = index_page(self.w, kind, preferred=ref)
            self.assertEqual(page['rows'][0]['revision'], ref['revision'])
            small = summary(self.w, ref)
            self.assertLessEqual(len(encoded(small)), MAX_BYTES)
            self.assertEqual(small['reference'], ref)
            for value in (page, small, inspect_page(self.w, ref)):
                self.assertLessEqual(len(encoded(value)), MAX_BYTES)
        self.assertEqual(summary(self.w, result['specialization'])['counts']['resolved_bindings'], 2)
        self.assertEqual(summary(self.w, result['family'])['counts']['repeats'], 1)
        self.assertEqual(self.state(), before); self.assertEqual(random.getstate(), rng)

    def test_cli_exact_stored_preview_and_import_preserve_hash_and_disk_atomicity(self):
        ref = self.w.put_spec('family', family())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'agent.vtw'; self.w.save(path); before = path.read_bytes()
            code, preview = self.cli(path, 'preview-stored-family', ref['id'], ref['revision'])
            self.assertEqual(code, 0, preview); self.assertEqual(preview['value']['status'], 'verified')
            self.assertEqual(path.read_bytes(), before)
            code, bad = self.cli(path, 'preview-stored-family', ref['id'], '0'*64)
            self.assertEqual(code, 1, bad); self.assertEqual(path.read_bytes(), before)
            edits = [{'op':'specialize-family','family_ref':ref,'bindings':{'hidden-width':4}}]
            with patch('vpw_cli.read_json', return_value=edits):
                code, reply = self.cli(path, 'apply', '-')
            self.assertEqual(code, 0, reply)
            retained = reply['value']['edits'][0]
            code, show = self.cli(path, 'spec-show', 'specialization', retained['specialization']['id'])
            self.assertEqual(code, 0, show); self.assertEqual(show['value']['bindings'], {'hidden-width':4,'hidden-depth':2})
            saved = path.read_bytes()
            edits[0]['bindings']['hidden-width'] = 6
            with patch('vpw_cli.read_json', return_value=edits):
                code, stale = self.cli(path, '--expect-revision', digest('not-disk'), 'apply', '-')
            self.assertEqual(code, 1, stale); self.assertEqual(path.read_bytes(), saved)
            with patch('vpw_cli.read_json', return_value=edits + [{'op':'not-supported','id':'not-real'}]):
                self.assertEqual(self.cli(path, 'apply', '-')[0], 1)
            self.assertEqual(path.read_bytes(), saved)

    def test_stdlib_storage_and_cli_import_torch_qt_neither(self):
        code = ('import json,sys; from app.core import Workspace; from app.family_records import retain; '
                'w=Workspace();r=retain(w,family=json.load(open("examples/families/width-depth.json"))); '
                'assert len(w.list_specs(all_revisions=True))==3; '
                'assert not any(k=="torch" or k.startswith("PyQt") for k in sys.modules); w.db.close()')
        result = subprocess.run([sys.executable, '-c', code], cwd=ROOT, text=True, capture_output=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
