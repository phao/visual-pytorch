"""Revision/hash/schema, migration, CLI and undo/redo evidence retention."""
import copy
import io
import json
import re
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from app.core import GraphError, Workspace, VERSION, default_model, default_cnn, generate_dataset, train_model, validate_model
from app.evidence import checkpoint_report, digest, identity
from app.revisions import (SCHEMA, frozen_document, model_document, validate_all,
                           validate_model_document, check_schema)
from test_datasets import data, task
from test_bindings import two_inputs
import vpw_cli


class RevisionTests(unittest.TestCase):
    def setUp(self):
        self.w = Workspace()

    def tearDown(self):
        self.w.db.close()
        if self.w.history.directory is not None:
            self.w.history.directory.cleanup()

    def test_dataset_and_task_are_independent_typed_revisions(self):
        d = data(); t = task(); original = copy.deepcopy((d, t))
        dr = self.w.put_spec('dataset', d); tr = self.w.put_spec('task', t)
        self.assertEqual((d, t), original)
        self.assertEqual(self.w.models(), {})
        self.assertEqual(dr, {'kind': 'dataset', 'id': d['id'], 'revision': digest(d)})
        self.assertEqual(tr['revision'], digest(t))
        copied = self.w.get_spec('dataset', dr['id']); copied['records'][0]['values']['x'][0] = 99.
        self.assertEqual(self.w.get_spec('dataset', dr['id'])['records'][0]['values']['x'], [0.])
        validate_all(self.w.db)

    def test_documented_sql_executes_and_matches_guarded_relational_schema(self):
        directory = Path(__file__).resolve().parents[1] / 'docs'
        for name in ('revision-store.md', 'vtw-format.md'):
            sql = re.findall(r'```sql\n(.*?)```', (directory / name).read_text(), re.S)[0]
            db = sqlite3.connect(':memory:')
            try:
                db.executescript(sql); check_schema(db)
                self.assertEqual(set(r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='trigger'")),
                                 set(SCHEMA) - {'spec_revisions', 'spec_heads'})
            finally: db.close()

    def test_task_intrinsic_grammar_is_checked_but_compatibility_not_guessed(self):
        t = task(); t['inputs'][0]['field'] = 'field-not-yet-in-a-dataset'
        self.w.put_spec('task', t)  # reusable definition; not a claim of compatibility
        for change in ('duplicate', 'loss', 'transform', 'version'):
            bad = copy.deepcopy(t)
            if change == 'duplicate': bad['inputs'] *= 2
            if change == 'loss': bad['heads'][0]['loss'] = 'implicitly-guessed'
            if change == 'transform': bad['inputs'][0]['transforms'] = [{'kind': 'affine', 'scale': 0., 'offset': 1.}]
            if change == 'version': bad['version'] = True
            with self.assertRaises(GraphError): self.w.put_spec('task', bad)
        with self.assertRaises(GraphError): self.w.put_spec('future-experiment', {})

    def test_content_is_idempotent_and_stale_hash_not_an_update_instruction(self):
        first = self.w.put_spec('dataset', data()); before = self.w.db.execute('SELECT * FROM spec_revisions').fetchone()
        self.assertEqual(self.w.put_spec('dataset', data()), first)
        self.assertEqual(self.w.db.execute('SELECT * FROM spec_revisions').fetchall(), [before])
        d = self.w.get_spec('dataset', first['id']); d['name'] = 'Edited name'
        with self.assertRaisesRegex(GraphError, 'revision'): self.w.put_spec('dataset', d)
        d.pop('revision'); second = self.w.put_spec('dataset', d)
        self.assertNotEqual(first['revision'], second['revision'])
        self.assertEqual(self.w.get_spec('dataset', first['id'], first['revision'])['name'], data()['name'])
        self.assertEqual(self.w.get_spec('dataset', first['id'])['name'], 'Edited name')

    def test_browsing_heads_archive_and_select_do_not_delete_bytes(self):
        d = data(); old = self.w.put_spec('dataset', d); d['name'] = 'Another revision'; new = self.w.put_spec('dataset', d)
        self.w.select_spec('dataset', old['id'], old['revision'])
        self.assertEqual(self.w.get_spec('dataset', old['id'])['revision'], old['revision'])
        self.w.archive_spec('dataset', old['id']); self.assertEqual(self.w.list_specs(), [])
        self.assertEqual(len(self.w.list_specs(all_revisions=True)), 2)
        with self.assertRaisesRegex(GraphError, 'head'): self.w.get_spec('dataset', old['id'])
        self.w.select_spec('dataset', new['id'], new['revision'])
        self.assertEqual(self.w.get_spec('dataset', old['id'])['revision'], new['revision'])
        with self.assertRaises(GraphError): self.w.select_spec('dataset', new['id'], '0' * 64)
        self.assertEqual(len(self.w.list_specs(all_revisions=True)), 2)

    def test_sorting_is_alphabetical_only_for_browsing_not_document_arrays(self):
        for name in ('Zulu', 'alpha', 'Beta'):
            d = data(); d['name'] = name; d['id'] = name
            d['fields'].reverse(); d['records'].reverse(); self.w.put_spec('dataset', d)
        self.assertEqual([r['name'] for r in self.w.list_specs()], ['alpha', 'Beta', 'Zulu'])
        self.assertEqual([r['id'] for r in self.w.get_spec('dataset', 'Zulu')['records']], ['r7','r6','r5','r4','r3','r2','r1','r0'])

    def test_sql_update_delete_replace_and_upsert_cannot_overwrite_retained_revision(self):
        ref = self.w.put_spec('dataset', data()); row = self.w.db.execute('SELECT * FROM spec_revisions').fetchone()
        self.w.db.execute('PRAGMA recursive_triggers=OFF')
        commands = [('UPDATE spec_revisions SET name=?', ('Corrupted',)), ('DELETE FROM spec_revisions', ()),
                    ('INSERT OR REPLACE INTO spec_revisions VALUES (?,?,?,?,?,?)', (*row[:3], 'Corrupted', *row[4:])),
                    ('''INSERT INTO spec_revisions VALUES (?,?,?,?,?,?) ON CONFLICT(kind,id,revision) DO UPDATE SET name=excluded.name''', row)]
        for sql, args in commands:
            with self.subTest(sql=sql), self.assertRaisesRegex(sqlite3.IntegrityError, 'Immutable'): self.w.db.execute(sql, args)
            self.w.db.rollback()
            self.assertEqual(self.w.get_spec('dataset', ref['id'])['name'], row[3])

    def test_head_foreign_keys_are_composite_and_enforced(self):
        ref = self.w.put_spec('dataset', data())
        with self.assertRaises(sqlite3.IntegrityError): self.w.db.execute('INSERT INTO spec_heads VALUES (?,?,?)', ('task', ref['id'], ref['revision']))
        self.w.db.rollback(); validate_all(self.w.db)

    def test_schema_and_hash_corruption_are_rejected_before_save_or_load(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'lab.vtw'; ref = self.w.put_spec('dataset', data()); self.w.save(path); original = path.read_bytes()
            self.w.db.execute('DROP TRIGGER spec_revisions_no_update')
            with self.assertRaisesRegex(GraphError, 'guard'): self.w.save(path)
            self.assertEqual(path.read_bytes(), original)
            with sqlite3.connect(path) as db: db.execute('DROP TRIGGER spec_revisions_no_replace')
            before = path.read_bytes()
            with self.assertRaisesRegex(GraphError, 'guard'): Workspace.load(path)
            self.assertEqual(path.read_bytes(), before)
        # Reinstalling a canonical guard after altering bytes cannot hide hash mismatch.
        w = Workspace()
        try:
            w.put_spec('dataset', data()); w.db.execute('DROP TRIGGER spec_revisions_no_update')
            row = w.db.execute('SELECT document FROM spec_revisions').fetchone()[0]; doc = json.loads(row); doc['records'][0]['values']['x'][0] += 1.
            w.db.execute('UPDATE spec_revisions SET document=?', (json.dumps(doc),)); w.db.execute(SCHEMA['spec_revisions_no_update']); w.db.commit()
            with self.assertRaisesRegex(GraphError, 'revision'): validate_all(w.db)
        finally: w.db.close()

    def test_loaded_relational_identity_duplicate_json_and_head_corruption_reject(self):
        for mutation in ('row-name', 'row-id', 'bad-time', 'duplicate-key', 'head'):
            with tempfile.TemporaryDirectory() as td:
                path = Path(td) / 'lab.vtw'; w = Workspace(); w.put_spec('dataset', data()); w.save(path); w.db.close()
                with sqlite3.connect(path) as db:
                    if mutation == 'head':
                        db.execute('PRAGMA foreign_keys=OFF'); db.execute('UPDATE spec_heads SET revision=?', ('0' * 64,))
                    else:
                        db.execute('DROP TRIGGER spec_revisions_no_update')
                        if mutation == 'row-name': db.execute("UPDATE spec_revisions SET name='wrong'")
                        if mutation == 'row-id': db.execute("UPDATE spec_revisions SET id='wrong'")
                        if mutation == 'bad-time': db.execute("UPDATE spec_revisions SET created_at='yesterday'")
                        if mutation == 'duplicate-key':
                            raw = db.execute('SELECT document FROM spec_revisions').fetchone()[0]
                            db.execute('UPDATE spec_revisions SET document=?', ('{"id":"overridden",' + raw[1:],))
                        db.execute(SCHEMA['spec_revisions_no_update'])
                before = path.read_bytes()
                with self.subTest(mutation=mutation), self.assertRaises(GraphError): Workspace.load(path)
                self.assertEqual(path.read_bytes(), before)

    def test_frozen_closure_survives_architecture_delete_and_keeps_scientific_hash(self):
        w = Workspace.demo()
        try:
            root = next(d for d in w.models().values() if d['name'] == 'Composed linear model')
            child = next(n['ref'] for n in root['graph']['nodes'] if n['type'] == 'Module')
            scientific = identity(w.models(), root['id'])['architecture']; ref = w.freeze_model(root['id'])
            frozen = w.get_spec('model', ref['id'], ref['revision'])
            self.assertEqual(set(frozen['models']), {root['id'], child}); self.assertEqual(frozen['architecture_hash'], scientific)
            self.assertEqual(set(frozen['models'][root['id']]), {'id', 'name', 'graph'})
            self.assertTrue(all('x' not in n and 'width' not in n and 'collapsed' not in n for m in frozen['models'].values() for n in m['graph']['nodes']))
            w.delete(root['id']); w.delete(child)
            self.assertEqual(w.get_spec('model', ref['id'], ref['revision']), frozen)
            validate_model(frozen['models'], frozen['root']); validate_all(w.db)
        finally: w.db.close()

    def test_model_presentation_does_not_create_revision_but_name_and_computation_do(self):
        d = default_cnn(); self.w.put(d); first = self.w.freeze_model(d['id'])
        d['graph']['nodes'][1].update(x=100, y=200, width=100, height=80, collapsed=True); self.w.put(d)
        self.assertEqual(self.w.freeze_model(d['id']), first)
        original = self.w.get_spec('model', first['id'], first['revision'])
        d['name'] = 'Renamed CNN'; self.w.put(d); renamed = self.w.freeze_model(d['id'])
        self.assertNotEqual(first['revision'], renamed['revision'])
        self.assertEqual(self.w.get_spec('model', renamed['id'])['architecture_hash'], original['architecture_hash'])
        d['graph']['nodes'][1]['bias'] = False; self.w.put(d); edited = self.w.freeze_model(d['id'])
        self.assertNotEqual(self.w.get_spec('model', edited['id'])['architecture_hash'], original['architecture_hash'])

    def test_frozen_model_invalid_closure_and_hash_fail_without_tensor_allocation(self):
        d = default_model(); doc = model_document({d['id']: d}, d['id'])
        for mutation in ('hash', 'unknown-version', 'root', 'presentation', 'unrelated', 'packing', 'foreign-doc'):
            bad = copy.deepcopy(doc)
            if mutation == 'hash': bad['architecture_hash'] = '0' * 64
            if mutation == 'unknown-version': bad['version'] = True
            if mutation == 'root': bad['root'] = 'missing'
            if mutation == 'presentation': bad['models'][d['id']]['graph']['nodes'][0]['x'] = 1
            if mutation == 'unrelated':
                extra = default_model(); bad['models'][extra['id']] = {'id': extra['id'], 'name': extra['name'], 'graph': extra['graph']}
            if mutation == 'packing': bad['models'][d['id']]['graph'].pop('input_order')
            if mutation == 'foreign-doc': bad['models'][d['id']]['dataset'] = {'x': [[1.]], 'y': [[1.]]}
            with self.subTest(mutation=mutation), self.assertRaises(GraphError): validate_model_document(bad)

    def test_actual_v1_to_v4_schema_migration_is_readonly_and_preserves_legacy_checkpoint(self):
        import torch
        torch.set_num_threads(1)
        for version in (1, 2, 3, 4):
            with tempfile.TemporaryDirectory() as td:
                path = Path(td) / 'old.vtw'; w = Workspace(); d = default_model(inputs=1, outputs=1); d['training']['epochs'] = 2
                w.put(d); d['dataset'] = generate_dataset(w.models(), d['id'], samples=16); w.put(d)
                result = train_model(w.models(), d['id'])
                d = w.models()[d['id']]; d['result'] = result; w.put(d); w.record_run(d['id'], result, d['graph']); w.save(path); w.db.close()
                with sqlite3.connect(path) as db:
                    db.execute('DROP TABLE spec_heads'); db.execute('DROP TABLE spec_revisions'); db.execute('PRAGMA user_version=' + str(version))
                    prior = list(db.execute('SELECT * FROM runs'))
                before = path.read_bytes(); loaded = Workspace.load(path)
                try:
                    self.assertEqual(path.read_bytes(), before); self.assertEqual(loaded.db.execute('PRAGMA user_version').fetchone()[0], VERSION)
                    self.assertEqual(list(loaded.db.execute('SELECT * FROM runs')), prior)
                    self.assertEqual(checkpoint_report(loaded.models(), d['id'])['status'], 'current')
                    self.assertEqual(loaded.list_specs(all_revisions=True), [])
                    loaded.save(path)
                finally: loaded.db.close()
                with sqlite3.connect(path) as db:
                    self.assertEqual(db.execute('PRAGMA user_version').fetchone()[0], VERSION)
                    self.assertEqual(list(db.execute('SELECT * FROM runs')), prior)

    def test_partial_schema_or_unknown_kind_cannot_hide_under_old_header(self):
        for mutation in ('partial', 'unknown'):
            with tempfile.TemporaryDirectory() as td:
                path = Path(td) / 'lab.vtw'; w = Workspace(); w.save(path); w.db.close()
                with sqlite3.connect(path) as db:
                    db.execute('PRAGMA user_version=4')
                    if mutation == 'partial': db.execute('DROP TRIGGER spec_revisions_no_delete')
                    else: db.execute('INSERT INTO spec_revisions VALUES (?,?,?,?,?,?)', ('custom', 'id', '0' * 64, 'name', '{}', '2026-10-10T00:00:00Z'))
                before = path.read_bytes()
                with self.assertRaises(GraphError): Workspace.load(path)
                self.assertEqual(path.read_bytes(), before)

    def test_actual_legacy_table_migration_pins_two_input_meaning_before_new_revisions(self):
        for version in (1, 2):
            with tempfile.TemporaryDirectory() as td:
                path = Path(td) / 'old.vtw'; w = Workspace(); d = two_inputs(); w.put(d); w.save(path); w.db.close()
                with sqlite3.connect(path) as db:
                    doc = json.loads(db.execute('SELECT document FROM architectures').fetchone()[0])
                    doc['graph'].pop('input_order'); doc.pop('dataset_binding')
                    db.execute('UPDATE architectures SET document=?', (json.dumps(doc),))
                    db.execute('DROP TABLE spec_heads'); db.execute('DROP TABLE spec_revisions'); db.execute('PRAGMA user_version=' + str(version))
                before = path.read_bytes(); loaded = Workspace.load(path)
                try:
                    self.assertEqual(path.read_bytes(), before)
                    stored = loaded.models()[d['id']]
                    self.assertEqual(stored['graph']['input_order'], ['b', 'a'])
                    self.assertEqual(stored['dataset_binding']['inputs'][0]['id'], 'b')
                    self.assertEqual(stored['dataset'], d['dataset'])
                    validate_all(loaded.db)
                finally: loaded.db.close()

    def test_roundtrip_head_changes_do_not_alias_documents_or_other_kinds(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'lab.vtw'; d = data(); t = task(); t['id'] = d['id']
            dr = self.w.put_spec('dataset', d); tr = self.w.put_spec('task', t); self.w.save(path)
            loaded = Workspace.load(path)
            try:
                self.assertEqual(loaded.get_spec('dataset', dr['id'])['revision'], dr['revision'])
                self.assertEqual(loaded.get_spec('task', tr['id'])['revision'], tr['revision'])
                self.assertNotEqual(dr['revision'], tr['revision'])
                validate_all(loaded.db)
            finally: loaded.db.close()

    def test_undo_and_redo_restore_heads_but_keep_new_immutable_revisions(self):
        first = self.w.put_spec('dataset', data())
        before = self.w.history.capture(self.w.db); d = data(); d['name'] = 'Revision two'; second = self.w.put_spec('dataset', d)
        self.w.history.record(before, self.w.db)
        t = task(); task_ref = self.w.put_spec('task', t)  # arrived after the edit snapshot
        self.assertTrue(self.w.history.restore(self.w))
        self.assertEqual(self.w.get_spec('dataset', first['id'])['revision'], first['revision'])
        self.assertEqual(self.w.get_spec('dataset', second['id'], second['revision'])['name'], 'Revision two')
        self.assertEqual(self.w.get_spec('task', task_ref['id'], task_ref['revision'])['name'], t['name'])
        with self.assertRaises(GraphError): self.w.get_spec('task', task_ref['id'])
        self.assertEqual(len(self.w.list_specs(all_revisions=True)), 3)
        self.assertTrue(self.w.history.restore(self.w, redo=True))
        self.assertEqual(self.w.get_spec('dataset', first['id'])['revision'], second['revision'])
        self.assertEqual(self.w.get_spec('task', task_ref['id'])['revision'], task_ref['revision'])
        validate_all(self.w.db)

    def test_undo_model_creation_keeps_frozen_closure_after_authoring_graph_removed(self):
        before = self.w.history.capture(self.w.db); d = default_model(); self.w.put(d); ref = self.w.freeze_model(d['id'])
        self.w.history.record(before, self.w.db); self.assertTrue(self.w.history.restore(self.w))
        self.assertEqual(self.w.models(), {}); frozen = self.w.get_spec('model', ref['id'], ref['revision'])
        validate_model(frozen['models'], frozen['root'])

    def test_real_cli_stores_revisions_freezes_model_and_replays_exact_selection(self):
        with tempfile.TemporaryDirectory() as td:
            directory = Path(td); path = directory / 'lab.vtw'; d = default_model(inputs=1, outputs=1)
            edits = [{'op': 'create', 'document': d}, {'op': 'put-spec', 'kind': 'dataset', 'document': data()},
                     {'op': 'put-spec', 'kind': 'task', 'document': task()}, {'op': 'freeze-model', 'id': d['id']}]
            batch = directory / 'edits.json'; batch.write_text(json.dumps(edits)); split = directory / 'split.json'
            split.write_text(json.dumps({'method': 'row', 'seed': 8, 'fractions': [.5, .25, .25]}))
            probe = "import sys,vpw_cli; code=vpw_cli.main(sys.argv[1:]); assert not any(k in sys.modules for k in ('torch','numpy','PyQt6')); raise SystemExit(code)"
            def cli(*args):
                proc = subprocess.run([sys.executable, '-c', probe, str(path), *map(str, args)], capture_output=True, text=True, timeout=10)
                return proc.returncode, json.loads(proc.stdout)
            self.assertEqual(cli('init')[0], 0); code, output = cli('apply', batch); self.assertEqual(code, 0, output)
            refs = {r['reference']['kind']: r['reference'] for r in output['value']['edits'] if 'reference' in r}
            args = ['prepare-stored']
            for kind in ('model', 'dataset', 'task'): args += ['--' + kind, refs[kind]['id'], refs[kind]['revision']]
            args += ['--split', str(split), '--memory-limit-percent', '10']
            before = path.read_bytes(); code, prepared = cli(*args); self.assertEqual(code, 0, prepared); self.assertEqual(path.read_bytes(), before)
            self.assertEqual(prepared['value']['references'], refs)
            changed = data(); changed['records'][0]['values']['x'][0] = 1000.; batch.write_text(json.dumps([{'op': 'put-spec', 'kind': 'dataset', 'document': changed}]))
            self.assertEqual(cli('apply', batch)[0], 0); code, replay = cli(*args); self.assertEqual(code, 0, replay)
            self.assertEqual(replay, prepared)
            self.assertEqual(len(cli('spec-list', '--all-revisions')[1]['value']), 4)
            self.assertEqual(cli('spec-show', 'dataset', refs['dataset']['id'], '--revision', refs['dataset']['revision'])[1]['value']['records'][0]['values']['x'], [0.])

    def test_cli_revision_batches_are_atomic_and_never_bypass_spec_validation(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'lab.vtw'; self.w.save(path); before = path.read_bytes()
            for invalid in ({'op': 'put-spec', 'kind': 'dataset', 'document': {'version': 1}},
                            {'op': 'select-spec', 'reference': {'kind': 'dataset', 'id': data()['id'], 'revision': '0' * 64}},
                            {'op': 'freeze-model', 'id': 'missing'}):
                edits = [{'op': 'put-spec', 'kind': 'dataset', 'document': data()}, invalid]
                output = io.StringIO()
                with patch('vpw_cli.read_json', return_value=edits), redirect_stdout(output): code = vpw_cli.main([str(path), 'apply', '-', '--allow-invalid'])
                self.assertEqual(code, 1, output.getvalue()); self.assertEqual(path.read_bytes(), before)
            output = io.StringIO()
            with redirect_stdout(output): code = vpw_cli.main([str(path), '--expect-revision', '0' * 64, 'spec-list'])
            self.assertEqual(code, 1); self.assertEqual(path.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
