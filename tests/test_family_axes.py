"""Concrete family sweep coordinates, pure previews and retained execution meaning."""
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

from app.core import GraphError, Workspace, VERSION, default_model, default_mlp
from app.evidence import digest
from app.experiment_objects import (admit_run, candidate, candidate_bundle, dependencies,
    execute_candidate, object_reference, preview_experiment, selection_at, selection_count)
from app.family_axes import at, count, selected, total, validate_choice
from app.revisions import validate_all
from app import authoring, experiment_drafts, candidate_views
from app.task_execution import training_protocol
import vpw_cli

ROOT = Path(__file__).resolve().parents[1]


def family_experiment():
    w = Workspace(); w.set_constrained_mode(False)
    model = default_model('Linear', inputs=1); w.put(model); model_ref = w.freeze_model(model['id'])
    pairs = []
    for i in range(2):
        dataset = {'version': 1, 'id': 'dataset' + str(i), 'name': 'Observations ' + str(i),
            'storage': {'kind': 'inline-records', 'version': 1},
            'fields': [{'id': 'x', 'dtype': 'float32', 'shape': [1]}, {'id': 'y', 'dtype': 'float32', 'shape': [1]}],
            'records': [{'id': 'r' + str(j), 'values': {'x': [float(j)], 'y': [float(2*j+1+i)]}} for j in range(8)]}
        task = {'version': 1, 'id': 'task' + str(i), 'name': 'Predict observations',
            'inputs': [{'port': 'input', 'field': 'x', 'transforms': []}],
            'heads': [{'id': 'prediction', 'port': 'output', 'field': 'y', 'loss': 'mse', 'weight': 1., 'transforms': []}]}
        pairs.append({'id': 'pair' + str(i), 'dataset': w.put_spec('dataset', dataset), 'task': w.put_spec('task', task)})
    e = {'version': 1, 'id': 'experiment', 'name': 'Concrete reference', 'question': 'How does width and depth affect learning?',
         'models': [model_ref], 'data_tasks': pairs,
         'splits': [{'id': 'split-a', 'protocol': {'method': 'row', 'seed': 4, 'fractions': [.5, .25, .25]}}],
         'initialization_seeds': [7, 13], 'protocols': [{'id': 'adam', 'protocol': training_protocol(epochs=None, updates=3, batch_size=3, evaluate_every=2)}],
         'selection': {'kind': 'crossed', 'pairs': [], 'exclusions': []}}
    w.put_spec('experiment', e)
    source = json.loads((ROOT / 'examples/families/width-depth.json').read_text())
    source['models']['mlp']['graph']['nodes'][0]['features'] = 1
    ref = w.put_spec('family', source)
    e.update(version=2, id='family-experiment', models=[{
        'id': 'width-depth', 'family': ref, 'bindings': {}, 'axes': [
            {'symbol': 'hidden-width', 'values': [2, 3]},
            {'symbol': 'hidden-depth', 'values': [1, 2]}]}])
    retained = w.put_spec('experiment', e)
    return w, w.get_spec('experiment', retained['id'], retained['revision'])


class FamilyAxisTests(unittest.TestCase):
    def setUp(self):
        self.w, self.e = family_experiment()
        self.addCleanup(self.w.db.close)
        self.addCleanup(lambda: self.w.history.directory.cleanup() if self.w.history.directory else None)

    def replace(self, **changes):
        doc = copy.deepcopy(self.e); doc.pop('revision'); doc.update(changes)
        ref = self.w.put_spec('experiment', doc)
        return self.w.get_spec('experiment', ref['id'], ref['revision'])

    def state(self): return list(self.w.db.iterdump()), self.w.dirty

    def test_ordered_width_depth_axes_exact_counts_and_last_axis_fastest(self):
        before, rng = self.state(), random.getstate()
        rows = preview_experiment(self.e, self.w.get_spec)['candidates']
        self.assertEqual(len(rows), 16)
        self.assertEqual([rows[i]['coordinates']['model']['bindings'] for i in (0, 4, 8, 12)], [
            {'hidden-width': 2, 'hidden-depth': 1}, {'hidden-width': 2, 'hidden-depth': 2},
            {'hidden-width': 3, 'hidden-depth': 1}, {'hidden-width': 3, 'hidden-depth': 2}])
        self.assertEqual([rows[i]['parameters_exact'] for i in (0, 4, 8, 12)], ['13', '19', '22', '34'])
        self.assertEqual({r['status'] for r in rows}, {'compatible'})
        self.assertEqual(len({r['id'] for r in rows}), 16)
        self.assertEqual(len({r['references']['model']['revision'] for r in rows}), 4)
        self.assertEqual(rows[0]['manifest'], rows[4]['manifest'])
        self.assertEqual(self.state(), before); self.assertEqual(random.getstate(), rng)
        self.assertEqual(self.w.list_specs('specialization'), [])

    def test_paging_and_direct_coordinates_reproduce_without_skipped_specialization(self):
        from app.families import specialize
        with patch('app.families.specialize', wraps=specialize) as called:
            page = preview_experiment(self.e, self.w.get_spec, offset=12, limit=1)
            self.assertEqual(called.call_count, 1)
        self.assertEqual(page['candidates'][0], candidate(self.e, selection_at(self.e, 12), self.w.get_spec))
        all_rows = preview_experiment(self.e, self.w.get_spec)['candidates']
        first = preview_experiment(self.e, self.w.get_spec, limit=5)
        next_page = preview_experiment(self.e, self.w.get_spec, offset=first['next_offset'], limit=5)
        self.assertEqual(first['candidates'] + next_page['candidates'], all_rows[:10])

    def test_intrinsic_axes_are_closed_distinct_exact_and_explicit(self):
        base = self.e['models'][0]
        for mutation in ('boolean', 'float', 'duplicate', 'empty', 'unknown-key', 'overlap', 'duplicate-symbol', 'text'):
            choice = copy.deepcopy(base)
            if mutation == 'boolean': choice['axes'][0]['values'] = [True]
            if mutation == 'float': choice['axes'][0]['values'] = [2.0]
            if mutation == 'duplicate': choice['axes'][0]['values'] = [2, 2]
            if mutation == 'empty': choice['axes'][0]['values'] = []
            if mutation == 'unknown-key': choice['solver'] = 'infer'
            if mutation == 'overlap': choice['bindings']['hidden-width'] = 2
            if mutation == 'duplicate-symbol': choice['axes'] *= 2
            if mutation == 'text': choice['axes'][0]['values'] = ['eval(2)']
            with self.subTest(mutation=mutation), self.assertRaises(GraphError): validate_choice(choice)

    def test_unknown_symbols_missing_family_and_duplicate_selection_ids_fail_admission(self):
        for mutation in ('unknown', 'missing', 'duplicate'):
            models = copy.deepcopy(self.e['models'])
            if mutation == 'unknown': models[0]['axes'][0]['symbol'] = 'absent'
            if mutation == 'missing': models[0]['family']['revision'] = '0' * 64
            if mutation == 'duplicate': models.append({**models[0], 'axes': []})
            before = self.state()
            with self.subTest(mutation=mutation), self.assertRaises(GraphError): self.replace(models=models)
            self.assertEqual(self.state(), before)

    def test_coordinate_membership_does_not_infer_omit_replace_or_accept_boolean(self):
        coordinate = selection_at(self.e, 0)
        for value in ({'id': 'absent', 'bindings': {}}, {'id': 'width-depth', 'bindings': {}},
                      {'id': 'width-depth', 'bindings': {'hidden-width': 4, 'hidden-depth': 1}},
                      {'id': 'width-depth', 'bindings': {'hidden-width': 2, 'hidden-depth': True}},
                      {'id': 'width-depth', 'bindings': {'hidden-width': 2, 'hidden-depth': 1}, 'extra': 0}):
            with self.subTest(value=value), self.assertRaises(GraphError):
                candidate(self.e, {**coordinate, 'model': value}, self.w.get_spec)

    def test_fixed_binding_and_defaults_are_distinct_and_fully_retained_in_provenance(self):
        choice = copy.deepcopy(self.e['models'][0]); choice.update(bindings={'hidden-depth': 2}, axes=choice['axes'][:1])
        e = self.replace(models=[choice]); coord = selection_at(e, 0)
        self.assertEqual(coord['model']['bindings'], {'hidden-depth': 2, 'hidden-width': 2})
        bad = copy.deepcopy(coord); bad['model']['bindings']['hidden-depth'] = 1
        with self.assertRaisesRegex(GraphError, 'fixed'): candidate(e, bad, self.w.get_spec)
        choice['bindings'] = {}; e = self.replace(models=[choice]); row = preview_experiment(e, self.w.get_spec)['candidates'][0]
        self.assertEqual(row['parameters_exact'], '19')
        self.assertEqual(set(row['coordinates']['model']['bindings']), {'hidden-width'})
        self.assertIn('specialization', row['references'])

    def test_unresolved_points_stay_conditional_and_out_of_bounds_stay_incompatible(self):
        source = self.w.get_spec('family', self.e['models'][0]['family']['id']); source.pop('revision')
        del source['symbols'][1]['default']; ref = self.w.put_spec('family', source)
        choice = copy.deepcopy(self.e['models'][0]); choice.update(family=ref, axes=choice['axes'][:1])
        e = self.replace(models=[choice]); rows = preview_experiment(e, self.w.get_spec)['candidates']
        self.assertEqual({r['status'] for r in rows}, {'conditional'})
        self.assertTrue(all(r['manifest'] is None and 'model' not in r['references'] for r in rows))
        choice['axes'].append({'symbol': 'hidden-depth', 'values': [0, 2]}); e = self.replace(models=[choice])
        rows = preview_experiment(e, self.w.get_spec)['candidates']
        self.assertEqual((rows[0]['status'], rows[4]['status']), ('incompatible', 'compatible'))
        self.assertIn('bounds', rows[0]['reason'])

    def test_pairings_and_exclusions_use_exact_family_coordinates_and_keep_reasons(self):
        point = selection_at(self.e, 0)['model']
        e = self.replace(selection={'kind': 'paired', 'pairs': [{'model': point, 'data_task': 'pair0'}],
            'exclusions': [{'model': point, 'data_task': 'pair0', 'reason': 'Explicit control'}]})
        rows = preview_experiment(e, self.w.get_spec)['candidates']
        self.assertEqual({r['status'] for r in rows}, {'excluded'})
        self.assertEqual(rows[0]['reason'], 'Explicit control'); self.assertIn('pairing', rows[4]['reason'])
        self.assertEqual(set(rows[0]['references']), {'family', 'dataset', 'task'})
        self.assertEqual(self.w.list_specs('specialization'), [])

    def test_exact_old_family_is_not_replaced_by_editing_or_archiving_head(self):
        before = preview_experiment(self.e, self.w.get_spec)
        source = self.w.get_spec('family', self.e['models'][0]['family']['id']); source.pop('revision')
        source['symbols'][1]['default'] = 4; source['name'] = 'Changed newer family'
        self.w.put_spec('family', source); self.w.archive_spec('family', source['id'])
        self.assertEqual(preview_experiment(self.e, self.w.get_spec), before)

    def test_model_axis_can_mix_frozen_choices_without_changing_legacy_identity(self):
        model = self.w.list_specs('model')[0]
        ref = {k: model[k] for k in ('kind', 'id', 'revision')}
        e = self.replace(models=[ref, *self.e['models']])
        self.assertEqual(selection_count(e), 20)
        self.assertEqual(selection_at(e, 0)['model'], ref)
        self.assertEqual(selection_at(e, 4)['model'], selection_at(self.e, 0)['model'])
        old = self.w.get_spec('experiment', 'experiment')
        rows = preview_experiment(old, self.w.get_spec)['candidates']
        self.assertNotIn('parameters_exact', rows[0]); self.assertEqual(old['version'], 1)
        with self.assertRaises(GraphError): self.replace(version=1)

    def test_arbitrarily_large_product_count_and_last_point_do_not_enumerate(self):
        choice = copy.deepcopy(self.e['models'][0]); choice['axes'] = [
            {'symbol': 's' + str(i), 'values': [1, 2]} for i in range(128)]
        validate_choice(choice)
        self.assertEqual(count(choice), 2**128); self.assertEqual(total([choice]), 2**128)
        coord = at([choice], 2**128 - 1)
        self.assertEqual(set(coord['bindings'].values()), {2}); selected([choice], coord)

    def test_huge_model_parameter_counts_are_exact_decimal_projections_without_allocation(self):
        choice = copy.deepcopy(self.e['models'][0]); choice['axes'][0]['values'] = [2**53-1]
        e = self.replace(models=[choice]); row = preview_experiment(e, self.w.get_spec, limit=1)['candidates'][0]
        width = 2**53-1
        self.assertEqual(row['parameters_exact'], str(width**2 + 4*width + 1))
        self.assertEqual(row['status'], 'compatible')
        native = candidate_views.stored(self.w, object_reference('experiment', e), limit=1)
        self.assertEqual(native['candidates'][0]['parameters_exact'], row['parameters_exact'])
        self.assertEqual(self.w.list_specs('specialization'), [])

    def test_typed_draft_pages_edit_family_choices_without_partial_admission(self):
        draft = authoring.begin(self.w, 'experiment', base=object_reference('experiment', self.e))
        header = {'session': 'owned', 'sequence': 0, 'id': draft['id'], 'workspace_id': draft['workspace_id']}
        page = experiment_drafts.page(draft['document'], header, 'models')
        self.assertEqual([p['key'] for p in page['rows'][0]['properties']], ['id', 'family', 'bindings', 'axes'])
        doc = experiment_drafts.edit(draft['document'], {'op': 'property', 'target': {'kind': 'entry', 'collection': 'models', 'ordinal': 0},
            'key': 'axes', 'text': '[{"symbol":"hidden-width","values":[3,2]}]'})
        draft['document'] = doc; authoring.check(self.w, draft)
        preview = experiment_drafts.preview(self.w, draft, header)
        self.assertEqual(preview['candidates'][0]['parameters_exact'], '34')
        self.assertEqual(preview['candidates'][0]['coordinates']['model']['bindings'], {'hidden-width': 3})

    def test_actual_owned_session_edits_v2_and_explicitly_upgrades_v1_atomically(self):
        from app.authoring_sessions import Sessions
        owner = object(); store = Sessions(); self.addCleanup(store.close_owner, owner)
        header = store.open(self.w, owner, 'experiment', base=object_reference('experiment', self.e))
        result = store.experiment_edit(self.w, header['session'], owner, 0, [{
            'op': 'property', 'target': {'kind': 'entry', 'collection': 'models', 'ordinal': 0}, 'key': 'id', 'text': '"renamed-choice"'}])
        self.assertEqual(result['sequence'], 1)
        prospective = store.check(self.w, header['session'], owner, 1); self.assertEqual(prospective['kind'], 'experiment')
        store.close(header['session'], owner)
        old = self.w.get_spec('experiment', 'experiment')
        header = store.open(self.w, owner, 'experiment', base=object_reference('experiment', old))
        page = store.experiment_page(header['session'], owner, 0, 'models')
        operations = [{'op': 'property', 'target': {'kind': 'document'}, 'key': 'version', 'text': '2'},
            {'op': 'insert-json', 'path': page['path'], 'index': page['total'], 'text': json.dumps(self.e['models'][0])}]
        result = store.experiment_edit(self.w, header['session'], owner, 0, operations)
        self.assertEqual(result['sequence'], 1)
        self.assertEqual(store.experiment_preview(self.w, header['session'], owner, 1)['total_combinations_exact'], '20')
        with self.assertRaises(GraphError): store.experiment_edit(self.w, header['session'], owner, 1, [{
            'op': 'property', 'target': {'kind': 'document'}, 'key': 'version', 'text': '1'}])
        self.assertEqual(store.describe(header['session'], owner)['sequence'], 1)

    def test_missing_worker_run_retains_derived_receipt_without_editable_drawing(self):
        row = preview_experiment(self.e, self.w.get_spec)['candidates'][0]
        drawings = set(self.w.models())
        result = execute_candidate(self.w, object_reference('experiment', self.e), row['coordinates'], 'no-worker', 'Missing worker', python='/missing/family-test-worker')
        record = result['record']; self.assertEqual(record['version'], 2)
        self.assertEqual((record['attempts'][0]['stage'], record['attempts'][0]['status']), ('worker', 'failed'))
        refs = record['resolved']['references']; self.assertEqual(refs, row['references'])
        specialization = self.w.get_spec('specialization', refs['specialization']['id'], refs['specialization']['revision'])
        self.assertEqual(specialization['bindings'], {'hidden-width': 2, 'hidden-depth': 1})
        self.assertEqual(set(self.w.models()), drawings); validate_all(self.w.db)
        from app.run_views import run_view
        view = run_view(self.w, result['reference'])
        self.assertEqual(view['objects']['specialization']['reference'], refs['specialization'])
        self.assertEqual(view['objects']['model']['parameters_exact'], '13')

    def test_external_completion_reconstructs_and_atomically_rejects_forged_provenance(self):
        row = preview_experiment(self.e, self.w.get_spec)['candidates'][0]
        other = Workspace(); self.addCleanup(other.db.close)
        from app.experiment_ui import capture
        snap = capture(self.w, object_reference('experiment', self.e), row['coordinates'], 'external', 'External')
        for entry in snap['documents']: other.put_spec(entry['kind'], entry['document'])
        result = execute_candidate(other, snap['experiment'], snap['coordinates'], 'external', 'External', python='/missing/family-test-worker')
        before = self.state(); bad = copy.deepcopy(result['record']); bad.pop('revision')
        bad['resolved']['references']['specialization']['revision'] = '0' * 64
        with self.assertRaises(GraphError): admit_run(self.w, bad)
        self.assertEqual(self.state(), before)
        # Later validation failure rolls back even newly admitted provenance.
        bad = copy.deepcopy(result['record']); bad.pop('revision'); bad['resolved']['protocol']['seeds']['shuffle'] += 1
        with self.assertRaises(GraphError): admit_run(self.w, bad)
        self.assertEqual(self.state(), before)
        response = vpw_cli.apply_edits(self.w, [{'op': 'admit-run', 'document': result['record'], 'select': False}])
        self.assertEqual(response[0]['reference'], result['reference'])
        self.assertEqual(self.w.list_specs('run'), [])
        self.assertEqual(self.w.get_spec('run', result['reference']['id'], result['reference']['revision']), result['record'])
        self.assertTrue(self.w.list_specs('specialization')); validate_all(self.w.db)

    def test_conditional_run_is_terminal_unavailable_without_phantom_model(self):
        source = self.w.get_spec('family', self.e['models'][0]['family']['id']); source.pop('revision'); del source['symbols'][1]['default']
        choice = copy.deepcopy(self.e['models'][0]); choice.update(family=self.w.put_spec('family', source), axes=choice['axes'][:1])
        e = self.replace(models=[choice]); point = selection_at(e, 0)
        result = execute_candidate(self.w, object_reference('experiment', e), point, 'unbound', 'Unbound')
        record = result['record']; self.assertIsNone(record['resolved'])
        self.assertEqual((record['attempts'][0]['stage'], record['attempts'][0]['status']), ('compatibility', 'unavailable'))
        self.assertEqual(self.w.list_specs('specialization'), []); validate_all(self.w.db)
        from app.run_views import run_view
        view = run_view(self.w, result['reference'])
        self.assertNotIn('model', view['objects']); self.assertIsNone(view['execution']); self.assertIn('family', view['objects'])

    def test_v8_roundtrip_and_genuine_v7_migration_preserve_exact_revisions(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'family.vtw'; self.w.save(path)
            with closing(Workspace.load(path).db) as db:
                self.assertEqual(db.execute('PRAGMA user_version').fetchone()[0], 8)
                validate_all(db)
            with closing(sqlite3.connect(path)) as db: db.execute('PRAGMA user_version=7'); db.commit()
            before = path.read_bytes()
            with self.assertRaisesRegex(GraphError, 'version-8'): Workspace.load(path)
            self.assertEqual(path.read_bytes(), before)
            old = Workspace(); self.addCleanup(old.db.close)
            for dep in dependencies('experiment', self.e): old.put_spec(dep['kind'], self.w.get_spec(dep['kind'], dep['id'], dep['revision']))
            path = Path(td) / 'genuine-v7.vtw'; old.save(path)
            with closing(sqlite3.connect(path)) as db: db.execute('PRAGMA user_version=7'); db.commit()
            before = path.read_bytes(); loaded = Workspace.load(path); self.addCleanup(loaded.db.close)
            self.assertEqual(path.read_bytes(), before); self.assertEqual(loaded.db.execute('PRAGMA user_version').fetchone()[0], VERSION)

    def test_cli_exact_preview_matches_bounded_native_projection(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'family.vtw'; self.w.save(path)
            stream = io.StringIO()
            with redirect_stdout(stream):
                result = vpw_cli.main([str(path), 'preview-experiment-page', self.e['id'], self.e['revision'], '--offset', '8', '--limit', '2'])
            self.assertEqual(result, 0)
            self.assertEqual(json.loads(stream.getvalue())['value'], candidate_views.stored(self.w, object_reference('experiment', self.e), offset=8, limit=2))

    def test_cli_missing_worker_run_persists_exact_family_provenance_without_host_torch(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'family.vtw'; self.w.save(path)
            coordinates = Path(td) / 'coordinates.json'; coordinates.write_text(json.dumps(selection_at(self.e, 0)))
            stream = io.StringIO()
            with redirect_stdout(stream):
                code = vpw_cli.main([str(path), 'run-candidate', '--experiment', self.e['id'], self.e['revision'],
                    '--coordinates', str(coordinates), '--run-id', 'cli-trial', '--name', 'CLI family trial', '--torch-python', '/missing/family-test-worker'])
            self.assertEqual(code, 0); value = json.loads(stream.getvalue())['value']; self.assertTrue(value['persisted'])
            loaded = Workspace.load(path); self.addCleanup(loaded.db.close)
            run = loaded.get_spec('run', 'cli-trial'); self.assertEqual(run['attempts'][0]['stage'], 'worker')
            ref = run['resolved']['references']['specialization']
            self.assertEqual(loaded.get_spec(ref['kind'], ref['id'], ref['revision'])['bindings'], {'hidden-width': 2, 'hidden-depth': 1})

    def test_preview_fresh_process_imports_no_numerical_or_qt_runtime(self):
        source = 'import sys; sys.path.insert(0,"tests"); from test_family_axes import family_experiment; from app.experiment_objects import preview_experiment; w,e=family_experiment(); preview_experiment(e,w.get_spec,limit=1); assert not any(n in sys.modules for n in ("torch","numpy","PyQt6")); w.db.close()'
        result = subprocess.run([sys.executable, '-c', source], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
