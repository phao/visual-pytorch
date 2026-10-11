"""Authoring is read-only until validated admission, with explicit stale guards."""
import copy
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.authoring import begin, check, save
from app.core import GraphError, Workspace, default_model
from app.task_execution import training_protocol


def measurements(sid='observations'):
    return {'version': 1, 'id': sid, 'name': 'Measurements',
        'storage': {'kind': 'inline-records', 'version': 1},
        'fields': [{'id': 'x', 'dtype': 'float32', 'shape': [1], 'axes': ['F']},
                   {'id': 'y', 'dtype': 'float32', 'shape': [1]}],
        'records': [{'id': 'row' + str(i), 'values': {'x': [float(i)], 'y': [2. * i + 1.]}} for i in range(8)],
        'provenance': {'procedure': 'independent deterministic measurements'}, 'usage': {'license': 'test-only'}}


class AuthoringTests(unittest.TestCase):
    def setUp(self):
        self.w = Workspace(); self.addCleanup(self.w.db.close)
        self.addCleanup(lambda: self.w.history.directory.cleanup() if self.w.history.directory else None)

    def draft(self):
        result = begin(self.w, 'dataset', sid='observations'); result['document'] = measurements(); return result

    def rejection(self, draft, match=None):
        before = self.w.db.serialize(); dirty = self.w.dirty
        with self.assertRaisesRegex(GraphError, match or '.'):
            save(self.w, draft)
        self.assertEqual(before, self.w.db.serialize()); self.assertEqual(dirty, self.w.dirty)

    def test_new_templates_all_three_kinds_are_incomplete_readonly_and_owned(self):
        before = self.w.db.serialize(); rng = random.getstate()
        for kind in ('dataset', 'task', 'experiment'):
            draft = begin(self.w, kind)
            self.assertEqual(draft['kind'], kind); self.assertIsNone(draft['base']); self.assertIsNone(draft['expected_head'])
            self.assertNotIn('revision', draft['document']); self.rejection(draft)
        self.assertEqual(before, self.w.db.serialize()); self.assertEqual(rng, random.getstate())

    def test_check_is_pure_then_save_matches_the_prospective_hash(self):
        draft = self.draft(); before = self.w.db.serialize(); rng = random.getstate()
        with patch('app.datasets._prepare_column', side_effect=AssertionError('No fitting')):
            ref = check(self.w, draft)
        self.assertEqual(before, self.w.db.serialize()); self.assertEqual(rng, random.getstate())
        self.assertEqual(ref, save(self.w, draft)); self.assertNotIn('revision', draft['document'])
        self.assertEqual(self.w.get_spec('dataset', ref['id'])['revision'], ref['revision'])
        draft['document']['records'][0]['values']['x'][0] = 9
        self.assertEqual(self.w.get_spec('dataset', ref['id'])['records'][0]['values']['x'], [0.])

    def test_edit_is_private_keeps_exact_base_and_preserves_metadata(self):
        original = save(self.w, self.draft()); before = self.w.db.serialize()
        draft = begin(self.w, 'dataset', base=original)
        self.assertEqual(draft['base'], original); self.assertEqual(draft['expected_head'], original['revision'])
        self.assertEqual(draft['document'], measurements()); self.assertEqual(before, self.w.db.serialize())
        draft['document']['name'] = 'Changed'; new = save(self.w, draft)
        self.assertNotEqual(original['revision'], new['revision'])
        self.assertEqual(self.w.get_spec('dataset', original['id'], original['revision'])['name'], 'Measurements')
        self.assertEqual(self.w.get_spec('dataset', new['id'])['usage'], {'license': 'test-only'})

    def test_noop_edit_is_no_sql_change_or_dirty_flag(self):
        ref = save(self.w, self.draft()); draft = begin(self.w, 'dataset', base=ref)
        self.w.dirty = False; before = self.w.db.serialize(); changes = self.w.db.total_changes
        self.assertEqual(save(self.w, draft), ref); self.assertEqual(self.w.db.total_changes, changes)
        self.assertEqual(self.w.db.serialize(), before); self.assertFalse(self.w.dirty)

    def test_changed_head_rejects_without_losing_draft_or_revision(self):
        original = save(self.w, self.draft()); draft = begin(self.w, 'dataset', base=original)
        other = begin(self.w, 'dataset', base=original); other['document']['name'] = 'Concurrent revision'; save(self.w, other)
        draft['document']['name'] = 'Unsaved text'; before = copy.deepcopy(draft)
        self.rejection(draft, 'head changed'); self.assertEqual(draft, before)
        self.assertEqual(len(self.w.list_specs('dataset', True)), 2)

    def test_historical_base_is_not_silently_replaced_by_current_head(self):
        original = save(self.w, self.draft()); edit = begin(self.w, 'dataset', base=original)
        edit['document']['records'][0]['values']['y'][0] = 100.; newer = save(self.w, edit)
        old = begin(self.w, 'dataset', base=original)
        self.assertEqual(old['expected_head'], newer['revision']); self.assertEqual(old['document']['records'][0]['values']['y'], [1.])
        old['document']['name'] = 'Historical branch'; saved = save(self.w, old)
        self.assertEqual(self.w.get_spec('dataset', saved['id'])['records'][0]['values']['y'], [1.])

    def test_archived_base_edit_is_explicit_but_new_cannot_shadow_retained_id(self):
        ref = save(self.w, self.draft()); self.w.archive_spec('dataset', ref['id'])
        with self.assertRaisesRegex(GraphError, 'already retained'): begin(self.w, 'dataset', sid=ref['id'])
        draft = begin(self.w, 'dataset', base=ref); self.assertIsNone(draft['expected_head'])
        draft['document']['name'] = 'Restored'; save(self.w, draft)
        self.assertEqual(len(self.w.list_specs('dataset', True)), 2)

    def test_new_draft_collision_after_other_admission_and_archive_is_rejected(self):
        draft = self.draft(); save(self.w, copy.deepcopy(draft)); self.w.archive_spec('dataset', draft['id'])
        self.rejection(draft, 'already retained')

    def test_workspace_switch_rejects_without_admitting_to_either_database(self):
        draft = self.draft(); other = Workspace(); self.addCleanup(other.db.close)
        before = other.db.serialize(); original = self.w.db.serialize()
        with self.assertRaisesRegex(GraphError, 'another workspace'): save(other, draft)
        self.assertEqual(other.db.serialize(), before); self.assertEqual(self.w.db.serialize(), original)

    def test_malformed_envelopes_and_forged_identity_reject_atomically(self):
        cases = [lambda d: d.update(version=True), lambda d: d.update(version=2), lambda d: d.update(kind=[]),
            lambda d: d.update(kind='run'), lambda d: d.update(id='different'), lambda d: d.update(extra=True),
            lambda d: d.update(expected_head='not-a-hash'), lambda d: d['document'].update(revision='0'*64),
            lambda d: d.update(base={'kind':'task','id':d['id'],'revision':'0'*64}), lambda d: d.update(document=[])]
        for change in cases:
            draft = self.draft(); change(draft)
            with self.subTest(draft=draft): self.rejection(draft)
        for kind in ('model', 'report', 'run', None):
            with self.assertRaises(GraphError): begin(self.w, kind)

    def test_typed_dataset_covers_ragged_missing_category_bool_string_and_metadata(self):
        draft = self.draft(); d = draft['document']
        d['fields'] += [{'id':'sequence','dtype':'float32','shape':[-1,2],'axes':['T','F'],'nullable':True,'unit':'K'},
                       {'id':'label','dtype':'category','shape':[],'vocabulary':{'version':1,'values':['1',1,'unused']}},
                       {'id':'flag','dtype':'bool','shape':[]}, {'id':'subject','dtype':'string','shape':[]},
                       {'id':'time','dtype':'int64','shape':[]}]
        for i, row in enumerate(d['records']):
            row['values'].update(sequence=[[None,1.]]*(i+1), label=1 if i%2 else '1', flag=bool(i%2), subject='s'+str(i//2), time=i)
        ref = save(self.w, draft); result = self.w.get_spec('dataset', ref['id'])
        self.assertEqual(result['fields'], d['fields']); self.assertEqual(result['records'], d['records'])
        self.assertEqual(result['fields'][3]['vocabulary']['values'], ['1',1,'unused'])

    def test_dataset_errors_are_not_coerced_or_partially_inserted(self):
        cases = [lambda d: d['fields'][0].update(shape=[2]), lambda d: d['fields'][0].update(axes=['F','C']),
            lambda d: d['records'][0]['values'].update(x=[None]), lambda d: d['records'][0]['values'].update(x=[float('inf')]),
            lambda d: d['records'][1].update(id='row0'), lambda d: d['fields'][0].update(id='silently-renamed'),
            lambda d: d['fields'].append(copy.deepcopy(d['fields'][0])), lambda d: d['records'][0]['values'].update(unknown=0)]
        for change in cases:
            draft = self.draft(); change(draft['document'])
            with self.subTest(change=change): self.rejection(draft)

    def test_task_and_experiment_drafts_use_existing_grammar_and_exact_dependencies(self):
        dataset = save(self.w, self.draft()); model = default_model(inputs=1); self.w.put(model); mref = self.w.freeze_model(model['id'])
        task = begin(self.w, 'task', sid='prediction')
        task['document'].update(inputs=[{'port':'input','field':'x','transforms':[{'kind':'standardize'}]}],
            heads=[{'id':'prediction','port':'output','field':'y','loss':'mse','weight':1.,'transforms':[]}])
        tref = save(self.w, task)
        experiment = begin(self.w, 'experiment', sid='comparison')
        experiment['document'].update(question='What does the measured relation show?', models=[mref],
            data_tasks=[{'id':'measurements','dataset':dataset,'task':tref}],
            splits=[{'id':'held-out','protocol':{'method':'row','seed':3,'fractions':[.5,.25,.25]}}],
            initialization_seeds=[5,7], protocols=[{'id':'adam','protocol':training_protocol(epochs=None,updates=3)}])
        with patch('app.datasets._prepare_column',side_effect=AssertionError('No fitting')):
            eref = save(self.w, experiment)
        old = begin(self.w, 'experiment', base=eref); old['document']['models'][0]['revision']='0'*64
        self.rejection(old, 'does not exist')
        self.assertEqual(self.w.get_spec('experiment', eref['id'])['models'], [mref])

    def test_undo_retains_new_bytes_and_restores_head_save_reload_preserves_all(self):
        ref = save(self.w, self.draft()); edit = begin(self.w, 'dataset', base=ref); edit['document']['name']='Edited'
        before = self.w.history.capture(self.w.db); new = save(self.w, edit); self.w.history.record(before, self.w.db)
        self.assertTrue(self.w.history.restore(self.w)); self.assertEqual(self.w.get_spec('dataset', ref['id'])['revision'], ref['revision'])
        self.assertEqual(self.w.get_spec('dataset', new['id'], new['revision'])['name'], 'Edited')
        self.assertTrue(self.w.history.restore(self.w, redo=True))
        with tempfile.TemporaryDirectory() as folder:
            self.w.save(Path(folder)/'author.vtw'); loaded=Workspace.load(Path(folder)/'author.vtw')
            try: self.assertEqual(loaded.get_spec('dataset', ref['id'])['revision'], new['revision']); self.assertEqual(len(loaded.list_specs('dataset',True)),2)
            finally: loaded.db.close()
