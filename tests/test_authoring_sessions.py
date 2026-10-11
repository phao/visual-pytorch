"""Owned drafts, bounded transport and transactional patches; stdlib host only."""
import copy
import random
import unittest
from unittest.mock import patch

from app import authoring, science_browser
from app.authoring_sessions import Sessions
from app.core import GraphError, Workspace
from test_authoring import measurements


class AuthoringSessionTests(unittest.TestCase):
    def setUp(self):
        self.w = Workspace(); self.addCleanup(self.w.db.close)
        self.addCleanup(lambda: self.w.history.directory.cleanup() if self.w.history.directory else None)
        self.store = Sessions(); self.owner = object(); self.other = object()

    def open(self, kind='dataset', **kwargs):
        return self.store.open(self.w, self.owner, kind, **kwargs)

    def dataset(self, large=False):
        doc = measurements()
        if large:
            doc['records'] *= 200
            doc['records'] = [dict(copy.deepcopy(row), id='row'+str(i)) for i, row in enumerate(doc['records'])]
            doc['usage'] = {'long-key-'+'λ'*10000: '<script>literal</script>'+'λ'*8000}
        ref = self.w.put_spec('dataset', doc)
        return self.open(base=ref), ref, doc

    def p(self, info, operations):
        return self.store.patch(self.w, info['session'], self.owner, info['sequence'], operations)

    def path(self, token, sequence, key):
        page = self.store.read(token, self.owner, sequence)
        return [next(row['ordinal'] for row in page['entries'] if row['key']['text'] == key)]

    def test_all_templates_are_owned_incomplete_and_projection_only(self):
        before = self.w.db.serialize(); rng = random.getstate()
        for kind in authoring.KINDS:
            info = self.open(kind)
            self.assertNotIn('document', info); self.assertEqual(info['sequence'], 0)
            self.assertEqual(info['kind'], kind); self.assertIsNone(info['base'])
            self.assertLessEqual(len(science_browser.encoded(info)), science_browser.MAX_BYTES)
            with self.assertRaises(GraphError): self.store.check(self.w, info['session'], self.owner, 0)
            self.store.close(info['session'], self.owner)
        self.assertEqual(before, self.w.db.serialize()); self.assertEqual(rng, random.getstate())

    def test_large_historical_pages_keys_scalar_and_records_keep_exact_order(self):
        info, ref, doc = self.dataset(large=True)
        newer = copy.deepcopy(doc); newer['name'] = 'New head'; new = self.w.put_spec('dataset', newer)
        # Opening occurred before the head moved: old base/head remain captured.
        self.assertEqual(info['base'], ref); self.assertEqual(info['expected_head'], ref['revision'])
        before = self.w.db.serialize()
        token = info['session']; records = self.path(token, 0, 'records')
        row = self.store.read(token, self.owner, 0, path=records, offset=30, limit=25)
        self.assertEqual([x['ordinal'] for x in row['entries']], list(range(30,55)))
        usage = self.path(token, 0, 'usage'); keypath = usage+[0]
        for part, expected in [('key', next(iter(doc['usage']))), ('value', next(iter(doc['usage'].values())))]:
            chunks = []; offset = 0
            while True:
                page = self.store.read(token, self.owner, 0, path=keypath, part=part, offset=offset)
                self.assertLessEqual(len(science_browser.encoded(page)), 65536)
                self.assertEqual(page['session'], token); self.assertEqual(page['sequence'], 0)
                chunks.append(page['scalar']['text'])
                if page['next_offset'] is None: break
                offset = page['next_offset']
            self.assertEqual(''.join(chunks), expected)
        self.assertEqual(before, self.w.db.serialize())
        with self.assertRaisesRegex(GraphError, 'head changed'): self.store.commit(self.w, token, self.owner, 0)
        self.assertEqual(self.store.describe(token, self.owner), info)
        self.assertEqual(self.w.get_spec('dataset', ref['id'])['revision'], new['revision'])

    def test_ordered_atomic_patch_operations_and_unrelated_hidden_values(self):
        info, _, doc = self.dataset(); token = info['session']
        path = self.path(token, 0, 'records')
        original = copy.deepcopy(doc['records'])
        info = self.p(info, [{'op':'move','path':path,'from':0,'to':7},
            {'op':'insert','path':path,'index':1,'value':{'id':'inserted','values':{'x':[100.], 'y':[201.]}}}])
        row = self.store._sessions[token]['draft']['document']
        self.assertEqual([r['id'] for r in row['records']], ['row1','inserted','row2','row3','row4','row5','row6','row7','row0'])
        self.assertEqual(row['records'][0], original[1]); self.assertEqual(row['records'][-1], original[0])
        self.assertEqual(row['fields'], doc['fields']); self.assertEqual(row['usage'], doc['usage'])
        info = self.p(info, [{'op':'remove','path':path+[1]}, {'op':'put','path':[], 'key':'usage','value':{'literal':'<script>'}}])
        result = self.store.commit(self.w, token, self.owner, info['sequence'])
        retained = self.w.get_spec('dataset', result['id'], result['revision'])
        self.assertEqual(retained['records'], original[1:]+original[:1]); self.assertEqual(retained['usage'], {'literal':'<script>'})
        with self.assertRaises(GraphError): self.store.describe(token, self.owner)

    def test_midbatch_failure_and_header_overflow_do_not_publish_any_change(self):
        info, _, _ = self.dataset(); before = copy.deepcopy(self.store._sessions[info['session']]['draft'])
        cases = [[{'op':'put','path':[],'key':'name','value':'Unsaved'}, {'op':'remove','path':[999]}],
                 [{'op':'put','path':[],'key':'name','value':'λ'*10880}]]
        for index, ops in enumerate(cases):
            with self.subTest(case=index), self.assertRaises(GraphError): self.p(info, ops)
            self.assertEqual(self.store.describe(info['session'], self.owner), info)
            self.assertEqual(self.store._sessions[info['session']]['draft'], before)

    def test_patch_values_are_detached_and_checks_do_not_fit_or_write(self):
        info = self.open(sid='observations'); ops = [{'op':'replace','path':[],'value':measurements()}]
        info = self.p(info, ops); ops[0]['value']['records'][0]['values']['x'][0] = -100.
        before = self.w.db.serialize(); rng = random.getstate()
        with patch('app.datasets._prepare_column', side_effect=AssertionError('No fit')):
            result = self.store.check(self.w, info['session'], self.owner, info['sequence'])
        self.assertEqual(self.store._sessions[info['session']]['draft']['document']['records'][0]['values']['x'], [0.])
        self.assertEqual(before, self.w.db.serialize()); self.assertEqual(rng, random.getstate())
        self.assertEqual(result, self.store.commit(self.w, info['session'], self.owner, info['sequence']))

    def test_cross_owner_tokens_duplicate_open_close_teardown_and_noop_commit(self):
        info, ref, _ = self.dataset(); token = info['session']; self.w.dirty = False; before = self.w.db.serialize()
        for fn in [lambda:self.store.describe(token,self.other), lambda:self.store.read(token,self.other,0),
                   lambda:self.store.patch(self.w,token,self.other,0,[]), lambda:self.store.close(token,self.other),
                   lambda:self.open('task')]:
            with self.assertRaises(GraphError): fn()
        self.assertEqual(self.store.commit(self.w, token, self.owner, 0), ref)
        self.assertEqual(before, self.w.db.serialize()); self.assertFalse(self.w.dirty)
        info = self.open(); other = self.store.open(self.w, self.other, 'task')
        self.store.close_owner(self.owner)
        with self.assertRaises(GraphError): self.store.describe(info['session'], self.owner)
        self.assertEqual(self.store.describe(other['session'], self.other), other)

    def test_stale_sequence_and_malformed_patch_paths_fail_without_mutation(self):
        info = self.open(); token = info['session']; before = self.w.db.serialize()
        for sequence in [None, True, -1, 1, '0']:
            with self.subTest(sequence=sequence), self.assertRaises(GraphError): self.store.read(token, self.owner, sequence)
        for ops in [[], [{}], [{'op':'replace','path':[True],'value':0}], [{'op':'replace','path':[-1],'value':0}],
                    [{'op':'move','path':[],'from':0,'to':0}], [{'op':'put','path':[], 'key':'id','value':'forged'}],
                    [{'op':'put','path':[], 'key':'version','value':True}], [{'op':'put','path':[], 'key':'revision','value':'0'*64}],
                    [{'op':'remove','path':[]}], [{'op':'replace','path':[],'value':None}],
                    [{'op':'put','path':[], 'key':'a','value':float('nan')}], [{'op':'put','path':[], 'key':'a','value':'x'*65536}]]:
            with self.subTest(ops=ops), self.assertRaises(GraphError): self.p(info, ops)
            self.assertEqual(self.store.describe(token, self.owner), info)
        changed = self.p(info, [{'op':'put','path':[],'key':'name','value':'Incomplete but kept'}])
        with self.assertRaisesRegex(GraphError, 'sequence changed'): self.p(info, [{'op':'put','path':[],'key':'name','value':'Stale'}])
        self.assertEqual(changed['sequence'], 1); self.assertEqual(before, self.w.db.serialize())

    def test_actual_workspace_and_connection_identity_not_uuid_fence_mutations(self):
        info, _, _ = self.dataset(); token = info['session']
        other = Workspace(); self.addCleanup(other.db.close)
        uuid_value = authoring.workspace_id(self.w)
        other.db.execute("UPDATE metadata SET value=? WHERE key='workspace_id'", (uuid_value,))
        self.assertEqual(authoring.workspace_id(other), uuid_value)
        before = other.db.serialize(); origin = self.w.db.serialize()
        for fn in [lambda:self.store.patch(other,token,self.owner,0,[{'op':'put','path':[],'key':'name','value':'Wrong'}]),
                   lambda:self.store.check(other,token,self.owner,0), lambda:self.store.commit(other,token,self.owner,0)]:
            with self.assertRaisesRegex(GraphError, 'origin changed'): fn()
        old_db = self.w.db; self.w.db = other.db
        try:
            with self.assertRaisesRegex(GraphError, 'origin changed'): self.store.check(self.w,token,self.owner,0)
        finally: self.w.db = old_db
        self.assertEqual(self.store.describe(token,self.owner), info)
        self.assertEqual(before,other.db.serialize()); self.assertEqual(origin,self.w.db.serialize())

    def test_historical_open_after_new_head_keeps_base_noop_and_invalid_draft_retention(self):
        _, old, doc = self.dataset(); self.store.close_owner(self.owner)
        doc['name'] = 'New head'; new = self.w.put_spec('dataset', doc)
        info = self.open(base=old)
        self.assertEqual(info['base'],old); self.assertEqual(info['expected_head'],new['revision'])
        path = self.path(info['session'], 0, 'fields')
        info = self.p(info, [{'op':'remove','path':path}])
        with self.assertRaises(GraphError): self.store.commit(self.w, info['session'],self.owner,info['sequence'])
        self.assertEqual(self.store.describe(info['session'],self.owner),info)
        self.assertEqual(self.w.get_spec('dataset',old['id'])['revision'],new['revision'])
