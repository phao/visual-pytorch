"""Projection guarantees for transient recovery evidence, without Qt/Torch."""
import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app import recovery_views as V
from app import science_browser as B


def value(sid, run_id='run'):
    return {'version': 1, 'format': 'visual-pytorch-experiment-recovery', 'id': sid,
            'snapshot': {'run_id': run_id, 'documents': [{'hidden': ['λ😀' * 10000],
                                                        'precise': 9007199254740993}]},
            'result': {'weights': ['λ😀' * 10000]}, 'error': 'diagnostic ' + 'λ😀' * 10000,
            'exported_to': None}


class RecoveryViewTests(unittest.TestCase):
    def setUp(self):
        self.w = SimpleNamespace(experiment_recoveries={})
        for i in range(80):
            sid = f'{i:032x}'
            self.w.experiment_recoveries[sid] = value(sid, f'run-{79-i:03}')
        self.h = V.changed(self.w)
        self.before = copy.deepcopy(self.w.experiment_recoveries)

    def page(self, **kw):
        return V.page(self.w, self.h['scope_id'], self.h['sequence'], **kw)

    def test_header_and_notification_omit_large_contents_without_encoding_them(self):
        with patch.object(B, 'encoded', wraps=B.encoded) as encoding:
            header = V.header(self.w)
            notice = V.notification(self.w, f'{0:032x}')
        self.assertLess(len(B.encoded(header)), 400)
        self.assertLess(len(B.encoded(notice)), 4000)
        self.assertNotIn('snapshot', notice)
        self.assertEqual(notice['recovery']['error']['total'], len(self.before[f'{0:032x}']['error']))
        self.assertIsNotNone(notice['recovery']['error']['next_offset'])
        self.assertTrue(all('documents' not in str(call.args[0]) for call in encoding.call_args_list))
        self.assertEqual(self.w.experiment_recoveries, self.before)

    def test_metadata_search_is_alphabetical_with_independent_outside_missing_pins(self):
        pin = f'{0:032x}'
        page = self.page(query='RUN-000', pin=pin)
        self.assertEqual([r['id'] for r in page['rows']], [f'{79:032x}'])
        self.assertEqual(page['pinned']['id'], pin)
        self.assertNotIn(pin, [r['id'] for r in page['rows']])
        missing = 'f' * 32
        result = self.page(query='absent', pin=missing)
        self.assertEqual(result['rows'], [])
        self.assertEqual(result['missing_pin'], missing)
        self.assertIsNone(result['pinned'])
        self.assertEqual(self.w.experiment_recoveries, self.before)

    def test_actual_byte_shortened_cursors_exhaust_every_row_once(self):
        seen = []
        with patch.object(V, 'MAX_BYTES', 8000), patch.object(B, 'MAX_BYTES', 8000):
            cursor = 0
            while cursor is not None:
                page = self.page(offset=cursor, pin=f'{0:032x}')
                self.assertLessEqual(len(B.encoded(page)), 8000)
                self.assertLess(len(page['rows']), 25)
                seen.extend(r['id'] for r in page['rows'])
                cursor = page['next_offset']
        self.assertEqual(seen, [f'{i:032x}' for i in range(79, -1, -1)])
        self.assertEqual(len(set(seen)), 80)

    def test_foreign_or_stale_sessions_and_boolean_sequences_fail_without_mutation(self):
        foreign = V.header(SimpleNamespace())
        for scope, seq in [(foreign['scope_id'], self.h['sequence']), (self.h['scope_id'], True),
                           (self.h['scope_id'], self.h['sequence'] + 1)]:
            with self.subTest(scope=scope, sequence=seq), self.assertRaisesRegex(ValueError, 'Refresh'):
                V.page(self.w, scope, seq)
        V.changed(self.w)
        with self.assertRaisesRegex(ValueError, 'Refresh'):
            self.page()
        self.assertEqual(self.w.experiment_recoveries, self.before)

    def test_projection_arguments_reject_invalid_types_ranges_and_unrepresentable_pages(self):
        for kw in [{'offset': True}, {'offset': -1}, {'limit': 26}, {'limit': False},
                   {'query': 'x' * 201}, {'query': '\0'}, {'pin': {}}, {'pin': 'wrong'}]:
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                self.page(**kw)
        with patch.object(V, 'MAX_BYTES', 250), patch.object(B, 'MAX_BYTES', 250):
            with self.assertRaises(ValueError):
                self.page()
        self.assertEqual(self.w.experiment_recoveries, self.before)

    def test_bounded_tree_inspection_reconstructs_unicode_and_exact_integer_tokens(self):
        sid = f'{0:032x}'
        def read(path, **kw):
            return V.inspect(self.w, self.h['scope_id'], self.h['sequence'], sid, path=path, **kw)
        # Ordinals follow native insertion order, not UI alphabetical order.
        root = read([])
        snapshot_index = next(r['ordinal'] for r in root['entries'] if r['key']['text'] == 'snapshot')
        snapshot = read([snapshot_index])
        documents = next(r['ordinal'] for r in snapshot['entries'] if r['key']['text'] == 'documents')
        path = [snapshot_index, documents, 0, 0, 0]
        text, cursor = '', 0
        while cursor is not None:
            page = read(path, offset=cursor)
            self.assertLessEqual(len(B.encoded(page)), 65536)
            text += page['scalar']['text']
            cursor = page['next_offset']
        self.assertEqual(text, 'λ😀' * 10000)
        exact = read([snapshot_index, documents, 0, 1])
        self.assertEqual(exact['representation'], 'json-token')
        self.assertEqual(exact['scalar']['text'], '9007199254740993')
        self.assertEqual(self.w.experiment_recoveries, self.before)

    def test_long_keys_are_chunked_and_wrong_ordinal_or_part_is_not_coerced(self):
        sid = f'{0:032x}'
        self.w.experiment_recoveries[sid]['snapshot'] = {'λ😀' * 1500: [None, True]}
        root = V.inspect(self.w, self.h['scope_id'], self.h['sequence'], sid)
        index = next(r['ordinal'] for r in root['entries'] if r['key']['text'] == 'snapshot')
        member = V.inspect(self.w, self.h['scope_id'], self.h['sequence'], sid, path=[index])
        self.assertIsNotNone(member['entries'][0]['key']['next_offset'])
        key = V.inspect(self.w, self.h['scope_id'], self.h['sequence'], sid, path=[index, 0], part='key')
        self.assertEqual(key['scalar']['text'], ('λ😀' * 1500)[:2000])
        for kw in [{'path': [True]}, {'path': [999]}, {'path': [], 'part': 'key'}, {'part': 'unknown'}]:
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                V.inspect(self.w, self.h['scope_id'], self.h['sequence'], sid, **kw)

    def test_export_marker_invalidation_does_not_change_evidence_or_invent_durability(self):
        sid = f'{0:032x}'
        self.w.experiment_recoveries[sid]['exported_to'] = '/very/long/' + 'λ' * 20000
        h = V.changed(self.w)
        page = V.page(self.w, h['scope_id'], h['sequence'])
        self.assertTrue(V.notification(self.w, sid)['recovery']['exported'])
        self.assertNotIn('exported_to', page['rows'][0])
        self.assertNotIn('receipt', V.header(self.w))
        with self.assertRaises(ValueError):
            V.inspect(self.w, self.h['scope_id'], self.h['sequence'], sid)

    def test_completion_contains_only_exact_reference_and_latest_terminal_metadata(self):
        ref = {'kind': 'run', 'id': 'λ😀' * 100, 'revision': 'a' * 64}
        record = {'attempts': [{'number': 1, 'status': 'failed', 'stage': 'worker'},
                              {'number': 2, 'status': 'completed', 'stage': 'training',
                               'result': {'hidden': ['λ😀' * 100000]}}]}
        result = V.completion('workspace', ref, record)
        self.assertEqual(result['reference'], ref)
        self.assertEqual(result['attempt'], {'number': 2, 'status': 'completed', 'stage': 'training'})
        self.assertLess(len(B.encoded(result)), 4000)
        self.assertNotIn('result', result)
        self.assertNotIn('record', result)


if __name__ == '__main__':
    unittest.main()
