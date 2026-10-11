"""Bounded projections of window-owned recovery evidence, not durable jobs.

Full snapshots/results remain in the native collection and explicit export.
Projection limits do not bound its native storage/copy/search costs.
"""
import uuid

from .datasets import _require
from .science_browser import MAX_BYTES, _bounded, _page, _text, encoded, inspect_value


def header(window):
    if not hasattr(window, '_experiment_recovery_scope'):
        window._experiment_recovery_scope = uuid.uuid4().hex
        window._experiment_recovery_sequence = 0
    return {'version': 1, 'scope': 'experiment-recovery-header',
            'scope_id': window._experiment_recovery_scope,
            'sequence': window._experiment_recovery_sequence,
            'total': len(getattr(window, 'experiment_recoveries', {})),
            'byte_limit': MAX_BYTES}


def changed(window):
    header(window)
    window._experiment_recovery_sequence += 1
    return header(window)


def _identity(window, scope_id, sequence):
    h = header(window)
    _require(scope_id == h['scope_id'] and type(sequence) is int and sequence == h['sequence'],
             'Recovery selection changed. Refresh its session pages.')
    return h


def _id(value):
    _require(isinstance(value, str) and len(value) == 32 and all(c in '0123456789abcdef' for c in value),
             'Choose an exact recovery evidence ID.')


def _row(sid, value):
    return {'id': sid, 'run_id': _text(value['snapshot']['run_id'], limit=128),
            'error': _text(value['error'], limit=128),
            'has_result': value['result'] is not None,
            'exported': bool(value.get('exported_to'))}


def notification(window, sid):
    h = header(window)
    return _bounded({**h, 'scope': 'experiment-recovery-notification',
                     'recovery': _row(sid, window.experiment_recoveries[sid])})


def page(window, scope_id, sequence, *, offset=0, limit=25, query='', pin=None):
    h = _identity(window, scope_id, sequence)
    _page(offset, limit)
    _require(isinstance(query, str) and len(query) <= 200 and '\0' not in query,
             'Recovery search needs at most 200 literal characters.')
    if pin is not None:
        _id(pin)
    values = getattr(window, 'experiment_recoveries', {})
    needle = query.casefold()
    selected = sorted((sid for sid, value in values.items()
                       if needle in value['snapshot']['run_id'].casefold() or needle in value['error'].casefold()),
                      key=lambda sid: (values[sid]['snapshot']['run_id'].casefold(), sid))
    result = {**h, 'scope': 'experiment-recovery-page', 'offset': offset, 'limit': limit,
              'query': query, 'total': len(selected), 'rows': [], 'pin': pin,
              'pinned': _row(pin, values[pin]) if pin in values else None,
              'missing_pin': pin if pin is not None and pin not in values else None,
              'next_offset': None}
    for sid in selected[offset:offset + limit]:
        result['rows'].append(_row(sid, values[sid]))
        end = offset + len(result['rows'])
        result['next_offset'] = end if end < len(selected) else None
        if len(encoded(result)) > MAX_BYTES:
            result['rows'].pop()
            break
    end = offset + len(result['rows'])
    result['next_offset'] = end if end < len(selected) else None
    _require(bool(result['rows']) or offset >= len(selected), 'Recovery entry cannot fit its page.')
    return _bounded(result)


def inspect(window, scope_id, sequence, sid, **arguments):
    h = _identity(window, scope_id, sequence)
    _id(sid)
    value = getattr(window, 'experiment_recoveries', {}).get(sid)
    _require(value is not None, 'Unknown recovery evidence ID.')
    return inspect_value(value, {'scope': 'experiment-recovery-inspection',
                                'scope_id': h['scope_id'], 'sequence': sequence, 'id': sid},
                         **arguments)


def completion(workspace_id, reference, record):
    last = record['attempts'][-1]
    return _bounded({'version': 1, 'scope': 'experiment-completion-notification',
                     'workspace_id': workspace_id, 'reference': dict(reference),
                     'attempt': {key: last[key] for key in ('number', 'status', 'stage')}})
