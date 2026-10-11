"""Bounded exact-revision read projections; not lazy storage or execution.

Metadata SQL does not fetch documents. Exact summary/tree reads still validate
the entire immutable document/dependency closure through the ordinary registry.
Limits apply to the compact ASCII JSON projection, not total validation costs,
SQLite's sort/offset work, the enclosing bridge or an explicit complete export.
"""
import copy
import itertools
import json

from .datasets import _identifier, _integer, _require
from .experiment_objects import reference
from .revisions import KINDS, _hash

MAX_BYTES = 65536
MAX_ROWS = 25
PREVIEW_CHARS = 128
SCALAR_CHARS = 2000


def encoded(value):
    return json.dumps(value, ensure_ascii=True, separators=(',', ':'), allow_nan=False).encode('ascii')


def _bounded(value):
    _require(len(encoded(value)) <= MAX_BYTES, 'Scientific read projection exceeds its byte limit.')
    return value


def _page(offset, limit):
    _require(_integer(offset) and _integer(limit, 1, MAX_ROWS),
             'Scientific page needs a nonnegative exact offset and limit in 1..25.')


def _text(value, offset=0, limit=SCALAR_CHARS):
    end = min(len(value), offset + limit)
    return {'offset': offset, 'text': value[offset:end], 'total': len(value),
            'next_offset': end if end < len(value) else None}


def _row(row):
    kind, sid, revision, name, created, selected = row
    _require(kind in KINDS and _identifier(sid) and _hash(revision) and
             _identifier(name) and isinstance(created, str) and len(created) <= 200,
             'Invalid scientific metadata row.')
    return {'kind': kind, 'id': sid, 'revision': revision, 'name': name,
            'created_at': created, 'selected': bool(selected)}


def index_page(workspace, kind, *, offset=0, limit=25, query='', preferred=None, heads_only=False):
    """Casefold-sorted SQL window with an independently pinned exact choice."""
    _page(offset, limit)
    _require(isinstance(kind, str) and kind in KINDS, 'Choose a scientific object kind.')
    _require(isinstance(query, str) and len(query) <= 200 and '\0' not in query and
             type(heads_only) is bool, 'Search text must be at most 200 characters; heads_only must be boolean.')
    if preferred is not None:
        reference(preferred, kind)
    db = workspace.db
    db.create_collation('VPW_CASEFOLD', lambda a, b: (a.casefold() > b.casefold()) - (a.casefold() < b.casefold()))
    db.create_function('VPW_CASEFOLD', 1, str.casefold, deterministic=True)
    columns = 'r.kind,r.id,r.revision,r.name,r.created_at,r.revision=h.revision'
    tables = ' FROM spec_revisions r LEFT JOIN spec_heads h ON r.kind=h.kind AND r.id=h.id'
    where = ' WHERE r.kind=? AND (instr(VPW_CASEFOLD(r.name),VPW_CASEFOLD(?))>0 OR instr(VPW_CASEFOLD(r.id),VPW_CASEFOLD(?))>0)'
    if heads_only:
        where += ' AND r.revision=h.revision'
    params = (kind, query, query)
    total = db.execute('SELECT count(*)' + tables + where, params).fetchone()[0]
    pinned = None
    if preferred is not None:
        row = db.execute('SELECT ' + columns + tables + ' WHERE r.kind=? AND r.id=? AND r.revision=?',
                         (kind, preferred['id'], preferred['revision'])).fetchone()
        _require(row is not None, 'Pinned exact revision no longer exists.')
        pinned = _row(row)
    result = {'version': 1, 'scope': 'scientific-metadata-page', 'kind': kind,
              'query': query, 'heads_only': heads_only, 'offset': offset, 'limit': limit,
              'total': total, 'rows': [], 'pinned': pinned, 'next_offset': None,
              'byte_limit': MAX_BYTES}
    rows = db.execute('SELECT ' + columns + tables + where +
                      ' ORDER BY r.name COLLATE VPW_CASEFOLD,r.id,r.revision LIMIT ? OFFSET ?',
                      (*params, limit, offset))
    for raw in rows:
        result['rows'].append(_row(raw))
        end = offset + len(result['rows']); result['next_offset'] = end if end < total else None
        if len(encoded(result)) > MAX_BYTES:
            result['rows'].pop()
            break
    end = offset + len(result['rows']); result['next_offset'] = end if end < total else None
    _require(bool(result['rows']) or offset >= total, 'Metadata entry cannot fit its page.')
    return _bounded(result)


def summary(workspace, ref):
    """Typed summary, deliberately not a partial document accepted by put_spec."""
    _require(isinstance(ref, dict) and ref.get('kind') in KINDS, 'Choose an exact scientific revision.')
    reference(ref, ref['kind'])
    doc = workspace.get_spec(ref['kind'], ref['id'], ref['revision'])
    result = {'version': 1, 'scope': 'exact-spec-summary', 'reference': copy.deepcopy(ref),
              'id': doc['id'], 'name': doc['name'], 'counts': {}, 'byte_limit': MAX_BYTES}
    kind = ref['kind']
    if kind == 'dataset':
        result['storage'] = doc['storage']['kind']
        result['counts'] = {'fields': len(doc['fields']), 'observations': len(doc['records'])}
    elif kind == 'task':
        result['counts'] = {'input_bindings': len(doc['inputs']), 'loss_heads': len(doc['heads'])}
    elif kind == 'model':
        result['root'] = doc['root']; result['architecture_hash'] = doc['architecture_hash']
        result['counts'] = {'architectures': len(doc['models']),
                            'blocks': sum(len(m['graph']['nodes']) for m in doc['models'].values())}
    elif kind == 'experiment':
        result['question'] = _text(doc['question'], limit=1024)
        result['counts'] = {key: len(doc[key]) for key in
                            ('models', 'data_tasks', 'splits', 'initialization_seeds', 'protocols')}
    elif kind == 'run':
        result['selection'] = copy.deepcopy(doc['selection']); result['attempt_count'] = len(doc['attempts'])
        last = doc['attempts'][-1]
        result['latest'] = {key: last[key] for key in ('number', 'stage', 'status', 'started_at', 'finished_at')}
        result['latest']['error'] = None if last['error'] is None else {
            'type': last['error']['type'], 'message': _text(last['error']['message'], limit=1024)}
    else:
        result['mode'] = doc['mode']; result['question'] = _text(doc['question'], limit=1024)
        result['experiment'] = copy.deepcopy(doc['experiment']); result['counts'] = {'selected_runs': len(doc['runs'])}
    return _bounded(result)


def _type(value):
    if isinstance(value, dict): return 'object'
    if isinstance(value, list): return 'array'
    if isinstance(value, str): return 'string'
    if value is None: return 'null'
    if type(value) is bool: return 'boolean'
    return 'number'


def _literal(value):
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, allow_nan=False)


def _preview(value):
    kind = _type(value)
    if kind in ('object', 'array'):
        return {'type': kind, 'size': len(value)}
    return {'type': kind, 'value': _text(_literal(value), limit=PREVIEW_CHARS)}


def inspect_page(workspace, ref, *, path=None, part='value', offset=0, limit=25):
    """One container window or scalar chunk; dictionary paths use stored ordinals.

    A numeric ordinal addresses a dictionary's immutable canonical stored key
    order or an array's unchanged scientific order. Keys themselves can be read
    as chunked scalars with part=key, avoiding huge keys in request/response paths.
    """
    _page(offset, limit)
    _require(isinstance(ref, dict) and ref.get('kind') in KINDS, 'Choose an exact scientific revision.')
    reference(ref, ref['kind'])
    path = [] if path is None else path
    _require(isinstance(path, list) and len(path) <= 1024 and all(_integer(i) for i in path),
             'Inspection path must contain at most 1024 exact nonnegative ordinals.')
    _require(part in ('value', 'key'), 'Inspection part must be value or key.')
    value = workspace.get_spec(ref['kind'], ref['id'], ref['revision']); key = None
    for index in path:
        _require(isinstance(value, (dict, list)) and index < len(value), 'Inspection path does not exist in this exact revision.')
        if isinstance(value, dict):
            key, value = next(itertools.islice(value.items(), index, index + 1))
        else:
            key = None; value = value[index]
    if part == 'key':
        _require(key is not None, 'Only a dictionary member has a key to inspect.')
        value = key
    result = {'version': 1, 'scope': 'exact-spec-inspection-page', 'reference': copy.deepcopy(ref),
              'path': list(path), 'part': part, 'type': _type(value), 'offset': offset,
              'limit': limit, 'byte_limit': MAX_BYTES, 'entries': [], 'scalar': None,
              'total': len(value) if isinstance(value, (dict, list)) else len(_literal(value)), 'next_offset': None}
    if isinstance(value, (dict, list)):
        items = itertools.islice(value.items(), offset, offset + limit) if isinstance(value, dict) else (
            (None, value[i]) for i in range(offset, min(len(value), offset + limit)))
        for index, (label, child) in enumerate(items, offset):
            row = {'ordinal': index, 'key': None if label is None else _text(label, limit=PREVIEW_CHARS),
                   'preview': _preview(child)}
            result['entries'].append(row)
            end = offset + len(result['entries']); result['next_offset'] = end if end < result['total'] else None
            if len(encoded(result)) > MAX_BYTES:
                result['entries'].pop(); break
        end = offset + len(result['entries']); result['next_offset'] = end if end < result['total'] else None
        _require(bool(result['entries']) or offset >= result['total'], 'Inspection entry cannot fit its page.')
    else:
        result['scalar'] = _text(_literal(value), offset, SCALAR_CHARS)
        result['next_offset'] = result['scalar']['next_offset']
        result['representation'] = 'native-string' if isinstance(value, str) else 'json-token'
    return _bounded(result)
