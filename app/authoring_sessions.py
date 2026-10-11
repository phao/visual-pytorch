"""Owned transient draft transport; no alternate SQLite or execution state.

Session and sequence identify mutable editor bytes; a content revision identifies
retained science. They are deliberately different. Complete validation remains
the existing authoring contract. This service bounds wire projections/patches,
not whole-document parsing, copies, validation or universal memory usage.
"""
import copy
import itertools
import uuid

from . import authoring, science_browser
from .core import GraphError
from .datasets import _integer, _keys, _require
from .revisions import _parse

_ANY_SEQUENCE = object()


def _json(value):
    try:
        data = science_browser.encoded(value)
    except (ValueError, TypeError, OverflowError, RecursionError) as error:
        raise GraphError('Draft patches must be finite JSON.') from error
    _require(len(data) <= science_browser.MAX_BYTES, 'Draft patch exceeds its 65536-byte compact JSON limit.')
    # Round trip excludes Python-only objects/aliases; duplicate keys have already
    # been rejected by the bridge's strict JSON decoder.
    return _parse(data.decode('ascii'))


def _path(path):
    _require(isinstance(path, list) and len(path) <= 1024 and all(_integer(i) for i in path),
             'Draft paths require at most 1024 exact nonnegative ordinals.')


def _child(value, index):
    _require(isinstance(value, (dict, list)) and index < len(value), 'Draft ordinal does not exist.')
    return next(itertools.islice(value, index, index + 1)) if isinstance(value, dict) else index


def _at(document, path):
    for index in path:
        document = document[_child(document, index)]
    return document


def _change(document, path, mutate):
    """Copy touched containers only; untargeted scientific arrays stay in order."""
    result = document.copy() if isinstance(document, (dict, list)) else document
    if not path:
        return mutate(result)
    key = _child(document, path[0])
    result[key] = _change(document[key], path[1:], mutate)
    return result


def _operation(document, op):
    _require(isinstance(op, dict) and isinstance(op.get('op'), str), 'Choose a draft patch operation.')
    kind = op['op']; path = op.get('path'); _path(path)
    if kind == 'replace':
        _keys(op, ('op', 'path', 'value'), context='Replace draft value')
        return _change(document, path, lambda _: op['value'])
    if kind == 'remove':
        _keys(op, ('op', 'path'), context='Remove draft value')
        _require(path, 'Cannot remove the draft document.')
        def remove(parent):
            key = _child(parent, path[-1]); del parent[key]; return parent
        return _change(document, path[:-1], remove)
    if kind == 'put':
        _keys(op, ('op', 'path', 'key', 'value'), context='Put draft property')
        _require(isinstance(op['key'], str), 'Draft property keys must be strings.')
        def put(parent):
            _require(isinstance(parent, dict), 'Put needs a dictionary container.')
            parent[op['key']] = op['value']; return parent
        return _change(document, path, put)
    if kind in ('insert', 'move'):
        _keys(op, ('op', 'path', 'index', 'value') if kind == 'insert' else ('op', 'path', 'from', 'to'),
              context='Draft array operation')
        def array(parent):
            _require(isinstance(parent, list), 'Array operation needs an array container.')
            if kind == 'insert':
                _require(_integer(op['index'], 0, len(parent)), 'Insert index must be within the array.')
                parent.insert(op['index'], op['value'])
            else:
                _require(_integer(op['from']) and op['from'] < len(parent) and
                         _integer(op['to']) and op['to'] < len(parent), 'Move indices must exist in the array.')
                parent.insert(op['to'], parent.pop(op['from']))
            return parent
        return _change(document, path, array)
    raise GraphError('Unsupported draft patch operation.')


class Sessions:
    """One active draft per owner; actual Workspace/connection identity fences edits.

    The Qt shell owns this store. Different tabs cannot use each other's tokens.
    Explicit close/owner teardown discards drafts; failed checks retain bytes.
    It is not a durable detached/crash recovery mechanism.
    """
    def __init__(self):
        self._sessions = {}

    def open(self, workspace, owner, kind, *, base=None, sid=None):
        _require(not any(row['owner'] is owner for row in self._sessions.values()),
                 'Close the owner\'s existing unsaved draft explicitly before opening another.')
        draft = authoring.begin(workspace, kind, base=base, sid=sid)
        token = uuid.uuid4().hex
        row = {'owner': owner, 'workspace': workspace, 'connection': workspace.db,
               'draft': draft, 'sequence': 0}
        result = self._header(token, row)  # Check projection before retaining it.
        self._sessions[token] = row
        return result

    def _get(self, token, owner, sequence=_ANY_SEQUENCE, workspace=None):
        _require(isinstance(token, str), 'Draft session requires its exact token.')
        row = self._sessions.get(token)
        _require(row is not None and row['owner'] is owner, 'Unknown draft session for this owner.')
        if sequence is not _ANY_SEQUENCE:
            _require(_integer(sequence) and sequence == row['sequence'], 'Draft sequence changed; read the current page before editing.')
        if workspace is not None:
            _require(row['workspace'] is workspace and row['connection'] is workspace.db and
                     row['draft']['workspace_id'] == authoring.workspace_id(workspace),
                     'Draft origin changed. Inspect or close it; do not admit it into another working copy.')
        return row

    def _header(self, token, row):
        draft = row['draft']
        return science_browser._bounded({'version': 1, 'scope': 'owned-authoring-session',
            'session': token, 'sequence': row['sequence'],
            **{key: copy.deepcopy(value) for key, value in draft.items() if key != 'document'},
            'name': draft['document'].get('name'), 'byte_limit': science_browser.MAX_BYTES})

    def describe(self, token, owner):
        return self._header(token, self._get(token, owner))

    def read(self, token, owner, sequence, **page):
        row = self._get(token, owner, sequence)
        return science_browser.inspect_value(row['draft']['document'],
            {'scope': 'owned-draft-inspection-page', 'session': token, 'sequence': row['sequence']}, **page)

    def patch(self, workspace, token, owner, sequence, operations):
        row = self._get(token, owner, sequence, workspace)
        operations = _json(operations)
        _require(isinstance(operations, list) and 1 <= len(operations) <= 25, 'Draft patch needs 1..25 ordered operations.')
        document = row['draft']['document']
        for operation in operations:
            document = _operation(document, operation)
        _require(isinstance(document, dict) and document.get('id') == row['draft']['id'] and
                 type(document.get('version')) is int and document['version'] == 1 and 'revision' not in document,
                 'Draft patches must preserve stable document ID/version and cannot forge a revision.')
        prospective = dict(row, sequence=row['sequence'] + 1,
                           draft=dict(row['draft'], document=document))
        result = self._header(token, prospective)
        self._sessions[token] = prospective  # Publish only after every operation/projection check.
        return result

    def check(self, workspace, token, owner, sequence):
        row = self._get(token, owner, sequence, workspace)
        return authoring.check(workspace, row['draft'])

    def commit(self, workspace, token, owner, sequence):
        row = self._get(token, owner, sequence, workspace)
        result = authoring.save(workspace, row['draft'])
        del self._sessions[token]  # Successful admission ends this editor session.
        return result

    def close(self, token, owner):
        self._get(token, owner)
        del self._sessions[token]
        return {'session': token, 'status': 'discarded-unsaved-draft'}

    def close_owner(self, owner):
        for token in [token for token, row in self._sessions.items() if row['owner'] is owner]:
            del self._sessions[token]
