"""Owned transient draft transport; no alternate SQLite or execution state.

Session and sequence identify mutable editor bytes; a content revision identifies
retained science. They are deliberately different. Complete validation remains
the existing authoring contract. This service bounds wire projections/patches,
not whole-document parsing, copies, validation or universal memory usage.
"""
import copy
import codecs
import itertools
import json
import tempfile
import uuid

from . import authoring, science_browser, authoring_memory
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
    def __init__(self, *, memory_limit_percent=10., available_fn=None):
        self._sessions = {}
        self.memory_limit_percent = memory_limit_percent
        self.available_fn = available_fn

    def _admit(self, workspace, document=None, *, excluding=None, source_bytes=0, staging_bytes=0):
        estimated = authoring_memory.estimate(document, source_bytes=source_bytes, staging_bytes=staging_bytes)
        estimated += sum(row['estimate'] for token, row in self._sessions.items() if token != excluding)
        return authoring_memory.admit(workspace, estimated, self.memory_limit_percent, available_fn=self.available_fn)

    def open(self, workspace, owner, kind, *, base=None, sid=None, model=None):
        _require(not any(row['owner'] is owner for row in self._sessions.values()),
                 'Close the owner\'s existing unsaved draft explicitly before opening another.')
        _require(kind in authoring.KINDS, 'Choose a dataset, task, experiment or architecture family draft.')
        source_bytes = 0
        if base is not None:
            authoring.reference(base, kind)
            size = workspace.db.execute('SELECT length(CAST(document AS BLOB)) FROM spec_revisions WHERE kind=? AND id=? AND revision=?',
                (kind, base['id'], base['revision'])).fetchone()
            _require(size is not None, 'Requested immutable revision does not exist.')
            source_bytes = size[0]
        self._admit(workspace, source_bytes=source_bytes)  # Before full base parsing/copy.
        draft = authoring.begin(workspace, kind, base=base, sid=sid, model=model,
                                seed_admit=lambda size: self._admit(workspace, source_bytes=size))
        self._admit(workspace, draft['document'])
        token = uuid.uuid4().hex
        row = {'owner': owner, 'workspace': workspace, 'connection': workspace.db,
               'draft': draft, 'sequence': 0, 'buffer':None,
               'estimate':authoring_memory.estimate(draft['document'])}
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

    def origin(self, workspace, token, owner, sequence):
        return self._header(token, self._get(token, owner, sequence, workspace))

    def read(self, token, owner, sequence, **page):
        row = self._get(token, owner, sequence)
        return science_browser.inspect_value(row['draft']['document'],
            {'scope': 'owned-draft-inspection-page', 'session': token, 'sequence': row['sequence']}, **page)

    def patch(self, workspace, token, owner, sequence, operations):
        row = self._get(token, owner, sequence, workspace)
        _require(row['buffer'] is None, 'Finish or discard the staged value before other draft edits.')
        operations = _json(operations)
        _require(isinstance(operations, list) and 1 <= len(operations) <= 25, 'Draft patch needs 1..25 ordered operations.')
        document = row['draft']['document']
        for operation in operations:
            document = _operation(document, operation)
        return self._publish(workspace, token, row, document)

    def _publish(self, workspace, token, row, document):
        versions = (row['draft']['document']['version'],)
        if row['draft']['kind'] in ('experiment', 'family'): versions = tuple(range(versions[0], 4))
        _require(isinstance(document, dict) and document.get('id') == row['draft']['id'] and
                 type(document.get('version')) is int and document['version'] in versions and 'revision' not in document,
                 'Draft patches must preserve stable document ID/version and cannot forge a revision.')
        self._admit(workspace, document, excluding=token)
        prospective = dict(row, sequence=row['sequence'] + 1, buffer=None,
                           estimate=authoring_memory.estimate(document),
                           draft=dict(row['draft'], document=document))
        result = self._header(token, prospective)
        self._sessions[token] = prospective  # Publish only after every operation/projection check.
        return result

    def check(self, workspace, token, owner, sequence):
        row = self._get(token, owner, sequence, workspace)
        _require(row['buffer'] is None, 'Finish or discard the staged value before validating the draft.')
        self._admit(workspace, row['draft']['document'], excluding=token)
        return authoring.check(workspace, row['draft'])

    def commit(self, workspace, token, owner, sequence):
        row = self._get(token, owner, sequence, workspace)
        _require(row['buffer'] is None, 'Finish or discard the staged value before retaining the draft.')
        self._admit(workspace, row['draft']['document'], excluding=token)
        result = authoring.save(workspace, row['draft'])
        del self._sessions[token]  # Successful admission ends this editor session.
        return result

    def close(self, token, owner):
        row = self._get(token, owner)
        if row['buffer'] is not None: row['buffer']['file'].close()
        del self._sessions[token]
        return {'session': token, 'status': 'discarded-unsaved-draft'}

    def close_owner(self, owner):
        for token in [token for token, row in self._sessions.items() if row['owner'] is owner]:
            self.close(token, owner)

    def _buffer(self, token, owner, sequence, value, revision, workspace=None):
        row = self._get(token, owner, sequence, workspace); buffer = row['buffer']
        _require(buffer is not None and buffer['id'] == value and _integer(revision) and
                 revision == buffer['revision'], 'Staged value identity/revision changed. Read its current fragment.')
        return row, buffer

    def _value_header(self, token, row, buffer):
        return science_browser._bounded({'version':1, 'scope':'staged-draft-value', 'session':token,
            'sequence':row['sequence'], 'value':buffer['id'], 'value_revision':buffer['revision'],
            'path':list(buffer['path']), 'total_bytes':buffer['size'], 'offset_units':'utf8-bytes',
            'source_path':list(buffer['source_path']), 'from_base':buffer['from_base'],
            'byte_limit':science_browser.MAX_BYTES})

    def start_value(self, workspace, token, owner, sequence, path, *, from_base=False):
        row = self._get(token, owner, sequence, workspace); _path(path)
        source_path = list(path)
        _require(type(from_base) is bool, 'from_base must be boolean.')
        if from_base:
            _require(row['draft']['base'] is not None and sequence == 0 and path,
                     'Base paths require an untouched exact-base draft and a non-root value.')
            # frozen_document returns canonical alphabetic dictionary keys; Begin
            # removes only its root revision hash. Descendant order is unchanged.
            ordinal = sorted([*row['draft']['document'], 'revision']).index('revision')
            _require(path[0] != ordinal, 'Cannot edit the retained content revision identity.')
            path = [path[0] - (1 if path[0] > ordinal else 0), *path[1:]]
        _require(row['buffer'] is None, 'A staged value is already open. Finish or discard it explicitly.')
        value = _at(row['draft']['document'], path)
        self._admit(workspace, row['draft']['document'], excluding=token)
        stream = tempfile.TemporaryFile('w+b')
        try:
            # JSON tokens retain large int64 values without JavaScript-number
            # conversion. Encoding/parsing is a host cost, not a wire-bound claim.
            encoder = json.JSONEncoder(ensure_ascii=False, allow_nan=False, separators=(',', ':'))
            for chunk in encoder.iterencode(value): stream.write(chunk.encode('utf-8'))
            stream.flush()
            buffer = {'id':uuid.uuid4().hex, 'revision':0, 'path':list(path), 'source_path':source_path,
                      'from_base':from_base, 'size':stream.tell(), 'file':stream}
            result = self._value_header(token, row, buffer)
            row['buffer'] = buffer
            return result
        except Exception:
            stream.close(); raise

    @staticmethod
    def _boundary(buffer, offset):
        _require(_integer(offset, 0, buffer['size']), 'Fragment offset must be within its UTF-8 byte range.')
        if offset == buffer['size']: return
        buffer['file'].seek(offset); first = buffer['file'].read(1)
        _require(not first or not 0x80 <= first[0] <= 0xbf, 'Fragment offset splits a UTF-8 character.')

    def read_value(self, token, owner, sequence, value, revision, *, offset=0):
        row, buffer = self._buffer(token, owner, sequence, value, revision)
        self._boundary(buffer, offset); buffer['file'].seek(offset)
        text = codecs.getincrementaldecoder('utf-8')().decode(buffer['file'].read(8192), final=False)[:2000]
        end = offset + len(text.encode('utf-8'))
        return science_browser._bounded({**self._value_header(token,row,buffer), 'offset':offset,
            'text':text, 'next_offset':end if end < buffer['size'] else None})

    def splice_value(self, workspace, token, owner, sequence, value, revision, *, offset, delete_bytes, text):
        row, buffer = self._buffer(token, owner, sequence, value, revision, workspace)
        _require(isinstance(text,str) and len(text) <= 2000 and _integer(delete_bytes),
                 'A fragment splice needs at most 2000 code points and an exact deletion byte count.')
        self._boundary(buffer,offset); self._boundary(buffer,offset+delete_bytes)
        try: data = text.encode('utf-8')
        except UnicodeError as error: raise GraphError('Fragment text must encode as valid UTF-8.') from error
        replacement = tempfile.TemporaryFile('w+b')
        try:
            source = buffer['file']; source.seek(0); remaining = offset
            while remaining:
                chunk = source.read(min(65536,remaining)); _require(chunk, 'Staged file ended unexpectedly.')
                replacement.write(chunk); remaining -= len(chunk)
            replacement.write(data); source.seek(offset+delete_bytes)
            while chunk := source.read(65536): replacement.write(chunk)
            replacement.flush()
            newer = dict(buffer, file=replacement, revision=revision+1, size=replacement.tell())
            result = self._value_header(token,row,newer)
            row['buffer'] = newer; source.close()
            return result
        except Exception:
            replacement.close(); raise

    def finish_value(self, workspace, token, owner, sequence, value, revision):
        row, buffer = self._buffer(token, owner, sequence, value, revision, workspace)
        self._admit(workspace, row['draft']['document'], excluding=token, staging_bytes=buffer['size'])
        buffer['file'].seek(0)
        try: parsed = _parse(buffer['file'].read().decode('utf-8'))
        except (UnicodeError,RecursionError) as error: raise GraphError('Staged value is not valid finite JSON.') from error
        authoring_memory.footprint(parsed)  # Reject exponent overflow, not just NaN constants.
        document = _operation(row['draft']['document'], {'op':'replace','path':buffer['path'],'value':parsed})
        result = self._publish(workspace,token,row,document)
        buffer['file'].close()
        return result

    def discard_value(self, token, owner, sequence, value, revision):
        row, buffer = self._buffer(token,owner,sequence,value,revision)
        buffer['file'].close(); row['buffer'] = None
        return self._header(token,row)

    def admit_commit(self, workspace, token, owner, sequence):
        row = self._get(token,owner,sequence,workspace)
        _require(row['buffer'] is None, 'Finish or discard the staged value before retaining the draft.')
        return self._admit(workspace,row['draft']['document'],excluding=token)

    def family_page(self, token, owner, sequence, collection, **cursor):
        from . import family_drafts
        row = self._get(token, owner, sequence)
        _require(row['draft']['kind'] == 'family', 'Family pages require an architecture family draft.')
        return family_drafts.page(row['draft']['document'], self._header(token, row), collection, **cursor)

    def family_edit(self, workspace, token, owner, sequence, operations):
        from . import family_drafts
        row = self._get(token, owner, sequence, workspace)
        _require(row['draft']['kind'] == 'family', 'Family edits require an architecture family draft.')
        _require(row['buffer'] is None, 'Finish or discard the staged value before family edits.')
        operations = _json(operations)
        _require(isinstance(operations, list) and 1 <= len(operations) <= 25, 'Family edit needs 1..25 ordered operations.')
        self._admit(workspace, row['draft']['document'], excluding=token)
        document = row['draft']['document']
        for index, operation in enumerate(operations):
            try: document = family_drafts.edit(document, operation)
            except (GraphError, ValueError) as error:
                raise GraphError('Family edit #' + str(index) + ': ' + str(error)) from error
        return self._publish(workspace, token, row, document)

    def family_preview(self, workspace, token, owner, sequence, bindings=None):
        from . import family_drafts
        row = self._get(token, owner, sequence, workspace)
        _require(row['draft']['kind'] == 'family', 'Family preview requires an architecture family draft.')
        _require(row['buffer'] is None, 'Finish or discard the staged value before family preview.')
        self._admit(workspace, row['draft']['document'], excluding=token)
        authoring.check(workspace, row['draft'])
        return family_drafts.preview(row['draft']['document'], self._header(token, row), bindings)

    def dataset_page(self, token, owner, sequence, **cursor):
        from . import dataset_drafts
        row=self._get(token,owner,sequence)
        _require(row['draft']['kind']=='dataset','Typed dataset pages require a dataset draft.')
        return dataset_drafts.page(row['draft']['document'],self._header(token,row),**cursor)

    def dataset_edit(self, workspace, token, owner, sequence, operations):
        from . import dataset_drafts
        row=self._get(token,owner,sequence,workspace)
        _require(row['draft']['kind']=='dataset','Typed dataset edits require a dataset draft.')
        _require(row['buffer'] is None,'Finish or discard the staged value before typed edits.')
        operations=_json(operations)
        _require(isinstance(operations,list) and 1<=len(operations)<=25,'Typed edit needs 1..25 ordered operations.')
        self._admit(workspace,row['draft']['document'],excluding=token)
        document=row['draft']['document']
        for index,operation in enumerate(operations):
            try:document=dataset_drafts.edit(document,operation)
            except (GraphError,ValueError) as error:raise GraphError('Dataset edit #'+str(index)+': '+str(error)) from error
        return self._publish(workspace,token,row,document)

    def dataset_value(self, token, owner, sequence, target):
        from . import dataset_drafts
        row=self._get(token,owner,sequence)
        _require(row['draft']['kind']=='dataset','Typed dataset values require a dataset draft.')
        return dataset_drafts.locate(row['draft']['document'],self._header(token,row),target)

    def task_page(self, token, owner, sequence, collection, **cursor):
        from . import task_drafts
        row = self._get(token, owner, sequence)
        _require(row['draft']['kind'] == 'task', 'Typed task pages require a task draft.')
        return task_drafts.page(row['draft']['document'], self._header(token, row), collection, **cursor)

    def task_transforms(self, token, owner, sequence, collection, binding, **cursor):
        from . import task_drafts
        row = self._get(token, owner, sequence)
        _require(row['draft']['kind'] == 'task', 'Typed task transforms require a task draft.')
        return task_drafts.transforms(row['draft']['document'], self._header(token, row),
                                      collection, binding, **cursor)

    def task_compatibility(self, workspace, token, owner, sequence, model, dataset):
        row = self._get(token, owner, sequence, workspace)
        _require(row['draft']['kind'] == 'task', 'Task compatibility requires a task draft.')
        _require(row['buffer'] is None, 'Finish or discard the staged value before checking compatibility.')
        self._admit(workspace, row['draft']['document'], excluding=token)
        return science_browser._bounded(authoring.task_compatibility(workspace, row['draft'], model, dataset))

    def task_edit(self, workspace, token, owner, sequence, operations):
        from . import task_drafts
        row = self._get(token, owner, sequence, workspace)
        _require(row['draft']['kind'] == 'task', 'Typed task edits require a task draft.')
        _require(row['buffer'] is None, 'Finish or discard the staged value before typed edits.')
        operations = _json(operations)
        _require(isinstance(operations, list) and 1 <= len(operations) <= 25,
                 'Typed task edit needs 1..25 ordered operations.')
        self._admit(workspace, row['draft']['document'], excluding=token)
        document = row['draft']['document']
        for index, operation in enumerate(operations):
            try: document = task_drafts.edit(document, operation)
            except (GraphError, ValueError) as error:
                raise GraphError('Task edit #'+str(index)+': '+str(error)) from error
        return self._publish(workspace, token, row, document)

    def experiment_page(self, token, owner, sequence, collection, **cursor):
        from . import experiment_drafts
        row = self._get(token, owner, sequence)
        _require(row['draft']['kind'] == 'experiment', 'Experiment cards require an experiment draft.')
        return experiment_drafts.page(row['draft']['document'], self._header(token, row), collection, **cursor)

    def experiment_value(self, token, owner, sequence, target, key):
        from . import experiment_drafts
        row = self._get(token, owner, sequence)
        _require(row['draft']['kind'] == 'experiment', 'Experiment values require an experiment draft.')
        return experiment_drafts.locate(row['draft']['document'], self._header(token, row), target, key)

    def experiment_choices(self, token, owner, sequence, collection, **cursor):
        from . import experiment_drafts
        row = self._get(token, owner, sequence)
        _require(row['draft']['kind'] == 'experiment', 'Local choices require an experiment draft.')
        return experiment_drafts.choices(row['draft']['document'], self._header(token, row), collection, **cursor)

    def experiment_preview(self, workspace, token, owner, sequence, **cursor):
        from . import experiment_drafts
        row = self._get(token, owner, sequence, workspace)
        _require(row['draft']['kind'] == 'experiment', 'Preview requires an experiment draft.')
        _require(row['buffer'] is None, 'Finish or discard the staged value before preview.')
        self._admit(workspace, row['draft']['document'], excluding=token)
        return experiment_drafts.preview(workspace, row['draft'], self._header(token, row), **cursor)

    def experiment_edit(self, workspace, token, owner, sequence, operations):
        from . import experiment_drafts
        row = self._get(token, owner, sequence, workspace)
        _require(row['draft']['kind'] == 'experiment', 'Experiment edits require an experiment draft.')
        _require(row['buffer'] is None, 'Finish or discard the staged value before typed edits.')
        operations = _json(operations)
        _require(isinstance(operations, list) and 1 <= len(operations) <= 25,
                 'Experiment edit needs 1..25 ordered operations.')
        self._admit(workspace, row['draft']['document'], excluding=token)
        document = row['draft']['document']
        for index, operation in enumerate(operations):
            try: document = experiment_drafts.edit(document, operation)
            except (GraphError, ValueError) as error:
                raise GraphError('Experiment edit #'+str(index)+': '+str(error)) from error
        return self._publish(workspace, token, row, document)
