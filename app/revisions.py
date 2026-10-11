"""Immutable scientific objects and mutable browsing selections, without Torch/Qt.

Version-1 scientific object validators are closed, not arbitrary JSON records.
Unknown future kinds fail explicitly until their own contract is implemented.
"""
import copy
import json
import re
from datetime import datetime

from .core import GraphError, check_document, timestamp, validate_model
from .datasets import _identifier, _keys, _require, _version, validate_dataset, validate_task_spec
from .evidence import PRESENTATION, digest, identity, pin_input_order
from .operators import normalized

KINDS = ('dataset', 'experiment', 'family', 'model', 'report', 'run', 'specialization', 'task')
SCHEMA = {
    'spec_revisions': """CREATE TABLE spec_revisions(
        kind TEXT NOT NULL, id TEXT NOT NULL, revision TEXT NOT NULL,
        name TEXT NOT NULL, document TEXT NOT NULL, created_at TEXT NOT NULL,
        PRIMARY KEY(kind,id,revision))""",
    'spec_heads': """CREATE TABLE spec_heads(
        kind TEXT NOT NULL, id TEXT NOT NULL, revision TEXT NOT NULL,
        PRIMARY KEY(kind,id),
        FOREIGN KEY(kind,id,revision) REFERENCES spec_revisions(kind,id,revision))""",
    'spec_revisions_no_update': """CREATE TRIGGER spec_revisions_no_update BEFORE UPDATE ON spec_revisions
        BEGIN SELECT RAISE(ABORT, 'Immutable revisions cannot be updated'); END""",
    'spec_revisions_no_delete': """CREATE TRIGGER spec_revisions_no_delete BEFORE DELETE ON spec_revisions
        BEGIN SELECT RAISE(ABORT, 'Immutable revisions cannot be deleted'); END""",
    'spec_revisions_no_replace': """CREATE TRIGGER spec_revisions_no_replace BEFORE INSERT ON spec_revisions
        WHEN EXISTS(SELECT 1 FROM spec_revisions WHERE kind=NEW.kind AND id=NEW.id AND revision=NEW.revision)
        BEGIN SELECT RAISE(ABORT, 'Immutable revisions cannot be replaced'); END""",
}


def _kind(kind):
    _require(isinstance(kind, str) and kind in KINDS, 'Unsupported independent revision kind.')


def _hash(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def _parse(text):
    def pairs(items):
        result = {}
        for key, value in items:
            _require(key not in result, 'Duplicate JSON key in revision document.')
            result[key] = value
        return result
    def constant(value):
        raise GraphError('Revision documents must be finite JSON: ' + value)
    try:
        return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)
    except GraphError:
        raise
    except (ValueError, TypeError, RecursionError) as error:
        raise GraphError('Invalid revision JSON.') from error


def create_schema(db):
    """Only new/migrated working copies; never silently repair a v5 guard."""
    db.executescript(';\n'.join(SCHEMA.values()) + ';')


def check_schema(db):
    expected_columns = {
        'spec_revisions': [('kind', 'TEXT', 1, 1), ('id', 'TEXT', 1, 2), ('revision', 'TEXT', 1, 3),
                           ('name', 'TEXT', 1, 0), ('document', 'TEXT', 1, 0), ('created_at', 'TEXT', 1, 0)],
        'spec_heads': [('kind', 'TEXT', 1, 1), ('id', 'TEXT', 1, 2), ('revision', 'TEXT', 1, 0)],
    }
    for table, columns in expected_columns.items():
        actual = [(r[1], r[2].upper(), r[3], r[5]) for r in db.execute('PRAGMA table_info(' + table + ')')]
        _require(actual == columns, 'Independent revision table schema does not match v5: ' + table + '.')
    foreign = list(db.execute('PRAGMA foreign_key_list(spec_heads)'))
    expected_foreign = [(0, i, 'spec_revisions', key, key, 'NO ACTION', 'NO ACTION', 'NONE')
                        for i, key in enumerate(('kind', 'id', 'revision'))]
    _require(foreign == expected_foreign and not list(db.execute('PRAGMA foreign_key_list(spec_revisions)')),
             'Independent revision foreign keys do not match v5.')
    normalize = lambda sql: re.sub(r'\s+', '', sql).lower().rstrip(';')
    for name in ('spec_revisions_no_update', 'spec_revisions_no_delete', 'spec_revisions_no_replace'):
        row = db.execute("SELECT sql FROM sqlite_master WHERE type='trigger' AND name=?", (name,)).fetchone()
        _require(row is not None and normalize(row[0]) == normalize(SCHEMA[name]), 'Missing or changed immutable revision guard: ' + name + '.')


def migrate_schema(db):
    names = {r[0] for r in db.execute('SELECT name FROM sqlite_master')}
    present = set(SCHEMA) & names
    _require(not present or present == set(SCHEMA), 'Partial independent revision schema; cannot safely migrate.')
    if not present:
        create_schema(db)
    check_schema(db)


def model_document(models, mid):
    """Capture only the complete reachable runnable closure, not editable data."""
    _require(_identifier(mid) and mid in models, 'Cannot freeze a missing architecture.')
    closure = {}
    def visit(current):
        if current in closure:
            return
        _require(current in models, 'Frozen model refers to a missing architecture.')
        source = models[current]
        graph = copy.deepcopy(source['graph'])
        pin_input_order(graph)
        graph['nodes'] = [{k: v for k, v in normalized(node).items() if k not in PRESENTATION} for node in graph['nodes']]
        closure[current] = {'id': current, 'name': source['name'], 'graph': graph}
        for node in graph['nodes']:
            if node['type'] == 'Module':
                visit(node['ref'])
    visit(mid)
    result = {'version': 1, 'id': mid, 'name': models[mid]['name'], 'root': mid, 'models': closure,
              'architecture_hash': identity(closure, mid)['architecture']}
    validate_model_document(result)
    return result


def validate_model_document(doc):
    _keys(doc, ('version', 'id', 'name', 'root', 'models', 'architecture_hash'), ('revision',), 'ModelRevision')
    _version(doc, 'ModelRevision')
    _require(_identifier(doc['id']) and _identifier(doc['name']) and doc['root'] == doc['id'] and
             isinstance(doc['models'], dict) and doc['root'] in doc['models'], 'Frozen model needs a root and reachable closure.')
    closure = doc['models']
    for mid, model in closure.items():
        _require(_identifier(mid), 'Frozen model needs stable architecture IDs.')
        _keys(model, ('id', 'name', 'graph'), context='Frozen architecture')
        _require(model['id'] == mid, 'Frozen architecture key/ID mismatch.')
        check_document(model)
        _require('input_order' in model['graph'] and all(not (PRESENTATION & set(n)) for n in model['graph']['nodes']),
                 'Frozen graph requires pinned packing and no node presentation fields.')
        _require(all(normalized(n) == n for n in model['graph']['nodes']), 'Frozen operator defaults must be normalized.')
    _require(closure[doc['root']]['name'] == doc['name'], 'Frozen root name does not match document.')
    reachable = set()
    def visit(mid):
        if mid in reachable:
            return
        _require(mid in closure, 'Frozen reference closure is incomplete.')
        reachable.add(mid)
        for node in closure[mid]['graph']['nodes']:
            if node['type'] == 'Module':
                visit(node['ref'])
    visit(doc['root'])
    _require(reachable == set(closure), 'Frozen closure contains unrelated architectures.')
    validate_model(closure, doc['root'])
    _require(_hash(doc['architecture_hash']) and doc['architecture_hash'] == identity(closure, doc['root'])['architecture'],
             'Frozen architecture hash does not match its closure.')


def frozen_document(kind, doc):
    """Closed validation plus private immutable JSON revision; no hidden repair."""
    _kind(kind)
    if kind in ('experiment', 'run', 'report'):
        from .experiment_objects import validate_experiment, validate_run, validate_report
        validator = {'experiment': validate_experiment, 'run': validate_run, 'report': validate_report}[kind]
    elif kind in ('family', 'specialization'):
        from .family_records import validate_family_revision, validate_specialization
        validator = {'family': validate_family_revision, 'specialization': validate_specialization}[kind]
    else:
        validator = {'dataset': validate_dataset, 'task': validate_task_spec, 'model': validate_model_document}[kind]
    validator(doc)
    try:
        value = digest({k: v for k, v in doc.items() if k != 'revision'})
        _require('revision' not in doc or doc['revision'] == value, 'Revision does not match document content.')
        result = copy.deepcopy(doc)
        result['revision'] = value
        # Roundtrip ensures the stored/public object has exactly JSON semantics.
        return _parse(json.dumps(result, allow_nan=False, sort_keys=True, separators=(',', ':')))
    except GraphError:
        raise
    except (ValueError, TypeError, RecursionError) as error:
        raise GraphError('Revision document must be finite JSON.') from error


def validate_all(db):
    """Verify relation, grammar and content meaning without tensor allocation."""
    check_schema(db)
    documents = {}
    for kind, sid, revision, name, text, created_at in db.execute('SELECT * FROM spec_revisions'):
        doc = frozen_document(kind, _parse(text))
        _require((doc['id'], doc['revision'], doc['name']) == (sid, revision, name), 'Revision row identity/name/hash mismatch.')
        try:
            parsed = datetime.fromisoformat(created_at)
            valid_time = parsed.utcoffset() is not None and parsed.utcoffset().total_seconds() == 0
        except (TypeError, ValueError, OverflowError):
            valid_time = False
        _require(valid_time, 'Revision creation time must be an explicit UTC timestamp.')
        documents[kind, sid, revision] = doc
    from .experiment_objects import validate_dependencies
    def resolve(kind, sid, revision):
        _require((kind, sid, revision) in documents, 'Scientific object refers to a missing immutable revision.')
        return documents[kind, sid, revision]
    for (kind, _, _), doc in documents.items():
        validate_dependencies(kind, doc, resolve)
    # Retained run revisions form an append-only attempt history independent of
    # mutable browsing heads, including after undo/reselection of an older head.
    for doc in (d for (k, _, _), d in documents.items() if k == 'run'):
        for other in (d for (k, sid, _), d in documents.items() if k == 'run' and sid == doc['id']):
            length = min(len(doc['attempts']), len(other['attempts']))
            _require(digest(doc['selection']) == digest(other['selection']) and digest(doc['resolved']) == digest(other['resolved']) and
                     digest(doc['attempts'][:length]) == digest(other['attempts'][:length]), 'Retained run attempts diverge or rewrite prior evidence.')
    for kind, sid, revision in db.execute('SELECT * FROM spec_heads'):
        _kind(kind)
        _require(db.execute('SELECT 1 FROM spec_revisions WHERE kind=? AND id=? AND revision=?', (kind, sid, revision)).fetchone(), 'Browsing head refers to a missing revision.')


class RevisionStore:
    """Use the workspace connection: Save and history remain one coherent file."""
    def __init__(self, workspace):
        self.workspace = workspace
        self.db = workspace.db

    def put(self, kind, document, select=True):
        _require(type(select) is bool, 'Revision select must be boolean.')
        doc = frozen_document(kind, document)
        from .experiment_objects import validate_dependencies
        validate_dependencies(kind, doc, self.get)
        if kind == 'run':
            already_retained = self.db.execute("SELECT 1 FROM spec_revisions WHERE kind='run' AND id=? AND revision=?", (doc['id'], doc['revision'])).fetchone()
            for row in self.db.execute("SELECT document FROM spec_revisions WHERE kind='run' AND id=?", (doc['id'],)):
                previous = _parse(row[0]); length = min(len(previous['attempts']), len(doc['attempts']))
                _require(digest(previous['selection']) == digest(doc['selection']) and digest(previous['resolved']) == digest(doc['resolved']) and
                         digest(previous['attempts'][:length]) == digest(doc['attempts'][:length]),
                         'Run revisions may append attempts, not rewrite their identity/evidence.')
                _require(already_retained or len(doc['attempts']) >= len(previous['attempts']), 'New run revisions cannot truncate retained attempt history.')
        reference = {'kind': kind, 'id': doc['id'], 'revision': doc['revision']}
        text = json.dumps(doc, allow_nan=False, sort_keys=True, separators=(',', ':'))
        with self.db:
            existing = self.db.execute('SELECT document FROM spec_revisions WHERE kind=? AND id=? AND revision=?', tuple(reference.values())).fetchone()
            if existing:
                _require(_parse(existing[0]) == doc, 'Immutable revision conflicts with retained bytes.')
            else:
                self.db.execute('INSERT INTO spec_revisions VALUES (?,?,?,?,?,?)', (kind, doc['id'], doc['revision'], doc['name'], text, timestamp()))
            if select:
                self.db.execute('''INSERT INTO spec_heads VALUES (?,?,?) ON CONFLICT(kind,id)
                                   DO UPDATE SET revision=excluded.revision''', tuple(reference.values()))
        self.workspace.dirty = True
        return reference

    def get(self, kind, sid, revision=None):
        _kind(kind)
        _require(_identifier(sid), 'Revision requires a stable object ID.')
        if revision is None:
            row = self.db.execute('SELECT revision FROM spec_heads WHERE kind=? AND id=?', (kind, sid)).fetchone()
            _require(row is not None, 'Object has no browsing head; request a retained revision explicitly.')
            revision = row[0]
        _require(_hash(revision), 'Revision must be a lowercase SHA-256 content hash.')
        row = self.db.execute('SELECT document FROM spec_revisions WHERE kind=? AND id=? AND revision=?', (kind, sid, revision)).fetchone()
        _require(row is not None, 'Requested immutable revision does not exist.')
        doc = frozen_document(kind, _parse(row[0]))
        _require(doc['id'] == sid and doc['revision'] == revision, 'Retained revision identity mismatch.')
        if kind in ('experiment', 'run', 'report', 'specialization'):
            from .experiment_objects import validate_dependencies
            validate_dependencies(kind, doc, self.get)
        return doc

    def select(self, kind, sid, revision):
        self.get(kind, sid, revision)
        with self.db:
            self.db.execute('''INSERT INTO spec_heads VALUES (?,?,?) ON CONFLICT(kind,id)
                               DO UPDATE SET revision=excluded.revision''', (kind, sid, revision))
        self.workspace.dirty = True

    def archive(self, kind, sid):
        self.get(kind, sid)
        with self.db:
            self.db.execute('DELETE FROM spec_heads WHERE kind=? AND id=?', (kind, sid))
        self.workspace.dirty = True

    def list(self, kind=None, all_revisions=False):
        _require(type(all_revisions) is bool, 'all_revisions must be boolean.')
        if kind is not None:
            _kind(kind)
        rows = self.db.execute('''SELECT r.kind,r.id,r.revision,r.name,r.created_at,h.revision
                                 FROM spec_revisions r LEFT JOIN spec_heads h ON r.kind=h.kind AND r.id=h.id''')
        result = [{'kind': k, 'id': sid, 'revision': rev, 'name': name, 'created_at': created,
                   'selected': rev == head} for k, sid, rev, name, created, head in rows
                  if (kind is None or k == kind) and (all_revisions or rev == head)]
        return sorted(result, key=lambda r: (r['name'].casefold(), r['id'], r['revision'], r['kind']))


def merge_retained(target, source):
    """Undo restores heads/drafts, NEVER removes newer retained revision bytes."""
    validate_all(target); validate_all(source)
    for row in source.execute('SELECT * FROM spec_revisions'):
        existing = target.execute('SELECT document FROM spec_revisions WHERE kind=? AND id=? AND revision=?', row[:3]).fetchone()
        if existing:
            _require(_parse(existing[0]) == _parse(row[4]), 'History immutable revision conflict.')
        else:
            target.execute('INSERT INTO spec_revisions VALUES (?,?,?,?,?,?)', row)
