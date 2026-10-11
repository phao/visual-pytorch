"""Immutable family/specialization provenance and atomic concrete admission.

Definitions and bindings belong to SQLite, not a sidecar or the mutable canvas.
No Torch, Qt, execution, data-driven binding or implicit model replacement here.
"""
import copy
import uuid
from contextlib import contextmanager

from .datasets import _identifier, _keys, _require, _version
from .evidence import digest
from .experiment_objects import reference
from .families import specialize, preview, validate_family, binding_atom


def definition(doc):
    """A verified registry hash is metadata, not part of the family grammar/hash."""
    return {k: copy.deepcopy(v) for k, v in doc.items() if k != 'revision'}


def validate_family_revision(doc):
    _require(isinstance(doc, dict), 'Family revision must be an object.')
    _require(_identifier(doc.get('id')) and _identifier(doc.get('name')), 'Family revision needs readable ID/name.')
    validate_family(definition(doc))


def validate_specialization(doc):
    _keys(doc, ('version', 'id', 'name', 'family', 'bindings', 'model') +
          (('instances',) if type(doc.get('version')) is int and doc['version'] >= 3 else ()) +
          (('runtime',) if doc.get('version') in (4,5,6) else ()), ('revision',), 'Specialization')
    _require(type(doc['version']) is int and doc['version'] in (1, 2, 3, 4, 5, 6), 'Specialization supports integer version 1 through 6.')
    _require(_identifier(doc['id']) and _identifier(doc['name']), 'Specialization needs a stable ID/name.')
    reference(doc['family'], 'family'); reference(doc['model'], 'model')
    _require(isinstance(doc['bindings'], dict) and len(doc['bindings']) <= 128,
             'Specialization must retain all resolved stable-symbol-ID bindings.')
    for sid, value in doc['bindings'].items():
        _require(_identifier(sid), 'Specialization binding key must be a stable symbol ID.')
        binding_atom(value, 'Specialization binding ' + sid, typed=doc['version'] >= 2)
    if doc['version'] >= 3:
        _require(isinstance(doc['instances'], list) and 1 <= len(doc['instances']) <= 64,
                 'Specialization needs 1..64 concrete constructor contexts.')
        seen = set()
        for row in doc['instances']:
            _keys(row, ('model', 'template', 'bindings'), context='Specialization constructor context')
            _require(_identifier(row['model']) and row['model'] not in seen and _identifier(row['template']),
                     'Specialization constructor context needs unique model and template identities.')
            seen.add(row['model'])
            _require(isinstance(row['bindings'], dict) and len(row['bindings']) <= 128,
                     'Specialization constructor context needs bounded local bindings.')
            for sid, value in row['bindings'].items():
                _require(_identifier(sid), 'Constructor context key must be a stable symbol ID.')
                binding_atom(value, 'Constructor context ' + sid, typed=True)
    if doc['version'] in (4,5,6) and doc['runtime'] is not None:
        value = doc['runtime']
        temporal = doc['version'] in (5,6)
        _keys(value, ('version','inputs','specification_sha256') + (('kind',) if temporal else ()), context='Specialization runtime contract')
        _require(type(value['version']) is int and value['version'] == (2 if temporal else 1) and isinstance(value['inputs'],dict) and
                 1 <= len(value['inputs']) <= 64, 'Runtime contract requires its exact grammar and explicit root Input IDs.')
        if temporal:
            _require(value['kind'] == 'runtime-temporal-v1', 'Specialization5 requires an explicit runtime-temporal-v1 witness.')
        for nid, axes in value['inputs'].items():
            _require(_identifier(nid) and isinstance(axes,list) and
                     (len(axes) == 1 and type(axes[0]) is int and axes[0] == 0 if temporal else
                      1 <= len(axes) <= 2 and all(type(v) is int and v in (1,2) for v in axes) and axes == sorted(set(axes))),
                     'Runtime contract must retain canonical T sample axis [0].' if temporal else
                     'Runtime contract must retain canonical unique H/W sample axes.')
        _require(isinstance(value['specification_sha256'],str) and len(value['specification_sha256']) == 64 and
                 all(v in '0123456789abcdef' for v in value['specification_sha256']), 'Runtime specification needs its exact SHA-256.')


def validate_specialization_dependencies(doc, resolve):
    """Reproduce meaning, not just existence or shape equality of pinned records."""
    from .revisions import model_document
    family = resolve(doc['family']['kind'], doc['family']['id'], doc['family']['revision'])
    model = resolve(doc['model']['kind'], doc['model']['id'], doc['model']['revision'])
    source = definition(family)
    _require(doc['version'] == (6 if source['version'] == 7 else 5 if source['version'] == 6 else 4 if source['version'] == 5 else 3 if source['version'] == 4 else 2 if source['version'] == 3 else 1),
             'Specialization version must match its exact family scalar grammar.')
    if doc['version'] < 3:
        _require(set(doc['bindings']) == {s['id'] for s in source['symbols']},
                 'Specialization bindings must explicitly include every symbol, including defaults.')
    resolved = specialize(source, doc['bindings'])
    if doc['version'] >= 3:
        _require(doc['bindings'] == resolved['bindings'] and digest(doc['instances']) == digest(resolved['instances']),
                 'Specialization must retain canonical fallback bindings and every exact constructor context.')
    if doc['version'] in (4,5,6):
        _require(digest(doc['runtime']) == digest(resolved['runtime']),
                 'Specialization runtime contract differs from its exact family revision and constructor bindings.')
    expected = model_document(resolved['models'], resolved['root'])
    _require(digest(definition(model)) == digest(expected),
             'Specialization frozen model does not match its exact family revision and bindings.')


def preview_stored(workspace, family_ref, bindings=None):
    reference(family_ref, 'family')
    family = workspace.get_spec('family', family_ref['id'], family_ref['revision'])
    result = preview(definition(family), bindings)
    return {'family': copy.deepcopy(family_ref), **result}


def compile_specialization(workspace, specialization_ref, *, constrained=None,
                           memory_limit_percent=10., initial_shapes=None):
    """Explicit native construction from exact retained provenance; no canvas writes."""
    from .core import compile_model
    reference(specialization_ref, 'specialization')
    doc = workspace.get_spec('specialization', specialization_ref['id'], specialization_ref['revision'])
    validate_specialization_dependencies(doc, workspace.get_spec)
    model = workspace.get_spec('model', doc['model']['id'], doc['model']['revision'])
    witness = doc.get('runtime')
    policy = {} if witness is None else {
        'runtime_temporal' if doc['version'] in (5,6) else 'runtime_inputs': copy.deepcopy(witness['inputs'])}
    return compile_model(model['models'], model['root'],
                         workspace.constrained_mode if constrained is None else constrained,
                         memory_limit_percent=memory_limit_percent, initial_shapes=initial_shapes, **policy)


class _SavepointConnection:
    """Keep existing `with db` writers inside ONE owned SQLite savepoint.

These methods normally commit their own operation. During this admission their
context exits must not release the enclosing savepoint halfway through a group.
Only this service's synchronous internal writes use the proxy; it is never kept
in the workspace, passed to a worker, or used across a Qt event-loop yield.
"""
    def __init__(self, db): self._db = db
    def __getattr__(self, name): return getattr(self._db, name)
    def __enter__(self): return self
    def __exit__(self, exc_type, exc_value, traceback): return False


@contextmanager
def _atomic(workspace):
    db, dirty = workspace.db, workspace.dirty
    name = 'family_' + uuid.uuid4().hex
    db.execute('SAVEPOINT ' + name)
    workspace.db = _SavepointConnection(db)
    try:
        yield
    except BaseException:
        db.execute('ROLLBACK TO ' + name); db.execute('RELEASE ' + name)
        workspace.dirty = dirty
        raise
    else:
        db.execute('RELEASE ' + name)
    finally:
        workspace.db = db


def resolution_records(frozen, resolved, *, sid=None, name=None):
    """Pure canonical records shared by preview and later atomic admission."""
    from .revisions import frozen_document, model_document
    family_ref = {'kind': 'family', 'id': frozen['id'], 'revision': frozen['revision']}
    model = frozen_document('model', model_document(resolved['models'], resolved['root']))
    model_ref = {'kind': 'model', 'id': model['id'], 'revision': model['revision']}
    record = frozen_document('specialization', {
        'version': 6 if frozen['version'] == 7 else 5 if frozen['version'] == 6 else 4 if frozen['version'] == 5 else 3 if frozen['version'] == 4 else 2 if frozen['version'] == 3 else 1, 'id': sid if sid is not None else 'specialization-' + resolved['specialization'],
        'name': name if name is not None else frozen['name'] + ' · resolved',
        'family': copy.deepcopy(family_ref), 'bindings': copy.deepcopy(resolved['bindings']), 'model': model_ref,
        **({'instances': copy.deepcopy(resolved['instances'])} if frozen['version'] >= 4 else {}),
        **({'runtime':copy.deepcopy(resolved['runtime'])} if frozen['version'] in (5,6,7) else {})})
    return {'family_ref': family_ref, 'model': model, 'model_ref': model_ref,
            'specialization': record, 'specialization_ref': {
                'kind': 'specialization', 'id': record['id'], 'revision': record['revision']}}


def retain(workspace, *, family=None, family_ref=None, bindings=None, sid=None, name=None,
           materialize=True):
    """Atomically retain exact provenance + frozen instance; optionally draw it.

The caller performs normal explicit Save/undo framing. A failed operation leaves
the working database and dirty flag unchanged, including when a later writer
raises. SQLite's diagnostic total_changes counter is not an undoable row value.
Family historical selection never moves its browsing head implicitly.
"""
    from .revisions import frozen_document
    _require((family is None) != (family_ref is None), 'Supply exactly one family definition or exact family reference.')
    _require(type(materialize) is bool, 'Family materialize must be boolean.')
    if family is not None:
        frozen = frozen_document('family', family)
        family_ref = {'kind': 'family', 'id': frozen['id'], 'revision': frozen['revision']}
    else:
        reference(family_ref, 'family')
        frozen = workspace.get_spec('family', family_ref['id'], family_ref['revision'])
    resolved = specialize(definition(frozen), bindings)
    records = resolution_records(frozen, resolved, sid=sid, name=name)
    model, model_ref, record = records['model'], records['model_ref'], records['specialization']
    if materialize:
        _require(not (set(resolved['models']) & set(workspace.models())),
                 'A resolved family architecture ID already exists; no implicit replacement.')
    with _atomic(workspace):
        if family is not None:
            workspace.put_spec('family', frozen)
        workspace.put_spec('model', model)
        ref = workspace.put_spec('specialization', record)
        if materialize:
            inserted = set()
            def insert(mid):
                if mid in inserted:
                    return
                doc = resolved['models'][mid]
                for node in doc['graph']['nodes']:
                    if node['type'] in {'Module', 'Recurrent'}:
                        insert(node['ref'])
                workspace.put(doc)
                inserted.add(mid)
            insert(resolved['root'])
    return {'family': family_ref, 'model': model_ref, 'specialization': ref,
            'materialized': materialize, 'resolved': resolved}
