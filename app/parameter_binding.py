"""Copied initial tensors with exact independent immutable lineage; never resume."""
import copy
import json

from .datasets import _identifier, _integer, _keys, _require
from .evidence import digest
from .experiment_objects import reference
from .family_records import _atomic
from .parameter_policy import resolve
from .parameter_sources import VALUE_FORMAT, float32_values, read
from .policy_authoring import TOKEN, build_changes, prepare
from .policy_graph import owner_key, plan
from .revisions import _hash, frozen_document, model_document


def _parameter_id(value):
    return isinstance(value, str) and 0 < len(value) <= 65536 and not any(ord(c) < 32 for c in value)


def _tensor_descriptor(doc):
    _require(doc['dtype'] == 'float32' and doc['value_format'] == VALUE_FORMAT and _hash(doc['values_sha256']),
             'Parameter binding requires exact finite float32 value-digest semantics.')
    resolve({'selected': {'id': 'shape-check', 'shape': doc['shape']}})


def validate_source(source):
    _keys(source, ('version', 'kind', 'run', 'attempt', 'checkpoint_id', 'model', 'parameter', 'shape',
                  'dtype', 'values_sha256', 'value_format', 'status', 'completed_updates', 'optimizer_resume'), context='Binding source')
    _require(type(source['version']) is int and source['version'] == 1 and source['kind'] == 'retained-run-parameter',
             'Unsupported retained binding source.')
    reference(source['run'], 'run'); reference(source['model'], 'model')
    _keys(source['parameter'], ('path', 'id'), context='Binding source parameter')
    _require(_parameter_id(source['parameter']['path']) and _parameter_id(source['parameter']['id']) and
             _integer(source['attempt'], 1) and _hash(source['checkpoint_id']) and
             source['status'] in ('completed', 'failed', 'cancelled', 'unavailable') and
             _integer(source['completed_updates']) and source['optimizer_resume'] is False,
             'Retained binding source fields are invalid.')
    _tensor_descriptor(source)


def validate(doc):
    _keys(doc, ('version', 'id', 'name', 'source', 'destination', 'optimizer_resume'), ('revision',), 'ParameterBinding')
    _require(type(doc['version']) is int and doc['version'] == 1 and _identifier(doc['id']) and _identifier(doc['name']),
             'ParameterBinding requires version1, stable ID and name.')
    _require(doc['optimizer_resume'] is False, 'Parameter binding is initialization, never optimizer resume.')
    source = doc['source']; validate_source(source)
    destination = doc['destination']
    _keys(destination, ('model', 'selector', 'id', 'shape', 'dtype', 'values_sha256', 'value_format'), context='Binding destination')
    reference(destination['model'], 'model'); _tensor_descriptor(destination)
    try: components = json.loads(destination['selector'])
    except (ValueError, TypeError): components = None
    _require(isinstance(components, list) and 2 <= len(components) <= 64 and all(isinstance(v, str) and v for v in components)
             and owner_key(components) == destination['selector'] and
             owner_key([destination['model']['id']] + components) == destination['id'],
             'Binding destination requires the exact canonical component selector and owner ID.')
    _require(all(destination[k] == source[k] for k in ('shape', 'dtype', 'values_sha256', 'value_format')),
             'Binding source/destination shape, dtype and copied values must match exactly.')


def dependencies(doc):
    return [doc['source']['run'], doc['source']['model'], doc['destination']['model']]


def check_dependencies(doc, resolver):
    """Prove the source and destination against their own pinned snapshots."""
    validate(doc)
    class Snapshot:
        get_spec = staticmethod(resolver)
    source = doc['source']
    selection = {key: source[key] for key in ('run', 'attempt', 'checkpoint_id', 'parameter')}
    actual = read(Snapshot(), selection)
    _require(actual['source'] == source, 'Binding source descriptor does not match exact retained evidence.')
    target = doc['destination']; ref = target['model']
    model = resolver('model', ref['id'], ref['revision'])
    _require(model['id'] == ref['id'] and model['revision'] == ref['revision'], 'Binding resolver substituted destination model.')
    resolved = plan(model['models'], model['root'])['resolved']['owners']
    row = resolved.get(target['id'])
    _require(row is not None and row['shape'] == target['shape'] and row['initializer']['kind'] == 'values',
             'Binding destination is not a compatible exact-value owner.')
    _, value_hash = float32_values(row['initializer']['values'], row['shape'])
    _require(value_hash == target['values_sha256'], 'Binding destination values differ from the pinned source.')


def preview(workspace, token, selector, selection, *, binding_id=None, name=None):
    """Read/prove a complete prospective edit; no writes, RNG or numerical import."""
    models = workspace.models()
    # Validate identity before inspecting a potentially larger source checkpoint.
    build_changes(models, token, selector, {})
    info = prepare(models, token['id'], token['node'])
    owner = next(row for row in info['owners'] if row['selector'] == selector)
    selected = read(workspace, selection); source = selected['source']
    _require(owner['shape'] == source['shape'], 'Source/destination parameter shapes must match exactly; no reshape or transpose.')
    doc = build_changes(models, token, selector, {'initializer': {'kind': 'values', 'dtype': 'float32', 'values': selected['values']}})
    prospective = dict(models); prospective[token['id']] = doc
    frozen = frozen_document('model', model_document(prospective, token['id']))
    modelref = {'kind': 'model', 'id': frozen['id'], 'revision': frozen['revision']}
    destination = {'model': modelref, 'selector': selector, 'id': owner['id'],
                   **{key: source[key] for key in ('shape', 'dtype', 'values_sha256', 'value_format')}}
    sid = binding_id if binding_id is not None else 'parameter-binding-' + digest({'source': source, 'destination': destination})[:32]
    binding = {'version': 1, 'id': sid, 'name': name if name is not None else ('Retained parameters: ' + owner['name'])[:200],
               'source': source, 'destination': destination, 'optimizer_resume': False}
    binding = frozen_document('parameter-binding', binding)
    return {'document': doc, 'model': frozen, 'binding': binding,
            'scope': 'copy-initial-values-only; trainability-preserved; no-optimizer-resume'}


def _retain(workspace, kind, document):
    ref = {'kind': kind, 'id': document['id'], 'revision': document['revision']}
    exists = workspace.db.execute('SELECT 1 FROM spec_revisions WHERE kind=? AND id=? AND revision=?', tuple(ref.values())).fetchone()
    head = workspace.db.execute('SELECT revision FROM spec_heads WHERE kind=? AND id=?', (kind, ref['id'])).fetchone()
    if not exists or head != (ref['revision'],): return workspace.put_spec(kind, document)
    return ref


def current_bindings(workspace, mid):
    """Newest selected receipt proving each owner's current initial-value bytes.

    No new index or model field: inspect small receipt metadata, then validate
    every matched reference. Historical receipts are still independently browsable.
    """
    from .revisions import _parse
    if not workspace.db.execute("SELECT 1 FROM spec_heads WHERE kind='parameter-binding' LIMIT 1").fetchone(): return {}
    owners = plan(workspace.models(), mid)['resolved']['owners']; result = {}; hashes = {}
    rows = workspace.db.execute("SELECT r.id,r.revision,r.document FROM spec_revisions r JOIN spec_heads h ON r.kind=h.kind AND r.id=h.id AND r.revision=h.revision WHERE r.kind='parameter-binding' ORDER BY r.created_at DESC,r.id")
    for sid, revision, text in rows:
        doc = _parse(text); destination = doc.get('destination', {})
        selector = destination.get('selector')
        if destination.get('model', {}).get('id') != mid or selector in result: continue
        row = owners.get(destination.get('id'))
        if row is None or row['shape'] != destination.get('shape') or row['initializer']['kind'] != 'values': continue
        if selector not in hashes: hashes[selector] = float32_values(row['initializer']['values'], row['shape'])[1]
        if hashes[selector] != destination.get('values_sha256'): continue
        doc = workspace.get_spec('parameter-binding', sid, revision)
        result[doc['destination']['selector']] = {'reference': {'kind': 'parameter-binding', 'id': sid, 'revision': revision},
            'source': copy.deepcopy(doc['source']), 'receipt_model': copy.deepcopy(doc['destination']['model']),
            'scope': 'current-initial-value-bytes-match-pinned-receipt; not-model-revision-equality'}
    return result


def admit(workspace, token, selector, selection, *, binding_id=None, name=None):
    candidate = preview(workspace, token, selector, selection, binding_id=binding_id, name=name)
    before = workspace.db.total_changes
    with _atomic(workspace):
        doc = candidate['document']
        if digest(doc) != digest(workspace.models()[token['id']]): workspace.put(doc)
        model = _retain(workspace, 'model', candidate['model'])
        binding = _retain(workspace, 'parameter-binding', candidate['binding'])
    return {'reference': binding, 'model': model, 'changed': workspace.db.total_changes != before,
            'source': copy.deepcopy(candidate['binding']['source']), 'scope': candidate['scope']}
