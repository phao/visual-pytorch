"""Guarded pure GUI/CLI owner edits; no numerical imports or user-file writes."""
import copy
import json
from .core import GraphError, validate_model
from .evidence import digest, identity, parameter_schema
from .graph import parameter_owners
from .parameter_policy import initializer, resolve
from .policy_graph import expansion_bound, owner_key, plan, present
from .revisions import _parse

TOKEN = ('id', 'node', 'architecture')
EDIT_LIMIT = 65536


def _resolved(models, mid):
    validate_model(models, mid)
    expansion_bound(models, mid)
    if present(models, mid): return plan(models, mid)['resolved']
    return resolve(parameter_schema(models, mid, identity_version=2))


def _context(models, mid, nid):
    if not isinstance(mid, str) or not isinstance(nid, str) or mid not in models or not any(n['id'] == nid for n in models[mid]['graph']['nodes']):
        raise GraphError('Choose an existing block for parameter policy.')
    return {'id': mid, 'node': nid, 'architecture': identity(models, mid)['architecture']}


def _small_json(value):
    fragments, size = [], 0
    for part in json.JSONEncoder(allow_nan=False, separators=(',', ':')).iterencode(value):
        size += len(part.encode('utf-8'))
        if size > EDIT_LIMIT: return None
        fragments.append(part)
    return ''.join(fragments)


def prepare(models, mid, nid):
    context = _context(models, mid, nid)
    from .graph import contract as shape_contract
    shape_contract(models, mid); expansion_bound(models, mid)
    policy_error = None
    try: resolved = _resolved(models, mid)
    except GraphError as error:
        policy_error = str(error)
        # Shapes can be valid while an edited layer no longer matches its old
        # initializer. Expose a repair editor, never a pretend effective policy.
        resolved = resolve(parameter_schema(models, mid, identity_version=2))
    canonical = parameter_owners(models[mid])[nid]
    owners = []
    for sid, row in resolved['owners'].items():
        parts = json.loads(sid)
        if parts[:2] != [mid, canonical]: continue
        current = mid
        for block in parts[1:-2]:
            node = next(n for n in models[current]['graph']['nodes'] if n['id'] == block)
            current = node['ref']
        selector = owner_key(parts[1:])
        raw = models[mid]['graph'].get('parameter_policy', {}).get('parameters', {}).get(selector, {}) if policy_error else {}
        init = raw.get('initializer', row['initializer'])
        encoded = _small_json(init)
        owners.append({'id': sid, 'selector': owner_key(parts[1:]), 'name': ' / '.join(parts[1:]),
                       'shape': row['shape'], 'trainable': row['trainable'],
                       'initializer_kind': init.get('kind', 'constructor'), 'initializer_json': encoded,
                       'effective_available': policy_error is None,
                       'definition': {'model': current, 'node': parts[-2]},
                       'shared': canonical != nid})
    return {**context, 'owners': sorted(owners, key=lambda row: row['selector']), 'policy_error': policy_error,
            'note': 'Initial parameters for a new run, not optimizer resume. Explicit conflicts across shared/reference scopes refuse.'}


def block_summary(models, mid):
    resolved = _resolved(models, mid); counts = {}
    for sid, row in resolved['owners'].items():
        block = json.loads(sid)[1]
        group = counts.setdefault(block, {'trainable': 0, 'frozen': 0, 'stored': 0})
        import math
        size = math.prod(row['shape']); group['stored'] += size
        group['trainable' if row['trainable'] else 'frozen'] += size
    owners = parameter_owners(models[mid])
    return {nid: {**counts[owner], 'owner': owner, 'shared': nid != owner}
            for nid, owner in owners.items() if owner in counts}


def build(models, token, selector, changes_json):
    if not isinstance(token, dict) or set(token) != set(TOKEN) or token != _context(models, token.get('id'), token.get('node')):
        raise GraphError('Architecture changed; reopen parameter policy.')
    if not isinstance(changes_json, str) or len(changes_json.encode('utf-8')) > EDIT_LIMIT:
        raise GraphError('Parameter policy edit must be JSON of at most 64 KiB; use the CLI for larger exact values.')
    changes = _parse(changes_json)
    if not isinstance(changes, dict) or set(changes) - {'trainable', 'initializer'}:
        raise GraphError('Parameter policy edit has unsupported fields.')
    info = prepare(models, token['id'], token['node'])
    selected = next((row for row in info['owners'] if row['selector'] == selector), None)
    if selected is None: raise GraphError('Parameter selector is not owned by the selected block.')
    effective = None if info['policy_error'] else _resolved(models, token['id'])['owners'][selected['id']]
    doc = copy.deepcopy(models[token['id']])
    policy = doc['graph'].get('parameter_policy', {'version': 1, 'parameters': {}})
    previous = policy['parameters'].get(selector, {})
    entry = copy.deepcopy(previous)
    for field, value in changes.items():
        if value is None: entry.pop(field, None); continue
        if field == 'trainable':
            if type(value) is not bool: raise GraphError('Trainable must be boolean.')
        else: value = initializer(value, selected['shape'])
        # Do not create a declaration/history edit merely to redisplay the
        # effective inherited default. Explicit pinning remains possible by CLI.
        if effective is None or value != effective[field] or field in previous: entry[field] = value
    if entry == previous: return doc
    if entry: policy['parameters'][selector] = entry
    else: policy['parameters'].pop(selector, None)
    doc['graph']['parameter_policy'] = policy
    prospective = dict(models); prospective[token['id']] = doc
    plan(prospective, token['id'])
    return doc
