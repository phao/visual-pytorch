"""Guarded pure GUI/CLI owner edits; no numerical imports or user-file writes."""
import copy
import json
from .core import GraphError, validate_model
from .evidence import digest, identity, parameter_schema
from .graph import parameter_owners
from .parameter_policy import graph_aliases, initializer, resolve
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
            'has_local_policy': any(json.loads(key)[0] in {block for block, target in parameter_owners(models[mid]).items() if target == canonical}
                for key in models[mid]['graph'].get('parameter_policy', {}).get('parameters', {})),
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


def trainable_block_counts(models, mid):
    """Visible-block optimizer ownership, counting each canonical owner once.

    Frozen storage still belongs to memory and stored-count contracts; only
    learnable-height presentation consumes this projection. Nested owners are
    attributed to their visible root Module/Recurrent block, not T occurrences.
    """
    import math
    resolved = _resolved(models, mid)
    counts = {node['id']: 0 for node in models[mid]['graph']['nodes']}
    for sid, row in resolved['owners'].items():
        block = json.loads(sid)[1]
        if block not in counts: raise GraphError('Parameter owner has no visible root block.')
        if row['trainable']: counts[block] += math.prod(row['shape'])
    if sum(counts.values()) != resolved['counts']['trainable']:
        raise GraphError('Visible learnable counts disagree with unique optimizer owners.')
    return {nid: str(count) for nid, count in counts.items()}


def build(models, token, selector, changes_json):
    if not isinstance(token, dict) or set(token) != set(TOKEN) or token != _context(models, token.get('id'), token.get('node')):
        raise GraphError('Architecture changed; reopen parameter policy.')
    if not isinstance(changes_json, str) or len(changes_json.encode('utf-8')) > EDIT_LIMIT:
        raise GraphError('Parameter policy edit must be JSON of at most 64 KiB; use the CLI for larger exact values.')
    changes = _parse(changes_json)
    return build_changes(models, token, selector, changes)


def build_changes(models, token, selector, changes):
    """Closed direct-data route for validated retained values beyond GUI text limit."""
    if not isinstance(token, dict) or set(token) != set(TOKEN) or token != _context(models, token.get('id'), token.get('node')):
        raise GraphError('Architecture changed; reopen parameter policy.')
    if not isinstance(changes, dict) or set(changes) - {'trainable', 'initializer'}:
        raise GraphError('Parameter policy edit has unsupported fields.')
    info = prepare(models, token['id'], token['node'])
    selected = next((row for row in info['owners'] if row['selector'] == selector), None)
    if selected is None: raise GraphError('Parameter selector is not owned by the selected block.')
    effective = None if info['policy_error'] else _resolved(models, token['id'])['owners'][selected['id']]
    doc = copy.deepcopy(models[token['id']])
    policy = doc['graph'].get('parameter_policy', {'version': 1, 'parameters': {}})
    aliases = graph_aliases(models, token['id'])
    matching = [key for key in policy['parameters']
                if aliases.get(owner_key([token['id']] + json.loads(key)), owner_key([token['id']] + json.loads(key))) == selected['id']]
    if not matching: matching = [selector]
    original = copy.deepcopy(policy)
    for field, value in changes.items():
        if value is not None:
            if field == 'trainable':
                if type(value) is not bool: raise GraphError('Trainable must be boolean.')
            else: value = initializer(value, selected['shape'])
        # Do not create a declaration/history edit merely to redisplay the
        # effective inherited default. Explicit pinning remains possible by CLI.
        for key in matching:
            entry = policy['parameters'].setdefault(key, {})
            if value is None: entry.pop(field, None)
            elif effective is None or digest(value) != digest(effective[field]): entry[field] = value
    for key in matching:
        if key in policy['parameters'] and not policy['parameters'][key]: policy['parameters'].pop(key)
    if digest(policy) == digest(original): return doc
    doc['graph']['parameter_policy'] = policy
    prospective = dict(models); prospective[token['id']] = doc
    plan(prospective, token['id'])
    return doc


def reset_block(models, token):
    """Explicit atomic local-scope reset, including orphaned tensor selectors."""
    if not isinstance(token, dict) or set(token) != set(TOKEN) or token != _context(models, token.get('id'), token.get('node')):
        raise GraphError('Architecture changed; reopen parameter policy.')
    info = prepare(models, token['id'], token['node'])
    doc = copy.deepcopy(models[token['id']])
    if not info['has_local_policy']: return doc
    owners = parameter_owners(doc); canonical = owners[token['node']]
    blocks = {block for block, owner in owners.items() if owner == canonical}
    entries = doc['graph']['parameter_policy']['parameters']
    for key in list(entries):
        if json.loads(key)[0] in blocks: del entries[key]
    prospective = dict(models); prospective[token['id']] = doc
    plan(prospective, token['id'])
    return doc
