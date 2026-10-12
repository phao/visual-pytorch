"""Internal owner-aware policy normalization; no tensors, RNG or file writes.

Not yet a persisted graph field or an automatic compile/training hook. Caller
supplies the existing semantic parameter schema and an explicit alias map.
Route/version admission must land before a workspace advertises this policy.
"""
import copy
import math
from .core import GraphError
from .evidence import digest

FLOAT32_MAX = 3.4028234663852886e38
MAX_SEED = 2147483647


def graph_aliases(models, mid):
    """Pure occurrence-ID -> canonical-owner-ID inventory for a concrete graph.

    Traverse structural reference/ownership scopes; never parse slash-separated
    IDs. Existing ambiguous canonical or occurrence spellings refuse. This is
    not a family-export schema adapter or a new persisted identity convention.
    """
    from .core import operation_nodes, validate_model
    from .evidence import parameter_schema
    from .graph import parameter_owners
    schema = parameter_schema(models, mid)
    canonical = resolve(schema)['owners']  # Also refuses old slash-ID collisions.
    aliases, occurrences = {}, {}

    def visit(current, occurrence_prefix, owner_prefix):
        contract = validate_model(models, current)
        owners = parameter_owners(models[current])
        for node in operation_nodes(models[current]):
            occurrence = occurrence_prefix + [node['id']]
            owner = owner_prefix + [owners[node['id']]]
            if node['type'] in {'Module', 'Recurrent'}:
                visit(node['ref'], occurrence, owner)
                continue
            if node['type'] == 'LSTM':
                names = contract['shapes'][node['id']]['lstm']['parameter_shapes']
            elif node['type'] in {'Linear', 'Conv2d'}:
                names = ['weight'] + (['bias'] if node.get('bias', True) else [])
            else:
                continue
            for name in names:
                sid = '/'.join(occurrence + [name]); target = '/'.join(owner + [name])
                components = tuple(occurrence + [name])
                _require(sid not in occurrences or occurrences[sid] == components,
                         'ambiguous occurrence identity; structural components collide.')
                occurrences[sid] = components
                _require(target in canonical, 'graph alias must resolve to a canonical owner.')
                _require(sid not in canonical or sid == target,
                         'graph alias cannot shadow a different canonical owner.')
                if sid != target:
                    _require(sid not in aliases or aliases[sid] == target,
                             'graph alias has conflicting owners.')
                    aliases[sid] = target
    visit(mid, [mid], [mid])
    return dict(sorted(aliases.items()))


def _require(condition, message):
    if not condition:
        raise GraphError('Parameter policy: ' + message)


def _keys(value, required, optional=()):
    _require(isinstance(value, dict) and set(required) <= set(value)
             and set(value) <= set(required) | set(optional), 'unsupported or missing fields.')


def _number(value):
    _require(type(value) in (int, float) and abs(value) <= FLOAT32_MAX
             and math.isfinite(value), 'values must be finite float32-range numbers, not booleans.')
    return float(value)


def _tensor(value, shape):
    if shape:
        _require(isinstance(value, list) and len(value) == shape[0], 'initial value shape mismatch.')
        return [_tensor(part, shape[1:]) for part in value]
    else:
        return _number(value)


def initializer(value, shape):
    """Normalize a closed declaration, without drawing values or consuming RNG."""
    _require(isinstance(value, dict), 'initializer must be an object.')
    kind = value.get('kind')
    result = None
    if kind in ('constructor', 'zeros'):
        _keys(value, ('kind',))
    elif kind == 'values':
        _keys(value, ('kind', 'dtype', 'values'))
        _require(value['dtype'] == 'float32', 'explicit values require dtype float32.')
        result = {'kind': 'values', 'dtype': 'float32', 'values': _tensor(value['values'], shape)}
    elif kind in ('normal', 'uniform'):
        fields = ('mean', 'std') if kind == 'normal' else ('low', 'high')
        _keys(value, ('kind', 'seed', *fields))
        _require(type(value['seed']) is int and 0 <= value['seed'] <= MAX_SEED,
                 'seed must be an integer from 0 to 2147483647.')
        a, b = (_number(value[key]) for key in fields)
        if kind == 'normal':
            _require(b > 0, 'normal standard deviation must be positive.')
        else:
            _require(a < b and b - a <= FLOAT32_MAX,
                     'uniform bounds must be ordered with a float32-range width.')
        result = {'kind': kind, 'seed': value['seed'], fields[0]: a, fields[1]: b}
    else:
        raise GraphError('Parameter policy: unsupported initializer kind.')
    return copy.deepcopy(value) if result is None else result


def resolve(schema, policy=None, *, aliases=None):
    """One policy per unique semantic owner, not per diagram occurrence.

    schema is the existing path -> {id, shape} pure parameter schema. aliases
    maps additional semantic IDs directly to a canonical owner ID. Partial
    declarations default to trainable/constructor. Equal alias declarations
    coalesce; incompatible declarations refuse regardless of insertion order.
    """
    _require(isinstance(schema, dict), 'parameter schema must be an object.')
    owners = {}
    for path, entry in schema.items():
        _require(isinstance(path, str) and bool(path), 'parameter path must be nonempty text.')
        _keys(entry, ('id', 'shape'))
        sid, shape = entry['id'], entry['shape']
        _require(isinstance(sid, str) and bool(sid) and sid not in owners,
                 'semantic owner IDs must be nonempty and unambiguous.')
        _require(isinstance(shape, list) and 1 <= len(shape) <= 8
                 and all(type(v) is int and 1 <= v <= 9007199254740991 for v in shape),
                 'owner shape must have one to eight positive integer dimensions.')
        owners[sid] = {'path': path, 'shape': list(shape)}
    aliases = {} if aliases is None else aliases
    _require(isinstance(aliases, dict), 'aliases must be an object.')
    for alias, target in aliases.items():
        _require(isinstance(alias, str) and bool(alias) and isinstance(target, str)
                 and target in owners and (alias not in owners or alias == target),
                 'aliases must refer directly to one known owner without shadowing another owner.')
    policy = {'version': 1, 'parameters': {}} if policy is None else policy
    _keys(policy, ('version', 'parameters'))
    _require(type(policy['version']) is int and policy['version'] == 1,
             'supports integer version 1 only.')
    _require(isinstance(policy['parameters'], dict), 'parameters must be an object.')
    declared = {}
    for sid, entry in policy['parameters'].items():
        _require(isinstance(sid, str), 'parameter identity must be text.')
        owner = aliases.get(sid, sid)
        _require(owner in owners, 'unknown parameter owner: ' + sid + '.')
        _keys(entry, (), ('trainable', 'initializer'))
        trainable = entry.get('trainable', True)
        _require(type(trainable) is bool, 'trainable must be a boolean.')
        init = initializer(entry.get('initializer', {'kind': 'constructor'}), owners[owner]['shape'])
        normalized = {'trainable': trainable, 'initializer': init}
        _require(owner not in declared or declared[owner] == normalized,
                 'conflicting declarations for shared owner: ' + owner + '.')
        declared[owner] = normalized
    resolved, stored, trainable = {}, 0, 0
    for sid in sorted(owners):
        entry = declared.get(sid, {'trainable': True, 'initializer': {'kind': 'constructor'}})
        count = math.prod(owners[sid]['shape'])
        stored += count
        if entry['trainable']:
            trainable += count
        resolved[sid] = {'id': sid, **owners[sid], **copy.deepcopy(entry), 'dtype': 'float32'}
    # Execution addresses are separately recorded, not semantic ownership.
    semantic = {sid: {k: v for k, v in entry.items() if k != 'path'}
                for sid, entry in resolved.items()}
    return {'version': 1, 'owners': resolved,
            'counts': {'stored': stored, 'trainable': trainable, 'frozen': stored - trainable},
            'identity': digest({'version': 1, 'owners': semantic})}
