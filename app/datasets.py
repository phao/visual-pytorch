"""Versioned, inspectable data/task contracts. Pure Python: no Torch or Qt.

This first adapter is deliberately inline and finite. Storage capability claims
are made by adapters, not guessed from a URI. Preparation never uses global RNGs
or fits on validation/test records. All returned objects own their data.
"""
import copy
import math
import random

from .core import GraphError
from .evidence import digest

ROLES = ('train', 'validation', 'test')
DTYPES = {'float32', 'int64', 'bool', 'category', 'string'}
MAX_FLOAT32 = 3.4028234663852886e38


def _require(ok, message):
    if not ok:
        raise GraphError(message)


def _keys(obj, required, optional=(), context='Object'):
    _require(isinstance(obj, dict), context + ' must be an object.')
    _require(set(required) <= set(obj) <= set(required) | set(optional),
             context + ' has missing or unsupported fields.')


def _integer(value, minimum=0, maximum=2**53 - 1):
    return type(value) is int and minimum <= value <= maximum


def _identifier(value):
    return isinstance(value, str) and 0 < len(value) <= 200 and not any(ord(c) < 32 for c in value)


def _number(value):
    try:
        return type(value) in (float, int) and math.isfinite(value) and abs(value) <= MAX_FLOAT32
    except OverflowError:
        return False


def _version(doc, kind):
    _require(type(doc.get('version')) is int and doc['version'] == 1,
             kind + ' requires integer version 1.')


def _shape(shape, ragged=False):
    _require(isinstance(shape, list) and len(shape) <= 8 and all(
        _integer(d, 1) or (ragged and i == 0 and type(d) is int and d == -1)
        for i, d in enumerate(shape)), 'Field shape must be sample dimensions; only the first axis may be -1.')


def _flat(value):
    if isinstance(value, list):
        return [v for item in value for v in _flat(item)]
    return [value]


def _unflatten(values, shape):
    if not shape:
        _require(len(values) == 1, 'Invalid scalar buffer.')
        return values[0]
    width = math.prod(shape[1:])
    _require(len(values) == math.prod(shape), 'Invalid tensor buffer.')
    return [_unflatten(values[i * width:(i + 1) * width], shape[1:]) for i in range(shape[0])]


def _value(value, field, shape=None):
    shape = field['shape'] if shape is None else shape
    if value is None:
        _require(field.get('nullable', False), 'Missing value in nonnullable field ' + field['id'] + '.')
        return
    if shape:
        _require(isinstance(value, list) and (shape[0] == -1 or len(value) == shape[0]),
                 'Tensor value does not match field shape: ' + field['id'] + '.')
        for item in value:
            _value(item, field, shape[1:])
        return
    dtype = field['dtype']
    valid = (_number(value) if dtype == 'float32' else
             _integer(value, -(2**63), 2**63-1) if dtype == 'int64' else
             type(value) is bool if dtype == 'bool' else
             isinstance(value, str) if dtype == 'string' else
             type(value) in (str, int) and any(type(value) is type(v) and value == v for v in field['vocabulary']['values']))
    _require(valid, 'Invalid ' + dtype + ' value in field ' + field['id'] + '.')


def validate_dataset(spec):
    """Validate a DatasetSpec without allocating tensor models or modifying it."""
    _keys(spec, ('version', 'id', 'name', 'storage', 'fields', 'records'),
          ('provenance', 'usage', 'revision'), 'DatasetSpec')
    _version(spec, 'DatasetSpec')
    _require(_identifier(spec['id']) and _identifier(spec['name']), 'Dataset needs an ID and name.')
    _require(spec['storage'] == {'kind': 'inline-records', 'version': 1} and
             type(spec['storage'].get('version')) is int, 'Only inline-records adapter version 1 is implemented.')
    fields = spec['fields']
    _require(isinstance(fields, list) and fields, 'Dataset needs typed fields.')
    lookup = {}
    for field in fields:
        _keys(field, ('id', 'dtype', 'shape'), ('name', 'axes', 'nullable', 'vocabulary', 'modality', 'unit', 'coordinates'), 'Field')
        _require(_identifier(field['id']) and field['id'] not in lookup, 'Field IDs must be unique.')
        _require(isinstance(field['dtype'], str) and field['dtype'] in DTYPES, 'Unsupported field dtype.')
        _shape(field['shape'], ragged=True)
        _require('nullable' not in field or type(field['nullable']) is bool, 'nullable must be boolean.')
        if 'axes' in field:
            axes = field['axes']
            _require(isinstance(axes, list) and len(axes) == len(field['shape']) and
                     all(_identifier(a) for a in axes) and len(set(axes)) == len(axes), 'Field axes must be distinct sample-axis names.')
        if field['dtype'] == 'category':
            vocab = field.get('vocabulary')
            _keys(vocab, ('version', 'values'), context='Vocabulary')
            _version(vocab, 'Vocabulary')
            values = vocab['values']
            _require(isinstance(values, list) and len(values) >= 2 and
                     all(type(v) in (str, int) for v in values) and len({(type(v), v) for v in values}) == len(values),
                     'Category vocabulary needs at least two distinct fixed labels.')
        else:
            _require('vocabulary' not in field, 'Only category fields have vocabularies.')
        lookup[field['id']] = field
    records = spec['records']
    _require(isinstance(records, list) and len(records) >= 2, 'Dataset needs at least two records.')
    ids = set()
    for record in records:
        _keys(record, ('id', 'values'), context='Record')
        _require(_identifier(record['id']) and record['id'] not in ids, 'Observation IDs must be unique.')
        _require(isinstance(record['values'], dict) and set(record['values']) == set(lookup), 'Record field IDs must exactly match the schema.')
        ids.add(record['id'])
        for fid, field in lookup.items():
            _value(record['values'][fid], field)
    try:
        digest(spec)
    except (TypeError, ValueError, RecursionError) as error:
        raise GraphError('Dataset metadata must be finite JSON.') from error
    if 'revision' in spec:
        _require(spec['revision'] == digest({k: v for k, v in spec.items() if k != 'revision'}), 'Dataset revision does not match its content.')
    return lookup


def freeze_dataset(spec):
    """Materialize a private content-addressed revision, not a mutable reference."""
    validate_dataset(spec)
    result = copy.deepcopy(spec)
    result['revision'] = digest({k: v for k, v in result.items() if k != 'revision'})
    return result


def capabilities(spec):
    validate_dataset(spec)
    return {'adapter': 'inline-records-v1', 'random_access': True, 'known_cardinality': len(spec['records']),
            'repeatable': True, 'resumable_position': False}


def _records(spec):
    return {r['id']: r['values'] for r in spec['records']}


def check_manifest(spec, manifest):
    """A manifest is a selection, not a seed. Explicit exclusions are required."""
    validate_dataset(spec)
    _keys(manifest, ('version', 'dataset_revision', 'method', 'roles', 'excluded', 'protocol'), context='SplitManifest')
    _version(manifest, 'SplitManifest')
    _require(manifest['dataset_revision'] == freeze_dataset(spec)['revision'], 'Split manifest is for a different dataset revision.')
    _require(manifest['method'] in ('row', 'group', 'temporal', 'external'), 'Unsupported split method.')
    _keys(manifest['roles'], ROLES, context='Split roles')
    groups = list(manifest['roles'].values()) + [manifest['excluded']]
    _require(all(isinstance(g, list) and all(_identifier(v) for v in g) for g in groups), 'Split selections must be observation ID arrays.')
    flattened = [v for g in groups for v in g]
    _require(len(set(flattened)) == len(flattened) and set(flattened) == set(_records(spec)), 'Split selections must be disjoint and cover records, including explicit exclusions.')
    _require(manifest['roles']['train'] and manifest['roles']['test'], 'Training and test populations must be nonempty.')
    _require(isinstance(manifest['protocol'], dict), 'Split protocol must be an object.')
    values = _records(spec)
    if manifest['method'] == 'group':
        field = manifest['protocol'].get('field')
        _require(_identifier(field) and field in validate_dataset(spec), 'Unknown grouping field.')
        owners = {}
        for role, rows in list(manifest['roles'].items()) + [('excluded', manifest['excluded'])]:
            for rid in rows:
                v = values[rid][field]
                _require(type(v) in (str, int, bool), 'Group field must be a nonmissing scalar ID.')
                key = (type(v).__name__, str(v))
                _require(key not in owners or owners[key] == role, 'A subject/group occurs in different split roles.')
                owners[key] = role
    elif manifest['method'] == 'temporal':
        field = manifest['protocol'].get('field')
        _require(_identifier(field) and field in validate_dataset(spec), 'Unknown time field.')
        times = []
        for role in ROLES:
            ts = [values[rid][field] for rid in manifest['roles'][role]]
            _require(all(_number(v) for v in ts), 'Time must be a finite numeric scalar.')
            if ts:
                times.append((min(ts), max(ts)))
        _require(all(a[1] < b[0] for a, b in zip(times, times[1:])), 'Temporal roles must be strictly chronological; equal timestamps cannot cross roles.')
    return copy.deepcopy(manifest)


def make_split(spec, protocol):
    """Row/group shuffles start from sorted IDs, independent of physical row order."""
    fields = validate_dataset(spec)
    _keys(protocol, ('method',), ('seed', 'fractions', 'field', 'roles', 'excluded'), 'Split protocol')
    method = protocol['method']
    _require(method in ('row', 'group', 'temporal', 'external'), 'Unsupported split method.')
    rows = _records(spec)
    if method == 'external':
        _require(set(protocol) == {'method', 'roles', 'excluded'}, 'External split needs only roles and exclusions.')
        roles, excluded = protocol['roles'], protocol['excluded']
    else:
        allowed = {'method', 'fractions'} | ({'field'} if method in ('group', 'temporal') else set()) | ({'seed'} if method in ('row', 'group') else set())
        _require(set(protocol) == allowed, 'Split protocol must explicitly declare fractions, field and independent seed as appropriate.')
        fractions = protocol['fractions']
        _require(isinstance(fractions, list) and len(fractions) == 3 and all(_number(f) and 0 <= f <= 1 for f in fractions)
                 and fractions[0] > 0 and fractions[2] > 0 and abs(sum(fractions) - 1) < 1e-12, 'Split fractions are [train, validation, test] and sum to one.')
        units = {rid: [rid] for rid in sorted(rows)}
        if method in ('group', 'temporal'):
            fid = protocol['field']
            _require(_identifier(fid) and fid in fields and fields[fid]['shape'] == [], 'Split field must be a declared scalar.')
            units = {}
            for rid in sorted(rows):
                value = rows[rid][fid]
                _require(_number(value) if method == 'temporal' else type(value) in (str, int, bool), 'Split field contains a missing or nonscalar value.')
                key = value if method == 'temporal' else (type(value).__name__, str(value))
                units.setdefault(key, []).append(rid)
        keys = sorted(units)
        if method != 'temporal':
            seed = protocol['seed']
            _require(_integer(seed, 0, 2**31 - 1), 'Split seed must be an integer from 0 to 2³¹ − 1.')
            random.Random(seed).shuffle(keys)
        ntrain = int(len(keys) * fractions[0]); nval = int(len(keys) * fractions[1])
        _require(ntrain > 0 and len(keys) - ntrain - nval > 0 and (fractions[1] == 0 or nval > 0), 'Too few independent split units for the requested fractions.')
        selected = (keys[:ntrain], keys[ntrain:ntrain + nval], keys[ntrain + nval:])
        roles = {role: [rid for key in chosen for rid in units[key]] for role, chosen in zip(ROLES, selected)}
        excluded = []
    manifest = {'version': 1, 'dataset_revision': freeze_dataset(spec)['revision'], 'method': method,
                'roles': copy.deepcopy(roles), 'excluded': copy.deepcopy(excluded), 'protocol': copy.deepcopy(protocol)}
    if method in ('row', 'group'):
        manifest['protocol']['algorithm'] = 'python-mt19937-fisher-yates-v1'
    if method in ('group', 'temporal'):
        manifest['protocol']['allocation_unit'] = 'distinct-field-values'
    return check_manifest(spec, manifest)


def validate_transforms(transforms):
    """Intrinsic syntax is independent of a particular field/model revision."""
    _require(isinstance(transforms, list), 'Transforms must be an ordered array.')
    for transform in transforms:
        _keys(transform, ('kind',), ('length', 'value', 'scale', 'offset'), 'Transform')
        kind = transform['kind']
        if kind in ('onehot', 'impute-mean', 'standardize'):
            _require(set(transform) == {'kind'}, 'Transform has unsupported fields.')
        elif kind == 'pad':
            _require(set(transform) == {'kind', 'length', 'value'} and _integer(transform['length'], 1) and
                     _number(transform['value']), 'pad requires positive length and finite value.')
        elif kind == 'affine':
            _require(set(transform) == {'kind', 'scale', 'offset'} and _number(transform['scale']) and
                     _number(transform['offset']) and transform['scale'] != 0, 'affine requires finite nonzero scale and finite offset.')
        else:
            raise GraphError('Unsupported deterministic transform: ' + str(kind))


def validate_task_spec(task):
    """Validate reusable task grammar; field/port compatibility is a later gate."""
    version = task.get('version') if isinstance(task, dict) else None
    _require(type(version) is int and version in (1, 2), 'TaskSpec supports integer versions 1 and 2.')
    _keys(task, ('version', 'id', 'name', 'inputs', 'heads') +
          (('diagnostic_outputs',) if version == 2 else ()), ('revision', 'description'), 'TaskSpec')
    if version == 2:
        diagnostics = task['diagnostic_outputs']
        _require(isinstance(diagnostics, list) and all(_identifier(p) for p in diagnostics) and
                 len(set(diagnostics)) == len(diagnostics), 'Diagnostic outputs must be unique stable port IDs.')
    _require(_identifier(task['id']) and _identifier(task['name']), 'Task needs an ID and name.')
    _require(isinstance(task['inputs'], list) and task['inputs'] and isinstance(task['heads'], list) and task['heads'], 'Task inputs and heads must be nonempty arrays.')
    for item in task['inputs']:
        _keys(item, ('port', 'field', 'transforms'), ('derived', 'length'), 'Input binding')
        _require(_identifier(item['port']) and _identifier(item['field']), 'Task bindings require stable port/field IDs.')
        validate_transforms(item['transforms'])
        if 'derived' in item:
            _require(item['derived'] in ('length', 'mask') and item['transforms'] == [] and _integer(item.get('length'), 1), 'Derived binding needs length/mask, explicit length and no transforms.')
        else:
            _require('length' not in item, 'Only derived bindings declare length outside pad transforms.')
    for item in task['heads']:
        likelihood = isinstance(item, dict) and item.get('loss') == 'negative-log-likelihood'
        _require(not likelihood or version == 2, 'Target-free likelihood requires TaskSpec version 2.')
        _keys(item, ('id', 'port', 'loss', 'weight') + (() if likelihood else ('field', 'transforms')),
              ('mask', 'sample_weight'), 'Loss head')
        _require(all(_identifier(item[key]) for key in (('id', 'port') if likelihood else ('id', 'port', 'field'))) and
                 item['loss'] in ('mse', 'mae', 'cross-entropy', 'negative-log-likelihood') and
                 _number(item['weight']) and item['weight'] > 0, 'Head needs stable IDs, supported loss and positive weight.')
        if not likelihood:
            validate_transforms(item['transforms'])
            _require(item['loss'] != 'cross-entropy' or item['transforms'] == [], 'Cross-entropy index targets do not have transforms.')
        _require(all(key not in item or _identifier(item[key]) for key in ('mask', 'sample_weight')), 'Mask/weight requires a stable field ID.')
    _require(len({i['port'] for i in task['inputs']}) == len(task['inputs']) and
             len({i['port'] for i in task['heads']}) == len(task['heads']) and
             len({i['id'] for i in task['heads']}) == len(task['heads']), 'Task input/output ports and head IDs must be unique.')
    try:
        value = digest({k: v for k, v in task.items() if k != 'revision'})
    except (TypeError, ValueError, RecursionError) as error:
        raise GraphError('Task metadata must be finite JSON.') from error
    _require('revision' not in task or task['revision'] == value, 'Task revision does not match content.')
    return copy.deepcopy(task)


def _pipeline(field, transforms):
    """Validate order, resolve output shape/axes, and return deterministic semantics."""
    validate_transforms(transforms)
    shape, dtype, axes = list(field['shape']), field['dtype'], field.get('axes')
    for transform in transforms:
        _keys(transform, ('kind',), ('length', 'value', 'scale', 'offset'), 'Transform')
        kind = transform['kind']
        if kind == 'onehot':
            _require(set(transform) == {'kind'} and dtype == 'category' and shape == [], 'onehot requires a scalar fixed-vocabulary category.')
            shape, dtype, axes = [len(field['vocabulary']['values'])], 'float32', None
        elif kind == 'pad':
            _require(set(transform) == {'kind', 'length', 'value'} and shape and shape[0] == -1 and dtype in ('float32', 'int64', 'bool') and
                     _integer(transform['length'], 1) and _number(transform['value']), 'pad requires a numeric ragged first axis, positive length and explicit value.')
            shape = [transform['length']] + shape[1:]
            dtype = 'float32'
        elif kind in ('impute-mean', 'standardize'):
            _require(set(transform) == {'kind'} and -1 not in shape and dtype in ('float32', 'int64', 'bool'), kind + ' requires a fixed numeric shape.')
            dtype = 'float32'
        elif kind == 'affine':
            _require(set(transform) == {'kind', 'scale', 'offset'} and -1 not in shape and dtype in ('float32', 'int64', 'bool') and
                     _number(transform['scale']) and _number(transform['offset']) and transform['scale'] != 0, 'affine needs explicit finite nonzero scale and offset.')
            dtype = 'float32'
        else:
            raise GraphError('Unsupported deterministic transform: ' + str(kind))
    _require(-1 not in shape, 'Variable-length numeric input needs explicit pad collation.')
    return shape or [1], dtype, axes if shape else None


def _column(values, width):
    rows = [[None] * width if v is None else _flat(v) for v in values]
    _require(all(len(row) == width for row in rows), 'Transform changed a tensor shape.')
    return rows


def _apply_step(values, shape, field, transform, state=None):
    kind = transform['kind']
    values = copy.deepcopy(values)
    if kind == 'onehot':
        vocab = field['vocabulary']['values']
        _require(all(v is not None for v in values), 'Missing category needs an explicitly defined ontology value.')
        return [[float(type(v) is type(c) and v == c) for c in vocab] for v in values], [len(vocab)], {'kind': kind, 'vocabulary': copy.deepcopy(field['vocabulary'])}
    if kind == 'pad':
        length = transform['length']; tail = shape[1:]
        for i, value in enumerate(values):
            _require(isinstance(value, list) and len(value) <= length, 'pad does not silently truncate or replace missing sequences.')
            fill = _unflatten([transform['value']] * math.prod(tail), tail)
            values[i] = value + [copy.deepcopy(fill) for _ in range(length - len(value))]
        return values, [length] + tail, copy.deepcopy(transform)
    width = math.prod(shape)
    rows = _column(values, width)
    if kind == 'impute-mean':
        if state is None:
            columns = [[row[i] for row in rows if row[i] is not None] for i in range(width)]
            _require(all(columns), 'Cannot fit imputation: a feature is missing in every training observation.')
            state = {'kind': kind, 'mean': [math.fsum(c) / len(c) for c in columns]}
        rows = [[state['mean'][i] if v is None else v for i, v in enumerate(row)] for row in rows]
    else:
        _require(all(v is not None for row in rows for v in row), 'Numeric missing values require impute-mean before this operation.')
        if kind == 'standardize':
            if state is None:
                mean = [math.fsum(row[i] for row in rows) / len(rows) for i in range(width)]
                std = [math.sqrt(math.fsum((row[i] - mean[i])**2 for row in rows) / len(rows)) for i in range(width)]
                state = {'kind': kind, 'mean': mean, 'scale': [s if s else 1. for s in std], 'variance': 'population-ddof0', 'constant_scale': 1.}
            rows = [[(v - state['mean'][i]) / state['scale'][i] for i, v in enumerate(row)] for row in rows]
        elif kind == 'affine':
            state = copy.deepcopy(transform)
            rows = [[v * state['scale'] + state['offset'] for v in row] for row in rows]
    _require(all(_number(v) for row in rows for v in row), 'Transform produced a nonfinite or non-float32 value.')
    return [_unflatten(row, shape) for row in rows], shape, state


def validate_task(task, dataset, contract):
    """Stable field-to-port compatibility; names and record positions are not keys."""
    fields = validate_dataset(dataset)
    validate_task_spec(task)
    _require(_identifier(task['id']) and _identifier(task['name']), 'Task needs an ID and name.')
    if 'revision' in task:
        _require(task['revision'] == digest({k: v for k, v in task.items() if k != 'revision'}), 'Task revision does not match content.')
    _require(isinstance(task['inputs'], list) and isinstance(task['heads'], list), 'Task inputs and heads must be arrays.')
    ports = {g['id']: g for g in contract['input_groups']}
    seen = set()
    for item in task['inputs']:
        _keys(item, ('port', 'field', 'transforms'), ('derived', 'length'), 'Input binding')
        _require(_identifier(item['port']) and item['port'] in ports and item['port'] not in seen, 'Input binding needs a unique existing stable port ID.')
        _require(_identifier(item['field']) and item['field'] in fields, 'Unknown input field ID.')
        seen.add(item['port']); field = fields[item['field']]
        if 'derived' in item:
            _require(field['shape'] and field['shape'][0] == -1 and item['transforms'] == [] and item['derived'] in ('length', 'mask') and
                     _integer(item.get('length'), 1), 'Derived sequence length/mask needs the explicit padded length and no transforms.')
            shape = [1] if item['derived'] == 'length' else [item['length']]
            dtype, axes = 'float32', None
        else:
            _require('length' not in item, 'Only derived bindings declare length outside pad transforms.')
            shape, dtype, axes = _pipeline(field, item['transforms'])
        _require(dtype in ('float32', 'int64', 'bool') and shape == ports[item['port']]['shape'], 'Input field/transform shape does not match port ' + item['port'] + '.')
        _require(not ports[item['port']].get('axes') or axes == ports[item['port']]['axes'], 'Input axes do not match port ' + item['port'] + '; use an explicit architecture adapter.')
    _require(seen == set(ports), 'Bind every model input exactly once.')
    for item in task['inputs']:
        if 'derived' in item:
            _require(any(b['field'] == item['field'] and 'derived' not in b and
                         any(t['kind'] == 'pad' and t['length'] == item['length'] for t in b['transforms'])
                         for b in task['inputs']), 'Derived length/mask must match an explicit pad binding for the same field and length.')
    ports = {g['id']: g for g in contract['output_groups']}
    seen = set()
    for item in task['heads']:
        _require(_identifier(item['id']) and _identifier(item['port']) and item['port'] in ports and item['port'] not in seen, 'Loss head needs a unique output port ID.')
        _require(_number(item['weight']) and item['weight'] > 0, 'Head weight must be positive and finite.')
        seen.add(item['port'])
        if item['loss'] == 'negative-log-likelihood':
            _require(ports[item['port']].get('representation') == 'log-evidence',
                     'Likelihood requires a proven log-evidence output, not logits, labels or posterior probabilities.')
            target_shape = ports[item['port']]['shape']
        else:
            _require(_identifier(item['field']) and item['field'] in fields, 'Unknown target field ID.')
            field = fields[item['field']]
        if item['loss'] == 'cross-entropy':
            _require(field['dtype'] == 'category' and item['transforms'] == [] and ports[item['port']]['shape'] == [len(field['vocabulary']['values'])] + field['shape'],
                     'Cross-entropy uses logits with first sample axis = fixed vocabulary and categorical index targets; no target transforms.')
            target_shape = field['shape']
        elif item['loss'] != 'negative-log-likelihood':
            _require(item['loss'] in ('mse', 'mae'), 'Supported heads are mse, mae and cross-entropy.')
            target_shape, dtype, axes = _pipeline(field, item['transforms'])
            _require(dtype in ('float32', 'int64', 'bool') and target_shape == ports[item['port']]['shape'], 'Regression target shape does not match output port.')
            _require(not ports[item['port']].get('axes') or axes == ports[item['port']]['axes'], 'Target axes do not match output port.')
        for key in ('mask', 'sample_weight'):
            if key in item:
                _require(_identifier(item[key]), 'Mask/weight requires a stable field ID.')
                f = fields.get(item[key])
                _require(f is not None and f['dtype'] in ('bool', 'float32', 'int64') and f['shape'] in ([], target_shape) and not f.get('nullable'), 'Mask/weight must be nonmissing numeric scalar or target-shaped field.')
    diagnostics = set(task.get('diagnostic_outputs', []))
    _require(not (diagnostics & seen) and diagnostics | seen == set(ports) and
             len({h['id'] for h in task['heads']}) == len(task['heads']),
             'Every output needs one loss head or an explicit diagnostic declaration, never both.')
    try:
        digest(task)
    except (TypeError, ValueError, RecursionError) as error:
        raise GraphError('Task metadata must be finite JSON.') from error
    return copy.deepcopy(task)


def _prepare_column(binding, field, rows, train_ids):
    if 'derived' in binding:
        values = {}
        length = binding['length']
        for rid, record in rows.items():
            value = record[field['id']]
            _require(isinstance(value, list) and len(value) <= length, 'Derived length/mask cannot silently truncate or infer a missing sequence.')
            values[rid] = [float(len(value))] if binding['derived'] == 'length' else [float(i < len(value)) for i in range(length)]
        return values, {'field': field['id'], 'derived': binding['derived'], 'length': length, 'steps': []}
    all_ids = sorted(rows)
    values = [rows[rid][field['id']] for rid in all_ids]
    training = [rows[rid][field['id']] for rid in train_ids]
    shape = list(field['shape']); steps = []
    for transform in binding['transforms']:
        training, next_shape, state = _apply_step(training, shape, field, transform)
        values, applied_shape, _ = _apply_step(values, shape, field, transform, state)
        _require(next_shape == applied_shape, 'Fitted/evaluation transform shape mismatch.')
        shape = next_shape; steps.append(state)
    width = math.prod(shape)
    numeric = _column(values, width)
    _require(all(v is not None and (type(v) is bool or _number(v)) for row in numeric for v in row), 'Prepared numeric binding contains missing/non-numeric values; add explicit imputation.')
    values = [_unflatten([float(v) for v in row], shape or [1]) for row in numeric]
    return dict(zip(all_ids, values)), {'field': field['id'], 'steps': steps, 'sample_shape': shape or [1]}


def prepare(dataset, task, contract, manifest, *, constrained=True, memory_limit_percent=10., available_bytes=None):
    """Execute deterministic collation and train-only fitting, before tensorization.

    Stable port IDs key batches. A runtime adapter may use the captured contract's
    human names, but cannot reinterpret the field binding when a node is renamed.
    Cross-entropy target indices always use the fixed vocabulary's declared order.
    """
    validate_task(task, dataset, contract)
    check_manifest(dataset, manifest)
    fields = validate_dataset(dataset); rows = _records(dataset)
    selected = {rid for ids in manifest['roles'].values() for rid in ids}
    rows = {rid: values for rid, values in rows.items() if rid in selected}
    # Check expanded collation BEFORE padding/copying; a tiny source sequence may
    # declare an enormous padded tensor. This is the same fail-closed RAM policy
    # as other CPU preflights, not a hidden maximum-feature authoring limit.
    from .memory import available_memory, memory_percent
    _require(type(constrained) is bool, 'Constrained mode must be boolean.')
    try:
        percentage = memory_percent(memory_limit_percent)
    except ValueError as error:
        raise GraphError(str(error)) from error
    prepared_elements = sum(math.prod(g['shape']) for g in contract['input_groups'])
    prepared_elements += 3 * sum(math.prod(g['shape']) for g in contract['output_groups'])
    estimate = 8 * 1024**2 + 128 * len(rows) * prepared_elements
    if constrained:
        available = available_memory() if available_bytes is None else available_bytes
        _require(_integer(available), 'Cannot observe available RAM for typed data preparation.')
        _require(estimate <= available * percentage / 100., 'Typed data preparation exceeds the constrained available-RAM budget; reduce padded shapes/population or explicitly disable constrained mode.')
    train_ids = manifest['roles']['train']
    inputs, targets, masks, weights = {}, {}, {}, {}
    fitted = {'version': 1, 'dataset_revision': freeze_dataset(dataset)['revision'],
              'task_revision': digest({k: v for k, v in task.items() if k != 'revision'}),
              'fit_observation_ids': list(train_ids), 'inputs': {}, 'heads': {}}
    for binding in task['inputs']:
        inputs[binding['port']], fitted['inputs'][binding['port']] = _prepare_column(binding, fields[binding['field']], rows, train_ids)
    for head in task['heads']:
        hid = head['id']
        field = fields[head['field']] if 'field' in head else None
        if head['loss'] == 'negative-log-likelihood':
            fitted['heads'][hid] = {'kind': 'target-free-log-evidence'}
        elif head['loss'] == 'cross-entropy':
            vocab = field['vocabulary']['values']
            def indices(v):
                if isinstance(v, list):
                    return [indices(x) for x in v]
                _require(v is not None, 'Missing categorical targets need a separately implemented masked-target adapter.')
                return next(i for i, label in enumerate(vocab) if type(v) is type(label) and v == label)
            targets[hid] = {rid: indices(r[head['field']]) for rid, r in rows.items()}
            fitted['heads'][hid] = {'field': field['id'], 'vocabulary': copy.deepcopy(field['vocabulary']), 'steps': []}
        else:
            targets[hid], fitted['heads'][hid] = _prepare_column(head, field, rows, train_ids)
        masks[hid] = {}; weights[hid] = {}
        for rid, record in rows.items():
            mask = copy.deepcopy(record[head['mask']]) if 'mask' in head else 1.
            weight = copy.deepcopy(record[head['sample_weight']]) if 'sample_weight' in head else 1.
            _require(all(type(v) is bool or (_number(v) and v in (0, 1)) for v in _flat(mask)), 'Mask elements must be boolean or exactly 0/1.')
            _require(all(_number(v) and v >= 0 for v in _flat(weight)), 'Sample weights must be nonnegative finite numbers.')
            masks[hid][rid] = mask; weights[hid][rid] = weight
    populations = {}
    for role, ids in manifest['roles'].items():
        populations[role] = {'observation_ids': list(ids),
            'inputs': {port: [column[rid] for rid in ids] for port, column in inputs.items()},
            'targets': {hid: [column[rid] for rid in ids] for hid, column in targets.items()},
            'masks': {hid: [column[rid] for rid in ids] for hid, column in masks.items()},
            'sample_weights': {hid: [column[rid] for rid in ids] for hid, column in weights.items()}}
    return {'version': 1, 'preparation_estimate': {'version': 1, 'estimated_bytes': estimate},
            'manifest': copy.deepcopy(manifest), 'fitted': fitted,
            'populations': populations, 'contract': copy.deepcopy(contract)}


def preview(dataset, limit=20):
    """Bounded diagnostics; no shared random state, fitting or split mutation."""
    fields = validate_dataset(dataset)
    _require(_integer(limit, 1, 100), 'Preview limit must be 1–100.')
    rows = sorted(dataset['records'], key=lambda r: r['id'])[:limit]
    summary = {}
    for fid, field in fields.items():
        vals = [v for row in rows for v in _flat(row['values'][fid])]
        numeric = [v for v in vals if _number(v)]
        info = {'dtype': field['dtype'], 'shape': copy.deepcopy(field['shape']), 'missing_leaves': sum(v is None for v in vals)}
        if numeric:
            info['range'] = [min(numeric), max(numeric)]
        if field['dtype'] == 'category':
            info['vocabulary'] = copy.deepcopy(field['vocabulary'])
            info['counts'] = [sum(type(v) is type(c) and v == c for v in vals) for c in field['vocabulary']['values']]
        summary[fid] = info
    return {'scope': 'bounded-preview-not-a-dataset-audit', 'total_records': len(dataset['records']),
            'selected_records': len(rows), 'records': copy.deepcopy(rows), 'fields': summary}
