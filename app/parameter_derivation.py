"""Explicit, deterministic float32 LoRA merge from exact retained tensor sources.

No numerical import, implicit checkpoint, destructive edit or optimizer resume.
This v1 local teaching kernel is bounded to one million scalar dot products.
"""
import copy
import json
import math
import struct
from .core import GraphError, default_model, validate_model
from .datasets import _identifier, _integer, _keys, _require
from .evidence import digest
from .experiment_objects import reference
from .family_records import _atomic
from .parameter_binding import _retain, _tensor_descriptor, validate_source
from .parameter_sources import read, float32_values, VALUE_FORMAT
from .policy_graph import owner_key, plan
from .revisions import _hash, frozen_document, model_document

ARITHMETIC = 'binary32-sequential-products-sum-scale-add-v1'
MAX_PRODUCTS = 1_000_000


def _f32(value):
    try: result = struct.unpack('<f', struct.pack('<f', value))[0]
    except (OverflowError, struct.error) as error: raise GraphError('LoRA merge overflow before admission.') from error
    _require(math.isfinite(result), 'LoRA merge must remain finite float32 before admission.')
    return result


def merged_values(base, a, b, *, alpha):
    _require(isinstance(base, list) and base and isinstance(base[0], list) and base[0] and
             isinstance(a, list) and a and isinstance(b, list), 'LoRA merge needs three matrices.')
    rows, columns, rank = len(base), len(base[0]), len(a)
    _require(_integer(rank, 1, min(rows, columns)) and rows * columns * rank <= MAX_PRODUCTS,
             'LoRA merge rank/work exceeds bounded local v1 arithmetic; no output allocated.')
    _require(type(alpha) in (int, float) and 0 < alpha <= 1e6 and math.isfinite(alpha), 'LoRA alpha must be finite positive, not boolean, at most1e6.')
    base, _ = float32_values(base, [rows, columns]); a, _ = float32_values(a, [rank, columns]); b, _ = float32_values(b, [rows, rank])
    scale = _f32(alpha / rank)
    result = []
    for i in range(rows):
        row = []
        for j in range(columns):
            total = 0.
            for k in range(rank): total = _f32(total + _f32(b[i][k] * a[k][j]))
            row.append(_f32(base[i][j] + _f32(scale * total)))
        result.append(row)
    return result


def validate(doc):
    _keys(doc, ('version', 'id', 'name', 'operation', 'sources', 'destination', 'optimizer_resume'), ('revision',), 'ParameterDerivation')
    _require(type(doc['version']) is int and doc['version'] == 1 and _identifier(doc['id']) and _identifier(doc['name']) and doc['optimizer_resume'] is False,
             'ParameterDerivation requires version1, ID/name and no optimizer resume.')
    op = doc['operation']; _keys(op, ('kind', 'alpha', 'rank', 'scale', 'arithmetic'), context='LoRA merge operation')
    _require(op['kind'] == 'lora-merge' and op['arithmetic'] == ARITHMETIC and _integer(op['rank'], 1) and
             type(op['alpha']) in (int, float) and 0 < op['alpha'] <= 1e6 and math.isfinite(op['alpha']) and
             type(op['scale']) in (int, float) and op['scale'] == op['alpha'] / op['rank'], 'Invalid explicit LoRA merge operation.')
    _keys(doc['sources'], ('base', 'A', 'B'), context='LoRA merge sources')
    for source in doc['sources'].values(): validate_source(source)
    base, a, b = (doc['sources'][key] for key in ('base', 'A', 'B'))
    _require(len(base['shape']) == len(a['shape']) == len(b['shape']) == 2, 'LoRA merge source parameters must be matrices.')
    rows, columns = base['shape']; rank = op['rank']
    _require(rank <= min(rows, columns) and a['shape'] == [rank, columns] and b['shape'] == [rows, rank] and rows * columns * rank <= MAX_PRODUCTS,
             'LoRA merge source matrix dimensions/work mismatch.')
    _require(all(all(source[key] == base[key] for key in ('run', 'attempt', 'checkpoint_id', 'model')) for source in (a, b)) and
             len({source['parameter']['id'] for source in (base, a, b)}) == 3,
             'LoRA merge v1 requires three distinct owners from the same exact retained checkpoint.')
    dest = doc['destination']; _keys(dest, ('model', 'selector', 'id', 'shape', 'dtype', 'values_sha256', 'value_format'), context='Derived destination')
    reference(dest['model'], 'model'); _tensor_descriptor(dest)
    _require(dest['shape'] == base['shape'] and dest['selector'] == owner_key(['linear1', 'weight']) and
             dest['id'] == owner_key([dest['model']['id'], 'linear1', 'weight']), 'Derived v1 destination must be its exact bias-free Linear owner.')


def dependencies(doc):
    return [doc['sources']['base']['run'], doc['sources']['base']['model'], doc['destination']['model']]


def _sources(resolver, selections):
    class Snapshot: get_spec = staticmethod(resolver)
    return {key: read(Snapshot(), value) for key, value in selections.items()}


def _topology(model, sources, alpha):
    """Prove a flat bias-free LoRA path, not three arbitrary tensors."""
    _require(model['version'] >= 6, 'LoRA merge requires executable component-identity freezing, not old ignored metadata.')
    root = model['root']; graph = model['models'][root]['graph']; nodes = {n['id']: n for n in graph['nodes']}
    ids = {}
    for key, source in sources.items():
        parts = json.loads(source['parameter']['id'])
        _require(len(parts) == 3 and parts[0] == root and parts[2] == 'weight', 'LoRA merge v1 requires flat component weight identities.')
        ids[key] = parts[1]
        n = nodes.get(parts[1]); _require(n and n['type'] == 'Linear' and n.get('bias', True) is False, 'LoRA source owners must be bias-free Linear boxes.')
    inputs = [n for n in nodes.values() if n['type'] == 'Input']; outputs = [n for n in nodes.values() if n['type'] == 'Output']
    scales = [n for n in nodes.values() if n['type'] == 'Scale']; sums = [n for n in nodes.values() if n['type'] == 'Add']
    _require(len(nodes) == 7 and len(inputs) == len(outputs) == len(scales) == len(sums) == 1,
             'LoRA merge v1 requires the complete flat two-branch architecture, no hidden nonlinear factor path.')
    x, y, scale, add = inputs[0], outputs[0], scales[0], sums[0]; rank = sources['A']['shape'][0]
    _require(scale['coefficient'] == alpha / rank and add['sources'] == [ids['base'], scale['id']], 'Retained source graph scale/Add ordering differs from derivation.')
    expected = {(x['id'], ids['base']), (x['id'], ids['A']), (ids['A'], ids['B']), (ids['B'], scale['id']),
                (ids['base'], add['id']), (scale['id'], add['id']), (add['id'], y['id'])}
    _require(len(graph['edges']) == len(expected) and {(e['from'], e['to']) for e in graph['edges']} == expected and
             not any(e.get('from_port') or e.get('to_port') for e in graph['edges']), 'Retained LoRA wiring does not match exact merge computation.')
    owners = plan(model['models'], root)['resolved']['owners']
    _require(owners[sources['base']['parameter']['id']]['trainable'] is False and
             all(owners[sources[key]['parameter']['id']]['trainable'] is True for key in ('A', 'B')),
             'Retained source must freeze the supplied base and train both adapter owners.')


def check_dependencies(doc, resolver):
    validate(doc); selections = {key: {field: source[field] for field in ('run', 'attempt', 'checkpoint_id', 'parameter')}
                                for key, source in doc['sources'].items()}
    actual = _sources(resolver, selections)
    _require(all(actual[key]['source'] == source for key, source in doc['sources'].items()), 'Derived source descriptors differ from exact retained evidence.')
    ref = doc['sources']['base']['model']; source_model = resolver('model', ref['id'], ref['revision'])
    _topology(source_model, doc['sources'], doc['operation']['alpha'])
    value = merged_values(*(actual[key]['values'] for key in ('base', 'A', 'B')), alpha=doc['operation']['alpha'])
    _, value_hash = float32_values(value, doc['destination']['shape'])
    ref = doc['destination']['model']; model = resolver('model', ref['id'], ref['revision'])
    _require(model['id'] == ref['id'] and model['revision'] == ref['revision'], 'Derived destination resolver substituted a model.')
    nodes = model['models'][model['root']]['graph']['nodes']
    _require(len(model['models']) == 1 and len(nodes) == 3 and sorted(n['type'] for n in nodes) == ['Input', 'Linear', 'Output'],
             'Derived destination must be only the corresponding Linear inference map.')
    linear = next(n for n in nodes if n['type'] == 'Linear')
    _require(linear['id'] == 'linear1' and linear.get('bias', True) is False, 'Merged inference map cannot add a bias.')
    expected_contract = validate_model(source_model['models'], source_model['root'])
    actual_contract = validate_model(model['models'], model['root'])
    _require(all(actual_contract[key] == expected_contract[key] for key in ('input_groups', 'output_groups')),
             'Merged inference must preserve source ports, sample shapes and axes.')
    row = plan(model['models'], model['root'])['resolved']['owners'].get(doc['destination']['id'])
    _require(row is not None and not row['trainable'] and row['initializer']['kind'] == 'values', 'Derived inference owner must be frozen exact values.')
    _, stored_hash = float32_values(row['initializer']['values'], row['shape'])
    _require(stored_hash == value_hash == doc['destination']['values_sha256'], 'Derived output differs from independently recomputed float32 merge.')


def preview(workspace, selections, *, alpha=4., model_id='lora-merged', name='LoRA merged inference · explicitly derived'):
    _keys(selections, ('base', 'A', 'B'), context='LoRA merge selection')
    _require(type(alpha) in (int, float) and 0 < alpha <= 1e6 and math.isfinite(alpha), 'LoRA alpha must be finite positive, not boolean, at most1e6.')
    _require(_identifier(model_id) and _identifier(name), 'Derived model ID/name invalid.')
    actual = _sources(workspace.get_spec, selections); sources = {key: value['source'] for key, value in actual.items()}
    source = sources['base']['model']; source_model = workspace.get_spec('model', source['id'], source['revision'])
    _topology(source_model, sources, alpha)
    values = merged_values(*(actual[key]['values'] for key in ('base', 'A', 'B')), alpha=alpha)
    shape = sources['base']['shape']; doc = default_model(name, inputs=shape[1], outputs=shape[0]); doc['id'] = model_id
    next(n for n in doc['graph']['nodes'] if n['id'] == 'linear1')['bias'] = False
    source_nodes = source_model['models'][source_model['root']]['graph']['nodes']
    x = copy.deepcopy(next(n for n in source_nodes if n['type'] == 'Input')); x.update(x=60, y=180)
    y = copy.deepcopy(next(n for n in source_nodes if n['type'] == 'Output')); y.update(x=580, y=180)
    dense = next(n for n in doc['graph']['nodes'] if n['type'] == 'Linear'); dense.update(x=320, y=180)
    doc['graph'] = {'nodes': [x, dense, y], 'input_order': [x['id']],
                    'edges': [{'from': x['id'], 'to': 'linear1'}, {'from': 'linear1', 'to': y['id']}]}
    doc['graph']['parameter_policy'] = {'version': 1, 'parameters': {owner_key(['linear1', 'weight']):
        {'trainable': False, 'initializer': {'kind': 'values', 'dtype': 'float32', 'values': values}}}}
    frozen = frozen_document('model', model_document({model_id: doc}, model_id)); ref = {'kind': 'model', 'id': model_id, 'revision': frozen['revision']}
    _, value_hash = float32_values(values, shape); rank = sources['A']['shape'][0]
    operation = {'kind': 'lora-merge', 'alpha': alpha, 'rank': rank, 'scale': alpha / rank, 'arithmetic': ARITHMETIC}
    destination = {'model': ref, 'selector': owner_key(['linear1', 'weight']), 'id': owner_key([model_id, 'linear1', 'weight']),
                   'shape': shape, 'dtype': 'float32', 'values_sha256': value_hash, 'value_format': VALUE_FORMAT}
    record = frozen_document('parameter-derivation', {'version': 1, 'id': 'lora-merge-' + digest({'operation': operation, 'sources': sources, 'destination': destination})[:32],
        'name': name, 'operation': operation, 'sources': sources, 'destination': destination, 'optimizer_resume': False})
    return {'document': doc, 'model': frozen, 'derivation': record, 'scope': 'explicit-new-derived-inference; original-evidence-unchanged'}


def admit(workspace, selections, *, alpha=4., model_id='lora-merged', name='LoRA merged inference · explicitly derived'):
    candidate = preview(workspace, selections, alpha=alpha, model_id=model_id, name=name)
    _require(model_id not in workspace.models(), 'Choose a new merged model ID; existing drawings are never overwritten.')
    with _atomic(workspace):
        workspace.put(candidate['document']); model = _retain(workspace, 'model', candidate['model'])
        record = _retain(workspace, 'parameter-derivation', candidate['derivation'])
    return {'reference': record, 'model': model, 'changed': True, 'scope': candidate['scope']}
