"""Pure, closed JSON architecture recipes; no execution, data or training.

Build returns ordinary create edits in dependency order. Workspace admission and
collision checks belong to the caller. Counts/shapes are proved by native graph
services, never retained as guessed metadata in the constructed documents.
"""
import copy
import json

from .core import GraphError, check_document, validate_model
from . import neural_msvar


MAX_DIMENSION = 4096
MAX_PARAMETERS = 1_000_000
MAX_WORK_ELEMENTS = 1_000_000
MAX_BLOCKS = 64


def _integer(default):
    return {'type': 'integer', 'minimum': 1, 'maximum': MAX_DIMENSION,
            'default': default}


def _boolean(default=False):
    return {'type': 'boolean', 'default': default}


_RECIPES = {
    'branching-mlp': {
        'kind': 'exact-model',
        'description': 'Two independent Tanh encoders, ordered concatenation, linear readout.',
        'limitations': 'A deterministic network; no probabilistic likelihood is declared.',
        'options': {'left': _integer(2), 'right': _integer(3),
                    'hidden': _integer(8), 'outputs': _integer(1)},
        'model_suffixes': [''], 'blocks': [9]},
    'shared-encoders': {
        'kind': 'parameter-component',
        'description': 'Two input encoders reuse one Linear parameter owner via actual share_with; embeddings are concatenated.',
        'limitations': 'An embedding component, not a latent regime model or likelihood.',
        'options': {'D': _integer(3), 'hidden': _integer(8)},
        'model_suffixes': [''], 'blocks': [8]},
    'recurrent-cell': {
        'kind': 'exact-model',
        'description': 'Explicit Tanh transition cell and recurrent parent, initial/final state and valid prefix lengths.',
        'limitations': 'Deterministic recurrence; controls use explicit exact adapters. No initial state is fabricated.',
        'options': {'D': _integer(3), 'hidden': _integer(8), 'T': _integer(12),
                    'mask': _boolean(), 'reset': _boolean()},
        'model_suffixes': ['/cell', ''], 'blocks': [6, 6],
        'control_semantics': {'lengths': 'Valid prefix in [0,T], exact-int64-v1.',
                             'mask': 'Exact 0/1 skip-step control, exact-bool-v1.',
                             'reset': 'Exact 0/1 reset to supplied initial state before active step.'}},
    'finite-window-regime-estimator': {
        'kind': 'approximate-estimator',
        'description': 'Flatten an explicitly supplied finite observation window, Tanh hidden layer, K-way Softmax supervised regime estimate.',
        'limitations': 'Approximate supervised classifier requiring externally supplied genuine regime targets. Not an HMM filter, sequence likelihood, or identifiability claim. No labels or dataset are created.',
        'options': {'window': _integer(4), 'D': _integer(3),
                    'K': _integer(5), 'hidden': _integer(16)},
        'model_suffixes': [''], 'blocks': [7]},
    'neural-msvar': {
        'kind': 'exact-model',
        'description': 'Full causal neural MS-VAR with learned transition law, affine state means, diagonal variances and exact conditional Gaussian filtering.',
        'limitations': 'Conditional on explicit observed history and controls; no missing-history marginalization or stationarity claim. p >= 1; the separate exact p=0 HMM is not this autoregressive recipe.',
        'options': {'K': _integer(5), 'p': _integer(3), 'D': _integer(3),
                    'hidden': _integer(32), 'T': _integer(12),
                    'controlled': _boolean()},
        'model_suffixes': [''], 'blocks': [23],
        'control_semantics': {
            'controlled=false': 'Prefix lengths; softplus variance with positive floor 0.0001.',
            'controlled=true': 'Prefix lengths plus execution/reset/emission masks and reset_history; exp of learned logvariance, floor 0.'}},
}


def show(name):
    """Return an independent JSON declaration, including defaults and bounds."""
    if type(name) is not str or name not in _RECIPES:
        raise GraphError('Unknown construction recipe; use catalogue().')
    result = copy.deepcopy(_RECIPES[name])
    properties = result.pop('options')
    result.update(name=name, version=1,
                  options={'type': 'object', 'additionalProperties': False,
                           'properties': properties},
                  bounds={'parameters': MAX_PARAMETERS,
                          'work_elements': MAX_WORK_ELEMENTS,
                          'blocks_per_document': MAX_BLOCKS},
                  edit_order='Dependencies before consumers; root uses the supplied model_id.')
    return result


def catalogue():
    """Alphabetical declarations; never reorder edits or tensor packing."""
    return [show(name) for name in sorted(_RECIPES)]


def _options(name, options):
    schema = show(name)['options']['properties']
    if options is None:
        options = {}
    if type(options) is not dict or any(type(k) is not str or k not in schema for k in options):
        raise GraphError('Recipe options must be a closed JSON object with only declared keys.')
    values = {key: options.get(key, spec['default']) for key, spec in schema.items()}
    for key, value in values.items():
        spec = schema[key]
        if spec['type'] == 'boolean':
            if type(value) is not bool:
                raise GraphError(key + ' must be a JSON boolean, without coercion.')
        elif type(value) is not int or not 1 <= value <= MAX_DIMENSION:
            if name == 'neural-msvar' and key == 'p' and type(value) is int and value == 0:
                raise GraphError('Autoregressive neural MS-VAR requires p >= 1; use the separate exact p=0 HMM.')
            raise GraphError(key + ' must be a positive bounded JSON integer (1..4096), without coercion.')
    return values


def _budget(parameters, work, blocks):
    if parameters > MAX_PARAMETERS or work > MAX_WORK_ELEMENTS or max(blocks) > MAX_BLOCKS:
        raise GraphError('Recipe exceeds bounded parameter/work products or number of blocks before construction.')


def _document(mid, title, nodes, edges, inputs):
    for i, node in enumerate(nodes):
        node.setdefault('x', 80 + 220 * (i % 6))
        node.setdefault('y', 80 + 160 * (i // 6))
    return {'id': mid, 'name': title,
            'graph': {'nodes': nodes, 'edges': edges, 'input_order': inputs}}


def _chain(*ids):
    return [{'from': a, 'to': b} for a, b in zip(ids, ids[1:])]


def _branches(mid, o, shared):
    left, right = (o['D'], o['D']) if shared else (o['left'], o['right'])
    h = o['hidden']
    parameters = (left + 1) * h
    if not shared:
        parameters += (right + 1) * h + (2 * h + 1) * o['outputs']
    _budget(parameters, left + right + 8 * h + (0 if shared else o['outputs']), [8 if shared else 9])
    nodes = [{'id': 'left', 'type': 'Input', 'shape': [left]},
             {'id': 'right', 'type': 'Input', 'shape': [right]},
             {'id': 'left_encoder', 'type': 'Linear', 'features': h, 'bias': True},
             {'id': 'left_nonlinearity', 'type': 'Activation', 'activation': 'Tanh'},
             {'id': 'right_encoder', 'type': 'Linear', 'features': h, 'bias': True},
             {'id': 'right_nonlinearity', 'type': 'Activation', 'activation': 'Tanh'},
             {'id': 'join', 'type': 'Concat', 'sources': ['left_nonlinearity', 'right_nonlinearity'], 'flatten': True}]
    edges = _chain('left', 'left_encoder', 'left_nonlinearity', 'join')
    edges += _chain('right', 'right_encoder', 'right_nonlinearity', 'join')
    if shared:
        nodes[4]['share_with'] = 'left_encoder'
        edges += _chain('join', 'output')
    else:
        nodes.append({'id': 'readout', 'type': 'Linear', 'features': o['outputs'], 'bias': True})
        edges += _chain('join', 'readout', 'output')
    nodes.append({'id': 'output', 'type': 'Output'})
    return [_document(mid, 'Shared encoders' if shared else 'Branching MLP', nodes, edges, ['left', 'right'])]


def _recurrent(mid, o):
    d, h, t = o['D'], o['hidden'], o['T']
    _budget((d + h + 1) * h, t * (d + 8 * h + 3), [6, 6 + o['mask'] + o['reset']])
    cell_id = mid + '/cell'
    cell_nodes = [{'id': 'sample', 'type': 'Input', 'shape': [d]},
                  {'id': 'state', 'type': 'Input', 'shape': [h]},
                  {'id': 'join', 'type': 'Concat', 'sources': ['sample', 'state'], 'flatten': True},
                  {'id': 'update', 'type': 'Linear', 'features': h, 'bias': True},
                  {'id': 'tanh', 'type': 'Activation', 'activation': 'Tanh'},
                  {'id': 'next_state', 'type': 'Output'}]
    cell = _document(cell_id, 'Explicit Tanh transition cell', cell_nodes,
                     _chain('sample', 'join', 'update', 'tanh', 'next_state') + _chain('state', 'join'),
                     ['sample', 'state'])
    nodes = [{'id': 'sequence', 'type': 'Input', 'shape': [t, d]},
             {'id': 'initial_state', 'type': 'Input', 'shape': [h]},
             {'id': 'lengths', 'type': 'Input', 'shape': [1]}]
    controls = {'lengths': {'source': 'lengths', 'adapter': 'exact-int64-v1'}}
    inputs = ['sequence', 'initial_state', 'lengths']
    for role in ('mask', 'reset'):
        if o[role]:
            nodes.append({'id': role, 'type': 'Input', 'shape': [t]})
            controls[role] = {'source': role, 'adapter': 'exact-bool-v1'}
            inputs.append(role)
    nodes.append({'id': 'recurrent', 'type': 'Recurrent', 'ref': cell_id,
                  'sequence_inputs': [{'input': 'sample', 'source': 'sequence'}],
                  'states': [{'input': 'state', 'output': 'next_state', 'port': 'state', 'source': 'initial_state'}],
                  'sequence_outputs': [{'output': 'next_state', 'port': 'sequence'}],
                  'controls': controls})
    edges = [{'from': source, 'to': 'recurrent'} for source in inputs]
    for port in ('sequence', 'state'):
        nodes.append({'id': 'out_' + port, 'type': 'Output'})
        edges.append({'from': 'recurrent', 'from_port': port, 'to': 'out_' + port})
    return [cell, _document(mid, 'Explicit recurrence with prefix controls', nodes, edges, inputs)]


def _estimator(mid, o):
    width, h, k = o['window'] * o['D'], o['hidden'], o['K']
    _budget((width + 1) * h + (h + 1) * k, 2 * width + 4 * h + 3 * k, [7])
    nodes = [{'id': 'window', 'type': 'Input', 'shape': [o['window'], o['D']]},
             {'id': 'flatten', 'type': 'Reshape', 'shape': [width]},
             {'id': 'hidden', 'type': 'Linear', 'features': h, 'bias': True},
             {'id': 'tanh', 'type': 'Activation', 'activation': 'Tanh'},
             {'id': 'logits', 'type': 'Linear', 'features': k, 'bias': True},
             {'id': 'probabilities', 'type': 'Activation', 'activation': 'Softmax', 'dim': 0},
             {'id': 'estimate', 'type': 'Output'}]
    return [_document(mid, 'Approximate supervised finite-window regime estimator', nodes,
                      _chain(*(n['id'] for n in nodes)), ['window'])]


def _msvar(mid, o):
    k, p, d, h, t = (o[key] for key in ('K', 'p', 'D', 'hidden', 'T'))
    parameters = (p * d + 1) * h + (h + 1) * k * k + (p * d + 1) * k * d + k + k * d
    # Conservative preallocation bound including native lag/filter scratch work,
    # activations and controlled reset histories, not merely dimension maxima.
    work = t * (16 * p * d + 12 * k * k + 24 * k * d + 24 * k + 8 * h + 8 * d + 48) + 16 * (t + p) * d + 8 * k
    _budget(parameters, work, [27 if o['controlled'] else 23])
    constructor = neural_msvar.controlled_drawing if o['controlled'] else neural_msvar.drawing
    doc = constructor(steps=t, states=k, lags=p, dimensions=d, hidden=h)
    doc['id'] = mid
    return [doc]


def build(name, *, model_id, options=None):
    """Construct and validate a fresh dependency-ordered list of create edits.

    No workspace is opened. IDs derive only from model_id (cell: model_id+'/cell').
    All options are strict JSON scalars; no executable declarations are accepted.
    """
    o = _options(name, options)
    if type(model_id) is not str or not model_id.strip() or len(model_id) > 200 or any(ord(c) < 32 or ord(c) == 127 for c in model_id):
        raise GraphError('Supply a nonempty explicit model_id of at most 200 characters without control characters.')
    if name in ('branching-mlp', 'shared-encoders'):
        documents = _branches(model_id, o, name == 'shared-encoders')
    elif name == 'recurrent-cell':
        documents = _recurrent(model_id, o)
    elif name == 'finite-window-regime-estimator':
        documents = _estimator(model_id, o)
    else:
        documents = _msvar(model_id, o)
    models = {}
    for doc in documents:
        check_document(doc)
        models[doc['id']] = doc
        contract = validate_model(models, doc['id'])  # recompute every shape/count
        _budget(contract['parameters'], contract['activation_elements'], [len(doc['graph']['nodes'])])
    edits = [{'op': 'create', 'document': doc} for doc in documents]
    json.dumps(edits, allow_nan=False)
    return edits
