"""Closed multiport recurrent graph references and explicit control adapters."""
import copy

FIELDS = {'ref', 'sequence_inputs', 'states', 'sequence_outputs', 'controls'}


def normalized(node):
    from .core import GraphError
    from .operators import COMMON
    from .recurrent import _keys, _ids
    if set(node) - COMMON - FIELDS:
        raise GraphError('Recurrent block has missing or unsupported fields.')
    if type(node.get('op_version', 1)) is not int or node.get('op_version', 1) != 1:
        raise GraphError('Recurrent supports integer op_version 1.')
    n = copy.deepcopy(node)
    n['op_version'] = 1
    if not isinstance(n.get('ref'), str) or not n['ref'].strip():
        raise GraphError('Recurrent needs a transition architecture reference ID.')
    for field, keys in (('sequence_inputs', ('input', 'source')),
                        ('states', ('input', 'output', 'port', 'source')),
                        ('sequence_outputs', ('output', 'port'))):
        if not isinstance(n.get(field), list) or (field != 'sequence_outputs' and not n[field]):
            raise GraphError('Recurrent ' + field + ' needs an explicit list.')
        for row in n[field]:
            _keys(row, keys, 'Recurrent ' + field)
            if any(not isinstance(v, str) or not v.strip() for v in row.values()):
                raise GraphError('Recurrent bindings require nonempty stable ID/source strings.')
    _ids([s['input'] for s in n['sequence_inputs']] + [s['input'] for s in n['states']], 'Recurrent bound inputs')
    _ids([s['output'] for s in n['states']], 'Recurrent next-state outputs')
    _ids([s['port'] for s in n['states'] + n['sequence_outputs']], 'Recurrent output ports')
    controls = n.setdefault('controls', {})
    if not isinstance(controls, dict) or set(controls) - {'lengths', 'mask', 'reset'}:
        raise GraphError('Recurrent controls are optional lengths, mask and reset bindings.')
    for role, row in controls.items():
        _keys(row, ('source', 'adapter'), 'Recurrent control ' + role)
        if not isinstance(row['source'], str) or not row['source'].strip():
            raise GraphError('Recurrent control needs a stable source.')
        expected = 'exact-int64-v1' if role == 'lengths' else 'exact-bool-v1'
        if row['adapter'] != expected:
            raise GraphError('Recurrent ' + role + ' requires explicit adapter ' + expected + '.')
    return n


def specification(node):
    n = normalized(node)
    return {'version': 1, 'sequence_inputs': [r['input'] for r in n['sequence_inputs']],
            'states': [{k: r[k] for k in ('input', 'output', 'port')} for r in n['states']],
            'sequence_outputs': n['sequence_outputs']}


def sources(node):
    n = normalized(node)
    ordered = [r['source'] for r in n['sequence_inputs'] + n['states']]
    ordered.extend(n['controls'][role]['source'] for role in ('lengths', 'mask', 'reset') if role in n['controls'])
    return list(dict.fromkeys(ordered))


def describe(node, child, incoming):
    """Shape/axis proof for one baseline or concrete runtime temporal case."""
    from .core import GraphError
    from .recurrent import interface_contract
    n = normalized(node)
    c = interface_contract(child, specification(n))
    steps = None
    for row in n['sequence_inputs']:
        shape, axes = incoming[row['source']]
        port = c['inputs'][row['input']]
        if len(shape) != len(port['shape']) + 1 or shape[1:] != port['shape']:
            raise GraphError('Recurrent sequence ' + row['input'] + ' needs [T, *transition_input_shape].')
        if steps is not None and steps != shape[0]:
            raise GraphError('Recurrent sequences must have equal T.')
        steps = shape[0]
        if axes is not None and axes[0] != 'T':
            raise GraphError('Recurrent sequence leading declared axis must be T.')
        expected = None if port.get('axes') is None else ['T', *port['axes']]
        if expected is not None and axes != expected:
            raise GraphError('Recurrent sequence ' + row['input'] + ' axes must be ' + repr(expected) + '.')
    for row in n['states']:
        shape, axes = incoming[row['source']]
        port = c['inputs'][row['input']]
        if shape != port['shape'] or axes != port.get('axes'):
            raise GraphError('Recurrent initial state ' + row['port'] + ' shape/axes differ from its transition Input.')
    for role, row in n['controls'].items():
        shape, axes = incoming[row['source']]
        if shape != ([1] if role == 'lengths' else [steps]):
            raise GraphError('Recurrent ' + role + ' needs sample shape ' + repr([1] if role == 'lengths' else [steps]) + '.')
        if role == 'lengths' and steps > 2**24:
            raise GraphError('exact-int64-v1 float32 lengths adapter requires T at most 2**24.')
    ports, axes = {}, {}
    for row in n['sequence_outputs']:
        port = c['outputs'][row['output']]
        ports[row['port']] = [steps, *port['shape']]
        axes[row['port']] = ['T', *(port.get('axes') or ['D' + str(i) for i in range(len(port['shape']))])]
    for row in n['states']:
        port = c['inputs'][row['input']]
        ports[row['port']] = port['shape']
        axes[row['port']] = port.get('axes')
    order = sources(n)
    plan = {'contract': c, 'steps': steps, 'ports': list(ports),
            'sequence_inputs': {r['input']: order.index(r['source']) for r in n['sequence_inputs']},
            'states': {r['port']: order.index(r['source']) for r in n['states']},
            'controls': {role: {'index': order.index(r['source']), 'adapter': r['adapter']} for role, r in n['controls'].items()}}
    return plan, ports, axes


def _control(value, adapter, steps):
    """Explicit validation + conversion of declared graph float32 controls."""
    import torch
    if not isinstance(value, torch.Tensor) or value.device.type != 'cpu' or value.dtype != torch.float32:
        raise ValueError('Recurrent control adapters require CPU float32 graph tensors.')
    if adapter == 'exact-bool-v1':
        if not bool(((value == 0) | (value == 1)).all()):
            raise ValueError('exact-bool-v1 requires exact 0/1 controls, not probabilities or NaNs.')
        return value.to(torch.bool)
    if adapter == 'exact-int64-v1':
        if steps > 2**24 or value.ndim != 2 or value.shape[1] != 1 or not bool(
                (torch.isfinite(value) & (value == value.trunc()) & (value >= 0) & (value <= steps)).all()):
            raise ValueError('exact-int64-v1 requires finite integer lengths [B,1] in [0,T], T <= 2**24.')
        return value[:, 0].to(torch.int64)
    raise ValueError('Unknown recurrent control adapter.')


def graph_forward(layer, parts, plan):
    """Same native/exported multiport projection; no implicit state initialization."""
    import torch
    # Imported at call time so generated helper can instead use captured globals.
    sequences = {key: parts[index] for key, index in plan['sequence_inputs'].items()}
    state = {key: parts[index] for key, index in plan['states'].items()}
    batch, steps = _recurrent_inputs(layer.contract, sequences, state, None, None, None)
    report = recurrent_memory_report(layer.contract, batch, steps, layer.constrained,
                                     memory_limit_percent=layer.memory_limit_percent,
                                     training=layer.training or torch.is_grad_enabled())
    if not report['allowed']:
        raise ValueError(report['reason'])
    controls = {role: _control(parts[row['index']], row['adapter'], steps) for role, row in plan['controls'].items()}
    result = layer(sequences, state, **controls)
    return {**result['sequences'], **result['state']}


# Runtime globals are deliberately supplied also by the app-free source emitter.
from .recurrent import _recurrent_inputs, recurrent_memory_report
