"""Explicit CPU float32 LSTM graph ports, allocation-free proof and execution.

Graph states are batch-leading [B, layers*directions, width]. The sole native
adapter transposes their first two axes; sequence remains [B,T,F]. No hidden
initialization, persistent state, guessed loss masks or implicit dropout.
"""
import copy
import math
from .memory import MIB, available_memory, memory_percent, format_bytes

FIELDS = {'inputs', 'hidden_size', 'num_layers', 'bidirectional', 'proj_size', 'bias', 'lengths'}
PORTS = ('sequence', 'hidden', 'cell')


def normalized(node):
    from .core import GraphError
    from .operators import COMMON, integer
    if set(node) - COMMON - FIELDS:
        raise GraphError('LSTM has unsupported fields.')
    if type(node.get('op_version', 1)) is not int or node.get('op_version', 1) != 1:
        raise GraphError('LSTM supports integer op_version 1.')
    n = copy.deepcopy(node)
    n['op_version'] = 1
    n['hidden_size'] = integer(n.get('hidden_size'), 'LSTM hidden_size')
    n['num_layers'] = integer(n.get('num_layers', 1), 'LSTM num_layers')
    if n['num_layers'] > 64:
        raise GraphError('This LSTM prototype supports at most 64 stacked layers.')
    n['proj_size'] = integer(n.get('proj_size', 0), 'LSTM proj_size', 0)
    if n['proj_size'] >= n['hidden_size']:
        raise GraphError('LSTM proj_size must be 0 or strictly less than hidden_size.')
    for field in ('bidirectional', 'bias'):
        n.setdefault(field, field == 'bias')
        if type(n[field]) is not bool:
            raise GraphError('LSTM ' + field + ' must be boolean.')
    bound = n.get('inputs')
    if not isinstance(bound, dict) or set(bound) != set(PORTS) or any(
            not isinstance(v, str) or not v.strip() for v in bound.values()):
        raise GraphError('LSTM inputs must explicitly bind sequence, hidden and cell source keys.')
    length = n.setdefault('lengths', None)
    if length is not None and (not isinstance(length, dict) or set(length) != {'source', 'adapter'} or
            not isinstance(length['source'], str) or not length['source'].strip() or
            length['adapter'] != 'exact-int64-v1'):
        raise GraphError('LSTM lengths requires source and explicit exact-int64-v1 adapter, or null.')
    return n


def sources(node):
    n = normalized(node)
    ordered = [n['inputs'][role] for role in PORTS]
    if n['lengths'] is not None:
        ordered.append(n['lengths']['source'])
    return list(dict.fromkeys(ordered))


def parameter_shapes(kwargs):
    """Match nn.LSTM state-dict order/shapes without importing Torch."""
    h, r = kwargs['hidden_size'], kwargs['proj_size'] or kwargs['hidden_size']
    directions = 2 if kwargs['bidirectional'] else 1
    shapes = {}
    for layer in range(kwargs['num_layers']):
        width = kwargs['input_size'] if layer == 0 else directions * r
        for direction in range(directions):
            suffix = '_l' + str(layer) + ('_reverse' if direction else '')
            shapes['weight_ih' + suffix] = [4 * h, width]
            shapes['weight_hh' + suffix] = [4 * h, r]
            if kwargs['bias']:
                shapes['bias_ih' + suffix] = [4 * h]
                shapes['bias_hh' + suffix] = [4 * h]
            if kwargs['proj_size']:
                shapes['weight_hr' + suffix] = [r, h]
    return shapes


def describe(node, incoming):
    from .core import GraphError, feature_count
    n = normalized(node)
    sequence, seq_axes = incoming[n['inputs']['sequence']]
    if len(sequence) != 2:
        raise GraphError('LSTM sequence needs sample shape [T,F]; batch B is implicit.')
    if seq_axes is not None and seq_axes[0] != 'T':
        raise GraphError('LSTM sequence leading declared axis must be T.')
    steps, width = sequence
    directions = 2 if n['bidirectional'] else 1
    h, r, ld = n['hidden_size'], n['proj_size'] or n['hidden_size'], n['num_layers'] * directions
    outputs = {'sequence': [steps, directions * r], 'hidden': [ld, r], 'cell': [ld, h]}
    axes = {'sequence': ['T', 'F'], 'hidden': ['LD', 'Hout'], 'cell': ['LD', 'Hcell']}
    for role in ('hidden', 'cell'):
        shape, declared = incoming[n['inputs'][role]]
        if shape != outputs[role] or (declared is not None and declared != axes[role]):
            raise GraphError('LSTM initial ' + role + ' requires batch-leading sample shape ' + repr(outputs[role]) +
                             ' and axes ' + repr(axes[role]) + ' when declared.')
    if n['lengths'] is not None:
        if incoming[n['lengths']['source']][0] != [1] or steps > 2**24:
            raise GraphError('LSTM exact-int64-v1 lengths needs sample shape [1], T <= 2**24.')
    for shape in outputs.values():
        feature_count(math.prod(shape))
    kwargs = {field: n[field] for field in ('hidden_size', 'num_layers', 'bidirectional', 'proj_size', 'bias')}
    kwargs.update(input_size=width, batch_first=True, dropout=0., device='cpu', dtype='float32')
    schema = parameter_shapes(kwargs)
    order = sources(n)
    activation_elements = steps * ld * (8 * h + 4 * r + width)
    return {'version': 1, 'kwargs': kwargs, 'steps': steps, 'input_shape': sequence,
            'ports': outputs, 'port_axes': axes, 'parameter_shapes': schema,
            'parameters': sum(math.prod(s) for s in schema.values()),
            'activation_elements': activation_elements,
            'inputs': {role: order.index(n['inputs'][role]) for role in PORTS},
            'lengths': None if n['lengths'] is None else order.index(n['lengths']['source'])}


def lstm_memory_report(plan, batch, *, constrained=True, memory_limit_percent=10.):
    """Current-call conservative admission, not an allocator or OOM guarantee."""
    percent = memory_percent(memory_limit_percent)
    components = {'runtime_headroom': 64 * MIB, 'parameters_optimizer_copies': 128 * plan['parameters'],
                  'gates_states_backward': 32 * batch * plan['activation_elements'],
                  'boundary_copies': 16 * batch * (math.prod(plan['input_shape']) +
                                      sum(math.prod(s) for s in plan['ports'].values()))}
    estimate = sum(components.values())
    available = available_memory() if constrained else None
    limit = None if available is None else int(available * percent / 100)
    return {'estimated_bytes': estimate, 'components': components, 'available_bytes': available,
            'limit_bytes': limit, 'allowed': not constrained or (limit is not None and estimate <= limit)}


def graph_forward(layer, parts, plan):
    """Direct nn.LSTM with explicit adapters; packed positive prefixes only."""
    import torch
    from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
    x, h, c = (parts[plan['inputs'][role]] for role in ('sequence', 'hidden', 'cell'))
    steps = plan['steps']
    for value, shape in ((x, plan['input_shape']), (h, plan['ports']['hidden']), (c, plan['ports']['cell'])):
        if not isinstance(value, torch.Tensor) or value.device.type != 'cpu' or value.dtype != torch.float32 or \
                value.ndim != len(shape) + 1 or list(value.shape[1:]) != shape:
            raise ValueError('LSTM requires CPU float32 tensors with explicit batch-leading sequence/state shapes.')
    batch = x.shape[0]
    if batch < 1 or h.shape[0] != batch or c.shape[0] != batch:
        raise ValueError('LSTM sequence and states require the same positive batch B.')
    report = lstm_memory_report(plan, batch, constrained=getattr(layer, 'constrained', False),
                                memory_limit_percent=getattr(layer, 'memory_limit_percent', 10.))
    if not report['allowed']:
        raise ValueError('Constrained LSTM call refused: estimated ' + format_bytes(report['estimated_bytes']) +
                         ', RAM allowance ' + format_bytes(report['limit_bytes']) + '.')
    if not bool(torch.isfinite(h).all()) or not bool(torch.isfinite(c).all()):
        raise ValueError('LSTM initial states must be finite; only padded sequence values may be nonfinite.')
    if plan['lengths'] is None:
        lengths = torch.full((batch,), steps, dtype=torch.int64, device='cpu')
    else:
        value = parts[plan['lengths']]
        if not isinstance(value, torch.Tensor) or value.device.type != 'cpu' or value.dtype != torch.float32 or \
                list(value.shape) != [batch, 1] or steps > 2**24 or not bool(
                (torch.isfinite(value) & (value == value.trunc()) & (value >= 0) & (value <= steps)).all()):
            raise ValueError('LSTM exact-int64-v1 requires finite integer lengths [B,1] in [0,T].')
        lengths = value[:, 0].to(torch.int64)
    # Empty slices create exact zero gradients without evaluating NaN padding.
    anchor = x.reshape(-1)[:0].sum() + h.reshape(-1)[:0].sum() + c.reshape(-1)[:0].sum()
    for parameter in layer.parameters():
        anchor = anchor + parameter.reshape(-1)[:0].sum()
    sequence = x.new_zeros(batch, *plan['ports']['sequence']) + anchor
    hidden, cell = h + anchor, c + anchor
    active = torch.nonzero(lengths > 0, as_tuple=False)[:, 0]
    if active.numel():
        packed = pack_padded_sequence(x.index_select(0, active), lengths.index_select(0, active),
                                       batch_first=True, enforce_sorted=False)
        if not bool(torch.isfinite(packed.data).all()):
            raise ValueError('LSTM active sequence values must be finite.')
        returned, (hn, cn) = layer(packed, (h.index_select(0, active).transpose(0, 1).contiguous(),
                                          c.index_select(0, active).transpose(0, 1).contiguous()))
        padded, _ = pad_packed_sequence(returned, batch_first=True, total_length=steps)
        sequence = sequence.index_copy(0, active, padded)
        hidden = hidden.index_copy(0, active, hn.transpose(0, 1))
        cell = cell.index_copy(0, active, cn.transpose(0, 1))
    return {'sequence': sequence, 'hidden': hidden, 'cell': cell}
