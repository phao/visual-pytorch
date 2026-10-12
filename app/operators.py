"""Version-1 structured operators: pure normalization, shapes and Torch specs.

Sample axis numbers exclude batch B. No Torch import or tensor allocation here.
The returned constructor arguments are shared by native execution and export.
"""
import math

TENSOR_OPERATORS = frozenset({
    'Conv2d', 'AvgPool2d', 'MaxPool2d', 'AdaptiveAvgPool2d', 'Flatten', 'Permute',
})
REFERENCE_TYPES = frozenset({'Module', 'Recurrent'})
LAYER_TYPES = TENSOR_OPERATORS | {'Linear', 'Activation', 'Module', 'Reshape', 'Recurrent', 'LSTM'}
ARITHMETIC_OPERATORS = frozenset({'Add', 'Scale'})
BLOCK_TYPES = LAYER_TYPES | ARITHMETIC_OPERATORS | {'Input', 'Output', 'Concat', 'Splitter', 'Replicator'}
COMMON = {'id', 'type', 'label', 'x', 'y', 'width', 'height', 'collapsed', 'op_version', 'share_with'}
FIELDS = {
    'Add': {'sources'}, 'Scale': {'coefficient'},
    'Conv2d': {'out_channels', 'kernel_size', 'stride', 'padding', 'dilation', 'groups', 'bias', 'padding_mode'},
    'AvgPool2d': {'kernel_size', 'stride', 'padding', 'ceil_mode', 'count_include_pad'},
    'MaxPool2d': {'kernel_size', 'stride', 'padding', 'dilation', 'ceil_mode'},
    'AdaptiveAvgPool2d': {'output_size'},
    'Flatten': {'start_dim', 'end_dim'},
    'Permute': {'dims'},
}


def integer(value, name, minimum=1):
    from .core import GraphError, MAX_INTEGER
    if type(value) is not int or not minimum <= value <= MAX_INTEGER:
        raise GraphError(f'{name} must be an integer from {minimum} to {MAX_INTEGER}.')
    return value


def pair(value, name, minimum=1):
    from .core import GraphError
    if type(value) is int:
        value = [value, value]
    if not isinstance(value, list) or len(value) != 2:
        raise GraphError(f'{name} must be an integer or a two-integer JSON array [height, width].')
    return [integer(v, name, minimum) for v in value]


def normalized(node):
    """Explicit defaults/pairs for new operators only; preserve legacy documents."""
    from .core import GraphError
    kind = node['type']
    if kind in ARITHMETIC_OPERATORS:
        if type(node.get('op_version', 1)) is not int or node.get('op_version', 1) != 1:
            raise GraphError(kind + ' supports integer op_version 1.')
        if set(node) - COMMON - FIELDS[kind] or 'share_with' in node:
            raise GraphError(kind + ' has unsupported fields; it owns no shareable parameters.')
        n = dict(node, op_version=1)
        if kind == 'Add':
            values = n.get('sources')
            if not isinstance(values, list) or any(type(s) is not str or not s or s != s.strip() for s in values) or len(set(values)) != len(values):
                raise GraphError('Add requires ordered unique nonempty source keys. Connect at least two to execute.')
            n['sources'] = list(values)
        else:
            value = n.get('coefficient')
            from .core import MAX_INTEGER
            if (type(value) not in (int, float) or
                    (type(value) is int and abs(value) > MAX_INTEGER) or
                    not math.isfinite(value) or abs(value) > 3.4028234663852886e38 or
                    (float(value).is_integer() and abs(value) > MAX_INTEGER)):
                raise GraphError('Scale coefficient must be finite, float32-range and editor-representable, not boolean.')
            n['coefficient'] = float(value)
        return n
    if kind == 'LSTM':
        from .lstm_nodes import normalized as lstm_normalized
        return lstm_normalized(node)
    if kind == 'Recurrent':
        from .recurrent_nodes import normalized as recurrent_normalized
        return recurrent_normalized(node)
    if kind == 'Activation':
        from .activations import normalize_activation
        return normalize_activation(node)
    if kind not in TENSOR_OPERATORS:
        return dict(node)
    if type(node.get('op_version', 1)) is not int or node.get('op_version', 1) != 1:
        raise GraphError(f'{kind} supports only integer op_version 1.')
    unknown = set(node) - COMMON - FIELDS[kind]
    if unknown:
        raise GraphError(f'{kind} has unsupported fields: {", ".join(sorted(unknown))}.')
    n = dict(node, op_version=1)
    if kind in {'Conv2d', 'AvgPool2d', 'MaxPool2d'}:
        n['kernel_size'] = pair(n.get('kernel_size'), 'kernel_size')
        n['stride'] = pair(n.get('stride', 1 if kind == 'Conv2d' else n['kernel_size']), 'stride')
        n['padding'] = pair(n.get('padding', 0), 'padding', 0)
    if kind in {'Conv2d', 'MaxPool2d'}:
        n['dilation'] = pair(n.get('dilation', 1), 'dilation')
    if kind == 'Conv2d':
        n['out_channels'] = integer(n.get('out_channels'), 'out_channels')
        n['groups'] = integer(n.get('groups', 1), 'groups')
        n['bias'] = n.get('bias', True)
        if type(n['bias']) is not bool:
            raise GraphError('Bias must be true or false.')
        n['padding_mode'] = n.get('padding_mode', 'zeros')
        if n['padding_mode'] != 'zeros':
            raise GraphError('Conv2d version 1 supports only zeros padding_mode.')
    if kind in {'AvgPool2d', 'MaxPool2d'}:
        n['ceil_mode'] = n.get('ceil_mode', False)
        if n['ceil_mode'] is not False:
            raise GraphError('Pooling version 1 supports only ceil_mode false.')
        if any(2 * p > k for p, k in zip(n['padding'], n['kernel_size'])):
            raise GraphError('Pooling padding must be at most half of kernel_size.')
    if kind == 'AvgPool2d':
        n['count_include_pad'] = n.get('count_include_pad', True)
        if type(n['count_include_pad']) is not bool:
            raise GraphError('count_include_pad must be true or false.')
    if kind == 'AdaptiveAvgPool2d':
        n['output_size'] = pair(n.get('output_size'), 'output_size')
    if kind == 'Flatten':
        for key, default in (('start_dim', 0), ('end_dim', -1)):
            v = n.get(key, default)
            if type(v) is not int or not -8 <= v < 8:
                raise GraphError('Flatten axes must be integers from -8 to 7, excluding batch B.')
            n[key] = v
    if kind == 'Permute':
        dims = n.get('dims')
        if (not isinstance(dims, list) or not 1 <= len(dims) <= 8 or
                any(type(d) is not int for d in dims) or set(dims) != set(range(len(dims)))):
            raise GraphError('Permute dims must list each sample axis 0..rank-1 exactly once; omit B.')
        n['dims'] = list(dims)
    return n


def axes_for(node, rank):
    from .core import GraphError
    axes = node.get('axes')
    if axes is not None and (not isinstance(axes, list) or len(axes) != rank or
            any(not isinstance(a, str) or not a or len(a) > 32 for a in axes) or len(set(axes)) != rank):
        raise GraphError('Input axes must give one unique nonempty name per sample dimension.')
    return list(axes) if axes is not None else None


def describe(node, before, axes=None):
    """Shape/statistics and executable constructor spec for one concrete input."""
    from .core import GraphError, tensor_shape
    n = normalized(node)
    kind = n['type']
    current, out_axes, params = list(before), axes, 0
    kwargs = {}
    weight_shape = None
    if kind in {'Conv2d', 'AvgPool2d', 'MaxPool2d', 'AdaptiveAvgPool2d'}:
        if len(before) != 3:
            raise GraphError(f'{kind} needs sample shape [C,H,W]; incoming shape {before}.')
        if axes is not None and axes != ['C', 'H', 'W']:
            raise GraphError(f'{kind} needs axes [C,H,W], not {axes}. Use Permute to reorder values.')
        out_axes = ['C', 'H', 'W']
        if kind == 'Conv2d':
            if before[0] % n['groups'] or n['out_channels'] % n['groups']:
                raise GraphError('Conv2d input and output channels must both be divisible by groups.')
            current[0] = n['out_channels']
            weight_shape = [current[0], before[0] // n['groups'], *n['kernel_size']]
            params = math.prod(weight_shape) + (current[0] if n['bias'] else 0)
            kwargs = {k: n[k] for k in FIELDS[kind]}
            kwargs.update(in_channels=before[0], device='cpu', dtype='float32')
        elif kind == 'AdaptiveAvgPool2d':
            current[1:] = n['output_size']
            kwargs = {'output_size': n['output_size']}
        else:
            kwargs = {k: n[k] for k in FIELDS[kind]}
        if kind != 'AdaptiveAvgPool2d':
            dilation = n.get('dilation', [1, 1])
            current[1:] = [(size + 2 * p - d * (k - 1) - 1) // s + 1
                           for size, p, d, k, s in zip(before[1:], n['padding'], dilation, n['kernel_size'], n['stride'])]
            if min(current[1:]) < 1:
                raise GraphError(f'{kind} produces nonpositive spatial dimensions {current[1:]}.')
    elif kind == 'Flatten':
        rank = len(before)
        start, end = n['start_dim'], n['end_dim']
        if not -rank <= start < rank or not -rank <= end < rank:
            raise GraphError('Flatten axes are outside the incoming sample rank; batch B is not an axis here.')
        start, end = start % rank, end % rank
        if start > end:
            raise GraphError('Flatten start_dim must not come after end_dim.')
        current = before[:start] + [math.prod(before[start:end + 1])] + before[end + 1:]
        # Names of merged dimensions are no longer channel/spatial axes.
        out_axes = None
        kwargs = {'start_dim': start + 1, 'end_dim': end + 1}
    elif kind == 'Permute':
        if len(n['dims']) != len(before):
            raise GraphError('Permute dims must match the incoming sample rank.')
        current = [before[d] for d in n['dims']]
        out_axes = [axes[d] for d in n['dims']] if axes else None
        kwargs = {'dims': [0, *[d + 1 for d in n['dims']]]}
    tensor_shape(current)
    return {'output_shape': current, 'output_axes': out_axes, 'parameters': params,
            'weight_shape': weight_shape, 'constructor': kind, 'kwargs': kwargs,
            'affine': kind != 'MaxPool2d', 'matrix_affine': False}


def constructor_code(spec):
    """Render only normalized, validated values; never user-supplied source."""
    name = spec['constructor']
    args = ', '.join(f'{key}=' + ('torch.float32' if key == 'dtype' else repr(value))
                     for key, value in sorted(spec['kwargs'].items()))
    return f'{"_Permute" if name == "Permute" else "nn." + name}({args})'
