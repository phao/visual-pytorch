"""Pure activation contracts; normalization axes exclude batch B."""
ACTIVATIONS = {
    'ReLU': 'nn.ReLU(inplace=False)',
    'Tanh': 'nn.Tanh()',
    'Sigmoid': 'nn.Sigmoid()',
    'GELU': "nn.GELU(approximate='none')",
    'LogSoftmax': 'nn.LogSoftmax(dim=-1)',
    'Softmax': 'nn.Softmax(dim=-1)',
    'Softplus': 'nn.Softplus(beta=1, threshold=20)',
}

AXIS_ACTIVATIONS = frozenset({'Softmax', 'LogSoftmax'})


def activation_name(value):
    from .core import GraphError
    if not isinstance(value, str) or value not in ACTIVATIONS:
        raise GraphError('Choose an activation: ' + ', '.join(sorted(ACTIVATIONS)) + '.')
    return value


def softmax_axis(node, rank=None):
    """Check the declared sample axis; with a rank return its nonnegative index."""
    from .core import GraphError
    name = node.get('activation', 'Softmax')
    value = node.get('dim', -1)
    if type(value) is not int or not -8 <= value <= 7:
        raise GraphError(f'{name} dim must be an integer sample axis -8..7, excluding batch B.')
    if rank is None:
        return value
    if type(rank) is not int or not 1 <= rank <= 8 or not -rank <= value < rank:
        raise GraphError(f'{name} sample axis {value} is out of range for rank {rank}; batch B is excluded.')
    return value % rank


def normalize_activation(node):
    name = activation_name(node.get('activation'))
    if name not in AXIS_ACTIVATIONS and name != 'Softplus':
        return dict(node)  # Preserve the original four activation documents.
    from .core import GraphError
    allowed = {'id', 'type', 'activation', 'label', 'x', 'y', 'width', 'height', 'collapsed'}
    if name in AXIS_ACTIVATIONS:
        allowed.add('dim')
    unknown = set(node) - allowed
    if unknown:
        raise GraphError(name + ' has unsupported fields: ' + ', '.join(sorted(unknown)) + '.')
    return dict(node, dim=softmax_axis(node)) if name in AXIS_ACTIVATIONS else dict(node)


def constructor_code(node, rank):
    node = normalize_activation(node)
    name = node['activation']
    return (f'nn.{name}(dim={softmax_axis(node, rank) + 1})'
            if name in AXIS_ACTIVATIONS else ACTIVATIONS[name])
