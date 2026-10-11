"""Pure activation contracts; Softmax axes index sample dimensions, excluding B."""
ACTIVATIONS = {
    'ReLU': 'nn.ReLU(inplace=False)',
    'Tanh': 'nn.Tanh()',
    'Sigmoid': 'nn.Sigmoid()',
    'GELU': "nn.GELU(approximate='none')",
    'Softmax': 'nn.Softmax(dim=-1)',
}


def activation_name(value):
    from .core import GraphError
    if not isinstance(value, str) or value not in ACTIVATIONS:
        raise GraphError('Choose an activation: ' + ', '.join(sorted(ACTIVATIONS)) + '.')
    return value


def softmax_axis(node, rank=None):
    """Check the declared sample axis; with a rank return its nonnegative index."""
    from .core import GraphError
    value = node.get('dim', -1)
    if type(value) is not int or not -8 <= value <= 7:
        raise GraphError('Softmax dim must be an integer sample axis -8..7, excluding batch B.')
    if rank is None:
        return value
    if not -rank <= value < rank:
        raise GraphError(f'Softmax sample axis {value} is out of range for rank {rank}; batch B is excluded.')
    return value % rank


def normalize_activation(node):
    name = activation_name(node.get('activation'))
    if name != 'Softmax':
        return dict(node)  # Preserve the original four activation documents.
    from .core import GraphError
    allowed = {'id', 'type', 'activation', 'dim', 'label', 'x', 'y', 'width', 'height', 'collapsed'}
    unknown = set(node) - allowed
    if unknown:
        raise GraphError('Softmax has unsupported fields: ' + ', '.join(sorted(unknown)) + '.')
    return dict(node, dim=softmax_axis(node))


def constructor_code(node, rank):
    name = activation_name(node.get('activation'))
    return (f'nn.Softmax(dim={softmax_axis(node, rank) + 1})'
            if name == 'Softmax' else ACTIVATIONS[name])
