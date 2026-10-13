"""Internal tensor-source kernel; no graph/operator/schema admission.

``normalized`` and ``constructor_args`` are allocation-free Python functions.
``build`` lazily constructs a CPU float32 source, using the same class factory
that ``runtime_source`` emits for app-free exports. Call forward(batch_size),
never pass dummy data. Shape includes sample axes only (including T, if any).

TensorParameter owns exactly one native Parameter, ``value``, initially zero.
Graph parameter_policy alone controls initialization/freezing; callers provide
its existing explicit path -> {id, shape} schema. ``share_with`` is preserved
as text during editing; construction refuses until graph-level resolution has
removed it or reused the actual owner module. Constant owns no parameters: its
nonpersistent buffer is reconstructed from immutable graph content, not weights.

The element guard precedes tensor allocations; it is a numerical allocation
guard, not an RSS estimate or permission to allocate memory. Expanded outputs
are views: consumers must not mutate them in place.

References: https://docs.pytorch.org/docs/2.13/generated/torch.nn.Module.html
and https://docs.pytorch.org/docs/2.13/generated/torch.Tensor.expand.html.
"""
import copy
import math
import struct

SOURCE_TYPES = frozenset({'Constant', 'TensorParameter'})
MAX_DIMENSION = 2**53 - 1
FLOAT32_MAX = 3.4028234663852886e38
DEFAULT_MAX_ELEMENTS = 1_000_000
COMMON = frozenset({'id', 'type', 'label', 'x', 'y', 'width', 'height',
                    'collapsed', 'op_version'})


class TensorSourceError(ValueError):
    """Invalid source declaration or unsupported source execution."""


def _shape(shape):
    if (not isinstance(shape, list) or not 1 <= len(shape) <= 8 or
            any(type(d) is not int or not 1 <= d <= MAX_DIMENSION for d in shape)):
        raise TensorSourceError('shape needs 1..8 positive exact integer dimensions up to 2**53-1.')
    return list(shape)


def _float32(value):
    # Round with the standard library, without importing Torch or NumPy. This
    # also preserves -0.0 (including negative underflow to signed zero).
    if type(value) not in (int, float) or abs(value) > FLOAT32_MAX:
        raise TensorSourceError('literal must be a finite float32-range number, not boolean.')
    number = float(value)
    if not math.isfinite(number):
        raise TensorSourceError('literal must be a finite float32-range number, not boolean.')
    return struct.unpack('!f', struct.pack('!f', number))[0]


def _values(values, shape):
    if not shape:
        return _float32(values)
    if not isinstance(values, list) or len(values) != shape[0]:
        raise TensorSourceError('values must be a rectangular nested array of exactly the declared shape.')
    return [_values(part, shape[1:]) for part in values]


def _budget(shape, max_elements, admitted_elements):
    for name, value in (('max_elements', max_elements), ('admitted_elements', admitted_elements)):
        if name == 'admitted_elements' and value is None:
            continue
        if type(value) is not int or value < 0:
            raise TensorSourceError(name + ' must be a nonnegative integer, not boolean.')
    limit = max_elements if admitted_elements is None else admitted_elements
    count = math.prod(shape)
    if count > limit:
        raise TensorSourceError(f'source needs {count} elements; allocation budget is {limit}.')
    return count


def normalized(node):
    """Validate the closed v1 declaration without expanding compact fills.

    Missing op_version defaults to 1. Axes, when present, name each sample
    dimension uniquely. Common presentation fields are opaque metadata. Numeric
    literals are canonical float32 values; dimensions are never feature-capped.
    The returned declaration does not alias the caller's arrays/metadata.
    """
    if not isinstance(node, dict) or type(node.get('type')) is not str or node['type'] not in SOURCE_TYPES:
        raise TensorSourceError('source type must be Constant or TensorParameter.')
    kind = node['type']
    version = node.get('op_version', 1)
    if type(version) is not int or version != 1:
        raise TensorSourceError('sources support integer op_version 1 only.')
    fields = {'shape', 'axes'} | ({'fill', 'values'} if kind == 'Constant' else {'share_with'})
    if set(node) - COMMON - fields:
        raise TensorSourceError('source has unsupported fields; initialization/freezing belong to parameter_policy.')
    shape = _shape(node.get('shape'))
    axes = node.get('axes')
    if 'axes' in node and (not isinstance(axes, list) or len(axes) != len(shape) or
                          any(type(a) is not str or not a or len(a) > 32 for a in axes) or
                          len(set(axes)) != len(axes)):
        raise TensorSourceError('axes need one unique nonempty name (up to 32 characters) per sample dimension.')
    if 'share_with' in node and type(node['share_with']) is not str:
        raise TensorSourceError('share_with must be a string; graph-level resolution is external.')
    literal = {}
    if kind == 'Constant':
        if ('fill' in node) == ('values' in node):
            raise TensorSourceError('Constant requires exactly one of fill or values.')
        literal = ({'fill': _float32(node['fill'])} if 'fill' in node else
                   {'values': _values(node['values'], shape)})
    result = copy.deepcopy({key: value for key, value in node.items() if key not in {'shape', 'axes', 'fill', 'values'}})
    result.update(op_version=1, shape=shape, **literal)
    if 'axes' in node:
        result['axes'] = list(axes)
    return result


def constructor_args(node):
    """Pure shared runtime kwargs; unresolved sharing refuses construction."""
    node = normalized(node)
    if 'share_with' in node:
        raise TensorSourceError('resolve share_with externally and reuse the actual owner module before construction.')
    result = {'kind': node['type'], 'shape': node['shape']}
    for key in ('fill', 'values'):
        if key in node:
            result[key] = node[key]
    return result


def runtime_class(nn):
    """Return the source class using the caller's nn; do not execute user code."""
    class TensorSource(nn.Module):
        def __init__(self, kind, shape, *, fill=Ellipsis, values=Ellipsis,
                     max_elements=DEFAULT_MAX_ELEMENTS, admitted_elements=None):
            shape = _shape(shape)
            # Do this before super/init, literal materialization or any tensor
            # constructor, including for direct/app-free class construction.
            _budget(shape, max_elements, admitted_elements)
            if type(kind) is not str or kind not in ('Constant', 'TensorParameter'):
                raise TensorSourceError('source type must be Constant or TensorParameter.')
            if kind == 'Constant':
                if (fill is Ellipsis) == (values is Ellipsis):
                    raise TensorSourceError('Constant requires exactly one of fill or values.')
                literal = _float32(fill) if values is Ellipsis else _values(values, shape)
            elif fill is not Ellipsis or values is not Ellipsis:
                raise TensorSourceError('TensorParameter initialization belongs to parameter_policy.')
            super().__init__()
            import torch
            self.kind = kind
            self.sample_shape = tuple(shape)
            if kind == 'TensorParameter':
                self.value = nn.Parameter(torch.zeros(shape, dtype=torch.float32, device='cpu'))
            else:
                value = (torch.full(shape, literal, dtype=torch.float32, device='cpu')
                         if values is Ellipsis else torch.tensor(literal, dtype=torch.float32, device='cpu'))
                self.register_buffer('value', value, persistent=False)

        def forward(self, batch_size):
            if type(batch_size) is not int or batch_size < 0:
                raise TensorSourceError('batch_size must be a nonnegative integer, not boolean or data.')
            import torch
            value = self.value
            if (value.device.type != 'cpu' or value.dtype != torch.float32 or
                    value.layout != torch.strided or tuple(value.shape) != self.sample_shape):
                raise TensorSourceError('source value must retain its declared shape and CPU float32 strided storage.')
            if self.kind == 'Constant' and value.requires_grad:
                raise TensorSourceError('Constant cannot require gradients.')
            return value.unsqueeze(0).expand(batch_size, *self.sample_shape)

    return TensorSource


def build(node, *, max_elements=DEFAULT_MAX_ELEMENTS, admitted_elements=None):
    """Lazily build one source; refuse unresolved sharing and leave policy external."""
    args = constructor_args(node)
    _budget(args['shape'], max_elements, admitted_elements)
    from torch import nn
    return runtime_class(nn)(**args, max_elements=max_elements, admitted_elements=admitted_elements)


def runtime_source():
    """Emit the audited helper closure; no eval/exec, app imports or user text.

    Exporters prepare ``kwargs = constructor_args(node)`` here, embed those
    kwargs as data, supply ``from torch import nn`` and instantiate the emitted
    ``runtime_class(nn)(**kwargs, admitted_elements=...)``. This function emits
    definitions only, not a graph or a policy/configure hook. Its stdlib closure
    is math and struct; Torch is imported locally by constructor/forward.
    """
    import inspect
    definitions = (TensorSourceError, _shape, _float32, _values, _budget, runtime_class)
    prelude = ('import math\nimport struct\n'
               f'MAX_DIMENSION = {MAX_DIMENSION!r}\n'
               f'FLOAT32_MAX = {FLOAT32_MAX!r}\n'
               f'DEFAULT_MAX_ELEMENTS = {DEFAULT_MAX_ELEMENTS!r}\n\n')
    return prelude + '\n\n'.join(inspect.getsource(obj) for obj in definitions)
