"""Closed PositiveVariance v1 helper; graph admission and registration are external.

Normalization and runtime emission use only the standard library. Build imports
Torch lazily and owns no parameters or buffers. Forward computes precisely
functional.softplus(raw, beta=1, threshold=20) + the explicit float32 floor.
CPU strided float32 is the application contract; float64 supports reference and
gradient checks. Overflow is rejected, never clipped. The default one-million
element guard bounds this primitive's work, not whole-model memory admission.
An external admission may explicitly replace that guard via admitted_elements.
"""
import copy
import math
import struct

FLOAT32_MAX = 3.4028234663852886e38
DEFAULT_MAX_ELEMENTS = 1_000_000
COMMON = frozenset({'id', 'type', 'label', 'x', 'y', 'width', 'height',
                    'collapsed', 'op_version'})


class PositiveVarianceError(ValueError):
    """Invalid declaration, unsupported input or allocation budget."""


def _floor(value):
    if type(value) not in (int, float):
        raise PositiveVarianceError('positive_floor must be an explicit finite positive float32-range number.')
    if not 0 < value <= FLOAT32_MAX:
        raise PositiveVarianceError('positive_floor must be finite, positive and representable in float32.')
    number = struct.unpack('!f', struct.pack('!f', float(value)))[0]
    if not math.isfinite(number) or number <= 0:
        raise PositiveVarianceError('positive_floor must remain positive in float32; no clamping is performed.')
    return number


def _limit(max_elements, admitted_elements):
    for name, value in (('max_elements', max_elements), ('admitted_elements', admitted_elements)):
        if name == 'admitted_elements' and value is None:
            continue
        if type(value) is not int or value < 0:
            raise PositiveVarianceError(name + ' must be a nonnegative integer, not boolean.')
    return max_elements if admitted_elements is None else admitted_elements


def _budget(count, max_elements, admitted_elements):
    limit = _limit(max_elements, admitted_elements)
    if count > limit:
        raise PositiveVarianceError(f'PositiveVariance needs {count} elements; allocation budget is {limit}.')


def normalized(node):
    """Validate v1 without Torch; canonicalize the required floor to float32.

    Missing op_version defaults to 1. Presentation metadata is opaque and copied.
    No trainable, sharing, shape, dtype or Softplus tuning fields are accepted.
    """
    if not isinstance(node, dict) or node.get('type') != 'PositiveVariance':
        raise PositiveVarianceError('operator type must be PositiveVariance.')
    if set(node) - COMMON - {'positive_floor'}:
        raise PositiveVarianceError('PositiveVariance has unsupported fields; it has no trainable parameters or sharing.')
    version = node.get('op_version', 1)
    if type(version) is not int or version != 1:
        raise PositiveVarianceError('PositiveVariance supports integer op_version 1 only.')
    if 'positive_floor' not in node:
        raise PositiveVarianceError('PositiveVariance requires an explicit positive_floor; there is no default.')
    floor = _floor(node['positive_floor'])
    result = copy.deepcopy(node)
    result.update(op_version=1, positive_floor=floor)
    return result


def constructor_args(node):
    """Pure constructor data, shared by eager and app-free runtime paths."""
    return {'positive_floor': normalized(node)['positive_floor']}


def runtime_class(nn):
    """Use the caller's nn; construction allocates no tensors and has no state."""
    class PositiveVariance(nn.Module):
        def __init__(self, *, positive_floor, max_elements=DEFAULT_MAX_ELEMENTS,
                     admitted_elements=None):
            floor = _floor(positive_floor)
            _limit(max_elements, admitted_elements)
            super().__init__()
            self.positive_floor = floor
            self.max_elements = max_elements
            self.admitted_elements = admitted_elements

        def forward(self, raw):
            import torch
            if not isinstance(raw, torch.Tensor):
                raise PositiveVarianceError('raw must be a Torch tensor.')
            if (raw.device.type != 'cpu' or raw.layout != torch.strided or
                    raw.dtype not in (torch.float32, torch.float64)):
                raise PositiveVarianceError('raw must be a strided CPU float32 or float64 tensor.')
            # Guard before finite-check masks, scalar tensor or output allocation.
            _budget(raw.numel(), self.max_elements, self.admitted_elements)
            floor = _floor(self.positive_floor)
            if not bool(torch.isfinite(raw).all()):
                raise PositiveVarianceError('raw must be finite.')
            result = torch.nn.functional.softplus(raw, beta=1, threshold=20) + floor
            if not bool(torch.isfinite(result).all()):
                raise FloatingPointError('PositiveVariance exceeds input dtype range; no clipping is performed.')
            return result

    return PositiveVariance


def build(node, *, max_elements=DEFAULT_MAX_ELEMENTS, admitted_elements=None):
    """Lazy, allocation-free construction; model admission remains external."""
    args = constructor_args(node)
    _limit(max_elements, admitted_elements)
    from torch import nn
    return runtime_class(nn)(**args, max_elements=max_elements, admitted_elements=admitted_elements)


def runtime_source():
    """Emit an app-free audited closure; constructor_args supplies node data.

    Exporters supply nn and instantiate runtime_class(nn)(**kwargs), with their
    explicit admitted_elements if appropriate. No graph or registry is emitted.
    """
    import inspect
    definitions = (PositiveVarianceError, _floor, _limit, _budget, runtime_class)
    prelude = ('import math\nimport struct\n'
               f'FLOAT32_MAX = {FLOAT32_MAX!r}\n'
               f'DEFAULT_MAX_ELEMENTS = {DEFAULT_MAX_ELEMENTS!r}\n\n')
    return prelude + '\n\n'.join(inspect.getsource(obj) for obj in definitions)
