"""Closed PositiveVariance v1/v2 helper; graph admission is external.

Normalization and runtime emission use only the standard library. Build imports
Torch lazily and owns no parameters or buffers. Forward computes precisely
functional.softplus(raw, beta=1, threshold=20) + the explicit float32 floor in v1.
Version 2 explicitly chooses exp or softplus and permits an exactly zero floor.
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


def _floor(value, allow_zero=False):
    if type(value) not in (int, float):
        raise PositiveVarianceError('positive_floor must be an explicit finite positive float32-range number.')
    if not (0 <= value <= FLOAT32_MAX if allow_zero else 0 < value <= FLOAT32_MAX):
        raise PositiveVarianceError('positive_floor must be finite, positive and representable in float32.')
    number = struct.unpack('!f', struct.pack('!f', float(value)))[0]
    if not math.isfinite(number) or (number < 0 if allow_zero else number <= 0) or value>0 and number==0:
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
    """Validate the versioned declaration without Torch; canonicalize its floor.

    Missing op_version defaults to 1. Presentation metadata is opaque and copied.
    No trainable, sharing, shape, dtype or Softplus tuning fields are accepted.
    """
    if not isinstance(node, dict) or node.get('type') != 'PositiveVariance':
        raise PositiveVarianceError('operator type must be PositiveVariance.')
    version = node.get('op_version', 1)
    if type(version) is not int or version not in (1,2):
        raise PositiveVarianceError('PositiveVariance supports integer op_version1/2.')
    if set(node) - COMMON - {'positive_floor'} - ({'transform'} if version==2 else set()):
        raise PositiveVarianceError('PositiveVariance has unsupported fields; it has no trainable parameters or sharing.')
    if version==2 and node.get('transform') not in ('exp','softplus'):
        raise PositiveVarianceError('PositiveVariance2 requires an explicit exp or softplus transform.')
    if 'positive_floor' not in node:
        raise PositiveVarianceError('PositiveVariance requires an explicit positive_floor; there is no default.')
    floor = _floor(node['positive_floor'], allow_zero=version==2)
    result = copy.deepcopy(node)
    result.update(op_version=version, positive_floor=floor)
    return result


def constructor_args(node):
    """Pure constructor data, shared by eager and app-free runtime paths."""
    n=normalized(node)
    return {'positive_floor':n['positive_floor'],**({'transform':n['transform']} if n['op_version']==2 else {})}


def runtime_class(nn):
    """Use the caller's nn; construction allocates no tensors and has no state."""
    class PositiveVariance(nn.Module):
        def __init__(self, *, positive_floor, transform=None, max_elements=DEFAULT_MAX_ELEMENTS,
                     admitted_elements=None):
            if transform is not None and transform not in ('exp','softplus'):
                raise PositiveVarianceError('Choose an explicit exp or softplus variance transform.')
            floor = _floor(positive_floor, allow_zero=transform is not None)
            _limit(max_elements, admitted_elements)
            super().__init__()
            self.positive_floor = floor
            self.transform=transform or 'softplus';self.explicit_transform=transform is not None
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
            floor = _floor(self.positive_floor, allow_zero=self.explicit_transform)
            if not bool(torch.isfinite(raw).all()):
                raise PositiveVarianceError('raw must be finite.')
            value=raw.exp() if self.transform=='exp' else torch.nn.functional.softplus(raw,beta=1,threshold=20)
            result=value if floor==0 else value+floor
            if not bool(torch.isfinite(result).all()) or not bool((result>0).all()):
                raise FloatingPointError('PositiveVariance overflows/underflows input dtype range; no clipping is performed.')
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
    import ast
    from .task_execution import _helper_source
    definitions = (PositiveVarianceError, _floor, _limit, _budget, runtime_class)
    prelude = ('import math\nimport struct\n'
               f'FLOAT32_MAX = {FLOAT32_MAX!r}\n'
               f'DEFAULT_MAX_ELEMENTS = {DEFAULT_MAX_ELEMENTS!r}\n\n')
    error_source=inspect.getsource(PositiveVarianceError);error_class=ast.parse(error_source).body[0]
    if (not isinstance(error_class,ast.ClassDef) or len(error_class.bases)!=1 or
        not isinstance(error_class.bases[0],ast.Name) or error_class.bases[0].id!='ValueError' or
        any(not isinstance(stmt,ast.Pass) and not (isinstance(stmt,ast.Expr) and isinstance(stmt.value,ast.Constant)
            and isinstance(stmt.value.value,str)) for stmt in error_class.body)):
        raise PositiveVarianceError('Exception source changed; restart before exporting.')
    return prelude+error_source+'\n\n'+'\n\n'.join(_helper_source(f) for f in definitions[1:])
