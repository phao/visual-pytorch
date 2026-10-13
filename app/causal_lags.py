"""Closed CausalLagWindow v1; graph admission and registration are external.

At active time t, concatenate the preceding p vectors, oldest to newest,
from explicit initial_history followed by observations. Never include x[t].
Lengths declare a prefix, including zero; there is no reset, missing-observation,
execution-mask or emission-mask inference. Inactive observations and the history
of zero-length sequences may be nonfinite and are sanitized before windowing.

The default guard bounds conservative element work, not RSS or model admission.
Its estimate is B + T + 3*B*T + 4*B*(T+p)*D + 2*B*T*p*D, accounting for
length checks, time/prefix masks, sanitized inputs, finite masks, staging and
packed output. Only an external admission may raise it via admitted_elements.
Normalization, description and checked source emission do not import Torch.
"""
import copy
from .sequence_controls import lag_run

DEFAULT_MAX_ELEMENTS = 1_000_000
MAX_DIMENSION = 2**53 - 1
COMMON = frozenset({'id', 'type', 'label', 'x', 'y', 'width', 'height',
                    'collapsed', 'op_version'})


class CausalLagWindowError(ValueError):
    """Invalid declaration, unsupported input or exceeded work budget."""


def _lags(value):
    if type(value) is not int or value <= 0:
        raise CausalLagWindowError('lags must be an explicit positive exact integer, not boolean.')
    return value


def _limit(max_elements, admitted_elements):
    for name, value in (('max_elements', max_elements), ('admitted_elements', admitted_elements)):
        if name == 'admitted_elements' and value is None:
            continue
        if type(value) is not int or value < 0:
            raise CausalLagWindowError(name + ' must be a nonnegative exact integer.')
    if max_elements > DEFAULT_MAX_ELEMENTS:
        raise CausalLagWindowError('raising the default budget requires external admission via admitted_elements.')
    return max_elements if admitted_elements is None else admitted_elements


def _budget(batch, time, features, lags, max_elements, admitted_elements):
    # Python integers do not overflow, including for adversarial compact shapes.
    dimensions = (batch, time, features, lags, time + lags, lags * features)
    if any(d < 0 or d > MAX_DIMENSION for d in dimensions) or features == 0:
        raise CausalLagWindowError('dimensions must be bounded by 2**53-1; D must be positive.')
    count = (batch + time + 3 * batch * time +
             4 * batch * (time + lags) * features +
             2 * batch * time * lags * features)
    limit = _limit(max_elements, admitted_elements)
    if count > limit:
        raise CausalLagWindowError(f'CausalLagWindow needs {count} work elements; allocation budget is {limit}.')
    return count


def normalized(node):
    """Copy a closed declaration; the only numerical constructor field is lags."""
    if not isinstance(node, dict) or node.get('type') != 'CausalLagWindow':
        raise CausalLagWindowError('operator type must be CausalLagWindow.')
    if set(node) - COMMON - {'lags'}:
        raise CausalLagWindowError('CausalLagWindow has unsupported fields; only lags and common presentation are allowed.')
    version = node.get('op_version', 1)
    if type(version) is not int or version not in (1,2):
        raise CausalLagWindowError('CausalLagWindow supports integer op_version1/2.')
    lags = _lags(node.get('lags'))
    result = copy.deepcopy(node)
    result.update(op_version=version, lags=lags)
    return result


def constructor_args(node):
    """Pure kwargs shared by eager and standalone construction."""
    n=normalized(node)
    return {'lags':n['lags'],**({'op_version':2} if n['op_version']==2 else {})}


def describe(node):
    """Pure human-readable contract for callers' presentation layers."""
    p = constructor_args(node)['lags']
    return (f'CausalLagWindow v1: {p} preceding observations, oldest to newest; '
            f'[B,T,D] + initial_history[B,{p},D] + int64 lengths[B] '
            f'-> [B,T,{p}*D]. Explicit active prefixes; inactive output is zero.')


def runtime_class(nn):
    """Construct a stateless module with the caller's nn; no tensor allocation."""
    class CausalLagWindow(nn.Module):
        def __init__(self, *, lags, max_elements=DEFAULT_MAX_ELEMENTS,
                     admitted_elements=None, op_version=1):
            lags = _lags(lags)
            _limit(max_elements, admitted_elements)
            if type(op_version) is not int or op_version not in (1,2):raise CausalLagWindowError('Unsupported causal operator version.')
            super().__init__()
            self.lags = lags
            self.max_elements = max_elements
            self.admitted_elements = admitted_elements
            self.op_version=op_version

        def forward(self, observations, initial_history, lengths, **controls):
            if self.op_version==2:
                try:return lag_run(observations,initial_history,lengths,lags=self.lags,
                    max_elements=self.max_elements,admitted_elements=self.admitted_elements,**controls)
                except ValueError as error:raise CausalLagWindowError(str(error)) from error
            if controls:raise TypeError('CausalLagWindow1 does not accept sequence controls.')
            import torch
            p = _lags(self.lags)
            for name, value in (('observations', observations),
                                ('initial_history', initial_history), ('lengths', lengths)):
                if not isinstance(value, torch.Tensor):
                    raise CausalLagWindowError(name + ' must be a Torch tensor.')
                if value.device.type != 'cpu' or value.layout != torch.strided:
                    raise CausalLagWindowError(name + ' must be a strided CPU tensor.')
            if observations.dtype not in (torch.float32, torch.float64):
                raise CausalLagWindowError('observations must be float32 or float64.')
            if initial_history.dtype != observations.dtype:
                raise CausalLagWindowError('initial_history must have the same dtype as observations.')
            if observations.ndim != 3:
                raise CausalLagWindowError('observations must have exact shape [B,T,D].')
            b, t, d = observations.shape
            if tuple(initial_history.shape) != (b, p, d):
                raise CausalLagWindowError('initial_history must have exact shape [B,lags,D]; no broadcasting.')
            if lengths.dtype != torch.int64 or tuple(lengths.shape) != (b,):
                raise CausalLagWindowError('lengths must have exact int64 dtype and shape [B].')
            # Before comparisons, finite masks, arange, sanitization or output.
            _budget(b, t, d, p, self.max_elements, self.admitted_elements)
            if not bool(((lengths >= 0) & (lengths <= t)).all()):
                raise CausalLagWindowError('lengths must declare prefixes within [0,T].')
            if b == 0 or t == 0:
                # Empty concatenation reads no values, including NaN history,
                # while retaining zero-gradient paths to both real inputs.
                empty = torch.cat((initial_history[:, :0, :], observations), dim=1)
                return empty.unsqueeze(2).expand(b, t, p, d).reshape(b, t, p * d)
            # Every expansion is explicit and all real input shapes were checked.
            times = torch.arange(t, device='cpu').reshape(1, t).expand(b, t)
            active = times < lengths.reshape(b, 1).expand(b, t)
            obs_mask = active.unsqueeze(-1).expand(b, t, d)
            history_mask = (lengths > 0).reshape(b, 1, 1).expand(b, p, d)
            zero = observations.new_zeros(())
            safe_obs = torch.where(obs_mask, observations, zero.expand(b, t, d))
            safe_history = torch.where(history_mask, initial_history, zero.expand(b, p, d))
            if not bool(torch.isfinite(safe_obs).all()):
                raise CausalLagWindowError('active observations must be finite.')
            if not bool(torch.isfinite(safe_history).all()):
                raise CausalLagWindowError('active initial_history must be finite.')
            stream = torch.cat((safe_history, safe_obs), dim=1)
            # unfold yields [B,T,D,p]; drop the final vector BEFORE windowing.
            windows = stream[:, :-1, :].unfold(1, p, 1)
            packed = windows.permute(0, 1, 3, 2).reshape(b, t, p * d)
            out_mask = active.unsqueeze(-1).expand(b, t, p * d)
            return torch.where(out_mask, packed, zero.expand(b, t, p * d))

    return CausalLagWindow


def build(node, *, max_elements=DEFAULT_MAX_ELEMENTS, admitted_elements=None):
    """Lazy construction; explicit external admission is the only raised guard."""
    args = constructor_args(node)
    _limit(max_elements, admitted_elements)
    from torch import nn
    return runtime_class(nn)(**args, max_elements=max_elements, admitted_elements=admitted_elements)


def runtime_source():
    """Emit the actual checked function closure, never arbitrary declaration code.

    Exporters supply nn and instantiate runtime_class(nn)(**constructor_args).
    The emitted closure has no app, registry, graph or presentation dependencies.
    """
    import inspect
    import ast
    from .task_execution import _helper_source
    from . import sequence_controls as controls
    definitions = (CausalLagWindowError, _lags, _limit, _budget, runtime_class)
    prelude = (f'DEFAULT_MAX_ELEMENTS = {DEFAULT_MAX_ELEMENTS!r}\n'
               f'MAX_DIMENSION = {MAX_DIMENSION!r}\n\n')
    error_source=inspect.getsource(CausalLagWindowError)
    error_class=ast.parse(error_source).body[0]
    if (not isinstance(error_class,ast.ClassDef) or len(error_class.bases)!=1 or
        not isinstance(error_class.bases[0],ast.Name) or error_class.bases[0].id!='ValueError' or
        any(not isinstance(stmt,ast.Pass) and not (isinstance(stmt,ast.Expr) and
            isinstance(stmt.value,ast.Constant) and isinstance(stmt.value.value,str)) for stmt in error_class.body)):
        raise CausalLagWindowError('Exception source changed; restart before exporting.')
    functions=(controls._float_roles,controls._mask_role,controls._prefix_role,controls.lag_run,*definitions[1:])
    return prelude + error_source + '\n\n' + '\n\n'.join(_helper_source(obj) for obj in functions)
