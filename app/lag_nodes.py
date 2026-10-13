"""Closed causal lag role bindings and float32 control adapter."""
import copy

ROLES = ('observations','history','lengths')


def normalized(node):
    from .core import GraphError
    from .causal_lags import normalized as kernel_normalized, CausalLagWindowError
    if not isinstance(node,dict):raise GraphError('CausalLagWindow must be a closed object.')
    primitive={k:v for k,v in node.items() if k!='inputs'}
    try:result=kernel_normalized(primitive)
    except CausalLagWindowError as error:raise GraphError(str(error)) from error
    bindings=node.get('inputs')
    if (not isinstance(bindings,dict) or set(bindings)!=set(ROLES) or
        any(not isinstance(v,str) or not v or v!=v.strip() for v in bindings.values())):
        raise GraphError('CausalLagWindow requires explicit observations/history/lengths source bindings.')
    result['inputs']=copy.deepcopy(bindings)
    return result


def sources(node):
    n=normalized(node)
    return list(dict.fromkeys(n['inputs'][r] for r in ROLES))


def describe(node,incoming):
    from .core import GraphError
    n=normalized(node);bound={r:incoming[k] for r,k in n['inputs'].items()}
    observed,axes=bound['observations'];history,history_axes=bound['history'];lengths,_=bound['lengths']
    if len(observed)!=2 or history!=[n['lags'],observed[-1]] or lengths!=[1]:
        raise GraphError('CausalLagWindow requires observations[T,D], history[p,D] and lengths[1], with identical D.')
    semantic={'T':'time','D':'coordinate'}
    if axes is not None and [semantic.get(a,a) for a in axes]!=['time','coordinate']:
        raise GraphError('Causal lag observation axes must be time/coordinate.')
    if history_axes is not None and history_axes!=['lag','coordinate']:
        raise GraphError('Causal history axes must be oldest-to-newest lag/coordinate.')
    t,d=observed;p=n['lags']
    return {'lags':p,'output_shape':[t,p*d],'output_axes':['time','lag-feature'],
            'work_elements':1+t+3*t+4*(t+p)*d+2*t*p*d,'parameters':0}


def _checked_lengths(observations,history,lengths,p):
    import torch
    for value in (observations,history,lengths):
        if not isinstance(value,torch.Tensor) or value.device.type!='cpu' or value.layout!=torch.strided or value.dtype!=torch.float32:
            raise ValueError('Causal lag diagram roles require dense CPU float32 tensors.')
    if observations.ndim!=3:raise ValueError('Causal observations require [B,T,D].')
    b,t,d=observations.shape
    if list(history.shape)!=[b,p,d] or list(lengths.shape)!=[b,1]:
        raise ValueError('Causal history/length shapes must match exactly; broadcasting is forbidden.')
    work=b+t+3*b*t+4*b*(t+p)*d+2*b*t*p*d
    if d<1 or max(b,t,d,p,t+p,p*d)>2**53-1 or work>1_000_000:
        raise ValueError('CausalLagWindow diagram exceeds its pre-conversion work/dimension budget.')
    if not bool((torch.isfinite(lengths)&(lengths>=0)&(lengths<=t)&(lengths==lengths.round())).all()):
        raise ValueError('Causal lengths require exact integer-valued prefixes in [0,T].')
    return lengths[:,0].to(torch.int64)


def runtime_class(nn,kernel_factory=None):
    if kernel_factory is None:
        from .causal_lags import runtime_class as kernel_factory
    kernel=kernel_factory(nn)
    class CausalLagWindow(nn.Module):
        def __init__(self,lags):
            super().__init__();self.lags=lags;self.kernel=kernel(lags=lags)
        def forward(self,observations,history,lengths):
            control=_checked_lengths(observations,history,lengths,self.lags)
            return self.kernel(observations,history,control)
    return CausalLagWindow


def runtime_source():
    from .task_execution import _helper_source
    return '\n\n'.join(_helper_source(f) for f in (_checked_lengths,runtime_class))
