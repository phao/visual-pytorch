"""Closed causal lag role bindings and float32 control adapter."""
import copy

ROLES = ('observations','history','lengths')
OPTIONAL_ROLES = ('execution_mask','reset_mask','reset_history')


def normalized(node):
    from .core import GraphError
    from .causal_lags import normalized as kernel_normalized, CausalLagWindowError
    if not isinstance(node,dict):raise GraphError('CausalLagWindow must be a closed object.')
    primitive={k:v for k,v in node.items() if k!='inputs'}
    try:result=kernel_normalized(primitive)
    except CausalLagWindowError as error:raise GraphError(str(error)) from error
    bindings=node.get('inputs')
    allowed=set(ROLES+(OPTIONAL_ROLES if result['op_version']==2 else ()))
    if (not isinstance(bindings,dict) or not set(ROLES)<=set(bindings)<=allowed or
        any(not isinstance(v,str) or not v or v!=v.strip() for v in bindings.values())):
        raise GraphError('CausalLagWindow requires explicit observations/history/lengths source bindings.')
    if ('reset_mask' in bindings)!=('reset_history' in bindings):raise GraphError('Causal resets require a mask and explicit reset histories together.')
    result['inputs']=copy.deepcopy(bindings)
    return result


def sources(node):
    n=normalized(node)
    return list(dict.fromkeys(n['inputs'][r] for r in ROLES+OPTIONAL_ROLES if r in n['inputs']))


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
    for role in OPTIONAL_ROLES:
        if role in bound:
            shape,names=bound[role]
            expected=[t,p,d] if role=='reset_history' else [t]
            axes_expected=['time','lag','coordinate'] if role=='reset_history' else ['time']
            if shape!=expected or names is not None and [semantic.get(a,a) for a in names]!=axes_expected:
                raise GraphError('Causal '+role+' requires exact '+str(expected)+' with matching time/history axes.')
    return {'lags':p,'output_shape':[t,p*d],'output_axes':['time','lag-feature'],
            'work_elements':1+t+10*t+8*(t+p)*d+8*t*p*d if n['op_version']==2 else 1+t+3*t+4*(t+p)*d+2*t*p*d,'parameters':0}


def _checked_lengths(observations,history,lengths,p,version=1):
    import torch
    for value in (observations,history,lengths):
        if not isinstance(value,torch.Tensor) or value.device.type!='cpu' or value.layout!=torch.strided or value.dtype!=torch.float32:
            raise ValueError('Causal lag diagram roles require dense CPU float32 tensors.')
    if observations.ndim!=3:raise ValueError('Causal observations require [B,T,D].')
    b,t,d=observations.shape
    if list(history.shape)!=[b,p,d] or list(lengths.shape)!=[b,1]:
        raise ValueError('Causal history/length shapes must match exactly; broadcasting is forbidden.')
    work=b+t+3*b*t+4*b*(t+p)*d+2*b*t*p*d
    if version==2:work=b+t+10*b*t+8*b*(t+p)*d+8*b*t*p*d
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
        def __init__(self,lags,op_version=1,bindings=None):
            super().__init__();self.lags=lags;self.op_version=op_version;self.bindings=bindings
            self.kernel=kernel(lags=lags,op_version=op_version)
        def forward(self,*parts):
            import torch
            if self.bindings is None:bound=dict(zip(('observations','history','lengths'),parts))
            else:
                keys=list(dict.fromkeys(self.bindings[r] for r in ('observations','history','lengths','execution_mask','reset_mask','reset_history') if r in self.bindings))
                bound={r:parts[keys.index(key)] for r,key in self.bindings.items()}
            observations,history,lengths=(bound[r] for r in ('observations','history','lengths'))
            control=_checked_lengths(observations,history,lengths,self.lags,self.op_version)
            extra={}
            for role in ('execution_mask','reset_mask'):
                if role in bound:
                    value=bound[role]
                    if (not isinstance(value,torch.Tensor) or value.device.type!='cpu' or value.layout!=torch.strided or value.dtype!=torch.float32 or
                        list(value.shape)!=list(observations.shape[:2]) or not bool(((value==0)|(value==1)).all())):
                        raise ValueError('Causal '+role+' must be exact CPU float32 0/1 [B,T].')
                    extra[role]=value.to(torch.bool)
            if 'reset_history' in bound:extra['reset_history']=bound['reset_history']
            return self.kernel(observations,history,control,**extra)
    return CausalLagWindow


def runtime_source():
    from .task_execution import _helper_source
    return '\n\n'.join(_helper_source(f) for f in (_checked_lengths,runtime_class))
