"""Closed GaussianFilter diagram contract; no numerical imports during editing.

This multiport layer owns no parameters. Explicit incoming tensors supply the
probability law; the shared audited Gaussian kernel performs log-space filtering.
Task objectives must select evidence outputs, never invented regime labels.
"""
import copy
import math

TYPE = 'GaussianFilter'
ROLES = ('observations','initial','transition','means','variances','lengths')
OPTIONAL_ROLES = ('emission_mask',)
PORTS = ('filtered','final','step_evidence','sequence_evidence')
COMMON = {'id','type','label','x','y','width','height','collapsed','op_version'}
FIELDS = {'inputs','initial_representation','transition_representation','variance_mode'}


def normalized(node):
    from .core import GraphError
    if (not isinstance(node,dict) or node.get('type') != TYPE or
            set(node) - COMMON - FIELDS or type(node.get('op_version',1)) is not int or
            node.get('op_version',1) != 1):
        raise GraphError('GaussianFilter supports closed integer op_version1 with no parameter owner/sharing.')
    bindings=node.get('inputs')
    if not isinstance(bindings,dict) or not set(ROLES) <= set(bindings) <= set(ROLES+OPTIONAL_ROLES):
        raise GraphError('GaussianFilter binds observations, initial, transition, means, variances and lengths; emission_mask is optional.')
    if any(type(v) is not str or not v or v != v.strip() for v in bindings.values()):
        raise GraphError('GaussianFilter roles require explicit stable source keys.')
    for key in ('initial_representation','transition_representation'):
        if node.get(key) not in ('probabilities','log-probabilities'):
            raise GraphError('GaussianFilter must explicitly declare '+key+' as probabilities or log-probabilities; no inferred logits.')
    if node.get('variance_mode') not in ('shared','sequence'):
        raise GraphError('GaussianFilter variance_mode is explicit shared [K,D] or sequence [T,K,D]; values are positive variances, not logs or standard deviations.')
    return dict(copy.deepcopy(node),op_version=1)


def sources(node):
    n=normalized(node)
    return list(dict.fromkeys(n['inputs'][role] for role in ROLES+OPTIONAL_ROLES if role in n['inputs']))


def ports(node):
    normalized(node)
    return list(PORTS)


def describe(node, incoming):
    from .core import GraphError
    n=normalized(node)
    bound={role:incoming[key] for role,key in n['inputs'].items()}
    observed,observed_axes=bound['observations']
    initial,initial_axes=bound['initial']
    if len(observed)!=2 or len(initial)!=1:
        raise GraphError('GaussianFilter observations need [T,D] and initial state law [K], excluding B.')
    steps,dimensions=observed;states=initial[0]
    expected={'observations':([steps,dimensions],['time','coordinate']),
              'initial':([states],['state']),
              'transition':([steps,states,states],['time','from_state','to_state']),
              'means':([steps,states,dimensions],['time','state','coordinate']),
              'variances':([states,dimensions],['state','coordinate']) if n['variance_mode']=='shared' else
                          ([steps,states,dimensions],['time','state','coordinate']),
              'lengths':([1],None),'emission_mask':([steps],['time'])}
    for role,(shape,axes) in bound.items():
        required,names=expected[role]
        if shape!=required or axes is not None and names is not None and axes!=names:
            raise GraphError('GaussianFilter '+role+' needs sample shape '+str(required)+
                             (' with unnamed axes or '+str(names) if names is not None else '')+'.')
    output_shapes={'filtered':[steps,states],'final':[states],
                   'step_evidence':[steps],'sequence_evidence':[1]}
    axes={'filtered':['time','state'],'final':['state'],
          'step_evidence':['time'],'sequence_evidence':None}
    representations={'filtered':'log-probabilities','final':'log-probabilities',
                     'step_evidence':'log-evidence','sequence_evidence':'log-evidence'}
    return {'ports':output_shapes,'port_axes':axes,'representations':representations,
            'parameters':0,'steps':steps,'states':states,'dimensions':dimensions,
            'work_elements':steps*(8*states*states+16*states*dimensions+12*states+4*dimensions+8)+4*states,
            'sources':sources(n),'inputs':copy.deepcopy(n['inputs']),
            'initial_representation':n['initial_representation'],
            'transition_representation':n['transition_representation'],'variance_mode':n['variance_mode']}


def _log_probabilities(value):
    """Exact structural zeros have -inf values and zero derivative, never NaN."""
    import torch
    if not bool(torch.isfinite(value).all()) or not bool((value>=0).all()):
        raise ValueError('GaussianFilter probabilities must be finite and nonnegative.')
    if not bool(((value.sum(-1)-1).abs() <= 2e-6).all()):
        raise ValueError('GaussianFilter probabilities must have normalized last-axis rows.')
    positive=value>0
    safe=torch.where(positive,value,torch.ones_like(value))
    return torch.where(positive,safe.log(),torch.full_like(value,-math.inf))


def forward(parts, plan):
    """Parameter-free CPU float32 diagram projection; explicit axis conventions."""
    import torch
    from .gaussian_filter import filter_diagonal_gaussian_hmm
    bound={role:parts[plan['sources'].index(key)] for role,key in plan['inputs'].items()}
    if any(not isinstance(v,torch.Tensor) or v.device.type!='cpu' or v.dtype!=torch.float32 or
           v.layout!=torch.strided for v in bound.values()):
        raise ValueError('GaussianFilter graph roles are dense CPU float32 tensors; controls use explicit exact adapters.')
    observed=bound['observations'];batch,steps,dimensions=observed.shape
    length_values=bound['lengths']
    if (tuple(length_values.shape)!=(batch,1) or not bool(torch.isfinite(length_values).all()) or
            not bool(((length_values==length_values.trunc()) & (length_values>=0) & (length_values<=steps)).all())):
        raise ValueError('GaussianFilter lengths are exact float32-encoded integers [B,1] in [0,T].')
    lengths=length_values[:,0].to(torch.int64)
    emission=None
    if 'emission_mask' in bound:
        mask=bound['emission_mask']
        if not bool(((mask==0)|(mask==1)).all()):
            raise ValueError('GaussianFilter emission_mask requires exact 0/1 [B,T] values; not an execution or loss mask.')
        emission=mask.to(torch.bool)
    initial=bound['initial'];transition=bound['transition']
    if plan['initial_representation']=='probabilities': initial=_log_probabilities(initial)
    if plan['transition_representation']=='probabilities':
        # Inactive padded probabilities may be NaN; sanitize before logs, just
        # as the numerical filter sanitizes inactive log transition entries.
        active=torch.arange(steps)[None,:]<lengths[:,None]
        states=initial.shape[-1]
        transition=torch.where(active[:,:,None,None],transition,transition.new_tensor(1/states))
        transition=_log_probabilities(transition)
    variance=bound['variances']
    if plan['variance_mode']=='shared':variance=variance[:,None].expand(batch,steps,*variance.shape[1:])
    result=filter_diagonal_gaussian_hmm(observed,log_initial=initial,log_transition=transition,
        means=bound['means'],positive_variances=variance,lengths=lengths,emission_mask=emission)
    return {'filtered':result.filtered_log_probs,'final':result.final_log_probs,
            'step_evidence':result.step_log_evidence,
            'sequence_evidence':result.sequence_log_likelihood[:,None]}


def runtime_class(nn):
    class GaussianFilter(nn.Module):
        def __init__(self,plan):
            super().__init__();self.plan=plan
        def forward(self,*parts):
            return forward(parts,self.plan)
    return GaussianFilter
