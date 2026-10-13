"""Closed GaussianFilter diagram contract; no numerical imports during editing.

This multiport layer owns no parameters. Explicit incoming tensors supply the
probability law; the shared audited Gaussian kernel performs log-space filtering.
Task objectives must select evidence outputs, never invented regime labels.
"""
import copy
import math

TYPE = 'GaussianFilter'
ROLES = ('observations','initial','transition','means','variances','lengths')
OPTIONAL_ROLES = ('emission_mask','execution_mask','reset_mask')
PORTS = ('filtered','final','step_evidence','sequence_evidence')
COMMON = {'id','type','label','x','y','width','height','collapsed','op_version'}
FIELDS = {'inputs','initial_representation','transition_representation','transition_mode','mean_mode','variance_mode'}


def normalized(node):
    from .core import GraphError
    if (not isinstance(node,dict) or node.get('type') != TYPE or
            set(node) - COMMON - FIELDS or type(node.get('op_version',1)) is not int or
            node.get('op_version',1) not in (1,2)):
        raise GraphError('GaussianFilter supports closed integer op_version1/2 with no parameter owner/sharing.')
    bindings=node.get('inputs')
    optional=OPTIONAL_ROLES if node.get('op_version',1)==2 else ('emission_mask',)
    if not isinstance(bindings,dict) or not set(ROLES) <= set(bindings) <= set(ROLES+optional):
        raise GraphError('GaussianFilter binds observations, initial, transition, means, variances and lengths; emission_mask is optional.')
    if any(type(v) is not str or not v or v != v.strip() for v in bindings.values()):
        raise GraphError('GaussianFilter roles require explicit stable source keys.')
    for key in ('initial_representation','transition_representation'):
        if node.get(key) not in ('probabilities','log-probabilities'):
            raise GraphError('GaussianFilter must explicitly declare '+key+' as probabilities or log-probabilities; no inferred logits.')
    for field in ('transition_mode','mean_mode','variance_mode'):
        if node.get(field) not in ('shared','sequence'):
            raise GraphError('GaussianFilter '+field+' must explicitly select shared across T or one tensor per sequence step.')
    return dict(copy.deepcopy(node),op_version=node.get('op_version',1))


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
              'transition':([states,states],['from_state','to_state']) if n['transition_mode']=='shared' else
                           ([steps,states,states],['time','from_state','to_state']),
              'means':([states,dimensions],['state','coordinate']) if n['mean_mode']=='shared' else
                      ([steps,states,dimensions],['time','state','coordinate']),
              'variances':([states,dimensions],['state','coordinate']) if n['variance_mode']=='shared' else
                          ([steps,states,dimensions],['time','state','coordinate']),
              'lengths':([1],None),'emission_mask':([steps],['time']),
              'execution_mask':([steps],['time']),'reset_mask':([steps],['time'])}
    for role,(shape,axes) in bound.items():
        required,names=expected[role]
        semantic={'T':'time','D':'coordinate'}
        actual=[semantic.get(a,a) for a in axes] if axes is not None else None
        if shape!=required or actual is not None and names is not None and actual!=names:
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
            'work_elements':steps*(8*states*states+16*states*dimensions+(18 if n['op_version']==2 else 12)*states+4*dimensions+(24 if n['op_version']==2 else 8))+(8 if n['op_version']==2 else 4)*states,
            **({'op_version':2} if n['op_version']==2 else {}),
            'sources':sources(n),'inputs':copy.deepcopy(n['inputs']),
            'initial_representation':n['initial_representation'],
            'transition_representation':n['transition_representation'],'transition_mode':n['transition_mode'],
            'mean_mode':n['mean_mode'],'variance_mode':n['variance_mode']}


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


def _forward(parts, plan, kernel, controlled_kernel=None, result_class=None):
    """Parameter-free CPU float32 diagram projection; explicit axis conventions."""
    import torch
    bound={role:parts[plan['sources'].index(key)] for role,key in plan['inputs'].items()}
    if any(not isinstance(v,torch.Tensor) or v.device.type!='cpu' or v.dtype!=torch.float32 or
           v.layout!=torch.strided for v in bound.values()):
        raise ValueError('GaussianFilter graph roles are dense CPU float32 tensors; controls use explicit exact adapters.')
    observed=bound['observations'];initial=bound['initial']
    if observed.ndim!=3 or initial.ndim!=2:
        raise ValueError('GaussianFilter needs observations [B,T,D] and initial [B,K], with no implicit batch broadcasting.')
    batch,steps,dimensions=observed.shape;states=initial.shape[1]
    if states<1 or dimensions<1:
        raise ValueError('GaussianFilter needs positive K and D.')
    expected={'initial':(batch,states),'transition':(batch,states,states) if plan['transition_mode']=='shared' else (batch,steps,states,states),
        'means':(batch,states,dimensions) if plan['mean_mode']=='shared' else (batch,steps,states,dimensions),
        'variances':(batch,states,dimensions) if plan['variance_mode']=='shared' else (batch,steps,states,dimensions),
        'lengths':(batch,1),'emission_mask':(batch,steps),'execution_mask':(batch,steps),'reset_mask':(batch,steps)}
    for role,value in bound.items():
        if role!='observations' and tuple(value.shape)!=expected[role]:
            raise ValueError('GaussianFilter '+role+' must have exact shape '+str(expected[role])+', including B; no broadcasting.')
    # Same numerical-work admission as the shared filter, BEFORE logarithms,
    # control conversion, masks or expanded allocations in this adapter.
    work=max(batch,1)*(max(steps,1)*(8*states*states+16*states*dimensions+12*states+4*dimensions+8)+4*states)
    if plan.get('op_version')==2:work=max(batch,1)*(max(steps,1)*(8*states*states+16*states*dimensions+18*states+4*dimensions+24)+8*states)
    if steps>4096 or work>1_000_000:
        raise ValueError('GaussianFilter work budget exceeded before adapter allocations.')
    length_values=bound['lengths']
    if (tuple(length_values.shape)!=(batch,1) or not bool(torch.isfinite(length_values).all()) or
            not bool(((length_values==length_values.trunc()) & (length_values>=0) & (length_values<=steps)).all())):
        raise ValueError('GaussianFilter lengths are exact float32-encoded integers [B,1] in [0,T].')
    lengths=length_values[:,0].to(torch.int64)
    controls={}
    for role in ('emission_mask','execution_mask','reset_mask'):
        if role in bound:
            mask=bound[role]
            if not bool(((mask==0)|(mask==1)).all()):raise ValueError('GaussianFilter '+role+' requires exact 0/1 [B,T].')
            controls[role]=mask.to(torch.bool)
    transition=bound['transition']
    if plan['transition_mode']=='shared':transition=transition[:,None].expand(batch,steps,*transition.shape[1:])
    if plan['initial_representation']=='probabilities': initial=_log_probabilities(initial)
    if plan['transition_representation']=='probabilities':
        # Inactive padded probabilities may be NaN; sanitize before logs, just
        # as the numerical filter sanitizes inactive log transition entries.
        active=torch.arange(steps)[None,:]<lengths[:,None]
        if 'execution_mask' in controls:active=active&controls['execution_mask']
        states=initial.shape[-1]
        transition=torch.where(active[:,:,None,None],transition,transition.new_tensor(1/states))
        transition=_log_probabilities(transition)
    variance=bound['variances']
    if plan['variance_mode']=='shared':variance=variance[:,None].expand(batch,steps,*variance.shape[1:])
    means=bound['means']
    if plan['mean_mode']=='shared':means=means[:,None].expand(batch,steps,*means.shape[1:])
    arguments={'log_initial':initial,'log_transition':transition,'means':means,
               'positive_variances':variance,'lengths':lengths,**controls}
    result=(controlled_kernel(observed,**arguments,kernel=kernel,result_class=result_class)
            if plan.get('op_version')==2 else kernel(observed,**arguments))
    return {'filtered':result.filtered_log_probs,'final':result.final_log_probs,
            'step_evidence':result.step_log_evidence,
            'sequence_evidence':result.sequence_log_likelihood[:,None]}


def forward(parts,plan):
    from .gaussian_filter import filter_diagonal_gaussian_hmm,GaussianFilterResult
    from .sequence_controls import filter_run
    return _forward(parts,plan,filter_diagonal_gaussian_hmm,filter_run,GaussianFilterResult)


def runtime_class(nn):
    class GaussianFilter(nn.Module):
        def __init__(self,plan):
            super().__init__();self.plan=plan
        def forward(self,*parts):
            return forward(parts,self.plan)
    return GaussianFilter


def runtime_source():
    """Capture checked native helper meanings in an app-free private namespace."""
    import inspect
    from . import gaussian_filter as kernel
    from .task_execution import _helper_source
    from . import sequence_controls as controls
    functions=(kernel._cpu_tensor,kernel._finite,kernel._safe_logsumexp,
               kernel._normalized_logs,kernel.filter_diagonal_gaussian_hmm,
               controls._float_roles,controls._mask_role,controls._prefix_role,controls.filter_run,
               _log_probabilities,_forward,runtime_class)
    prelude=('from dataclasses import dataclass\nimport math\n'+
             f'MAX_WORK_ELEMENTS = {kernel.MAX_WORK_ELEMENTS!r}\n'+
             f'MAX_STEPS = {kernel.MAX_STEPS!r}\n'+f'_LOG_TWO_PI = {kernel._LOG_TWO_PI!r}\n')
    return (prelude+inspect.getsource(kernel.GaussianFilterResult)+'\n'+
            '\n\n'.join(_helper_source(f) for f in functions)+'\n'+
            'forward = lambda parts, plan: _forward(parts, plan, filter_diagonal_gaussian_hmm, filter_run, GaussianFilterResult)\n')
