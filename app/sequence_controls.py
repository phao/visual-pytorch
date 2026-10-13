"""Explicit version-2 sequence controls, without numerical imports on load.

Execution False skips transition, density and observed-history updates. Resets
on skipped/padded slots are ignored. An executed reset is before transition and
before constructing causal features. Density masks never imply execution masks.
"""


def _float_roles(torch, roles):
    dtype=None
    for name,value in roles:
        if not isinstance(value,torch.Tensor) or value.device.type!='cpu' or value.layout!=torch.strided:
            raise ValueError(name+' requires a dense CPU tensor.')
        if value.dtype not in (torch.float32,torch.float64) or dtype is not None and value.dtype!=dtype:
            raise ValueError('Real sequence roles must have the same float32/float64 dtype.')
        dtype=value.dtype


def _mask_role(torch,value,name,shape):
    if value is not None and (not isinstance(value,torch.Tensor) or value.device.type!='cpu' or
        value.layout!=torch.strided or value.dtype!=torch.bool or tuple(value.shape)!=shape):
        raise ValueError(name+' must be an exact bool CPU tensor '+str(shape)+'.')


def _prefix_role(torch,lengths,batch,steps):
    if (not isinstance(lengths,torch.Tensor) or lengths.device.type!='cpu' or lengths.layout!=torch.strided or
        lengths.dtype not in (torch.int32,torch.int64) or tuple(lengths.shape)!=(batch,)):
        raise ValueError('Lengths require exact int32/int64 CPU [B] prefixes.')


def filter_run(observations, *, log_initial, log_transition, means, positive_variances, lengths,
               execution_mask=None, reset_mask=None, emission_mask=None,
               kernel, result_class, max_work_elements=1_000_000):
    """Partition executed observations at resets; reuse the exact audited filter.

    Selecting only executed time slots implements identity carry without ever
    evaluating skipped NaNs. Each reset segment starts with the supplied initial
    law before its first transition. Evidence sums over all independent segments.
    Result beliefs are scattered back with identity carry on unexecuted slots.
    """
    import torch
    _float_roles(torch,(('observations',observations),('initial',log_initial),
        ('transition',log_transition),('means',means),('variances',positive_variances)))
    if observations.ndim!=3 or log_initial.ndim!=2:raise ValueError('Observations/initial need [B,T,D]/[B,K].')
    b,t,d=observations.shape;k=log_initial.shape[-1]
    if d<1 or k<1 or tuple(log_initial.shape)!=(b,k):raise ValueError('K,D positive and matching batches required.')
    for value,shape in ((log_transition,(b,t,k,k)),(means,(b,t,k,d)),(positive_variances,(b,t,k,d))):
        if tuple(value.shape)!=shape:raise ValueError('Filter roles must match exactly; no broadcasting.')
    _prefix_role(torch,lengths,b,t)
    for name,value in (('execution_mask',execution_mask),('reset_mask',reset_mask),('emission_mask',emission_mask)):
        _mask_role(torch,value,name,(b,t))
    work=max(b,1)*(max(t,1)*(8*k*k+16*k*d+18*k+4*d+24)+8*k)
    if (type(max_work_elements) is not int or not 1<=max_work_elements<=1_000_000 or
        t>4096 or work>max_work_elements):raise ValueError('Controlled filter work budget exceeded before allocation.')
    if not bool(((lengths>=0)&(lengths<=t)).all()):raise ValueError('Lengths must lie in [0,T].')
    if b==0 or execution_mask is None and reset_mask is None:
        return kernel(observations,log_initial=log_initial,log_transition=log_transition,
                      means=means,positive_variances=positive_variances,lengths=lengths,emission_mask=emission_mask)
    # Validate every initial law, including an entirely skipped/empty trajectory.
    empty=kernel(observations[:,:0],log_initial=log_initial,log_transition=log_transition[:,:0],
                 means=means[:,:0],positive_variances=positive_variances[:,:0],lengths=torch.zeros_like(lengths))
    if b==0 or t==0:return empty
    active=torch.arange(t)[None,:]<lengths[:,None]
    execute=active if execution_mask is None else active&execution_mask
    reset=torch.zeros_like(execute) if reset_mask is None else execute&reset_mask
    emitted=torch.ones_like(execute) if emission_mask is None else emission_mask
    filtered_rows=[];evidence_rows=[];final_rows=[]
    for row in range(b):
        slots=execute[row].nonzero().flatten().tolist();segments=[]
        for slot in slots:
            if not segments or bool(reset[row,slot]):segments.append([])
            segments[-1].append(slot)
        posterior_by_time={};evidence_by_time={}
        for segment in segments:
            index=torch.tensor(segment,dtype=torch.int64)
            result=kernel(observations[row:row+1,index],log_initial=log_initial[row:row+1],
                log_transition=log_transition[row:row+1,index],means=means[row:row+1,index],
                positive_variances=positive_variances[row:row+1,index],
                lengths=lengths.new_tensor([len(segment)]),emission_mask=emitted[row:row+1,index])
            for position,slot in enumerate(segment):
                posterior_by_time[slot]=result.filtered_log_probs[0,position]
                evidence_by_time[slot]=result.step_log_evidence[0,position]
        belief=log_initial[row];neutral=empty.sequence_log_likelihood[row]
        row_beliefs=[];row_evidence=[]
        for slot in range(t):
            if slot in posterior_by_time:belief=posterior_by_time[slot]
            row_beliefs.append(belief);row_evidence.append(evidence_by_time.get(slot,neutral))
        filtered_rows.append(torch.stack(row_beliefs));evidence_rows.append(torch.stack(row_evidence));final_rows.append(belief)
    filtered=torch.stack(filtered_rows);step=torch.stack(evidence_rows);final=torch.stack(final_rows)
    sequence=step.sum(1)
    if not bool(torch.isfinite(sequence).all()):raise FloatingPointError('Controlled sequence evidence exceeds dtype range.')
    return result_class(filtered,final,step,sequence)


def lag_run(observations,initial_history,lengths, *, lags, execution_mask=None, reset_mask=None,
            reset_history=None, max_elements=1_000_000, admitted_elements=None):
    """Causal histories follow executed observations, with explicit reset history."""
    import torch
    _float_roles(torch,(('observations',observations),('initial_history',initial_history)))
    if type(lags) is not int or lags<1 or observations.ndim!=3:
        raise ValueError('Causal lags require positive exact p and observations[B,T,D].')
    b,t,d=observations.shape;p=lags
    if tuple(initial_history.shape)!=(b,p,d) or d<1:raise ValueError('Initial history must be exactly [B,p,D].')
    _prefix_role(torch,lengths,b,t)
    _mask_role(torch,execution_mask,'execution_mask',(b,t));_mask_role(torch,reset_mask,'reset_mask',(b,t))
    if (reset_mask is None)!=(reset_history is None):raise ValueError('Reset mask and explicit reset histories must be supplied together.')
    if reset_history is not None:
        _float_roles(torch,(('observations',observations),('reset_history',reset_history)))
        if tuple(reset_history.shape)!=(b,t,p,d):raise ValueError('Reset histories require exact [B,T,p,D], not inferred values.')
    limit=max_elements if admitted_elements is None else admitted_elements
    if (type(max_elements) is not int or not 0<=max_elements<=1_000_000 or
        type(limit) is not int or limit<0):raise ValueError('Controlled lag work admission must be explicit exact integers.')
    work=b+t+10*b*t+8*b*(t+p)*d+8*b*t*p*d
    if t>4096 or max(b,t,d,p,t+p,p*d)>2**53-1 or work>limit:
        raise ValueError('Controlled causal lag budget exceeded before allocation.')
    if not bool(((lengths>=0)&(lengths<=t)).all()):raise ValueError('Lengths must lie in [0,T].')
    neutral=observations[:,:0].sum((1,2))+initial_history[:,:0].sum((1,2))
    if reset_history is not None:neutral=neutral+reset_history[:,:0].sum((1,2,3))
    if b==0 or t==0:
        empty=torch.cat((initial_history[:,:0],observations),1)
        return empty.unsqueeze(2).expand(b,t,p,d).reshape(b,t,p*d)+neutral[:,None,None]
    active=torch.arange(t)[None,:]<lengths[:,None]
    execute=active if execution_mask is None else active&execution_mask
    reset=torch.zeros_like(execute) if reset_mask is None else execute&reset_mask
    safe_observations=torch.where(execute[:,:,None],observations,torch.zeros_like(observations))
    if not bool(torch.isfinite(safe_observations).all()):
        raise ValueError('Executed observations must be finite: missing autoregressive history cannot be silently imputed.')
    first=execute.to(torch.int64).argmax(1);has=execute.any(1)
    first_reset=reset[torch.arange(b),first]
    uses_initial=has&~first_reset
    history=torch.where(uses_initial[:,None,None],initial_history,torch.zeros_like(initial_history))
    if not bool(torch.isfinite(history).all()):raise ValueError('Used initial observed histories must be finite.')
    if reset_history is not None:
        safe_resets=torch.where(reset[:,:,None,None],reset_history,torch.zeros_like(reset_history))
        if not bool(torch.isfinite(safe_resets).all()):raise ValueError('Executed reset histories must be finite.')
    outputs=[]
    for slot in range(t):
        if reset_history is not None:history=torch.where(reset[:,slot,None,None],safe_resets[:,slot],history)
        outputs.append(torch.where(execute[:,slot,None],history.reshape(b,p*d),history.new_zeros((b,p*d))))
        updated=torch.cat((history[:,1:],safe_observations[:,slot,None]),1)
        history=torch.where(execute[:,slot,None,None],updated,history)
    return torch.stack(outputs,1)+neutral[:,None,None]
