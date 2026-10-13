"""Ephemeral predictive-mixture summaries; no fitting, storage or regime labels.

Parameters are supplied explicitly in the audited filter's expanded [B,T,...]
layout. This explains that computation; graph/task admission separately proves
its conditioning. No claim of smoothing, state recovery or calibration follows.
"""
import math


def inspect_model(model,inputs,filter_id):
    """Read a compiled/generated graph's actual conditional parameter tensors.

    Returns ephemeral JSON only. Does not change weights, graph, datasets or run
    evidence. The ordinary model's guarded intermediate_values supplies the same
    operations used by forward, including explicit runtime temporal contracts.
    """
    import torch
    from .gaussian_nodes import _log_probabilities
    contract=model.contract
    plan=contract.get('shapes',{}).get(filter_id,{}).get('gaussian')
    if plan is None:raise ValueError('Choose an existing local GaussianFilter block, not a caption or opaque body.')
    if not hasattr(model,'intermediate_values'):raise ValueError('Model must expose checked graph intermediate_values.')
    if isinstance(inputs,dict) and set(inputs)=={g['id'] for g in contract['input_groups']}:
        inputs={g['name']:inputs[g['id']] for g in contract['input_groups']}
    item=next(item for item in contract['plan'] if item['id']==filter_id)
    source_map=dict(zip(plan['sources'],item['sources']))
    with torch.no_grad():
        values=model.intermediate_values(inputs)
        def read(key):
            source=source_map[key];value=values[source['node']]
            return value[source['port']] if source['port'] is not None else value
        bound={role:read(key) for role,key in plan['inputs'].items()}
        observations=bound['observations'];b,t,d=observations.shape;k=bound['initial'].shape[-1]
        lengths=bound['lengths'].reshape(b).to(torch.int64)
        active=torch.arange(t)[None,:]<lengths[:,None]
        execute=active if 'execution_mask' not in bound else active&bound['execution_mask'].to(torch.bool)
        transition=bound['transition']
        if plan['transition_mode']=='shared':transition=transition[:,None].expand(b,t,k,k)
        if plan['transition_representation']=='probabilities':
            transition=_log_probabilities(torch.where(execute[:,:,None,None],transition,transition.new_tensor(1./k)))
        initial=bound['initial']
        if plan['initial_representation']=='probabilities':initial=_log_probabilities(initial)
        means=bound['means'];variance=bound['variances']
        if plan['mean_mode']=='shared':means=means[:,None].expand(b,t,k,d)
        if plan['variance_mode']=='shared':variance=variance[:,None].expand(b,t,k,d)
        controls={key:bound[key].to(torch.bool) for key in ('execution_mask','reset_mask','emission_mask') if key in bound}
        report=summarize(observations,log_initial=initial,log_transition=transition,means=means,
            positive_variances=variance,lengths=lengths,**controls)
    report['filter_id']=filter_id
    report['conditioning_proofs']=[value for value in contract.get('density_conditioning',{}).values()
        if filter_id in value['filters']]
    return report


def summarize(observations, *, log_initial,log_transition,means,positive_variances,lengths,
              execution_mask=None,reset_mask=None,emission_mask=None,max_diagnostic_elements=1_000_000):
    from .gaussian_filter import filter_diagonal_gaussian_hmm,GaussianFilterResult
    from .sequence_controls import filter_run
    import torch
    if not isinstance(observations,torch.Tensor) or observations.ndim!=3:
        raise ValueError('Predictive-mixture diagnostics need observations[B,T,D].')
    batch,steps,dimensions=observations.shape
    states=log_initial.shape[-1] if isinstance(log_initial,torch.Tensor) and log_initial.ndim==2 else 0
    output_elements=batch*steps*(4*states+3*dimensions+dimensions*dimensions+12)
    if type(max_diagnostic_elements) is not int or not 1<=max_diagnostic_elements<=1_000_000 or output_elements>max_diagnostic_elements:
        raise ValueError('Predictive-mixture diagnostic budget exceeded before filter evaluation/covariance allocation.')
    result=filter_run(observations,log_initial=log_initial,log_transition=log_transition,
        means=means,positive_variances=positive_variances,lengths=lengths,
        execution_mask=execution_mask,reset_mask=reset_mask,emission_mask=emission_mask,
        kernel=filter_diagonal_gaussian_hmm,result_class=GaussianFilterResult)
    rows=[]
    with torch.no_grad():
        for row in range(batch):
            prior=log_initial[row].double();records=[];scored=executed=0
            for slot in range(steps):
                active=slot<int(lengths[row]) and (execution_mask is None or bool(execution_mask[row,slot]))
                reset=active and reset_mask is not None and bool(reset_mask[row,slot])
                emission=active and (emission_mask is None or bool(emission_mask[row,slot]))
                if reset:prior=log_initial[row].double()
                prediction=mean=covariance=density=None
                if active:
                    executed+=1
                    predicted=torch.logsumexp(prior[:,None]+log_transition[row,slot].double(),0)
                    weights=predicted.exp();prediction=weights.tolist()
                    if emission:
                        scored+=1;mu=means[row,slot].double();variance=positive_variances[row,slot].double()
                        mixture_mean=(weights[:,None]*mu).sum(0)
                        residual=mu-mixture_mean
                        # Total covariance: within-component plus between-state.
                        total=torch.diag((weights[:,None]*variance).sum(0))
                        total=total+torch.einsum('k,ki,kj->ij',weights,residual,residual)
                        mean=mixture_mean.tolist();covariance=total.tolist()
                        terms=-.5*(math.log(2*math.pi)+variance.log()+
                            ((observations[row,slot,None].double()-mu)/variance.sqrt()).square()).sum(-1)
                        density=float(torch.logsumexp(predicted+terms,0))
                        if not torch.isfinite(mixture_mean).all() or not torch.isfinite(total).all() or not math.isfinite(density):
                            raise FloatingPointError('Predictive-mixture diagnostics exceed input dtype range.')
                    prior=result.filtered_log_probs[row,slot].double()
                records.append({'time_index':slot,'executed':active,'reset_applied':reset,'scored':emission,
                    'predictive_state_probabilities':prediction,
                    'filtered_state_probabilities':result.filtered_log_probs[row,slot].exp().tolist(),
                    'predictive_mean':mean,'predictive_covariance':covariance,
                    'mixture_log_density':density,'filter_step_log_evidence':float(result.step_log_evidence[row,slot])})
            rows.append({'batch_index':row,'prefix_length':int(lengths[row]),'executed_steps':executed,
                'scored_steps':scored,'sequence_log_evidence':float(result.sequence_log_likelihood[row]),
                'final_state_probabilities':result.final_log_probs[row].exp().tolist(),'steps':records})
    return {'version':1,'kind':'visual-pytorch-predictive-mixture-diagnostics','unit':'trajectory',
        'dimensions':dimensions,'trajectories':rows,
        'scope':'conditional mixture density before each scored observation, not mixture-mean error or smoothed states',
        'assumptions':'supplied conditional parameters; graph/task conditioning admission remains separate',
        'limitations':'no regime labels, calibration, identifiability, missing-history integration or hidden state carry'}
