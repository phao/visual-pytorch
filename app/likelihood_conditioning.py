"""Pure temporal-conditioning proof for Gaussian evidence, not a new operator.

Shapes/normalized mixture weights do not prove a density in the observations
when its parameters see those same current/future observations. Track supported
pointwise/causal paths; unknown temporal mixing remains a located NLL refusal.
Different root inputs are explicit external conditioning, not inferred independence.
"""
import math

_RANK={'past':0,'current':1,'mixed':2}


def _value(deps=None,axis=None,filters=None,adaptive=False):
    return {'deps':dict(deps or {}),'axis':axis,'filters':set(filters or ()), 'adaptive':adaptive}


def _join(values):
    deps={};filters=set();axes=set()
    for value in values:
        filters.update(value['filters'])
        if value['deps']:axes.add(value['axis'])
        for root,status in value['deps'].items():
            if root not in deps or _RANK[status]>_RANK[deps[root]]:deps[root]=status
    return _value(deps,next(iter(axes)) if len(axes)==1 else None,filters,any(v['adaptive'] for v in values))


def _mixed(value):
    return _value({root:'mixed' for root in value['deps']},None,value['filters'],value['adaptive'])


def _analyse(models,mid,contract,cache,seed=None):
    nodes={n['id']:n for n in models[mid]['graph']['nodes']};values={};proofs={}
    incoming={nid:[] for nid in nodes}
    for edge in models[mid]['graph']['edges']:
        incoming[edge['to']].append((edge['from'],edge.get('from_port')))
    plan={item['id']:item for item in contract['plan']}
    for nid in contract['order']:
        node=nodes[nid];kind=node['type'];shape=contract['shapes'][nid]['output_shape']
        parts=[values[key] for key in incoming[nid]];value=_join(parts)
        bindings={}
        if kind in ('CausalLagWindow','GaussianFilter'):
            keys={node+('@'+port if port is not None else ''):(node,port) for node,port in incoming[nid]}
            bindings={role:keys[key] for role,key in node['inputs'].items()}
        before=contract['shapes'][nid].get('input_shape',[])
        if kind=='Input':
            value=_value(seed[nid]['deps'],seed[nid]['axis'],seed[nid]['filters'],seed[nid]['adaptive']) if seed is not None else _value({nid:'current'},0 if len(shape)>1 else None)
        elif kind in ('Constant','TensorParameter'):value=_value(adaptive=kind=='TensorParameter')
        elif kind=='CausalLagWindow':
            observed=values[bindings['observations']]
            deps={root:'past' if status!='mixed' and observed['axis']==0 else 'mixed'
                  for root,status in observed['deps'].items()}
            # Histories and controls are external context. They cannot repair a
            # history that already depends on the scored current/future value.
            for role,key in bindings.items():
                if role=='observations':continue
                for root,status in values[key]['deps'].items():
                    deps[root]='mixed' if root in observed['deps'] else status
            value=_value(deps,0,adaptive=value['adaptive'])
        elif kind=='Linear':
            if value['axis'] is not None and value['axis']==len(before)-1:value=_mixed(value)
            value['adaptive']=True
        elif kind=='Activation':
            if node['activation'] in ('Softmax','LogSoftmax') and before:
                if node.get('dim',-1)%len(before)==value['axis']:value=_mixed(value)
        elif kind=='PositiveVariance' or kind in ('Scale','Output','Add'):pass
        elif kind=='Reshape':
            if not (value['axis']==0 and before and shape and before[0]==shape[0]
                    and math.prod(before[1:])==math.prod(shape[1:])) and value['deps']:
                value=_mixed(value)
        elif kind=='Permute':
            if value['axis'] is not None:value['axis']=node['dims'].index(value['axis'])
        elif kind=='Flatten':
            a=node.get('start_dim',0);b=node.get('end_dim',-1)
            a%=len(before);b%=len(before)
            if value['axis'] is not None:
                if a<=value['axis']<=b and a!=b:value=_mixed(value)
                elif value['axis']>b:value['axis']-=b-a
        elif kind=='Concat':
            if node.get('flatten',True) or value['axis']==len(before)-1:value=_mixed(value)
        elif kind=='Splitter':
            if value['axis']==len(before)-1:value=_mixed(value)
            for part in node['parts']:values[(nid,part['output'])]=_value(value['deps'],value['axis'],value['filters'],value['adaptive'])
        elif kind=='Replicator':
            for port in node['outputs']:values[(nid,port)]=_value(value['deps'],value['axis'],value['filters'],value['adaptive'])
        elif kind=='Module':
            child=plan[nid].get('runtime_contract') or cache.get(node['ref'])
            if child and len(child['input_groups'])<=1 and len(child['output_groups'])==1:
                seeds={g['id']:value for g in child['input_groups']}
                child_values,_=_analyse(models,node['ref'],child,cache,seeds)
                value=child_values[(child['output_groups'][0]['id'],None)]
            else:value=_mixed(value)
        elif kind=='GaussianFilter':
            observed=values[bindings['observations']];roots=set(observed['deps'])
            findings=[];context=set()
            if observed['adaptive']:
                findings.append({'code':'ADAPTIVE_OBSERVATION_LIKELIHOOD','path':['architectures',mid,'nodes',nid,'inputs','observations'],
                    'message':'Scored observations pass through model parameters; no change-of-variables/Jacobian objective is declared.',
                    'repair':'Score an explicit external observation input; adaptive observation transforms need a separately specified density objective.'})
            for role in ('initial','transition','means','variances'):
                parameter=values[bindings[role]];context.update(set(parameter['deps'])-roots)
                for root in sorted(roots & parameter['deps'].keys()):
                    if role=='initial' or parameter['deps'][root]!='past' or parameter['axis']!=0:
                        findings.append({'code':'NONCAUSAL_LIKELIHOOD','path':['architectures',mid,'nodes',nid,'inputs',role],
                            'message':role+' depends on current/future or unproved temporal mixing of observations '+root+'.',
                            'repair':'Use explicit causal lag features or an independent conditioning input; do not infer a density from a caption.'})
            # Context used by lag construction includes histories and controls.
            for role in ('transition','means','variances'):
                context.update(set(values[bindings[role]]['deps'])-roots)
            proofs[nid]={'version':1,'status':'unproved' if findings else 'conditional',
                'observation_inputs':sorted(roots),'conditioning_inputs':sorted(context),
                'findings':findings,'scope':'supported causal paths; external inputs are explicitly supplied conditioning'}
            joined=_join([values[key] for key in bindings.values()])
            for port in ('filtered','final','step_evidence','sequence_evidence'):
                values[(nid,port)]=_value(joined['deps'],0 if port in ('filtered','step_evidence') else None,{nid},joined['adaptive'])
            value=joined
        else:
            value=_mixed(value)
            if kind in ('Conv2d','LSTM','Recurrent'):value['adaptive']=True
        for port in contract['shapes'][nid].get('ports',{}):
            if (nid,port) not in values:values[(nid,port)]=_mixed(value)
        values[(nid,None)]=value
    return values,proofs


def attach(models,mid,contract,cache):
    values,proofs=_analyse(models,mid,contract,cache)
    result={}
    for group in contract['output_groups']:
        if group.get('representation')!='log-evidence':continue
        filters=values[(group['id'],None)]['filters'];entries=[proofs[key] for key in sorted(filters) if key in proofs]
        findings=[issue for entry in entries for issue in entry['findings']]
        result[group['id']]={'version':1,'status':'conditional' if entries and not findings else 'unproved',
            'observation_inputs':sorted({root for entry in entries for root in entry['observation_inputs']}),
            'conditioning_inputs':sorted({root for entry in entries for root in entry['conditioning_inputs']}),
            'findings':findings,'filters':sorted(filters)}
    return result


def check_task(task,contract,head):
    """Reject noncausal parameters and observable field aliases before a worker."""
    from .core import GraphError
    proof=contract.get('density_conditioning',{}).get(head['port'])
    if proof is None or proof['status']!='conditional':
        details=proof.get('findings',[]) if proof else []
        reason=details[0]['message'] if details else 'No supported temporal-conditioning proof for this output.'
        raise GraphError('Likelihood conditioning refused for '+head['port']+': '+reason)
    bindings={binding['port']:binding for binding in task['inputs']}
    observation_fields={bindings[key]['field'] for key in proof['observation_inputs'] if key in bindings}
    for key in proof['conditioning_inputs']:
        binding=bindings.get(key)
        if binding and 'derived' not in binding and binding['field'] in observation_fields:
            raise GraphError('Likelihood conditioning input '+key+' aliases the scored observation field; supply independent explicit context.')
