"""Pure inspectable HMM component authoring; no data, fitting or hidden labels."""
import copy
import uuid
from .core import GraphError,check_document,validate_model
from .evidence import identity


def component(prefix, *, steps=16, states=2, dimensions=1):
    """A concrete p=0 filter graph, not the requested neural p=3 MS-VAR."""
    if any(type(v) is not int or not 1<=v<=64 for v in (steps,states,dimensions)):
        raise GraphError('Prototype component needs exact T,K,D integers1..64; ordinary graph shape editing remains larger.')
    if not isinstance(prefix,str) or not prefix or len(prefix)>30:
        raise GraphError('Choose a bounded component ID prefix.')
    nodes=[{'id':'observations','type':'Input','label':'Observations','shape':[steps,dimensions],'axes':['T','D']},
        {'id':'lengths','type':'Input','label':'Prefix lengths','shape':[1]},
        {'id':'initial_logits','type':'TensorParameter','shape':[states],'axes':['state']},
        {'id':'initial','type':'Activation','activation':'LogSoftmax','dim':0},
        {'id':'transition_logits','type':'TensorParameter','shape':[states,states],'axes':['from_state','to_state']},
        {'id':'transition','type':'Activation','activation':'LogSoftmax','dim':1},
        {'id':'means','type':'TensorParameter','shape':[states,dimensions],'axes':['state','coordinate']},
        {'id':'raw_variances','type':'TensorParameter','shape':[states,dimensions],'axes':['state','coordinate']},
        {'id':'variances','type':'PositiveVariance','positive_floor':.01},
        {'id':'filter','type':'GaussianFilter','inputs':{r:r for r in ('observations','initial','transition','means','variances','lengths')},
         'initial_representation':'log-probabilities','transition_representation':'log-probabilities',
         'transition_mode':'shared','mean_mode':'shared','variance_mode':'shared'}]
    edges=[{'from':'initial_logits','to':'initial'},{'from':'transition_logits','to':'transition'},
           {'from':'raw_variances','to':'variances'}]
    edges.extend({'from':r,'to':'filter'} for r in ('observations','initial','transition','means','variances','lengths'))
    for port in ('filtered','final','step_evidence','sequence_evidence'):
        nodes.append({'id':'out_'+port,'type':'Output','label':port.replace('_',' ')})
        edges.append({'from':'filter','from_port':port,'to':'out_'+port})
    ids={n['id']:prefix+'-'+n['id'] for n in nodes}
    for i,n in enumerate(nodes):
        n['id']=ids[n['id']];n.update(x=40+(i%4)*210,y=40+(i//4)*125)
        if n.get('inputs'):n['inputs']={r:ids[s] for r,s in n['inputs'].items()}
        if 'label' in n:n['label']=prefix+' '+n['label']
    edges=[{**e,'from':ids[e['from']],'to':ids[e['to']]} for e in edges]
    selector='["'+ids['means']+'","value"]'
    values=[[(k-(states-1)/2) for _ in range(dimensions)] for k in range(states)]
    return {'nodes':nodes,'edges':edges,'policy':{selector:{'initializer':{'kind':'values','dtype':'float32','values':values}}},
            'filter':ids['filter'],'scope':'exact diagonal-Gaussian HMM p=0; not neural MS-VAR; no fit or dataset'}


def add(models, mid, expected):
    if mid not in models or expected!=identity(models,mid)['architecture']:
        raise GraphError('Architecture changed; reopen Gaussian component authoring.')
    block=component('g'+uuid.uuid4().hex[:8]);doc=copy.deepcopy(models[mid])
    if len(doc['graph']['nodes'])+len(block['nodes'])>64:
        raise GraphError('Leave room for the14-block inspectable Gaussian component.')
    offset=max((n.get('y',0)+150 for n in doc['graph']['nodes']),default=0)
    for n in block['nodes']:n['y']+=offset
    doc['graph']['nodes'].extend(block['nodes']);doc['graph']['edges'].extend(block['edges'])
    policy=doc['graph'].setdefault('parameter_policy',{'version':1,'parameters':{}})
    policy['parameters'].update(block['policy'])
    doc['graph'].pop('input_order',None)  # Workspace.put preserves existing pin and appends new root IDs.
    doc['dataset']=None;doc['result']=None
    check_document(doc);validate_model({**models,mid:doc},mid)
    return doc,block['filter']


def edit(models,mid,nid,expected,fields):
    from .gaussian_nodes import FIELDS,normalized
    if mid not in models or expected!=identity(models,mid)['architecture']:
        raise GraphError('Architecture changed; reopen filter inspection.')
    if not isinstance(fields,dict) or set(fields)!=FIELDS:
        raise GraphError('Filter editing needs exactly its six configuration fields.')
    doc=copy.deepcopy(models[mid]);node=next((n for n in doc['graph']['nodes'] if n['id']==nid),None)
    if node is None or node['type']!='GaussianFilter':raise GraphError('Select an existing Gaussian filter.')
    node.update(copy.deepcopy(fields));normalized(node)
    check_document(doc);validate_model({**models,mid:doc},mid)
    doc['result']=None
    return doc
