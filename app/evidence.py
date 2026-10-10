"""Allocation-free identities and bindings; no Qt or Torch imports."""
import copy
import hashlib
import json

PRESENTATION = {'x', 'y', 'width', 'height', 'collapsed'}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                      allow_nan=False).encode('utf-8')).hexdigest()


def legacy_input_order(graph):
    roots=[n['id'] for n in graph['nodes'] if n['type']=='Input']
    join=next((n for n in graph['nodes'] if n['type']=='Concat' and
               set(n.get('sources',[]))==set(roots)),None)
    return list(join['sources']) if join else roots


def pin_input_order(graph, previous=None):
    """Pin legacy meaning once; append new IDs, never silently rebind data."""
    roots={n['id'] for n in graph['nodes'] if n['type']=='Input'}
    chosen=graph.get('input_order')
    if chosen is None:
        chosen=previous.get('input_order',legacy_input_order(previous)) if previous else legacy_input_order(graph)
    graph['input_order']=[nid for nid in chosen if nid in roots]
    graph['input_order'].extend(nid for nid in legacy_input_order(graph) if nid not in graph['input_order'])


def graph_signature(graph):
    from .operators import normalized
    return {'input_order':graph.get('input_order',legacy_input_order(graph)),
            'nodes':sorted(({k:v for k,v in normalized(n).items() if k not in PRESENTATION | {'label'}}
                            for n in graph['nodes']),key=lambda n:n['id']),
            # Preserve declared computational ordering, including Concat/router ports.
            'edges':graph['edges']}


def binding_for(contract):
    def groups(key):
        return [{k:g[k] for k in ('id','shape','start','stop','axes') if k in g} for g in contract[key]]
    return {'version':1,'inputs':groups('input_groups'),'outputs':groups('output_groups'),
            'output_mode':contract['output_mode']}


def check_binding(doc, contract):
    from .core import GraphError
    binding=doc.get('dataset_binding')
    try:
        matches=binding is None or digest(binding)==digest(binding_for(contract))
    except (TypeError,ValueError,RecursionError):
        matches=False
    if not matches:
        raise GraphError('Dataset binding is incompatible with the declared input/output port IDs or shapes. '
                         'Keep the original ports, replace the dataset, or explicitly rebind its columns.')


def identity(models, mid):
    """Identity of scientific meaning, not display geometry or human labels."""
    closure={}
    def visit(nid):
        if nid in closure:return
        doc=models[nid]
        closure[nid]=graph_signature(doc['graph'])
        for n in doc['graph']['nodes']:
            if n['type']=='Module':visit(n['ref'])
    visit(mid)
    doc=models[mid]
    return {'version':1,'architecture':digest(closure),
            'dataset':digest(doc.get('dataset')),
            'binding':digest(doc.get('dataset_binding')),
            'training':digest(doc.get('training',{}))}


def normalize_models(models, mid):
    """Private snapshot for direct Python callers; original dictionaries stay intact."""
    result=copy.deepcopy(models)
    from .core import validate_model
    from .operators import normalized
    for doc in result.values():
        pin_input_order(doc['graph'])
        doc['graph']['nodes'] = [normalized(n) for n in doc['graph']['nodes']]
    root=result[mid]
    if root.get('dataset') is not None and root.get('dataset_binding') is None:
        root['dataset_binding']=binding_for(validate_model(result,mid))
    return result


def parameter_schema(models, mid, prefix='', stable_prefix=None):
    """Pure expected state-dict shapes/IDs, without allocating a Torch model."""
    from .core import operation_nodes, validate_model
    contract=validate_model(models,mid)
    stable_prefix=mid if stable_prefix is None else stable_prefix
    schema={}
    for i,node in enumerate(operation_nodes(models[mid])):
        path=prefix+f'layer_{i}'
        stable=stable_prefix+'/'+node['id']
        if node['type']=='Module':
            schema.update(parameter_schema(models,node['ref'],path+'.',stable))
        elif node['type'] in {'Linear', 'Conv2d'}:
            shape=contract['shapes'][node['id']]
            weight_shape = shape['operator']['weight_shape'] if node['type'] == 'Conv2d' else [shape['output'],shape['input']]
            width = weight_shape[0]
            schema[path+'.weight']={'shape':weight_shape, 'id':stable+'/weight'}
            if node.get('bias',True):
                schema[path+'.bias']={'shape':[width], 'id':stable+'/bias'}
    return schema


def check_checkpoint(models, mid, result):
    """Validate a v1 float32 inference checkpoint against a normalized snapshot.

    This proves compatibility/representation, not that a caller's manually
    constructed checkpoint originated in a trusted training process.
    """
    import math
    from .core import GraphError, validate_model
    try:
        matches=isinstance(result,dict) and digest(result.get('identity'))==digest(identity(models,mid))
    except (TypeError,ValueError,RecursionError):
        matches=False
    if not matches:
        raise GraphError('Checkpoint identity does not match this architecture, data, binding and training snapshot.')
    if type(result.get('checkpoint_version')) is not int or result['checkpoint_version']!=1:
        raise GraphError('Checkpoint version must be integer 1.')
    contract=validate_model(models,mid)
    check_binding(models[mid],contract)
    if isinstance(models[mid].get('dataset'), dict):
        from .tensor_data import check_adapter
        check_adapter(models[mid]['dataset'], contract)
    preprocessing={'kind':'identity','input_adapter':'declared-row-major',
                   'binding':models[mid].get('dataset_binding')}
    try:
        correct_preprocessing=digest(result.get('preprocessing'))==digest(preprocessing)
    except (TypeError,ValueError,RecursionError):
        correct_preprocessing=False
    if not correct_preprocessing:
        raise GraphError('Checkpoint preprocessing does not match the supported identity adapter and dataset binding.')
    schema=parameter_schema(models,mid)
    if result.get('parameter_ids')!={key:spec['id'] for key,spec in schema.items()}:
        raise GraphError('Checkpoint stable parameter IDs do not match this architecture.')
    weights=result.get('weights')
    if not isinstance(weights,dict) or set(weights)!=set(schema):
        raise GraphError('Checkpoint weights must contain exactly the declared parameter keys.')

    def tensor(value, shape, path):
        if shape:
            if not isinstance(value,list) or len(value)!=shape[0]:
                raise GraphError('Checkpoint tensor shape is incorrect: '+path)
            for item in value:tensor(item,shape[1:],path)
        elif (type(value) not in (int,float) or abs(value)>3.4028234663852886e38
              or not math.isfinite(value)):
            raise GraphError('Checkpoint weights must be finite float32-representable numbers: '+path)
    for path,spec in schema.items():tensor(weights[path],spec['shape'],path)
    return schema


def dataset_report(doc, contract):
    """Read-only admission status; no guessing/repacking of existing columns."""
    from .core import GraphError, validate_dataset
    expected=binding_for(contract)
    report={'allowed':False,'binding':doc.get('dataset_binding'),
            'declared_binding':expected,'can_rebind':False}
    if doc.get('dataset') is None:
        return {**report,'status':'none','reason':'Generate synthetic data or import x/y data first.'}
    try:
        validate_dataset(doc['dataset'],contract['inputs'],contract['outputs'])
        from .tensor_data import check_adapter
        check_adapter(doc['dataset'], contract)
    except GraphError as error:
        return {**report,'status':'invalid','reason':str(error)}
    if doc.get('dataset_binding') is None:
        return {**report,'status':'unbound','can_rebind':True,
                'reason':'Dataset column meaning is not recorded. Explicitly bind columns before training.'}
    try:
        check_binding(doc,contract)
    except GraphError as error:
        return {**report,'status':'incompatible','can_rebind':True,'reason':str(error)}
    return {**report,'allowed':True,'status':'compatible',
            'reason':'Dataset columns match the declared stable port IDs, shapes and packing order.'}


def checkpoint_report(models, mid):
    """Distinguish compatible current evidence from legacy/stale/invalid results."""
    from .core import GraphError
    result=models[mid].get('result')
    if result is None:return {'status':'none','reason':'No current trained checkpoint.'}
    if not isinstance(result,dict):return {'status':'invalid','reason':'Result must be an object.'}
    if not result.get('identity'):
        return {'status':'unverified','reason':'Legacy result has no snapshot identity; retrain to obtain a verified current checkpoint.'}
    try:
        normalized=normalize_models(models,mid)
        check_checkpoint(normalized,mid,result)
    except GraphError as error:
        return {'status':'stale' if str(error).startswith('Checkpoint identity does not match') else 'invalid','reason':str(error)}
    return {'status':'current','reason':'Checkpoint matches this architecture, dataset, binding and training snapshot.'}
