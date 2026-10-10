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
    return {'input_order':graph.get('input_order',legacy_input_order(graph)),
            'nodes':sorted(({k:v for k,v in n.items() if k not in PRESENTATION | {'label'}}
                            for n in graph['nodes']),key=lambda n:n['id']),
            # Preserve declared computational ordering, including Concat/router ports.
            'edges':graph['edges']}


def binding_for(contract):
    def groups(key):
        return [{k:g[k] for k in ('id','shape','start','stop')} for g in contract[key]]
    return {'version':1,'inputs':groups('input_groups'),'outputs':groups('output_groups'),
            'output_mode':contract['output_mode']}


def check_binding(doc, contract):
    from .core import GraphError
    binding=doc.get('dataset_binding')
    if binding is not None and binding!=binding_for(contract):
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
    for doc in result.values():pin_input_order(doc['graph'])
    root=result[mid]
    if root.get('dataset') is not None and root.get('dataset_binding') is None:
        root['dataset_binding']=binding_for(validate_model(result,mid))
    return result
