"""Bounded exact task suggestions, not a partial DatasetSpec/model contract.

Registry decoding/validation still has complete host costs. Response descriptors
omit observations, vocabulary values, weights and optional field metadata.
Alphabetic browsing order never changes scientific binding/packing order.
"""
import copy

from .core import validate_model
from .authoring import workspace_id
from .datasets import _identifier, _require
from .experiment_objects import reference
from . import science_browser

COLLECTIONS=('fields','input_ports','output_ports')
SHAPE_LIMIT=4
NAME_LIMIT=128


def _name(value):
    name=value.get('name')
    return name if isinstance(name,str) and name else value['id']


def _row(value,index,collection):
    shape=value['shape'];axes=value.get('axes');name=_name(value)
    return {'id':value['id'],'name':name[:NAME_LIMIT],'name_complete':len(name)<=NAME_LIMIT,
        'name_source':'name' if isinstance(value.get('name'),str) and value['name'] else 'id',
        'ordinal':index,'dtype':value.get('dtype','float32'),
        'shape':shape[:SHAPE_LIMIT],'shape_rank':len(shape),'shape_complete':len(shape)<=SHAPE_LIMIT,
        'axes':axes[:SHAPE_LIMIT] if axes is not None else None,
        'axes_complete':axes is None or len(axes)<=SHAPE_LIMIT,
        'vocabulary_count':len(value['vocabulary']['values']) if collection=='fields' and value['dtype']=='category' else None}


def page(workspace,model,dataset,collection,*,query='',offset=0,limit=25,pin=None):
    reference(model,'model');reference(dataset,'dataset');science_browser._page(offset,limit)
    _require(isinstance(collection,str) and collection in COLLECTIONS,'Choose fields, input_ports or output_ports.')
    _require(isinstance(query,str) and len(query)<=200 and '\0' not in query,'Context search is limited to 200 characters without NUL.')
    _require(pin is None or _identifier(pin),'Context pin must be an exact stable ID.')
    observations=workspace.get_spec('dataset',dataset['id'],dataset['revision'])
    if collection=='fields':values=observations['fields']
    else:
        architecture=workspace.get_spec('model',model['id'],model['revision'])
        contract=validate_model(architecture['models'],architecture['root'])
        values=contract['input_groups' if collection=='input_ports' else 'output_groups']
    # Validate model existence even for fields without compiling every port just
    # to suggest a source field. Exact-reference lookup cannot substitute a head.
    if collection=='fields':workspace.get_spec('model',model['id'],model['revision'])
    rows=[(index,value) for index,value in enumerate(values)
        if query.casefold() in _name(value).casefold() or query.casefold() in value['id'].casefold()]
    rows.sort(key=lambda entry:(_name(entry[1]).casefold(),entry[1]['id']))
    pinned=next((_row(value,index,collection) for index,value in enumerate(values) if value['id']==pin),None) if pin else None
    result={'version':1,'scope':'task-context-page','workspace_id':workspace_id(workspace),
        'model':copy.deepcopy(model),'dataset':copy.deepcopy(dataset),'collection':collection,
        'query':query,'offset':offset,'limit':limit,'total':len(rows),'source_total':len(values),
        'record_count':len(observations['records']),'rows':[],'pinned':pinned,
        'missing_pin':pin if pin and pinned is None else None,'requested_pin':pin,'next_offset':None,
        'shape_limit':SHAPE_LIMIT,'name_limit':NAME_LIMIT,'byte_limit':science_browser.MAX_BYTES}
    for index,value in rows[offset:offset+limit]:
        item=_row(value,index,collection);end=offset+len(result['rows'])+1
        prospective={**result,'rows':[*result['rows'],item],'next_offset':end if end<len(rows) else None}
        if len(science_browser.encoded(prospective))>science_browser.MAX_BYTES:break
        result['rows'].append(item)
    end=offset+len(result['rows']);result['next_offset']=end if end<len(rows) else None
    _require(result['rows'] or offset>=len(rows),'Task suggestion cannot fit its page.')
    return science_browser._bounded(result)
