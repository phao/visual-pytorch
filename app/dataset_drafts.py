"""Typed dataset draft projections/edits; not partial persisted specifications.

Wire/DOM bounds are independent from complete host storage and final validation.
Large/unsafe-for-Number values remain raw JSON text or explicit staged targets.
"""
import json

from .datasets import DTYPES, _identifier, _integer, _keys, _require
from .revisions import _parse
from . import science_browser

FIELD_KEYS = ('name', 'dtype', 'shape', 'axes', 'nullable', 'vocabulary', 'modality', 'unit', 'coordinates')
MAX_TEXT = 256


def _fits(value, budget=MAX_TEXT):
    """Cheap conservative eligibility before encoding a small complete JSON value.

No full large-string/array encoding just to discover it exceeds a preview limit.
This is a bounded walk, not a whole-object size calculation.
"""
    stack = [value]
    while stack and budget >= 0:
        item = stack.pop()
        if isinstance(item, dict):
            budget -= 2 + 4*len(item)
            if budget < 0: return False
            stack.extend(item.keys()); stack.extend(item.values())
        elif isinstance(item, list):
            budget -= 2 + len(item)
            if budget < 0: return False
            stack.extend(item)
        elif isinstance(item, str): budget -= 2 + 6*len(item)
        elif type(item) is int:
            if item.bit_length() > 1024: return False
            budget -= len(str(item))
        elif type(item) is float: budget -= 32
        elif item is None or type(item) is bool: budget -= 5
        else: return False
    return budget >= 0


def _descriptor(value, path, *, present=True, key=None, limit=MAX_TEXT):
    result = {'path':path, 'present':present, 'text':None, 'summary':None}
    if key is not None: result['key'] = key
    if not present: return result
    if _fits(value,limit):
        text = json.dumps(value,ensure_ascii=False,allow_nan=False,separators=(',',':'))
        if len(text) <= limit: result['text'] = text
    if result['text'] is None:
        kind = 'array' if isinstance(value,list) else 'object' if isinstance(value,dict) else 'string' if isinstance(value,str) else 'scalar'
        result['summary'] = {'type':kind, 'size':len(value) if isinstance(value,(dict,list,str)) else None,
            'preview':value[:64] if isinstance(value,str) else None}
    return result


def _ordinal(document,key):
    _require(isinstance(document,dict) and key in document,'Dataset draft property is missing: '+key)
    return list(document).index(key)


def _containers(document):
    fields=document.get('fields'); records=document.get('records')
    _require(isinstance(fields,list) and isinstance(records,list),'Dataset draft needs field/record arrays for typed editing.')
    return fields,records


def _property(container,key,path,limit=MAX_TEXT):
    return _descriptor(container.get(key),path+([list(container).index(key)] if key in container else []),
        present=key in container,key=key,limit=limit)


def _field_property(field,key,path,limit=MAX_TEXT):
    if key=='vocabulary' and isinstance(field.get(key),dict) and 'values' in field[key]:
        return _descriptor(field[key]['values'],path+[_ordinal(field,key),_ordinal(field[key],'values')],key=key,limit=limit)
    return _property(field,key,path,limit)


def page(document, header, *, field_offset=0, row_offset=0, field_limit=6, row_limit=20):
    _require(_integer(field_offset) and _integer(row_offset) and _integer(field_limit,1,6) and
        _integer(row_limit,1,20),'Typed dataset page needs exact nonnegative offsets, 1..6 fields and 1..20 observations.')
    fields,records=_containers(document)
    fi=_ordinal(document,'fields'); ri=_ordinal(document,'records')
    result={'version':1,'scope':'owned-dataset-page','session':header['session'],'sequence':header['sequence'],
        'workspace_id':header['workspace_id'],'id':header['id'],'name':header['name'],
        'field_offset':field_offset,'row_offset':row_offset,'field_total':len(fields),'row_total':len(records),
        'field_limit':field_limit,'row_limit':row_limit,'next_field_offset':None,'next_row_offset':None,
        'fields':[],'rows':[],'metadata':[_property(document,key,[]) for key in ('provenance','usage')],
        'byte_limit':science_browser.MAX_BYTES,'text_limit':MAX_TEXT}
    for index in range(field_offset,min(len(fields),field_offset+field_limit)):
        field=fields[index]
        _require(isinstance(field,dict) and _identifier(field.get('id')),'Typed field needs its stable ID.')
        item={'ordinal':index,'id':field['id'],'path':[fi,index],
            'properties':[_field_property(field,key,[fi,index]) for key in FIELD_KEYS]}
        prospective={**result,'fields':[*result['fields'],item]}
        # Reserve space for at least one complete observation. If necessary,
        # mark optional long properties for explicit raw editing, never truncate.
        if len(science_browser.encoded(prospective))>32768:
            item['properties']=[_field_property(field,key,[fi,index],limit=64) for key in FIELD_KEYS]
            prospective={**result,'fields':[*result['fields'],item]}
        if len(science_browser.encoded(prospective))>32768 and result['fields']: break
        _require(len(science_browser.encoded(prospective))<=32768,'Typed field projection exceeds its reserved byte budget.')
        result['fields'].append(item)
    end_field=min(len(fields),field_offset+len(result['fields']))
    if end_field<len(fields): result['next_field_offset']=end_field
    for index in range(row_offset,min(len(records),row_offset+row_limit)):
        record=records[index]
        _require(isinstance(record,dict) and _identifier(record.get('id')) and isinstance(record.get('values'),dict),
            'Typed observation needs its stable ID and values object.')
        vp=[ri,index,_ordinal(record,'values')]
        item={'ordinal':index,'id':record['id'],'path':[ri,index],
            'cells':[_property(record['values'],field['id'],vp) for field in result['fields']]}
        prospective={**result,'rows':[*result['rows'],item]}
        if len(science_browser.encoded(prospective))>science_browser.MAX_BYTES:
            item['cells']=[_property(record['values'],field['id'],vp,limit=64) for field in result['fields']]
            prospective={**result,'rows':[*result['rows'],item]}
        if len(science_browser.encoded(prospective))>science_browser.MAX_BYTES and result['rows']: break
        _require(len(science_browser.encoded(prospective))<=science_browser.MAX_BYTES,'Typed observation projection exceeds its byte budget.')
        result['rows'].append(item)
    end_row=min(len(records),row_offset+len(result['rows']))
    if end_row<len(records): result['next_row_offset']=end_row
    return science_browser._bounded(result)


def _find(items,sid,kind):
    _require(_identifier(sid),'Use an exact stable '+kind+' ID.')
    matches=[i for i,item in enumerate(items) if isinstance(item,dict) and item.get('id')==sid]
    _require(len(matches)==1,'Stable '+kind+' ID is missing or ambiguous.')
    return matches[0]


def locate(document,header,target):
    _require(isinstance(target,dict),'Choose a typed value target.')
    kind=target.get('kind');fields,records=_containers(document)
    if kind=='metadata':
        _keys(target,('kind','key'),context='Dataset metadata target')
        _require(target['key'] in ('provenance','usage'),'Choose provenance or usage.')
        descriptor=_property(document,target['key'],[])
    elif kind=='field':
        _keys(target,('kind','field','key'),context='Dataset field target')
        _require(target['key'] in FIELD_KEYS,'Choose a field property.')
        index=_find(fields,target['field'],'field')
        descriptor=_field_property(fields[index],target['key'],[_ordinal(document,'fields'),index])
    elif kind=='cell':
        _keys(target,('kind','field','record'),context='Dataset cell target')
        _find(fields,target['field'],'field'); index=_find(records,target['record'],'observation')
        descriptor=_property(records[index]['values'],target['field'],[_ordinal(document,'records'),index,_ordinal(records[index],'values')])
    else:raise ValueError('Unsupported typed dataset target.')
    _require(descriptor['present'],'Add the missing typed value explicitly before staging it.')
    return science_browser._bounded({'version':1,'scope':'owned-dataset-value','session':header['session'],
        'sequence':header['sequence'],'target':dict(target),**descriptor,'byte_limit':science_browser.MAX_BYTES})


def _text(text):
    _require(isinstance(text,str) and len(text)<=2000,'Typed edit text is limited to 2000 code points; use staged value editing for larger values.')
    return _parse(text)


def edit(document, operation):
    _require(isinstance(operation,dict) and isinstance(operation.get('op'),str),'Choose a typed dataset edit.')
    kind=operation['op']; fields,records=_containers(document); result=dict(document)
    if kind=='name':
        _keys(operation,('op','text'),context='Dataset name edit')
        result['name']=_text(operation['text']); return result
    if kind=='metadata':
        _keys(operation,('op','key','text'),context='Dataset metadata edit')
        _require(operation['key'] in ('provenance','usage'),'Choose provenance or usage.')
        if operation['text']=='':result.pop(operation['key'],None)
        else:result[operation['key']]=_text(operation['text'])
        return result
    if kind=='field-add':
        _keys(operation,('op','id'),context='Add dataset field')
        sid=operation['id'];_require(_identifier(sid) and not any(f.get('id')==sid for f in fields),'Use a unique stable field ID.')
        result['fields']=[*fields,{'id':sid,'dtype':'float32','shape':[]}]
        result['records']=[{**row,'values':{**row['values'],sid:None}} for row in records]; return result
    if kind=='field-remove':
        _keys(operation,('op','field'),context='Remove dataset field')
        index=_find(fields,operation['field'],'field'); sid=operation['field']
        result['fields']=[f for i,f in enumerate(fields) if i!=index]
        result['records']=[{**row,'values':{k:v for k,v in row['values'].items() if k!=sid}} for row in records]; return result
    if kind=='record-add':
        _keys(operation,('op','id'),context='Add dataset observation')
        sid=operation['id'];_require(_identifier(sid) and not any(r.get('id')==sid for r in records),'Use a unique stable observation ID.')
        result['records']=[*records,{'id':sid,'values':{f['id']:None for f in fields}}]; return result
    if kind=='record-remove':
        _keys(operation,('op','record'),context='Remove dataset observation')
        index=_find(records,operation['record'],'observation')
        result['records']=[r for i,r in enumerate(records) if i!=index]; return result
    if kind=='field':
        _keys(operation,('op','field','key','text'),context='Dataset field property edit')
        index=_find(fields,operation['field'],'field'); key=operation['key']
        _require(key in FIELD_KEYS,'Typed fields preserve IDs; choose a supported field property.')
        field=dict(fields[index]); text=operation['text']
        if text=='':
            _require(key not in ('dtype','shape'),'Dtype and sample shape are required.');field.pop(key,None)
        else:
            value=_text(text)
            if key=='vocabulary':value={'version':1,'values':value}
            if key=='dtype':
                _require(isinstance(value,str) and value in DTYPES,'Unsupported field dtype.')
                if value!='category':field.pop('vocabulary',None)
                elif 'vocabulary' not in field:field['vocabulary']={'version':1,'values':[]}
            field[key]=value
        result['fields']=list(fields);result['fields'][index]=field; return result
    if kind=='cell':
        _keys(operation,('op','field','record','text'),context='Dataset cell edit')
        _find(fields,operation['field'],'field'); index=_find(records,operation['record'],'observation')
        row=records[index];result['records']=list(records)
        result['records'][index]={**row,'values':{**row['values'],operation['field']:_text(operation['text'])}}
        return result
    raise ValueError('Unsupported typed dataset edit.')
