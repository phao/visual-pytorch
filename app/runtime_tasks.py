"""Explicit spatial-runtime task proofs/case planning; no Torch on import.

Frozen model shapes remain the constructor baseline. Dataset fields determine an
actual rectangular execution case only inside an opted-in runtime task contract.
"""
import copy
import math

from .core import GraphError, MAX_INTEGER, validate_model
from .datasets import _keys, _require, _pipeline, _value
from .evidence import digest
from .runtime_shapes import specification, shape_contract


def proof(family, specialization, model, *, execution='runtime-spatial-v1'):
    """Closed frozen provenance; no lookup of mutable heads or inferred bindings."""
    from .family_records import validate_specialization_dependencies
    from .revisions import frozen_document
    f=frozen_document('family',family);s=frozen_document('specialization',specialization)
    m=frozen_document('model',model)
    _require(execution in ('runtime-spatial-v1','runtime-temporal-v1','runtime-temporal-v2'),
             'Unknown explicit runtime task execution policy.')
    if execution=='runtime-temporal-v2':
        _require((f['version'],s['version']) in ((8,7),(9,8),(10,9),(11,10)) and s['runtime'] is not None,
                 'Temporal-v2 execution needs exact family8 and a nonnull specialization7 witness.')
    else:
        temporal=execution=='runtime-temporal-v1'
        _require(f['version']==(6 if temporal else 5) and s['version']==(5 if temporal else 4) and s['runtime'] is not None,
                 'Temporal runtime execution needs family6 and a nonnull specialization5 witness.' if temporal else
                 'Runtime task execution needs an exact family5 declaration and specialization4 witness.')
    records={('family',f['id'],f['revision']):f,('model',m['id'],m['revision']):m}
    def resolve(kind,sid,revision):
        _require((kind,sid,revision) in records,'Runtime task references another frozen family/model.')
        return records[(kind,sid,revision)]
    validate_specialization_dependencies(s,resolve)
    return {'family':f,'specialization':s}


def contract(model,dataset,task,runtime, *, execution=None):
    """Plan the dataset's actual shapes, then ordinary task checks validate meaning."""
    _keys(runtime,('family','specialization'),context='Runtime task provenance')
    if execution is None:
        sv=runtime['specialization'].get('version')
        execution='runtime-temporal-v2' if sv in (7,8,9,10) else 'runtime-temporal-v1' if sv==5 else 'runtime-spatial-v1'
    value=proof(runtime['family'],runtime['specialization'],model,execution=execution)
    _require(digest(value)==digest(runtime),'Runtime task provenance must retain exact frozen revisions.')
    models=model['models'];root=model['root'];decl=value['specialization']['runtime']['inputs']
    temporal=execution in ('runtime-temporal-v1','runtime-temporal-v2')
    if execution=='runtime-temporal-v2':
        from .temporal_shapes_v2 import specification as make_spec, shape_contract as plan
    elif temporal:
        from .temporal_shapes import specification as make_spec, shape_contract as plan
    else:make_spec,plan=specification,shape_contract
    spec=make_spec(models,root,decl);fields={f['id']:f for f in dataset['fields']}
    baseline=validate_model(models,root);shapes={g['id']:g['shape'] for g in baseline['input_groups']}
    for binding in task['inputs']:
        if binding.get('port') not in decl:continue
        _require(binding.get('field') in fields and 'derived' not in binding,
                 'Runtime ports need explicit numeric field bindings, not inferred or derived length/mask inputs.')
        field=fields[binding['field']]
        _require(field['dtype'] in ('float32','int64','bool') and
                 (1<=len(field['shape'])<=8 if temporal else len(field['shape'])==3 and all(v>0 for v in field['shape'])),
                 'Temporal fields need numeric sample tensors and explicit rectangular collation.' if temporal else
                 'Runtime image fields need a fixed rectangular numeric CHW shape; ragged collation requires a separate policy.')
        shape,_,_= _pipeline(field,binding['transforms'])
        shapes[binding['port']]=shape
    return plan(models,root,spec,shapes)


def input_policy(bundle):
    model=bundle['model'];baseline=validate_model(model['models'],model['root'])
    if bundle['version'] in (3,4):
        kind='runtime-temporal-v2' if bundle['version']==4 else 'runtime-temporal-v1'
        actual=contract(model,bundle['dataset'],bundle['task'],bundle['runtime'],execution=kind)
        return {'kind':kind,'runtime_temporal':copy.deepcopy(bundle['runtime']['specialization']['runtime']['inputs']),
                'baseline_input_groups':copy.deepcopy(baseline['input_groups']),
                'fitted_input_shapes':{g['id']:g['shape'] for g in actual['input_groups']}}
    return {'runtime_inputs':copy.deepcopy(bundle['runtime']['specialization']['runtime']['inputs']),
            'baseline_input_groups':copy.deepcopy(baseline['input_groups'])}


def record_shapes(input_spec,records):
    """Private rectangular actual-case shapes, checked before tensor allocation.

    Runtime inference with fitted per-feature preprocessing is deliberately denied:
    a different image size must not silently reinterpret a saved feature vector.
    """
    _require(isinstance(records,list) and records,'Runtime inference needs a nonempty rectangular record batch.')
    policy=input_spec['runtime_policy'];temporal=policy.get('kind') in ('runtime-temporal-v1','runtime-temporal-v2')
    decl=policy['runtime_temporal'] if temporal else policy['runtime_inputs']
    groups={g['id']:g for g in policy['baseline_input_groups']}
    fields={f['id']:f for f in input_spec['dataset']['fields']}
    shapes={g['id']:list(g['shape']) for g in groups.values()}
    def shape(value, depth=0):
        _require(depth<(8 if temporal else 3),'Runtime sample rank exceeds the declared tensor boundary.')
        _require(isinstance(value,list) and value,'Runtime image sample axes must be positive rectangular lists.')
        tail=shape(value[0],depth+1) if isinstance(value[0],list) else []
        for v in value:
            _require((isinstance(v,list) and shape(v,depth+1)==tail) if tail else not isinstance(v,list),
                     'Runtime image samples must be rectangular; ragged batches are not implicit padding.')
        return [len(value),*tail]
    for binding in input_spec['task']['inputs']:
        pid=binding['port']
        if pid not in decl:continue
        fid=binding['field'];field=fields[fid]
        _require(all(isinstance(r,dict) and fid in r for r in records),'Inference record is missing input field '+fid+'.')
        if temporal and binding['transforms']:
            for row in records:_value(row[fid],field)
            shapes[pid]=list(policy['fitted_input_shapes'][pid])
            continue
        values=[shape(r[fid]) for r in records]
        actual=values[0];fixed=groups[pid]['shape']
        _require(len(actual)==(len(groups[pid]['shape']) if temporal else 3) and all(v==actual for v in values),
                 'Each runtime port needs one common rectangular sample shape per batch; padding must be explicit.')
        _require(math.prod(actual)<=MAX_INTEGER and all(i in decl[pid] or v==fixed[i] for i,v in enumerate(actual)),
                 'Runtime inference changed a fixed channel/spatial dimension.')
        _require(not binding['transforms'] or actual==field['shape'],
                 'Changing image resolution with fitted preprocessing requires an explicit resolution-independent adapter.')
        shapes[pid]=actual
    return shapes
