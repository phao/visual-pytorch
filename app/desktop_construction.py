"""Qt bridge access to the same pure recipe/planning services as the CLI."""
from .datasets import _keys
from .construction_recipes import catalogue,show,build
from .edit_planning import preview,admit

ACTIONS=frozenset({'templateList','templateShow','previewTemplate','instantiateTemplate','previewEdits','applyEdits'})
EDITS=frozenset({'instantiateTemplate','applyEdits'})


def dispatch(window,req):
    action=req['action']
    if action=='templateList':
        _keys(req,('action',),context='Template catalogue request');return catalogue()
    if action=='templateShow':
        _keys(req,('action','name'),context='Template declaration request');return show(req['name'])
    if window.workspace is None:raise ValueError('Open a workspace before constructing or planning edits.')
    if action in ('previewTemplate','instantiateTemplate'):
        _keys(req,('action','name','id'),('options','expected_snapshot'),context='Template construction request')
        edits=build(req['name'],model_id=req['id'],options=req.get('options'))
        options={'expected_snapshot':req.get('expected_snapshot')} if action in EDITS else {}
    else:
        _keys(req,('action','edits'),('allow_invalid','expected_snapshot'),context='Edit plan request')
        edits=req['edits'];options={'allow_invalid':req.get('allow_invalid',False)}
        if action in EDITS:options['expected_snapshot']=req.get('expected_snapshot')
    result=(admit if action in EDITS else preview)(window.workspace,edits,**options)
    if action in EDITS:window.broadcast()
    if action in ('previewTemplate','instantiateTemplate'):result['recipe']=show(req['name']);result['id']=req['id']
    return result
