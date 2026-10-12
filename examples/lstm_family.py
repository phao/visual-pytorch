"""Create a checked LSTM architecture family and its concrete specialization.

Pure stdlib host; no Torch/Qt or training. Each fixed-T binding explicitly
derives hidden/cell/sequence shapes from the same constructor symbols.
"""
import argparse
import copy
import json
import sys
from pathlib import Path

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core import Workspace, validate_model
from app.family_records import retain
from examples.projected_lstm import models


def definition():
    templates, root = models()
    contract = validate_model(templates, root)
    for node in templates[root]['graph']['nodes']:
        if node['type'] == 'Output':
            node['shape'] = copy.deepcopy(contract['shapes'][node['id']]['output_shape'])
    source = {'version': 7, 'id': 'lstm-family', 'name': 'Explicit LSTM constructor family',
              'root': root, 'models': {k: {p: copy.deepcopy(m[p]) for p in ('id','name','graph')}
                                      for k,m in templates.items()},
              'symbols': [], 'assignments': [], 'repeats': [], 'constructors': [], 'runtime_policy': None}
    for sid, name, default, minimum, maximum in (
        ('time','Concrete sequence length',5,1,32), ('width','Input features',3,1,32),
        ('cell-width','Cell width',7,1,32), ('layers','Stacked layers',2,1,4),
        ('projection','Projection width (0 means none)',4,0,31)):
        source['symbols'].append({'id':sid,'scope':root,'name':name,'default':default,
                                  'minimum':minimum,'maximum':maximum})
    for sid, name in (('directions','Bidirectional'), ('bias','Learn biases')):
        source['symbols'].append({'id':sid,'scope':root,'name':name,'type':'boolean','default':True})
    sym = lambda sid: {'symbol': sid}
    directions = {'select':sym('directions'),'then':2,'else':1}
    hidden = {'select':{'cmp':'gt','args':[sym('projection'),0]},'then':sym('projection'),'else':sym('cell-width')}
    state_rows = {'op':'mul','args':[sym('layers'),directions]}
    sequence_width = {'op':'mul','args':[directions,hidden]}
    for nid, field, value in (
        ('sequence','shape',[sym('time'),sym('width')]),
        ('hidden','shape',[state_rows,hidden]), ('cell','shape',[state_rows,sym('cell-width')]),
        ('lstm','hidden_size',sym('cell-width')), ('lstm','num_layers',sym('layers')),
        ('lstm','proj_size',sym('projection')), ('lstm','bidirectional',sym('directions')),
        ('lstm','bias',sym('bias')),
        ('out_sequence','shape',[sym('time'),sequence_width]),
        ('out_hidden','shape',[state_rows,hidden]), ('out_cell','shape',[state_rows,sym('cell-width')])):
        source['assignments'].append({'model':root,'node':nid,'field':field,'value':copy.deepcopy(value)})
    return source


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('workspace',type=Path)
    parser.add_argument('--bindings',type=Path,help='Exact JSON symbol-ID bindings; defaults otherwise')
    args=parser.parse_args()
    if args.workspace.exists():parser.error('Refusing to overwrite an existing workspace.')
    from app.revisions import _parse
    bindings=None if args.bindings is None else _parse(args.bindings.read_text())
    workspace=Workspace()
    try:
        result=retain(workspace,family=definition(),bindings=bindings,materialize=True)
        workspace.save(args.workspace)
        print(json.dumps({'workspace':str(args.workspace),'family':result['family'],
                          'specialization':result['specialization'],'model':result['model'],
                          'parameters':result['resolved']['contract']['parameters'],
                          'bindings':result['resolved']['bindings'],
                          'status':'concrete-family-retained; not-trained; states-and-task-heads-remain-explicit'}))
    finally:
        workspace.db.close()


if __name__=='__main__':main()
