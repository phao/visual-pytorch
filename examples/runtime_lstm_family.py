"""Create an explicit Family8 runtime-T LSTM with supplied states and lengths.

The time symbol chooses the baseline drawing, not the runtime T. Input/state
constructor widths stay fixed for each specialization; all runtime inputs remain
named. This example does not create a dataset, task, optimizer or hidden state.
"""
import argparse
import json
import sys
from pathlib import Path

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core import Workspace
from app.family_records import retain
from examples.lstm_family import definition as fixed_definition


def definition():
    doc = fixed_definition()
    doc.update(version=8, id='runtime-lstm-family', name='Runtime-T LSTM constructor family',
               runtime_policy={'version':2, 'kind':'runtime-temporal-v2', 'inputs':{'sequence':[0]}})
    # A fixed Output annotation is intentionally not a variable-time contract.
    next(n for n in doc['models'][doc['root']]['graph']['nodes'] if n['id']=='out_sequence').pop('shape')
    doc['assignments'] = [a for a in doc['assignments'] if a['node'] != 'out_sequence']
    return doc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('workspace',type=Path)
    parser.add_argument('--bindings',type=Path)
    args = parser.parse_args()
    if args.workspace.exists(): parser.error('Refusing to overwrite an existing workspace.')
    from app.revisions import _parse
    bindings = None if args.bindings is None else _parse(args.bindings.read_text())
    workspace = Workspace()
    try:
        result = retain(workspace,family=definition(),bindings=bindings,materialize=True)
        workspace.save(args.workspace)
        print(json.dumps({'workspace':str(args.workspace),'family':result['family'],
                          'specialization':result['specialization'],'model':result['model'],
                          'parameters':result['resolved']['contract']['parameters'],
                          'runtime':result['resolved']['runtime'],
                          'status':'runtime-v2-family-retained; not-trained; no-dataset-task-or-experiment-created'}))
    finally:
        workspace.db.close()


if __name__=='__main__': main()
