"""Create a small explicit projected/bidirectional LSTM drawing (no Torch).

Run from the repository root: python examples/projected_lstm.py example.vtw
Initial hidden/cell states and optional lengths are data, not persistent state.
"""
import argparse
import sys
from pathlib import Path

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core import Workspace, default_model, validate_model


def models(*, steps=5, width=3, hidden=7, projection=4, layers=2, bidirectional=True, bias=True, lengths=True):
    ld = layers * (2 if bidirectional else 1)
    root = default_model('Projected bidirectional LSTM'); root['id'] = 'lstm-example'
    nodes = [
        {'id': 'sequence', 'type': 'Input', 'label': 'Frames', 'shape': [steps, width], 'axes': ['T', 'F'], 'x': 64, 'y': 80},
        {'id': 'hidden', 'type': 'Input', 'label': 'Initial hidden', 'shape': [ld, projection or hidden], 'x': 64, 'y': 190},
        {'id': 'cell', 'type': 'Input', 'label': 'Initial cell', 'shape': [ld, hidden], 'x': 64, 'y': 300},
        {'id': 'lstm', 'type': 'LSTM', 'label': 'Projected LSTM', 'hidden_size': hidden, 'proj_size': projection,
         'num_layers': layers, 'bidirectional': bidirectional, 'bias': bias,
         'inputs': {role: role for role in ('sequence', 'hidden', 'cell')},
         'x': 340, 'y': 160, 'height': 190},
        *[{'id': 'out_' + port, 'type': 'Output', 'label': 'Returned ' + port, 'x': 630, 'y': y}
          for port, y in zip(('sequence', 'hidden', 'cell'), (80, 210, 340))],
    ]
    edges = [{'from': role, 'to': 'lstm'} for role in ('sequence', 'hidden', 'cell')]
    edges += [{'from': 'lstm', 'from_port': port, 'to': 'out_' + port} for port in ('sequence', 'hidden', 'cell')]
    if lengths:
        nodes.append({'id': 'lengths', 'type': 'Input', 'label': 'Valid lengths', 'shape': [1], 'x': 64, 'y': 410})
        nodes[3]['lengths'] = {'source': 'lengths', 'adapter': 'exact-int64-v1'}
        edges.append({'from': 'lengths', 'to': 'lstm'})
    root['graph'] = {'nodes': nodes, 'edges': edges, 'input_order': [n['id'] for n in nodes if n['type'] == 'Input']}
    return {root['id']: root}, root['id']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('workspace', type=Path)
    args = parser.parse_args()
    if args.workspace.exists():
        parser.error('Refusing to overwrite an existing workspace.')
    collection, mid = models()
    contract = validate_model(collection, mid)
    workspace = Workspace()
    try:
        workspace.put(collection[mid]); workspace.freeze_model(mid); workspace.save(args.workspace)
    finally:
        workspace.db.close()
    print('Created:', args.workspace)
    print('Learned parameters:', contract['parameters'])
    print('Returned port sample shapes:', contract['shapes']['lstm']['ports'])
    print('Initial states and lengths are explicit; synthetic legacy data is not a length-aware dataset.')


if __name__ == '__main__':
    main()
