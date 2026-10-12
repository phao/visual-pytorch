"""Concrete recurrent graph and .vtw round-trip; PYTHONPATH=. python this_file."""
import tempfile
from pathlib import Path
from app.core import Workspace, compile_model, default_model, generate_code, validate_model
from examples.recurrent_transition import transition_example


def graph_example():
    models, cell, _ = transition_example()
    root = default_model('Concrete sequence'); root['id'] = 'concrete-sequence'
    root['graph'] = {
        'input_order': ['sequence', 'initial'],
        'nodes': [
            {'id': 'sequence', 'type': 'Input', 'shape': [5, 2], 'label': 'observations'},
            {'id': 'initial', 'type': 'Input', 'shape': [2], 'label': 'initial hidden'},
            {'id': 'scan', 'type': 'Recurrent', 'ref': cell,
             'sequence_inputs': [{'input': 'observation', 'source': 'sequence'}],
             'states': [{'input': 'previous', 'output': 'next', 'port': 'hidden', 'source': 'initial'}],
             'sequence_outputs': [{'output': 'next', 'port': 'trajectory'}]},
            {'id': 'trajectory', 'type': 'Output', 'label': 'trajectory'},
            {'id': 'hidden', 'type': 'Output', 'label': 'final hidden'},
        ],
        'edges': [{'from': 'sequence', 'to': 'scan'}, {'from': 'initial', 'to': 'scan'},
                  {'from': 'scan', 'from_port': 'trajectory', 'to': 'trajectory'},
                  {'from': 'scan', 'from_port': 'hidden', 'to': 'hidden'}],
    }
    models[root['id']] = root
    return models, root['id']


def main():
    import torch
    torch.set_num_threads(1); torch.manual_seed(7)
    models, mid = graph_example()
    workspace = Workspace()
    try:
        for doc in models.values(): workspace.put(doc)
        frozen = workspace.freeze_model(mid)
        print('ModelRevision:', workspace.get_spec('model', frozen['id'], frozen['revision'])['version'])
        # Only an isolated disposable fixture is written; no user workspace changes.
        with tempfile.TemporaryDirectory(prefix='vpw-sequence-example-') as directory:
            path = Path(directory) / 'example.vtw'; workspace.save(path)
            reopened = Workspace.load(path)
            try: models = reopened.models()
            finally: reopened.db.close()
        model = compile_model(models, mid); model.eval()
        with torch.no_grad(): result = model({'observations': torch.randn(3, 5, 2), 'initial hidden': torch.zeros(3, 2)})
        for name, value in result.items(): print(name + ':', tuple(value.shape))
        print('Parameters:', validate_model(models, mid)['parameters'])
        print('Standalone source bytes:', len(generate_code(models, mid).encode('utf-8')))
    finally: workspace.db.close()


if __name__ == '__main__': main()
