"""Small explicit-state graph example; run from the repository with PYTHONPATH=."""
from app.core import default_model
from app.recurrent import compile_recurrent_transition, generate_recurrent_code


def transition_example():
    cell = default_model('Tanh recurrent transition'); cell['id'] = 'tanh-transition'
    cell['graph'] = {
        'input_order': ['observation', 'previous'],
        'nodes': [
            {'id': 'observation', 'type': 'Input', 'features': 2, 'x': 60, 'y': 70},
            {'id': 'previous', 'type': 'Input', 'features': 2, 'x': 60, 'y': 220},
            {'id': 'joined', 'type': 'Concat', 'sources': ['observation', 'previous'],
             'flatten': True, 'x': 280, 'y': 150},
            {'id': 'affine', 'type': 'Linear', 'features': 2, 'bias': True, 'x': 500, 'y': 150},
            {'id': 'nonlinear', 'type': 'Activation', 'activation': 'Tanh', 'x': 720, 'y': 150},
            {'id': 'next', 'type': 'Output', 'x': 940, 'y': 150},
        ],
        'edges': [{'from': a, 'to': b} for a, b in [
            ('observation', 'joined'), ('previous', 'joined'), ('joined', 'affine'),
            ('affine', 'nonlinear'), ('nonlinear', 'next')]],
    }
    spec = {'version': 1, 'sequence_inputs': ['observation'],
            'states': [{'input': 'previous', 'output': 'next', 'port': 'hidden'}],
            'sequence_outputs': [{'output': 'next', 'port': 'trajectory'}]}
    return {cell['id']: cell}, cell['id'], spec


def main():
    import torch
    torch.set_num_threads(1); torch.manual_seed(7)
    network = compile_recurrent_transition(*transition_example())
    network.eval()
    observations = torch.randn(3, 5, 2)
    initial = torch.zeros(3, 2)
    lengths = torch.tensor([0, 3, 5], dtype=torch.int64)
    with torch.no_grad():
        result = network({'observation': observations}, {'hidden': initial}, lengths=lengths)
    print('Sequence shape:', tuple(result['sequences']['trajectory'].shape))
    print('Final state shape:', tuple(result['state']['hidden'].shape))
    print('Learnable parameters:', sum(p.numel() for p in network.parameters()))
    print('Zero-length state unchanged:', torch.equal(result['state']['hidden'][0], initial[0]))
    # Ordinary PyTorch optimizer/loss code is possible. A production trainer must
    # define masked loss/counts and skip all-invalid optimizer updates explicitly.
    source = generate_recurrent_code(*transition_example())
    print('Standalone source bytes:', len(source.encode('utf-8')))


if __name__ == '__main__': main()
